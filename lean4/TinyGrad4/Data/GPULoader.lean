import TinyGrad4.Shape
import TinyGrad4.Data.Slice
import TinyGrad4.Data.Dataset
import TinyGrad4.Data.Prefetch
import TinyGrad4.Data.Shard
import TinyGrad4.Data.Lease
import TinyGrad4.Backend.DeviceBuffer
import TinyGrad4.Backend.Metal
import TinyGrad4.Backend.Cuda

/-!
# Heterogeneous GPU Data Loader

Multi-backend GPU data loading with:
- Unified DeviceId abstraction (Metal, CUDA, CPU)
- GPUBuffer with backend dispatch
- ByteSlice → GPU bridge with zero-copy support
- Double-buffered async loading via IOQueue
- Optional GPU buffer pool for reuse (lower alloc overhead)
- Multi-device sharding for distributed training

## Design Principles
- Reuse existing IOQueue from Prefetch (no new concurrency primitives)
- Transforms happen in tensor graph, loader just moves bytes to GPU
- Zero-copy on Apple Silicon, fallback copy elsewhere
-/

namespace TinyGrad4.Data.GPULoader

open TinyGrad4.Data
open TinyGrad4.Backend.DeviceBuffer

/-! ## ByteArray helpers -/

/-- Concatenate chunks into one ByteArray with a single allocation. -/
private def concatByteArraysWithTotal (chunks : Array ByteArray) (total : Nat) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity total
  let mut offset := 0
  for chunk in chunks do
    out := ByteArray.copySlice chunk 0 out offset chunk.size false
    offset := offset + chunk.size
  out

/-- Concatenate chunks into one ByteArray with a single allocation. -/
private def concatByteArrays (chunks : Array ByteArray) : ByteArray := Id.run do
  let total := chunks.foldl (fun acc b => acc + b.size) 0
  concatByteArraysWithTotal chunks total

/-! ## Device Identification -/

/-- Target device for data loading -/
inductive DeviceId where
  | cpu
  | metal
  | cuda (idx : Nat := 0)
  | tpu (idx : Nat := 0)
  deriving BEq, Hashable, Repr, Inhabited

namespace DeviceId

/-- Check device availability -/
def isAvailable : DeviceId → IO Bool
  | .cpu => pure true
  | .metal => TinyGrad4.Backend.Metal.isAvailable
  | .cuda _ => TinyGrad4.Backend.Cuda.isAvailable
  | .tpu _ => do
      let env1 ← IO.getEnv "TPU_VISIBLE_DEVICES"
      let env2 ← IO.getEnv "TPU_NAME"
      pure (env1.isSome || env2.isSome)

/-- Get device name -/
def name : DeviceId → IO String
  | .cpu => pure "CPU"
  | .metal => TinyGrad4.Backend.Metal.deviceInfo
  | .cuda idx => do
      let info ← TinyGrad4.Backend.Cuda.deviceInfoFor idx
      pure s!"{info}:{idx}"
  | .tpu idx => pure s!"TPU:{idx}"

/-- Sync device (wait for all operations to complete) -/
def sync : DeviceId → IO Unit
  | .cpu => pure ()
  | .metal => TinyGrad4.Backend.Metal.metalSync
  | .cuda idx => do
      TinyGrad4.Backend.Cuda.setDevice idx
      TinyGrad4.Backend.Cuda.cudaSync
  | .tpu _ => pure ()

instance : ToString DeviceId where
  toString
    | .cpu => "cpu"
    | .metal => "metal"
    | .cuda idx => s!"cuda:{idx}"
    | .tpu idx => s!"tpu:{idx}"

end DeviceId

/-! ## GPU Buffer Handle (Multi-Backend) -/

/-- Opaque handle to a GPU buffer on any backend -/
inductive GPUHandle where
  | metal (buf : TinyGrad4.Backend.Metal.MetalBuffer)
  | cuda (buf : TinyGrad4.Backend.Cuda.CUDABuffer)

namespace GPUHandle

instance : Repr GPUHandle where
  reprPrec h _ := match h with
    | .metal _ => "GPUHandle.metal(...)"
    | .cuda _ => "GPUHandle.cuda(...)"

end GPUHandle

/-- GPU memory allocation (owns device memory). -/
structure GPUAllocation where
  handle : GPUHandle
  byteSize : Nat
  device : DeviceId
  deriving Repr

namespace GPUAllocation

/-- Allocate raw bytes on device. -/
def alloc (device : DeviceId) (byteSize : Nat) : IO GPUAllocation := do
  match device with
  | .metal =>
      let buf ← TinyGrad4.Backend.Metal.metalAllocBytes byteSize
      pure { handle := .metal buf, byteSize, device }
  | .cuda idx =>
      TinyGrad4.Backend.Cuda.setDevice idx
      let buf ← TinyGrad4.Backend.Cuda.cudaAllocBytes byteSize
      pure { handle := .cuda buf, byteSize, device := .cuda idx }
  | .tpu _ =>
      throw (IO.userError "GPUAllocation.alloc: TPU backend not implemented")
  | .cpu =>
      throw (IO.userError "GPUAllocation.alloc: use RawBuffer for CPU data")

/-- Free allocation. -/
def free (a : GPUAllocation) : IO Unit := do
  match a.handle with
  | .metal buf => TinyGrad4.Backend.Metal.metalFree buf
  | .cuda buf => TinyGrad4.Backend.Cuda.cudaFree buf

/-- Copy from CPU ByteArray to GPU allocation. -/
def copyIn (a : GPUAllocation) (data : ByteArray) : IO Unit := do
  match a.handle with
  | .metal buf => TinyGrad4.Backend.Metal.metalCopyInBytes buf data
  | .cuda buf => TinyGrad4.Backend.Cuda.cudaCopyInBytes buf data

/-- Copy from GPU allocation to CPU ByteArray. -/
def copyOut (a : GPUAllocation) (byteSize : Nat) : IO ByteArray := do
  match a.handle with
  | .metal buf => TinyGrad4.Backend.Metal.metalCopyOutBytes buf byteSize
  | .cuda buf => TinyGrad4.Backend.Cuda.cudaCopyOutBytes buf byteSize

/-- Sync device for this allocation. -/
def sync (a : GPUAllocation) : IO Unit :=
  a.device.sync

end GPUAllocation

/-- Typed GPU buffer view (shape/dtype at type level). -/
structure GPUBuffer (shape : Shape) (dtype : DType) where
  allocation : GPUAllocation
  /-- Strides in elements (None = C-contiguous). -/
  strides : Option (List Nat) := none
  /-- Byte offset into allocation. -/
  byteOffset : Nat := 0
  deriving Repr

namespace GPUBuffer

/-- Total number of elements. -/
def numel (_buf : GPUBuffer shape dtype) : Nat :=
  Shape.numel shape

/-- Byte size of the view. -/
def bytes (_buf : GPUBuffer shape dtype) : Nat :=
  Shape.numel shape * dtype.itemsize

/-- Device where buffer resides. -/
def device (b : GPUBuffer shape dtype) : DeviceId :=
  b.allocation.device

/-- Allocate buffer on device for a typed shape/dtype. -/
def alloc (device : DeviceId) (shape : Shape) (dtype : DType := .float32) : IO (GPUBuffer shape dtype) := do
  let byteSize := Shape.numel shape * dtype.itemsize
  let alloc ← GPUAllocation.alloc device byteSize
  pure { allocation := alloc, strides := none, byteOffset := 0 }

/-- Free the underlying allocation. -/
def free (b : GPUBuffer shape dtype) : IO Unit :=
  b.allocation.free

/-- Copy from CPU ByteArray to GPU buffer (exact size). -/
def copyIn (b : GPUBuffer shape dtype) (data : ByteArray) : IO Unit := do
  let expected := b.bytes
  if data.size != expected then
    throw (IO.userError s!"GPUBuffer.copyIn: expected {expected} bytes, got {data.size}")
  b.allocation.copyIn data

/-- Copy from GPU to CPU ByteArray. -/
def copyOut (b : GPUBuffer shape dtype) : IO ByteArray :=
  b.allocation.copyOut b.bytes

/-- Sync the device this buffer is on. -/
def sync (b : GPUBuffer shape dtype) : IO Unit :=
  b.allocation.sync

/-- Ensure a buffer is on the expected device. -/
def assertDevice (b : GPUBuffer shape dtype) (device : DeviceId) : IO Unit := do
  if b.device != device then
    throw (IO.userError s!"GPUBuffer on {b.device}, expected {device}")

end GPUBuffer

abbrev GPULease (shape : Shape) (dtype : DType) := Lease (GPUBuffer shape dtype)

/-! ## ByteSlice → GPU Bridge -/

/-- Upload ByteSlice to GPU buffer with explicit shape/dtype (copies data). -/
def ByteSlice.toGPUBuffer (slice : ByteSlice) (device : DeviceId)
    (shape : Shape) (dtype : DType := .uint8) : IO (GPUBuffer shape dtype) := do
  let expected := Shape.numel shape * dtype.itemsize
  if slice.length != expected then
    throw (IO.userError s!"ByteSlice.toGPUBuffer: expected {expected} bytes, got {slice.length}")
  let buf ← GPUBuffer.alloc device shape dtype
  buf.copyIn slice.toByteArray
  pure buf

/-- Upload ByteSlice to GPU with zero-copy if possible.
    Zero-copy works on Apple Silicon Metal (unified memory).
    Falls back to copy on CUDA and other platforms. -/
def ByteSlice.toGPUBufferZeroCopy (slice : ByteSlice) (device : DeviceId)
    (shape : Shape) (dtype : DType := .uint8) : IO (GPUBuffer shape dtype) := do
  let expected := Shape.numel shape * dtype.itemsize
  if slice.length != expected then
    throw (IO.userError s!"ByteSlice.toGPUBufferZeroCopy: expected {expected} bytes, got {slice.length}")
  match device with
  | .metal =>
    -- Use Metal zero-copy via unified memory
    let buf ← TinyGrad4.Backend.Metal.metalWrapBytesNoCopy slice.parent slice.offset slice.length
    let alloc : GPUAllocation := { handle := .metal buf, byteSize := slice.length, device }
    pure { allocation := alloc, strides := none, byteOffset := 0 }
  | _ =>
    -- Fall back to copy
    ByteSlice.toGPUBuffer slice device shape dtype

/-- Upload ByteArray to GPU -/
def ByteArray.toGPUBuffer (data : ByteArray) (device : DeviceId)
    (shape : Shape) (dtype : DType := .uint8) : IO (GPUBuffer shape dtype) := do
  let expected := Shape.numel shape * dtype.itemsize
  if data.size != expected then
    throw (IO.userError s!"ByteArray.toGPUBuffer: expected {expected} bytes, got {data.size}")
  let buf ← GPUBuffer.alloc device shape dtype
  buf.copyIn data
  pure buf

/-! ## GPU Buffer Pool -/

/-- Pool of reusable GPU buffers (one allocation per slot). -/
structure GPUPool (shape : Shape) (dtype : DType) where
  device : DeviceId
  buffers : Array (GPUBuffer shape dtype)
  freeSlots : IOQueue Nat

namespace GPUPool

/-- Create a pool with `slots` pre-allocated buffers. -/
def create (device : DeviceId) (shape : Shape) (dtype : DType) (slots : Nat) : IO (GPUPool shape dtype) := do
  if slots == 0 then
    throw (IO.userError "GPUPool.create: slots must be > 0")
  let freeSlots ← IOQueue.new slots
  let mut buffers : Array (GPUBuffer shape dtype) := Array.mkEmpty slots
  for i in [:slots] do
    let buf ← GPUBuffer.alloc device shape dtype
    buffers := buffers.push buf
    let _ ← freeSlots.push i
  pure { device, buffers, freeSlots }

/-- Release a slot back to the pool. -/
def release (pool : GPUPool shape dtype) (idx : Nat) : IO Unit := do
  let _ ← pool.freeSlots.push idx
  pure ()

/-- Acquire a buffer lease from the pool (blocks until available or finished). -/
def acquire (pool : GPUPool shape dtype) : IO (Option (Nat × GPULease shape dtype)) := do
  match ← pool.freeSlots.pop with
  | some idx =>
      if h : idx < pool.buffers.size then
        let buf := pool.buffers[idx]'(h)
        let lease : GPULease shape dtype := { value := buf, release := pool.release idx }
        pure (some (idx, lease))
      else
        throw (IO.userError s!"GPUPool.acquire: invalid slot {idx}")
  | none => pure none

/-- Stop the pool (unblocks any waiting acquire). -/
def stop (pool : GPUPool shape dtype) : IO Unit :=
  pool.freeSlots.finish

/-- Free all buffers owned by the pool. -/
def freeAll (pool : GPUPool shape dtype) : IO Unit := do
  for buf in pool.buffers do
    buf.free

end GPUPool

/-! ## GPU Data Queue

Reuses `IOQueue` from Prefetch, but records iterator state with each buffer
so we can resume deterministically after interruption.
-/

/-- Queue item with checkpoint state after this batch. -/
structure QueuedBuffer (shape : Shape) (dtype : DType) where
  lease : GPULease shape dtype
  state : IteratorState

/-! ## Single-Device GPU Loader -/

/-- Background GPU data loader for a single device -/
structure GPUDataLoader (batch : Nat) (itemShape : Shape) (dtype : DType) where
  /-- Output queue of GPU buffers -/
  queue : IOQueue (QueuedBuffer (batch :: itemShape) dtype)
  /-- Worker task -/
  worker : Task (Except IO.Error Unit)
  /-- Target device -/
  device : DeviceId
  /-- Number of batches to produce -/
  numBatches : Nat
  /-- Last consumed iterator state for checkpointing -/
  lastState : IO.Ref IteratorState
  /-- Optional batch prefetcher (for staged CPU → GPU pipelines). -/
  prefetcher? : Option (BatchPrefetcher ByteArray)
  /-- Optional GPU buffer pool (reused allocations). -/
  pool? : Option (GPUPool (batch :: itemShape) dtype)

namespace GPUDataLoader

/-- Build one batch from an iterator, returning the state after the batch. -/
private def nextBatch (iter : DataIterator ByteArray) (batchSize : Nat) (itemBytes : Nat) :
    IO (Option (ByteArray × IteratorState)) := do
  let mut chunks := Array.mkEmpty batchSize
  for _ in [:batchSize] do
    match ← iter.next with
    | some item =>
        if item.size != itemBytes then
          throw (IO.userError s!"GPUDataLoader.nextBatch: expected {itemBytes} bytes, got {item.size}")
        chunks := chunks.push item
    | none => return none
  let state ← iter.checkpoint
  let batchData := concatByteArraysWithTotal chunks (batchSize * itemBytes)
  pure (some (batchData, state))

/-- Fetch one prefetched batch with its checkpoint state. -/
private def nextBatchFromPrefetcher (prefetcher : BatchPrefetcher ByteArray) :
    IO (Option (ByteArray × IteratorState)) := do
  match ← prefetcher.nextWithState with
  | some (batch, state) => pure (some (batch, state))
  | none => pure none

/-- Create loader from a pre-built iterator. -/
private def createFromIterator (iter : DataIterator ByteArray) (initState : IteratorState)
    (totalItems : Nat) (device : DeviceId) (batchSize : Nat) (itemShape : Shape)
    (dtype : DType) (bufferSize : Nat) (poolSlots : Nat) :
    IO (GPUDataLoader batchSize itemShape dtype) := do
  match device with
  | .cpu => throw (IO.userError "GPUDataLoader: use CPU datasets directly")
  | .tpu _ => throw (IO.userError "GPUDataLoader: use TPULoader for TPU devices")
  | _ => pure ()
  let itemBytes := Shape.numel itemShape * dtype.itemsize
  let batchShape : Shape := batchSize :: itemShape
  let queue ← IOQueue.new bufferSize
  let poolSlots' := if poolSlots == 0 then 0 else Nat.max poolSlots bufferSize
  let pool? ←
    if poolSlots' == 0 then
      pure none
    else
      some <$> GPUPool.create device batchShape dtype poolSlots'
  let lastState ← IO.mkRef initState
  let numBatches := totalItems / batchSize

  let worker ← IO.asTask (prio := .dedicated) do
    match device with
    | .cuda idx => TinyGrad4.Backend.Cuda.setDevice idx
    | _ => pure ()
    for _ in [:numBatches] do
      if ← IO.checkCanceled then break
      match ← nextBatch iter batchSize itemBytes with
      | some (batchData, state) =>
          let lease? ←
            match pool? with
            | some pool =>
                match ← pool.acquire with
                | some (_, lease) =>
                    lease.value.copyIn batchData
                    pure (some lease)
                | none => pure none
            | none =>
                let gpuBuf ← ByteArray.toGPUBuffer batchData device batchShape dtype
                let lease : GPULease _ _ := { value := gpuBuf, release := gpuBuf.free }
                pure (some lease)
          match lease? with
          | some lease =>
              lease.value.assertDevice device
              let ok ← queue.push { lease, state }
              if !ok then break
          | none => break
      | none => break
    queue.finish

  pure { queue, worker, device, numBatches, lastState, prefetcher? := none, pool? }

/-- Create loader from a pre-built iterator prefetcher (staged CPU → GPU). -/
private def createFromPrefetcher (prefetcher : BatchPrefetcher ByteArray) (initState : IteratorState)
    (totalItems : Nat) (device : DeviceId) (batchSize : Nat) (itemShape : Shape)
    (dtype : DType) (bufferSize : Nat) (poolSlots : Nat) :
    IO (GPUDataLoader batchSize itemShape dtype) := do
  match device with
  | .cpu => throw (IO.userError "GPUDataLoader: use CPU datasets directly")
  | .tpu _ => throw (IO.userError "GPUDataLoader: use TPULoader for TPU devices")
  | _ => pure ()
  let itemBytes := Shape.numel itemShape * dtype.itemsize
  let batchShape : Shape := batchSize :: itemShape
  let queue ← IOQueue.new bufferSize
  let poolSlots' := if poolSlots == 0 then 0 else Nat.max poolSlots bufferSize
  let pool? ←
    if poolSlots' == 0 then
      pure none
    else
      some <$> GPUPool.create device batchShape dtype poolSlots'
  let lastState ← IO.mkRef initState
  let numBatches := totalItems / batchSize

  let worker ← IO.asTask (prio := .dedicated) do
    match device with
    | .cuda idx => TinyGrad4.Backend.Cuda.setDevice idx
    | _ => pure ()
    for _ in [:numBatches] do
      if ← IO.checkCanceled then break
      match ← nextBatchFromPrefetcher prefetcher with
      | some (batchData, state) =>
          if batchData.size != batchSize * itemBytes then
            throw (IO.userError s!"GPUDataLoader.prefetch: expected {batchSize * itemBytes} bytes, got {batchData.size}")
          let lease? ←
            match pool? with
            | some pool =>
                match ← pool.acquire with
                | some (_, lease) =>
                    lease.value.copyIn batchData
                    pure (some lease)
                | none => pure none
            | none =>
                let gpuBuf ← ByteArray.toGPUBuffer batchData device batchShape dtype
                let lease : GPULease _ _ := { value := gpuBuf, release := gpuBuf.free }
                pure (some lease)
          match lease? with
          | some lease =>
              lease.value.assertDevice device
              let ok ← queue.push { lease, state }
              if !ok then break
          | none => break
      | none => break
    queue.finish

  pure { queue, worker, device, numBatches, lastState, prefetcher? := some prefetcher, pool? }

/-- Create loader from an iterator config (supports checkpoint/resume). -/
def createFromIteratorCfg [Dataset D ByteArray] (cfg : IteratorConfig D) (device : DeviceId)
    (batchSize : Nat) (itemShape : Shape) (dtype : DType := .float32) (bufferSize : Nat := 4)
    (poolSlots : Nat := 0) : IO (GPUDataLoader batchSize itemShape dtype) := do
  let iter ← Dataset.toIteratorCfg cfg
  let initState ← iter.checkpoint
  let n := Dataset.len cfg.base
  let remaining := remainingItems n cfg.epochs initState
  createFromIterator iter initState remaining device batchSize itemShape dtype bufferSize poolSlots

/-- Create loader from an iterator config with a CPU-side batch prefetcher.
    `prefetchSize` counts batches (not individual items). -/
def createFromIteratorCfgPrefetch [Dataset D ByteArray] (cfg : IteratorConfig D) (device : DeviceId)
    (batchSize : Nat) (itemShape : Shape) (prefetchSize : Nat := 8)
    (dtype : DType := .float32) (bufferSize : Nat := 4) (poolSlots : Nat := 0)
    : IO (GPUDataLoader batchSize itemShape dtype) := do
  let itemBytes := Shape.numel itemShape * dtype.itemsize
  let prefetcher ← BatchPrefetcher.createFromIteratorCfg cfg batchSize
    (fun chunks => pure (concatByteArraysWithTotal chunks (batchSize * itemBytes))) true prefetchSize
  let initState ← prefetcher.checkpoint
  let n := Dataset.len cfg.base
  let remaining := remainingItems n cfg.epochs initState
  createFromPrefetcher prefetcher initState remaining device batchSize itemShape dtype bufferSize poolSlots

/-- Create loader from a dataset of ByteArrays -/
def create [Dataset D ByteArray] (ds : D) (device : DeviceId)
    (batchSize : Nat) (itemShape : Shape) (dtype : DType := .float32)
    (bufferSize : Nat := 4) (poolSlots : Nat := 0) : IO (GPUDataLoader batchSize itemShape dtype) := do
  let cfg : IteratorConfig D := { base := ds }
  createFromIteratorCfg cfg device batchSize itemShape dtype bufferSize poolSlots

/-- Create loader from array of ByteSlices (zero-copy if possible) -/
def fromSlices (slices : Array ByteSlice) (device : DeviceId)
    (batchSize : Nat) (itemShape : Shape) (dtype : DType := .float32) (bufferSize : Nat := 4)
    : IO (GPUDataLoader batchSize itemShape dtype) := do
  match device with
  | .cpu => throw (IO.userError "GPUDataLoader.fromSlices: use CPU datasets directly")
  | .tpu _ => throw (IO.userError "GPUDataLoader.fromSlices: use TPULoader for TPU devices")
  | _ => pure ()
  let queue ← IOQueue.new bufferSize
  let lastState ← IO.mkRef { position := 0, epoch := 0, key := RandKey.new 0 }
  let batchShape : Shape := batchSize :: itemShape

  let worker ← IO.asTask (prio := .dedicated) do
    match device with
    | .cuda idx => TinyGrad4.Backend.Cuda.setDevice idx
    | _ => pure ()
    let mut pos := 0
    for slice in slices do
      if ← IO.checkCanceled then break
      let gpuBuf ← ByteSlice.toGPUBufferZeroCopy slice device batchShape dtype
      gpuBuf.assertDevice device
      pos := pos + 1
      let state : IteratorState := { position := pos, epoch := 0, key := RandKey.new 0 }
      let lease : GPULease _ _ := { value := gpuBuf, release := gpuBuf.free }
      let ok ← queue.push { lease, state }
      if !ok then break
    queue.finish

  pure { queue, worker, device, numBatches := slices.size, lastState, prefetcher? := none, pool? := none }

/-- Get next GPU buffer lease (blocks until available).
    Call `lease.release` when done (ForIn/forEach auto-release). -/
def next (loader : GPUDataLoader batch itemShape dtype) :
    IO (Option (GPULease (batch :: itemShape) dtype)) := do
  match ← loader.queue.pop with
  | some item =>
      loader.lastState.set item.state
      pure (some item.lease)
  | none => pure none

/-- Get next buffer and wait time spent in queue (ns). -/
def nextWithWait (loader : GPUDataLoader batch itemShape dtype) :
    IO (Option (GPULease (batch :: itemShape) dtype) × Nat) := do
  let (item?, waitNs) ← loader.queue.popWithWait
  match item? with
  | some item =>
      loader.lastState.set item.state
      pure (some item.lease, waitNs)
  | none => pure (none, waitNs)

/-- Stop the loader -/
def stop (loader : GPUDataLoader batch itemShape dtype) : IO Unit := do
  loader.queue.finish
  IO.cancel loader.worker
  match loader.prefetcher? with
  | some p => p.cancel
  | none => pure ()
  match loader.pool? with
  | some pool => pool.stop
  | none => pure ()

/-- Wait for loader to complete -/
def wait (loader : GPUDataLoader batch itemShape dtype) : IO Unit := do
  let _ ← IO.wait loader.worker
  match loader.prefetcher? with
  | some p => p.wait
  | none => pure ()

/-- Get checkpoint state for deterministic resume. -/
def checkpoint (loader : GPUDataLoader batch itemShape dtype) : IO IteratorState :=
  loader.lastState.get

/-- Drain any queued buffers (freeing GPU memory). -/
def drain (loader : GPUDataLoader batch itemShape dtype) : IO Unit := do
  repeat do
    match ← loader.queue.pop with
    | some item => item.lease.release
    | none => break
  match loader.prefetcher? with
  | some p => p.drain
  | none => pure ()
  match loader.pool? with
  | some pool => pool.freeAll
  | none => pure ()

/-- Iterate over all batches -/
def forEach (loader : GPUDataLoader batch itemShape dtype)
    (f : GPUBuffer (batch :: itemShape) dtype → IO Unit) : IO Unit := do
  repeat do
    match ← loader.next with
    | some lease => lease.withLease f
    | none => break

end GPUDataLoader

/-- Worker state for a single device -/
structure DeviceWorker (batch : Nat) (itemShape : Shape) (dtype : DType) where
  device : DeviceId
  shard : ShardConfig
  queue : IOQueue (QueuedBuffer (batch :: itemShape) dtype)
  worker : Task (Except IO.Error Unit)
  lastState : IO.Ref IteratorState
  prefetcher? : Option (BatchPrefetcher ByteArray)
  pool? : Option (GPUPool (batch :: itemShape) dtype)

/-- Pool of loaders across multiple GPUs -/
structure MultiGPULoader (batch : Nat) (itemShape : Shape) (dtype : DType) where
  workers : Array (DeviceWorker batch itemShape dtype)
  numBatchesPerDevice : Nat

namespace MultiGPULoader

/-- Build one batch from an iterator, returning the state after the batch. -/
private def nextBatch (iter : DataIterator ByteArray) (batchSize : Nat) (itemBytes : Nat) :
    IO (Option (ByteArray × IteratorState)) := do
  let mut chunks := Array.mkEmpty batchSize
  for _ in [:batchSize] do
    match ← iter.next with
    | some item =>
        if item.size != itemBytes then
          throw (IO.userError s!"MultiGPULoader.nextBatch: expected {itemBytes} bytes, got {item.size}")
        chunks := chunks.push item
    | none => return none
  let state ← iter.checkpoint
  let batchData := concatByteArraysWithTotal chunks (batchSize * itemBytes)
  pure (some (batchData, state))

/-- Fetch one prefetched batch with its checkpoint state. -/
private def nextBatchFromPrefetcher (prefetcher : BatchPrefetcher ByteArray) :
    IO (Option (ByteArray × IteratorState)) := do
  match ← prefetcher.nextWithState with
  | some (batch, state) => pure (some (batch, state))
  | none => pure none

/-- Initialization bundle for a device worker. -/
private structure WorkerInit where
  device : DeviceId
  shard : ShardConfig
  iter : DataIterator ByteArray
  initState : IteratorState
  shardBatches : Nat

private structure WorkerInitPrefetch where
  device : DeviceId
  shard : ShardConfig
  prefetcher : BatchPrefetcher ByteArray
  initState : IteratorState
  shardBatches : Nat

/-- Create multi-GPU loader from an iterator config (supports checkpoint/resume). -/
def createFromIteratorCfg [Dataset D ByteArray] (cfg : IteratorConfig D) (devices : Array DeviceId)
    (batchSize : Nat) (itemShape : Shape) (dtype : DType := .float32) (bufferSize : Nat := 4)
    (poolSlots : Nat := 0)
    (world? : Option WorldConfig := none)
    (states? : Option (Array (DeviceId × IteratorState)) := none) :
    IO (MultiGPULoader batchSize itemShape dtype) := do
  if devices.isEmpty then
    throw (IO.userError "MultiGPULoader: no devices specified")
  if devices.any fun d => d == .cpu || (match d with | .tpu _ => true | _ => false) then
    throw (IO.userError "MultiGPULoader: only GPU devices supported")
  if let some states := states? then
    for d in devices do
      if (states.find? fun (sd, _) => sd == d).isNone then
        throw (IO.userError s!"MultiGPULoader: missing state for device {d}")

  let world := world?.getD { rank := 0, worldSize := 1 }
  let numShards := world.worldSize * devices.size
  let baseShard := world.rank * devices.size

  let itemBytes := Shape.numel itemShape * dtype.itemsize
  let batchShape : Shape := batchSize :: itemShape
  let poolSlots' := if poolSlots == 0 then 0 else Nat.max poolSlots bufferSize
  let mut inits : Array WorkerInit := #[]

  for idx in [:devices.size] do
    let device := devices[idx]!
    let shard : ShardConfig := { shardIndex := baseShard + idx, numShards, mode := .interleaved, dropRemainder := true }
    let state? := states?.bind fun arr =>
      (arr.find? fun (sd, _) => sd == device).map (·.2)
    let shardBase := shardWithConfig shard cfg.base
    let shardCfg : IteratorConfig (ShardedDataset D ByteArray) := {
      base := shardBase
      startPos := state?.map (·.position) |>.getD cfg.startPos
      startEpoch := state?.map (·.epoch) |>.getD cfg.startEpoch
      epochs := cfg.epochs
      key := state?.map (·.key) |>.getD cfg.key
      updateKey := cfg.updateKey
      datasetAtEpoch := fun ds k epoch => shardWithConfig shard (cfg.datasetAtEpoch ds.inner k epoch)
    }
    let iter ← Dataset.toIteratorCfg shardCfg
    let initState ← iter.checkpoint
    let shardItems := Dataset.len shardBase
    let remaining := remainingItems shardItems cfg.epochs initState
    let shardBatches := remaining / batchSize
    inits := inits.push { device, shard, iter, initState, shardBatches }

  let mut numBatchesPerDevice : Nat := 0
  let mut first := true
  for init in inits do
    if first then
      numBatchesPerDevice := init.shardBatches
      first := false
    else
      numBatchesPerDevice := Nat.min numBatchesPerDevice init.shardBatches

  let mut workers : Array (DeviceWorker batchSize itemShape dtype) := #[]
  for init in inits do
    let queue ← IOQueue.new bufferSize
    let pool? ←
      if poolSlots' == 0 then
        pure none
      else
        some <$> GPUPool.create init.device batchShape dtype poolSlots'
    let lastState ← IO.mkRef init.initState
    let iter := init.iter
    let worker ← IO.asTask (prio := .dedicated) do
      match init.device with
      | .cuda idx => TinyGrad4.Backend.Cuda.setDevice idx
      | _ => pure ()
      -- Process this shard's batches (synchronized across devices).
      for _ in [:numBatchesPerDevice] do
        if ← IO.checkCanceled then break
        match ← nextBatch iter batchSize itemBytes with
        | some (batchData, state) =>
            let lease? ←
              match pool? with
              | some pool =>
                  match ← pool.acquire with
                  | some (_, lease) =>
                      lease.value.copyIn batchData
                      pure (some lease)
                  | none => pure none
              | none =>
                  let gpuBuf ← ByteArray.toGPUBuffer batchData init.device batchShape dtype
                  let lease : GPULease _ _ := { value := gpuBuf, release := gpuBuf.free }
                  pure (some lease)
            match lease? with
            | some lease =>
                lease.value.assertDevice init.device
                let ok ← queue.push { lease, state }
                if !ok then break
            | none => break
        | none => break

      queue.finish

    workers := workers.push {
      device := init.device,
      shard := init.shard,
      queue,
      worker,
      lastState,
      prefetcher? := none,
      pool?
    }

  pure { workers, numBatchesPerDevice }

/-- Create multi-GPU loader from an iterator config with CPU-side batch prefetchers.
    `prefetchSize` counts batches (not individual items). -/
def createFromIteratorCfgPrefetch [Dataset D ByteArray] (cfg : IteratorConfig D) (devices : Array DeviceId)
    (batchSize : Nat) (itemShape : Shape) (prefetchSize : Nat := 8)
    (dtype : DType := .float32) (bufferSize : Nat := 4) (poolSlots : Nat := 0)
    (world? : Option WorldConfig := none)
    (states? : Option (Array (DeviceId × IteratorState)) := none) :
    IO (MultiGPULoader batchSize itemShape dtype) := do
  if devices.isEmpty then
    throw (IO.userError "MultiGPULoader: no devices specified")
  if devices.any fun d => d == .cpu || (match d with | .tpu _ => true | _ => false) then
    throw (IO.userError "MultiGPULoader: only GPU devices supported")
  if let some states := states? then
    for d in devices do
      if (states.find? fun (sd, _) => sd == d).isNone then
        throw (IO.userError s!"MultiGPULoader: missing state for device {d}")

  let world := world?.getD { rank := 0, worldSize := 1 }
  let numShards := world.worldSize * devices.size
  let baseShard := world.rank * devices.size

  let itemBytes := Shape.numel itemShape * dtype.itemsize
  let batchShape : Shape := batchSize :: itemShape
  let poolSlots' := if poolSlots == 0 then 0 else Nat.max poolSlots bufferSize
  let mut inits : Array WorkerInitPrefetch := #[]

  for idx in [:devices.size] do
    let device := devices[idx]!
    let shard : ShardConfig := { shardIndex := baseShard + idx, numShards, mode := .interleaved, dropRemainder := true }
    let state? := states?.bind fun arr =>
      (arr.find? fun (sd, _) => sd == device).map (·.2)
    let shardBase := shardWithConfig shard cfg.base
    let shardCfg : IteratorConfig (ShardedDataset D ByteArray) := {
      base := shardBase
      startPos := state?.map (·.position) |>.getD cfg.startPos
      startEpoch := state?.map (·.epoch) |>.getD cfg.startEpoch
      epochs := cfg.epochs
      key := state?.map (·.key) |>.getD cfg.key
      updateKey := cfg.updateKey
      datasetAtEpoch := fun ds k epoch => shardWithConfig shard (cfg.datasetAtEpoch ds.inner k epoch)
    }
    let prefetcher ← BatchPrefetcher.createFromIteratorCfg shardCfg batchSize
      (fun chunks => pure (concatByteArraysWithTotal chunks (batchSize * itemBytes))) true prefetchSize
    let initState ← prefetcher.checkpoint
    let shardItems := Dataset.len shardBase
    let remaining := remainingItems shardItems cfg.epochs initState
    let shardBatches := remaining / batchSize
    inits := inits.push { device, shard, prefetcher, initState, shardBatches }

  let mut numBatchesPerDevice : Nat := 0
  let mut first := true
  for init in inits do
    if first then
      numBatchesPerDevice := init.shardBatches
      first := false
    else
      numBatchesPerDevice := Nat.min numBatchesPerDevice init.shardBatches

  let mut workers : Array (DeviceWorker batchSize itemShape dtype) := #[]
  for init in inits do
    let queue ← IOQueue.new bufferSize
    let pool? ←
      if poolSlots' == 0 then
        pure none
      else
        some <$> GPUPool.create init.device batchShape dtype poolSlots'
    let lastState ← IO.mkRef init.initState
    let prefetcher := init.prefetcher
    let worker ← IO.asTask (prio := .dedicated) do
      match init.device with
      | .cuda idx => TinyGrad4.Backend.Cuda.setDevice idx
      | _ => pure ()
      -- Process this shard's batches (synchronized across devices).
      for _ in [:numBatchesPerDevice] do
        if ← IO.checkCanceled then break
        match ← nextBatchFromPrefetcher prefetcher with
        | some (batchData, state) =>
            if batchData.size != batchSize * itemBytes then
              throw (IO.userError s!"MultiGPULoader.prefetch: expected {batchSize * itemBytes} bytes, got {batchData.size}")
            let lease? ←
              match pool? with
              | some pool =>
                  match ← pool.acquire with
                  | some (_, lease) =>
                      lease.value.copyIn batchData
                      pure (some lease)
                  | none => pure none
              | none =>
                  let gpuBuf ← ByteArray.toGPUBuffer batchData init.device batchShape dtype
                  let lease : GPULease _ _ := { value := gpuBuf, release := gpuBuf.free }
                  pure (some lease)
            match lease? with
            | some lease =>
                lease.value.assertDevice init.device
                let ok ← queue.push { lease, state }
                if !ok then break
            | none => break
        | none => break

      queue.finish

    workers := workers.push {
      device := init.device,
      shard := init.shard,
      queue,
      worker,
      lastState,
      prefetcher? := some prefetcher,
      pool?
    }

  pure { workers, numBatchesPerDevice }

/-- Create multi-GPU loader with round-robin sharding -/
def create [Dataset D ByteArray] (ds : D) (devices : Array DeviceId)
    (batchSize : Nat) (itemShape : Shape) (dtype : DType := .float32)
    (bufferSize : Nat := 4) (poolSlots : Nat := 0) (world? : Option WorldConfig := none) :
    IO (MultiGPULoader batchSize itemShape dtype) := do
  let cfg : IteratorConfig D := { base := ds }
  createFromIteratorCfg cfg devices batchSize itemShape dtype bufferSize poolSlots world?

/-- Get next batch from a specific device -/
def nextFrom (pool : MultiGPULoader batch itemShape dtype) (deviceIdx : Nat) :
    IO (Option (GPULease (batch :: itemShape) dtype)) := do
  if h : deviceIdx < pool.workers.size then
    match ← pool.workers[deviceIdx].queue.pop with
    | some item =>
        pool.workers[deviceIdx].lastState.set item.state
        pure (some item.lease)
    | none => pure none
  else
    pure none

/-- Get next batch from all devices (synchronized) -/
def nextAll (pool : MultiGPULoader batch itemShape dtype) :
    IO (Array (DeviceId × Option (GPULease (batch :: itemShape) dtype))) := do
  let mut results : Array (DeviceId × Option (GPULease (batch :: itemShape) dtype)) := #[]
  for w in pool.workers do
    let buf ← w.queue.pop
    match buf with
    | some item =>
        w.lastState.set item.state
        results := results.push (w.device, some item.lease)
    | none =>
        results := results.push (w.device, none)
  pure results

/-- Get next batch from all devices with total wait time (ns). -/
def nextAllWithWait (pool : MultiGPULoader batch itemShape dtype) :
    IO (Array (DeviceId × Option (GPULease (batch :: itemShape) dtype)) × Nat) := do
  let mut results : Array (DeviceId × Option (GPULease (batch :: itemShape) dtype)) := #[]
  let mut waitNs : Nat := 0
  for w in pool.workers do
    let (item?, wait) ← w.queue.popWithWait
    waitNs := waitNs + wait
    match item? with
    | some item =>
        w.lastState.set item.state
        results := results.push (w.device, some item.lease)
    | none =>
        results := results.push (w.device, none)
  pure (results, waitNs)

/-- Get next available batch from any device (first ready wins) -/
partial def nextAny (pool : MultiGPULoader batch itemShape dtype) :
    IO (Option (DeviceId × GPULease (batch :: itemShape) dtype)) := do
  -- Poll all queues, return first non-empty
  for w in pool.workers do
    match ← w.queue.tryPop with
    | some item =>
        w.lastState.set item.state
        return some (w.device, item.lease)
    | none => pure ()

  -- All empty - check if any still producing
  let allDone ← pool.workers.allM fun w => IOQueue.isClosed w.queue
  if allDone then return none

  -- Wait and retry (could be smarter with condition variables)
  IO.sleep 1
  pool.nextAny

/-- Stop all workers -/
def stop (pool : MultiGPULoader batch itemShape dtype) : IO Unit := do
  for w in pool.workers do
    w.queue.finish
    IO.cancel w.worker
    match w.prefetcher? with
    | some p => p.cancel
    | none => pure ()
    match w.pool? with
    | some p => p.stop
    | none => pure ()

/-- Drain queued buffers (freeing GPU memory). -/
def drain (pool : MultiGPULoader batch itemShape dtype) : IO Unit := do
  for w in pool.workers do
    repeat do
      match ← w.queue.pop with
      | some item => item.lease.release
      | none => break
    match w.prefetcher? with
    | some p => p.drain
    | none => pure ()
    match w.pool? with
    | some p => p.freeAll
    | none => pure ()

/-- Wait for all workers to complete -/
def wait (pool : MultiGPULoader batch itemShape dtype) : IO Unit := do
  for w in pool.workers do
    let _ ← IO.wait w.worker
    match w.prefetcher? with
    | some p => p.wait
    | none => pure ()

/-- Snapshot last consumed iterator states per device. -/
def checkpoint (pool : MultiGPULoader batch itemShape dtype) :
    IO (Array (DeviceId × IteratorState)) := do
  let mut out := Array.mkEmpty pool.workers.size
  for w in pool.workers do
    let st ← w.lastState.get
    out := out.push (w.device, st)
  pure out

/-- Iterate synchronized across all devices -/
def forEachSync (pool : MultiGPULoader batch itemShape dtype)
    (f : Array (DeviceId × GPUBuffer (batch :: itemShape) dtype) → IO Unit) : IO Unit := do
  repeat do
    let batches ← pool.nextAll
    if batches.all fun (_, b) => b.isNone then break
    let actualLeases := batches.filterMap fun (d, b) => b.map (d, ·)
    if actualLeases.size > 0 then
      let buffers := actualLeases.map (fun (d, lease) => (d, lease.value))
      try
        f buffers
      finally
        for (_, lease) in actualLeases do
          lease.release

end MultiGPULoader

/-! ## ForIn Instances -/

instance : ForIn IO (GPUDataLoader batch itemShape dtype) (GPULease (batch :: itemShape) dtype) where
  forIn loader init f := do
    let mut acc := init
    repeat do
      match ← loader.next with
      | none => break
      | some lease =>
        match ← f lease acc with
        | .done a =>
            lease.release
            return a
        | .yield a =>
            lease.release
            acc := a
    pure acc

/-! ## Convenience Functions -/

/-- Discover available GPU devices -/
def discoverDevices : IO (Array DeviceId) := do
  let mut devices : Array DeviceId := #[]

  -- Check Metal
  if ← DeviceId.metal.isAvailable then
    devices := devices.push .metal

  -- Check CUDA (could enumerate multiple GPUs)
  if ← DeviceId.cuda.isAvailable then
    let count ← TinyGrad4.Backend.Cuda.deviceCount
    for idx in [:count] do
      devices := devices.push (.cuda idx)

  -- Check TPU (best-effort via env)
  if ← DeviceId.isAvailable (.tpu 0) then
    devices := devices.push (.tpu 0)

  pure devices

/-- Create loader on best available device -/
def createOnBestDevice [Dataset D ByteArray] (ds : D)
    (batchSize : Nat) (itemShape : Shape) (dtype : DType := .float32)
    (bufferSize : Nat := 4) (poolSlots : Nat := 0) : IO (GPUDataLoader batchSize itemShape dtype) := do
  let devices ← discoverDevices
  if devices.isEmpty then
    throw (IO.userError "No GPU devices available")
  GPUDataLoader.create ds devices[0]! batchSize itemShape dtype bufferSize poolSlots

end TinyGrad4.Data.GPULoader
