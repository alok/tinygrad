import TinyGrad4.Data.Slice
import TinyGrad4.Data.Dataset
import TinyGrad4.Data.Prefetch
import TinyGrad4.Data.Shard
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
private def concatByteArrays (chunks : Array ByteArray) : ByteArray := Id.run do
  let total := chunks.foldl (fun acc b => acc + b.size) 0
  let mut out := ByteArray.emptyWithCapacity total
  let mut offset := 0
  for chunk in chunks do
    out := ByteArray.copySlice chunk 0 out offset chunk.size false
    offset := offset + chunk.size
  out

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

/-- Unified GPU buffer with backend-agnostic operations -/
structure GPUBuffer where
  handle : GPUHandle
  byteSize : Nat
  dtype : DType
  device : DeviceId
  deriving Repr

namespace GPUBuffer

/-- Allocate buffer on device -/
def alloc (device : DeviceId) (byteSize : Nat) (dtype : DType := .float32) : IO GPUBuffer := do
  match device with
  | .metal =>
    let buf ← TinyGrad4.Backend.Metal.metalAllocBytes byteSize
    pure { handle := .metal buf, byteSize, dtype, device }
  | .cuda idx =>
    TinyGrad4.Backend.Cuda.setDevice idx
    let buf ← TinyGrad4.Backend.Cuda.cudaAllocBytes byteSize
    pure { handle := .cuda buf, byteSize, dtype, device := .cuda idx }
  | .tpu _ =>
    throw (IO.userError "GPUBuffer.alloc: TPU backend not implemented")
  | .cpu =>
    throw (IO.userError "GPUBuffer.alloc: use RawBuffer for CPU data")

/-- Free buffer -/
def free (b : GPUBuffer) : IO Unit := do
  match b.handle with
  | .metal buf => TinyGrad4.Backend.Metal.metalFree buf
  | .cuda buf => TinyGrad4.Backend.Cuda.cudaFree buf

/-- Copy from CPU ByteArray to GPU -/
def copyIn (b : GPUBuffer) (data : ByteArray) : IO Unit := do
  match b.handle with
  | .metal buf => TinyGrad4.Backend.Metal.metalCopyInBytes buf data
  | .cuda buf => TinyGrad4.Backend.Cuda.cudaCopyInBytes buf data

/-- Copy from GPU to CPU ByteArray -/
def copyOut (b : GPUBuffer) : IO ByteArray := do
  match b.handle with
  | .metal buf => TinyGrad4.Backend.Metal.metalCopyOutBytes buf b.byteSize
  | .cuda buf => TinyGrad4.Backend.Cuda.cudaCopyOutBytes buf b.byteSize

/-- Sync the device this buffer is on -/
def sync (b : GPUBuffer) : IO Unit := b.device.sync

/-- Number of elements (byteSize / dtype itemsize) -/
def numel (b : GPUBuffer) : Nat := b.byteSize / b.dtype.itemsize

/-- Ensure a buffer is on the expected device. -/
def assertDevice (b : GPUBuffer) (device : DeviceId) : IO Unit := do
  if b.device != device then
    throw (IO.userError s!"GPUBuffer on {b.device}, expected {device}")

end GPUBuffer

/-! ## ByteSlice → GPU Bridge -/

/-- Upload ByteSlice to GPU buffer (copies data) -/
def ByteSlice.toGPUBuffer (slice : ByteSlice) (device : DeviceId)
    (dtype : DType := .uint8) : IO GPUBuffer := do
  let buf ← GPUBuffer.alloc device slice.length dtype
  buf.copyIn slice.toByteArray
  pure buf

/-- Upload ByteSlice to GPU with zero-copy if possible.
    Zero-copy works on Apple Silicon Metal (unified memory).
    Falls back to copy on CUDA and other platforms. -/
def ByteSlice.toGPUBufferZeroCopy (slice : ByteSlice) (device : DeviceId)
    (dtype : DType := .uint8) : IO GPUBuffer := do
  match device with
  | .metal =>
    -- Use Metal zero-copy via unified memory
    let buf ← TinyGrad4.Backend.Metal.metalWrapBytesNoCopy slice.parent slice.offset slice.length
    pure { handle := .metal buf, byteSize := slice.length, dtype, device }
  | _ =>
    -- Fall back to copy
    ByteSlice.toGPUBuffer slice device dtype

/-- Upload ByteArray to GPU -/
def ByteArray.toGPUBuffer (data : ByteArray) (device : DeviceId)
    (dtype : DType := .uint8) : IO GPUBuffer := do
  let buf ← GPUBuffer.alloc device data.size dtype
  buf.copyIn data
  pure buf

/-! ## GPU Data Queue

Reuses `IOQueue` from Prefetch, but records iterator state with each buffer
so we can resume deterministically after interruption.
-/

/-- Queue item with checkpoint state after this batch. -/
structure QueuedBuffer where
  buffer : GPUBuffer
  state : IteratorState

/-! ## Single-Device GPU Loader -/

/-- Background GPU data loader for a single device -/
structure GPUDataLoader where
  /-- Output queue of GPU buffers -/
  queue : IOQueue QueuedBuffer
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

namespace GPUDataLoader

/-- Build one batch from an iterator, returning the state after the batch. -/
private def nextBatch (iter : DataIterator ByteArray) (batchSize : Nat) :
    IO (Option (ByteArray × IteratorState)) := do
  let mut chunks := Array.mkEmpty batchSize
  for _ in [:batchSize] do
    match ← iter.next with
    | some item => chunks := chunks.push item
    | none => return none
  let state ← iter.checkpoint
  let batchData := concatByteArrays chunks
  pure (some (batchData, state))

/-- Fetch one prefetched batch with its checkpoint state. -/
private def nextBatchFromPrefetcher (prefetcher : BatchPrefetcher ByteArray) :
    IO (Option (ByteArray × IteratorState)) := do
  match ← prefetcher.nextWithState with
  | some (batch, state) => pure (some (batch, state))
  | none => pure none

/-- Create loader from a pre-built iterator. -/
private def createFromIterator (iter : DataIterator ByteArray) (initState : IteratorState)
    (totalItems : Nat) (device : DeviceId) (batchSize : Nat) (bufferSize : Nat) (dtype : DType)
    : IO GPUDataLoader := do
  match device with
  | .cpu => throw (IO.userError "GPUDataLoader: use CPU datasets directly")
  | .tpu _ => throw (IO.userError "GPUDataLoader: use TPULoader for TPU devices")
  | _ => pure ()
  let queue ← IOQueue.new bufferSize
  let lastState ← IO.mkRef initState
  let numBatches := totalItems / batchSize

  let worker ← IO.asTask (prio := .dedicated) do
    match device with
    | .cuda idx => TinyGrad4.Backend.Cuda.setDevice idx
    | _ => pure ()
    for _ in [:numBatches] do
      if ← IO.checkCanceled then break
      match ← nextBatch iter batchSize with
      | some (batchData, state) =>
          let gpuBuf ← ByteArray.toGPUBuffer batchData device dtype
          gpuBuf.assertDevice device
          let ok ← queue.push { buffer := gpuBuf, state }
          if !ok then break
      | none => break
    queue.finish

  pure { queue, worker, device, numBatches, lastState, prefetcher? := none }

/-- Create loader from a pre-built iterator prefetcher (staged CPU → GPU). -/
private def createFromPrefetcher (prefetcher : BatchPrefetcher ByteArray) (initState : IteratorState)
    (totalItems : Nat) (device : DeviceId) (batchSize : Nat) (bufferSize : Nat) (dtype : DType)
    : IO GPUDataLoader := do
  match device with
  | .cpu => throw (IO.userError "GPUDataLoader: use CPU datasets directly")
  | .tpu _ => throw (IO.userError "GPUDataLoader: use TPULoader for TPU devices")
  | _ => pure ()
  let queue ← IOQueue.new bufferSize
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
          let gpuBuf ← ByteArray.toGPUBuffer batchData device dtype
          gpuBuf.assertDevice device
          let ok ← queue.push { buffer := gpuBuf, state }
          if !ok then break
      | none => break
    queue.finish

  pure { queue, worker, device, numBatches, lastState, prefetcher? := some prefetcher }

/-- Create loader from an iterator config (supports checkpoint/resume). -/
def createFromIteratorCfg [Dataset D ByteArray] (cfg : IteratorConfig D) (device : DeviceId)
    (batchSize : Nat) (bufferSize : Nat := 4) (dtype : DType := .float32)
    : IO GPUDataLoader := do
  let iter ← Dataset.toIteratorCfg cfg
  let initState ← iter.checkpoint
  let n := Dataset.len cfg.base
  let remaining := remainingItems n cfg.epochs initState
  createFromIterator iter initState remaining device batchSize bufferSize dtype

/-- Create loader from an iterator config with a CPU-side batch prefetcher.
    `prefetchSize` counts batches (not individual items). -/
def createFromIteratorCfgPrefetch [Dataset D ByteArray] (cfg : IteratorConfig D) (device : DeviceId)
    (batchSize : Nat) (prefetchSize : Nat := 8) (bufferSize : Nat := 4) (dtype : DType := .float32)
    : IO GPUDataLoader := do
  let prefetcher ← BatchPrefetcher.createFromIteratorCfg cfg batchSize
    (fun chunks => pure (concatByteArrays chunks)) true prefetchSize
  let initState ← prefetcher.checkpoint
  let n := Dataset.len cfg.base
  let remaining := remainingItems n cfg.epochs initState
  createFromPrefetcher prefetcher initState remaining device batchSize bufferSize dtype

/-- Create loader from a dataset of ByteArrays -/
def create [Dataset D ByteArray] (ds : D) (device : DeviceId)
    (batchSize : Nat) (bufferSize : Nat := 4) (dtype : DType := .float32)
    : IO GPUDataLoader := do
  let cfg : IteratorConfig D := { base := ds }
  createFromIteratorCfg cfg device batchSize bufferSize dtype

/-- Create loader from array of ByteSlices (zero-copy if possible) -/
def fromSlices (slices : Array ByteSlice) (device : DeviceId)
    (bufferSize : Nat := 4) (dtype : DType := .float32)
    : IO GPUDataLoader := do
  match device with
  | .cpu => throw (IO.userError "GPUDataLoader.fromSlices: use CPU datasets directly")
  | .tpu _ => throw (IO.userError "GPUDataLoader.fromSlices: use TPULoader for TPU devices")
  | _ => pure ()
  let queue ← IOQueue.new bufferSize
  let lastState ← IO.mkRef { position := 0, epoch := 0, key := RandKey.new 0 }

  let worker ← IO.asTask (prio := .dedicated) do
    match device with
    | .cuda idx => TinyGrad4.Backend.Cuda.setDevice idx
    | _ => pure ()
    let mut pos := 0
    for slice in slices do
      if ← IO.checkCanceled then break
      let gpuBuf ← ByteSlice.toGPUBufferZeroCopy slice device dtype
      gpuBuf.assertDevice device
      pos := pos + 1
      let state : IteratorState := { position := pos, epoch := 0, key := RandKey.new 0 }
      let ok ← queue.push { buffer := gpuBuf, state }
      if !ok then break
    queue.finish

  pure { queue, worker, device, numBatches := slices.size, lastState, prefetcher? := none }

/-- Get next GPU buffer (blocks until available) -/
def next (loader : GPUDataLoader) : IO (Option GPUBuffer) := do
  match ← loader.queue.pop with
  | some item =>
      loader.lastState.set item.state
      pure (some item.buffer)
  | none => pure none

/-- Get next buffer and wait time spent in queue (ns). -/
def nextWithWait (loader : GPUDataLoader) : IO (Option GPUBuffer × Nat) := do
  let (item?, waitNs) ← loader.queue.popWithWait
  match item? with
  | some item =>
      loader.lastState.set item.state
      pure (some item.buffer, waitNs)
  | none => pure (none, waitNs)

/-- Stop the loader -/
def stop (loader : GPUDataLoader) : IO Unit := do
  loader.queue.finish
  IO.cancel loader.worker
  match loader.prefetcher? with
  | some p => p.cancel
  | none => pure ()

/-- Wait for loader to complete -/
def wait (loader : GPUDataLoader) : IO Unit := do
  let _ ← IO.wait loader.worker
  match loader.prefetcher? with
  | some p => p.wait
  | none => pure ()

/-- Get checkpoint state for deterministic resume. -/
def checkpoint (loader : GPUDataLoader) : IO IteratorState :=
  loader.lastState.get

/-- Drain any queued buffers (freeing GPU memory). -/
def drain (loader : GPUDataLoader) : IO Unit := do
  repeat do
    match ← loader.queue.pop with
    | some item => item.buffer.free
    | none => break
  match loader.prefetcher? with
  | some p => p.drain
  | none => pure ()

/-- Iterate over all batches -/
def forEach (loader : GPUDataLoader) (f : GPUBuffer → IO Unit) : IO Unit := do
  repeat do
    match ← loader.next with
    | some buf => f buf
    | none => break

end GPUDataLoader

/-- Worker state for a single device -/
structure DeviceWorker where
  device : DeviceId
  shard : ShardConfig
  queue : IOQueue QueuedBuffer
  worker : Task (Except IO.Error Unit)
  lastState : IO.Ref IteratorState
  prefetcher? : Option (BatchPrefetcher ByteArray)

/-- Pool of loaders across multiple GPUs -/
structure MultiGPULoader where
  workers : Array DeviceWorker
  numBatchesPerDevice : Nat

namespace MultiGPULoader

/-- Build one batch from an iterator, returning the state after the batch. -/
private def nextBatch (iter : DataIterator ByteArray) (batchSize : Nat) :
    IO (Option (ByteArray × IteratorState)) := do
  let mut chunks := Array.mkEmpty batchSize
  for _ in [:batchSize] do
    match ← iter.next with
    | some item => chunks := chunks.push item
    | none => return none
  let state ← iter.checkpoint
  let batchData := concatByteArrays chunks
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
    (batchSize : Nat) (bufferSize : Nat := 4) (dtype : DType := .float32)
    (world? : Option WorldConfig := none)
    (states? : Option (Array (DeviceId × IteratorState)) := none) : IO MultiGPULoader := do
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

  let mut workers : Array DeviceWorker := #[]
  for init in inits do
    let queue ← IOQueue.new bufferSize
    let lastState ← IO.mkRef init.initState
    let iter := init.iter
    let worker ← IO.asTask (prio := .dedicated) do
      match init.device with
      | .cuda idx => TinyGrad4.Backend.Cuda.setDevice idx
      | _ => pure ()
      -- Process this shard's batches (synchronized across devices).
      for _ in [:numBatchesPerDevice] do
        if ← IO.checkCanceled then break
        match ← nextBatch iter batchSize with
        | some (batchData, state) =>
            let gpuBuf ← ByteArray.toGPUBuffer batchData init.device dtype
            gpuBuf.assertDevice init.device
            let ok ← queue.push { buffer := gpuBuf, state }
            if !ok then break
        | none => break

      queue.finish

    workers := workers.push {
      device := init.device,
      shard := init.shard,
      queue,
      worker,
      lastState,
      prefetcher? := none
    }

  pure { workers, numBatchesPerDevice }

/-- Create multi-GPU loader from an iterator config with CPU-side batch prefetchers.
    `prefetchSize` counts batches (not individual items). -/
def createFromIteratorCfgPrefetch [Dataset D ByteArray] (cfg : IteratorConfig D) (devices : Array DeviceId)
    (batchSize : Nat) (prefetchSize : Nat := 8) (bufferSize : Nat := 4) (dtype : DType := .float32)
    (world? : Option WorldConfig := none)
    (states? : Option (Array (DeviceId × IteratorState)) := none) : IO MultiGPULoader := do
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
      (fun chunks => pure (concatByteArrays chunks)) true prefetchSize
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

  let mut workers : Array DeviceWorker := #[]
  for init in inits do
    let queue ← IOQueue.new bufferSize
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
            let gpuBuf ← ByteArray.toGPUBuffer batchData init.device dtype
            gpuBuf.assertDevice init.device
            let ok ← queue.push { buffer := gpuBuf, state }
            if !ok then break
        | none => break

      queue.finish

    workers := workers.push {
      device := init.device,
      shard := init.shard,
      queue,
      worker,
      lastState,
      prefetcher? := some prefetcher
    }

  pure { workers, numBatchesPerDevice }

/-- Create multi-GPU loader with round-robin sharding -/
def create [Dataset D ByteArray] (ds : D) (devices : Array DeviceId)
    (batchSize : Nat) (bufferSize : Nat := 4) (dtype : DType := .float32)
    (world? : Option WorldConfig := none) : IO MultiGPULoader := do
  let cfg : IteratorConfig D := { base := ds }
  createFromIteratorCfg cfg devices batchSize bufferSize dtype world?

/-- Get next batch from a specific device -/
def nextFrom (pool : MultiGPULoader) (deviceIdx : Nat) : IO (Option GPUBuffer) := do
  if h : deviceIdx < pool.workers.size then
    match ← pool.workers[deviceIdx].queue.pop with
    | some item =>
        pool.workers[deviceIdx].lastState.set item.state
        pure (some item.buffer)
    | none => pure none
  else
    pure none

/-- Get next batch from all devices (synchronized) -/
def nextAll (pool : MultiGPULoader) : IO (Array (DeviceId × Option GPUBuffer)) := do
  let mut results : Array (DeviceId × Option GPUBuffer) := #[]
  for w in pool.workers do
    let buf ← w.queue.pop
    match buf with
    | some item =>
        w.lastState.set item.state
        results := results.push (w.device, some item.buffer)
    | none =>
        results := results.push (w.device, none)
  pure results

/-- Get next batch from all devices with total wait time (ns). -/
def nextAllWithWait (pool : MultiGPULoader) : IO (Array (DeviceId × Option GPUBuffer) × Nat) := do
  let mut results : Array (DeviceId × Option GPUBuffer) := #[]
  let mut waitNs : Nat := 0
  for w in pool.workers do
    let (item?, wait) ← w.queue.popWithWait
    waitNs := waitNs + wait
    match item? with
    | some item =>
        w.lastState.set item.state
        results := results.push (w.device, some item.buffer)
    | none =>
        results := results.push (w.device, none)
  pure (results, waitNs)

/-- Get next available batch from any device (first ready wins) -/
partial def nextAny (pool : MultiGPULoader) : IO (Option (DeviceId × GPUBuffer)) := do
  -- Poll all queues, return first non-empty
  for w in pool.workers do
    match ← w.queue.tryPop with
    | some item =>
        w.lastState.set item.state
        return some (w.device, item.buffer)
    | none => pure ()

  -- All empty - check if any still producing
  let allDone ← pool.workers.allM fun w => w.queue.done.get
  if allDone then return none

  -- Wait and retry (could be smarter with condition variables)
  IO.sleep 1
  pool.nextAny

/-- Stop all workers -/
def stop (pool : MultiGPULoader) : IO Unit := do
  for w in pool.workers do
    w.queue.finish
    IO.cancel w.worker
    match w.prefetcher? with
    | some p => p.cancel
    | none => pure ()

/-- Drain queued buffers (freeing GPU memory). -/
def drain (pool : MultiGPULoader) : IO Unit := do
  for w in pool.workers do
    repeat do
      match ← w.queue.pop with
      | some item => item.buffer.free
      | none => break
    match w.prefetcher? with
    | some p => p.drain
    | none => pure ()

/-- Wait for all workers to complete -/
def wait (pool : MultiGPULoader) : IO Unit := do
  for w in pool.workers do
    let _ ← IO.wait w.worker
    match w.prefetcher? with
    | some p => p.wait
    | none => pure ()

/-- Snapshot last consumed iterator states per device. -/
def checkpoint (pool : MultiGPULoader) : IO (Array (DeviceId × IteratorState)) := do
  let mut out := Array.mkEmpty pool.workers.size
  for w in pool.workers do
    let st ← w.lastState.get
    out := out.push (w.device, st)
  pure out

/-- Iterate synchronized across all devices -/
def forEachSync (pool : MultiGPULoader)
    (f : Array (DeviceId × GPUBuffer) → IO Unit) : IO Unit := do
  repeat do
    let batches ← pool.nextAll
    -- Check if all are none
    if batches.all fun (_, b) => b.isNone then break
    -- Filter to actual batches
    let actual := batches.filterMap fun (d, b) => b.map (d, ·)
    if actual.size > 0 then f actual

end MultiGPULoader

/-! ## ForIn Instances -/

instance : ForIn IO GPUDataLoader GPUBuffer where
  forIn loader init f := do
    let mut acc := init
    repeat do
      match ← loader.next with
      | none => break
      | some buf =>
        match ← f buf acc with
        | .done a => return a
        | .yield a => acc := a
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
    (batchSize : Nat) (bufferSize : Nat := 4) : IO GPUDataLoader := do
  let devices ← discoverDevices
  if devices.isEmpty then
    throw (IO.userError "No GPU devices available")
  GPUDataLoader.create ds devices[0]! batchSize bufferSize

end TinyGrad4.Data.GPULoader
