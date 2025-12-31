import TinyGrad4.Data.Slice
import TinyGrad4.Data.Dataset
import TinyGrad4.Data.Prefetch
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

/-! ## Device Identification -/

/-- Target device for data loading -/
inductive DeviceId where
  | cpu
  | metal
  | cuda (idx : Nat := 0)
  deriving BEq, Hashable, Repr, Inhabited

namespace DeviceId

/-- Check device availability -/
def isAvailable : DeviceId → IO Bool
  | .cpu => pure true
  | .metal => TinyGrad4.Backend.Metal.isAvailable
  | .cuda _ => TinyGrad4.Backend.Cuda.isAvailable

/-- Get device name -/
def name : DeviceId → IO String
  | .cpu => pure "CPU"
  | .metal => TinyGrad4.Backend.Metal.deviceInfo
  | .cuda idx => do
      let info ← TinyGrad4.Backend.Cuda.deviceInfo
      pure s!"{info}:{idx}"

/-- Sync device (wait for all operations to complete) -/
def sync : DeviceId → IO Unit
  | .cpu => pure ()
  | .metal => TinyGrad4.Backend.Metal.metalSync
  | .cuda _ => TinyGrad4.Backend.Cuda.cudaSync

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
    let buf ← TinyGrad4.Backend.Cuda.cudaAllocBytes byteSize
    pure { handle := .cuda buf, byteSize, dtype, device := .cuda idx }
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

Thread-safe queue for GPU buffers, extending IOQueue pattern.
-/

/-- Queue of GPU buffers with producer-consumer semantics -/
structure GPUQueue where
  items : IO.Ref (Array GPUBuffer)
  done : IO.Ref Bool
  maxSize : Nat

namespace GPUQueue

/-- Create new GPU queue -/
def new (maxSize : Nat := 4) : IO GPUQueue := do
  let items ← IO.mkRef #[]
  let done ← IO.mkRef false
  pure { items, done, maxSize }

/-- Push buffer to queue (spins if full for backpressure) -/
def push (q : GPUQueue) (buf : GPUBuffer) : IO Unit := do
  repeat do
    let arr ← q.items.get
    if arr.size < q.maxSize then break
    IO.sleep 1  -- 1ms backoff
  q.items.modify (·.push buf)

/-- Mark queue as complete (no more items coming) -/
def finish (q : GPUQueue) : IO Unit := q.done.set true

/-- Pop buffer from queue (blocks until available, none if done) -/
def pop (q : GPUQueue) : IO (Option GPUBuffer) := do
  repeat do
    let arr ← q.items.get
    if h : arr.size > 0 then
      let item := arr[0]'(by omega)
      q.items.set (arr.eraseIdx 0)
      return some item
    if ← q.done.get then return none
    IO.sleep 1
  pure none

/-- Check if empty and done -/
def isEmpty (q : GPUQueue) : IO Bool := do
  let arr ← q.items.get
  let d ← q.done.get
  pure (arr.isEmpty && d)

end GPUQueue

/-! ## Single-Device GPU Loader -/

/-- Background GPU data loader for a single device -/
structure GPUDataLoader where
  /-- Output queue of GPU buffers -/
  queue : GPUQueue
  /-- Worker task -/
  worker : Task (Except IO.Error Unit)
  /-- Target device -/
  device : DeviceId
  /-- Number of batches to produce -/
  numBatches : Nat

namespace GPUDataLoader

/-- Create loader from a dataset of ByteArrays -/
def create [Dataset D ByteArray] (ds : D) (device : DeviceId)
    (batchSize : Nat) (bufferSize : Nat := 4) (dtype : DType := .float32)
    : IO GPUDataLoader := do
  let queue ← GPUQueue.new bufferSize
  let n := Dataset.len ds
  let numBatches := n / batchSize

  let worker ← IO.asTask (prio := .dedicated) do
    for batchIdx in [:numBatches] do
      if ← IO.checkCanceled then break

      -- Gather batch
      let start := batchIdx * batchSize
      let mut batchData := ByteArray.empty
      for i in [start : start + batchSize] do
        if h : i < n then
          let item ← Dataset.getItem ds i h
          batchData := batchData.append item

      -- Upload to GPU
      let gpuBuf ← ByteArray.toGPUBuffer batchData device dtype
      queue.push gpuBuf

    queue.finish

  pure { queue, worker, device, numBatches }

/-- Create loader from array of ByteSlices (zero-copy if possible) -/
def fromSlices (slices : Array ByteSlice) (device : DeviceId)
    (bufferSize : Nat := 4) (dtype : DType := .float32)
    : IO GPUDataLoader := do
  let queue ← GPUQueue.new bufferSize

  let worker ← IO.asTask (prio := .dedicated) do
    for slice in slices do
      if ← IO.checkCanceled then break
      let gpuBuf ← ByteSlice.toGPUBufferZeroCopy slice device dtype
      queue.push gpuBuf
    queue.finish

  pure { queue, worker, device, numBatches := slices.size }

/-- Get next GPU buffer (blocks until available) -/
def next (loader : GPUDataLoader) : IO (Option GPUBuffer) :=
  loader.queue.pop

/-- Stop the loader -/
def stop (loader : GPUDataLoader) : IO Unit := do
  loader.queue.finish
  IO.cancel loader.worker

/-- Wait for loader to complete -/
def wait (loader : GPUDataLoader) : IO Unit := do
  let _ ← IO.wait loader.worker

/-- Iterate over all batches -/
def forEach (loader : GPUDataLoader) (f : GPUBuffer → IO Unit) : IO Unit := do
  repeat do
    match ← loader.next with
    | some buf => f buf
    | none => break

end GPUDataLoader

/-! ## Multi-Device Sharding -/

/-- Shard configuration -/
structure ShardConfig where
  shardIndex : Nat
  numShards : Nat
  deriving Repr, BEq, Inhabited

namespace ShardConfig

/-- Indices belonging to this shard (round-robin) -/
def indices (cfg : ShardConfig) (totalItems : Nat) : Array Nat :=
  (Array.range totalItems).filter fun i => i % cfg.numShards == cfg.shardIndex

/-- Number of items in this shard -/
def size (cfg : ShardConfig) (totalItems : Nat) : Nat :=
  (totalItems + cfg.numShards - 1 - cfg.shardIndex) / cfg.numShards

end ShardConfig

/-- Device assignment with shard -/
structure DeviceAssignment where
  device : DeviceId
  shard : ShardConfig
  deriving Repr

/-- Assign shards to devices (round-robin) -/
def assignDevices (devices : Array DeviceId) : Array DeviceAssignment :=
  devices.mapIdx fun i dev => {
    device := dev
    shard := { shardIndex := i, numShards := devices.size }
  }

/-! ## Multi-GPU Loader Pool -/

/-- Worker state for a single device -/
structure DeviceWorker where
  device : DeviceId
  shard : ShardConfig
  queue : GPUQueue
  worker : Task (Except IO.Error Unit)

/-- Pool of loaders across multiple GPUs -/
structure MultiGPULoader where
  workers : Array DeviceWorker
  numBatchesPerDevice : Nat

namespace MultiGPULoader

/-- Create multi-GPU loader with round-robin sharding -/
def create [Dataset D ByteArray] (ds : D) (devices : Array DeviceId)
    (batchSize : Nat) (bufferSize : Nat := 4) (dtype : DType := .float32)
    : IO MultiGPULoader := do
  if devices.isEmpty then
    throw (IO.userError "MultiGPULoader: no devices specified")

  let n := Dataset.len ds
  let numShards := devices.size
  let numBatchesPerShard := n / (batchSize * numShards)

  let mut workers : Array DeviceWorker := #[]

  for idx in [:devices.size] do
    let device := devices[idx]!
    let shard : ShardConfig := { shardIndex := idx, numShards }
    let queue ← GPUQueue.new bufferSize

    let worker ← IO.asTask (prio := .dedicated) do
      -- Process this shard's batches
      for batchIdx in [:numBatchesPerShard] do
        if ← IO.checkCanceled then break

        -- Global index for this batch in this shard
        let globalBatchStart := batchIdx * batchSize * numShards + idx * batchSize

        -- Gather batch data
        let mut batchData := ByteArray.empty
        for i in [globalBatchStart : globalBatchStart + batchSize] do
          if h : i < n then
            let item ← Dataset.getItem ds i h
            batchData := batchData.append item

        -- Upload to this device
        let gpuBuf ← ByteArray.toGPUBuffer batchData device dtype
        queue.push gpuBuf

      queue.finish

    workers := workers.push { device, shard, queue, worker }

  pure { workers, numBatchesPerDevice := numBatchesPerShard }

/-- Get next batch from a specific device -/
def nextFrom (pool : MultiGPULoader) (deviceIdx : Nat) : IO (Option GPUBuffer) := do
  if h : deviceIdx < pool.workers.size then
    pool.workers[deviceIdx].queue.pop
  else
    pure none

/-- Get next batch from all devices (synchronized) -/
def nextAll (pool : MultiGPULoader) : IO (Array (DeviceId × Option GPUBuffer)) := do
  let mut results : Array (DeviceId × Option GPUBuffer) := #[]
  for w in pool.workers do
    let buf ← w.queue.pop
    results := results.push (w.device, buf)
  pure results

/-- Get next available batch from any device (first ready wins) -/
partial def nextAny (pool : MultiGPULoader) : IO (Option (DeviceId × GPUBuffer)) := do
  -- Poll all queues, return first non-empty
  for w in pool.workers do
    let arr ← w.queue.items.get
    if h : arr.size > 0 then
      let buf := arr[0]'(by omega)
      w.queue.items.set (arr.eraseIdx 0)
      return some (w.device, buf)

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

/-- Wait for all workers to complete -/
def wait (pool : MultiGPULoader) : IO Unit := do
  for w in pool.workers do
    let _ ← IO.wait w.worker

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
    devices := devices.push (.cuda 0)
    -- TODO: enumerate all CUDA devices

  pure devices

/-- Create loader on best available device -/
def createOnBestDevice [Dataset D ByteArray] (ds : D)
    (batchSize : Nat) (bufferSize : Nat := 4) : IO GPUDataLoader := do
  let devices ← discoverDevices
  if devices.isEmpty then
    throw (IO.userError "No GPU devices available")
  GPUDataLoader.create ds devices[0]! batchSize bufferSize

end TinyGrad4.Data.GPULoader
