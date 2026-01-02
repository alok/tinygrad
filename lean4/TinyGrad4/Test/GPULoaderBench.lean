import TinyGrad4.Data.GPULoader
import TinyGrad4.Data.Shard
-- Disable IO.monoNanosNow linter: benchmark timing uses raw monotonic clocks.
set_option linter.monoNanosNow false

/-!
# GPU Data Loader Benchmark

Measures GPU data loading performance:
- Single-device Metal loading
- Zero-copy vs copy throughput
- Multi-device synchronization overhead
-/

open TinyGrad4.Data.GPULoader
open TinyGrad4.Data

namespace TinyGrad4.Test

private instance : Inhabited TinyGrad4.Data.ByteSlice :=
  ⟨TinyGrad4.Data.ByteSlice.mk' ByteArray.empty 0 0⟩

private def getEnvNat (key : String) (default : Nat) : IO Nat := do
  match (← IO.getEnv key) with
  | some v =>
    match v.toNat? with
    | some n => pure n
    | none => pure default
  | none => pure default

private def getEnvString (key : String) (default : String) : IO String := do
  match (← IO.getEnv key) with
  | some v => pure v
  | none => pure default

private def getEnvBool (key : String) (default : Bool) : IO Bool := do
  match (← IO.getEnv key) with
  | some v =>
    let v' := v.trimAscii.toString.toLower
    pure (v' == "1" || v' == "true" || v' == "yes")
  | none => pure default

private def parseDevice (raw : String) : Option DeviceId :=
  let s := raw.trimAscii.toString.toLower
  if s == "auto" then
    none
  else if s == "metal" then
    some .metal
  else if s == "cpu" then
    some .cpu
  else if s.startsWith "cuda" then
    let parts := s.splitOn ":"
    let idx := match parts with
      | _ :: v :: _ => (String.toNat? v).getD 0
      | _ => 0
    some (.cuda idx)
  else if s.startsWith "tpu" then
    let parts := s.splitOn ":"
    let idx := match parts with
      | _ :: v :: _ => (String.toNat? v).getD 0
      | _ => 0
    some (.tpu idx)
  else
    none

/-- Measure time in milliseconds -/
def timeMs (action : IO α) : IO (Float × α) := do
  let start ← IO.monoNanosNow
  let result ← action
  let stop ← IO.monoNanosNow
  let ms := (stop - start).toFloat / 1_000_000.0
  pure (ms, result)

/-- Benchmark GPU buffer allocation and copy -/
def benchBufferAlloc (device : DeviceId) (sizes : Array Nat) (iters : Nat := 100) : IO Unit := do
  IO.println s!"=== Buffer Alloc/Copy Benchmark ({repr device}) ==="

  for size in sizes do
    -- Create test data
    let data := ByteArray.mk (Array.replicate size 42)

    -- Warm up
    for _ in [:5] do
      let buf ← GPUBuffer.alloc device [size] .uint8
      buf.copyIn data
      buf.free

    -- Measure allocation + copy
    let (allocMs, _) ← timeMs do
      for _ in [:iters] do
        let buf ← GPUBuffer.alloc device [size] .uint8
        buf.copyIn data
        buf.free

    let avgUs := allocMs * 1000.0 / iters.toFloat
    let mbPerSec := size.toFloat * iters.toFloat / allocMs / 1000.0
    IO.println s!"  {size / 1024}KB: {avgUs.toString.take 6}μs/op, {mbPerSec.toString.take 6} MB/s"

  device.sync
  IO.println ""

/-- Benchmark zero-copy vs regular copy on Metal -/
def benchZeroCopy (device : DeviceId) (imageData : ByteArray) (batchSize : Nat := 64) (iters : Nat := 100) : IO Unit := do
  let label := if device == .metal then "Metal" else s!"{device}"
  IO.println s!"=== Zero-Copy vs Regular Copy ({label}) ==="
  if device != .metal then
    IO.println "  note: zero-copy only supported on Metal (fallback copy on other backends)"

  let numImages := imageData.size / 784
  let numBatches := numImages / batchSize
  let batchBytes := batchSize * 784

  -- Create slices for zero-copy test
  let slices := Array.range numBatches |>.map fun i =>
    ByteSlice.mk' imageData (i * batchBytes) batchBytes

  let testCount := min iters slices.size

  -- Benchmark regular copy
  IO.println "  Regular copy:"
  let (copyMs, _) ← timeMs do
    for i in [:testCount] do
      if h : i < slices.size then
        let slice := slices[i]
        let buf ← ByteSlice.toGPUBuffer slice device [batchSize, 784] .uint8
        buf.free

  let copyBatchPerSec := testCount.toFloat * 1000.0 / copyMs
  let copyMBPerSec := testCount.toFloat * batchBytes.toFloat / copyMs / 1000.0
  IO.println s!"    {testCount} batches in {copyMs.toString.take 6}ms"
  IO.println s!"    {copyBatchPerSec.toString.take 6} batch/s, {copyMBPerSec.toString.take 6} MB/s"

  -- Benchmark zero-copy
  IO.println "  Zero-copy:"
  let (zcMs, _) ← timeMs do
    for i in [:testCount] do
      if h : i < slices.size then
        let slice := slices[i]
        let buf ← ByteSlice.toGPUBufferZeroCopy slice device [batchSize, 784] .uint8
        buf.free

  let zcBatchPerSec := testCount.toFloat * 1000.0 / zcMs
  let zcMBPerSec := testCount.toFloat * batchBytes.toFloat / zcMs / 1000.0
  IO.println s!"    {testCount} batches in {zcMs.toString.take 6}ms"
  IO.println s!"    {zcBatchPerSec.toString.take 6} batch/s, {zcMBPerSec.toString.take 6} MB/s"

  let speedup := zcBatchPerSec / copyBatchPerSec
  IO.println s!"  Speedup: {speedup.toString.take 5}x"
  IO.println ""

/-- Dataset adapter for raw ByteArray chunks -/
structure ByteChunkDataset where
  chunks : Array ByteArray
  deriving Inhabited

instance : Dataset ByteChunkDataset ByteArray where
  len ds := ds.chunks.size
  getItem ds idx _ := pure ds.chunks[idx]!

/-- Dataset adapter for image-sized ByteArray slices. -/
structure ByteImageDataset where
  data : ByteArray
  imageSize : Nat
  numImages : Nat
  deriving Inhabited

instance : Dataset ByteImageDataset ByteArray where
  len ds := ds.numImages
  getItem ds idx _ :=
    let start := idx * ds.imageSize
    pure (ds.data.extract start (start + ds.imageSize))

/-- Benchmark GPUDataLoader throughput -/
def benchGPULoader (device : DeviceId) (imageData : ByteArray) (batchSize : Nat := 64) (bufferSize : Nat := 4)
    (poolSlots : Nat := 0) (world? : Option WorldConfig := none) : IO Unit := do
  IO.println s!"=== GPUDataLoader Throughput (batch={batchSize}, buffer={bufferSize}, pool={poolSlots}) ==="

  let numImages := imageData.size / 784
  let batchBytes := batchSize * 784

  -- Sample-level dataset (GPUDataLoader batches internally)
  let baseDs : ByteImageDataset := { data := imageData, imageSize := 784, numImages }
  let cfg := match world? with
    | some world => world.toShardConfig .interleaved true
    | none => { shardIndex := 0, numShards := 1, mode := .interleaved, dropRemainder := true }
  let ds : ShardedDataset ByteImageDataset ByteArray := shardWithConfig cfg baseDs

  -- Create loader and consume all batches
  let (totalMs, totalBatches) ← timeMs do
    let loader ← GPUDataLoader.create ds device batchSize [784] .uint8 bufferSize poolSlots
    let mut count := 0
    for _lease in loader do
      count := count + 1
    pure count

  let batchPerSec := totalBatches.toFloat * 1000.0 / totalMs
  let mbPerSec := totalBatches.toFloat * batchBytes.toFloat / totalMs / 1000.0
  IO.println s!"  {totalBatches} batches in {totalMs.toString.take 6}ms"
  IO.println s!"  {batchPerSec.toString.take 6} batch/s, {mbPerSec.toString.take 6} MB/s"
  IO.println ""

/-- Benchmark GPUDataLoader throughput with batch prefetching. -/
def benchGPULoaderPrefetch (device : DeviceId) (imageData : ByteArray) (batchSize : Nat := 64)
    (prefetchSize : Nat := 8) (bufferSize : Nat := 4) (poolSlots : Nat := 0)
    (world? : Option WorldConfig := none) : IO Unit := do
  IO.println s!"=== GPUDataLoader Prefetch Throughput (batch={batchSize}, prefetch={prefetchSize}, buffer={bufferSize}, pool={poolSlots}) ==="

  let numImages := imageData.size / 784
  let batchBytes := batchSize * 784

  let baseDs : ByteImageDataset := { data := imageData, imageSize := 784, numImages }
  let cfg := match world? with
    | some world => world.toShardConfig .interleaved true
    | none => { shardIndex := 0, numShards := 1, mode := .interleaved, dropRemainder := true }
  let ds : ShardedDataset ByteImageDataset ByteArray := shardWithConfig cfg baseDs
  let iterCfg : IteratorConfig (ShardedDataset ByteImageDataset ByteArray) := { base := ds }

  let (totalMs, totalBatches) ← timeMs do
    let loader ← GPUDataLoader.createFromIteratorCfgPrefetch iterCfg device batchSize [784]
      prefetchSize .uint8 bufferSize poolSlots
    let mut count := 0
    for _lease in loader do
      count := count + 1
    pure count

  let batchPerSec := totalBatches.toFloat * 1000.0 / totalMs
  let mbPerSec := totalBatches.toFloat * batchBytes.toFloat / totalMs / 1000.0
  IO.println s!"  {totalBatches} batches in {totalMs.toString.take 6}ms"
  IO.println s!"  {batchPerSec.toString.take 6} batch/s, {mbPerSec.toString.take 6} MB/s"
  IO.println ""

/-- Benchmark ByteSlice loader with zero-copy -/
def benchSliceLoader (device : DeviceId) (imageData : ByteArray) (batchSize : Nat := 64) (bufferSize : Nat := 4)
    (world? : Option WorldConfig := none) : IO Unit := do
  IO.println s!"=== GPUDataLoader from Slices (zero-copy, batch={batchSize}) ==="

  let numImages := imageData.size / 784
  let numBatches := numImages / batchSize
  let batchBytes := batchSize * 784

  -- Create slices (zero-copy views)
  let baseSlices := Array.range numBatches |>.map fun i =>
    ByteSlice.mk' imageData (i * batchBytes) batchBytes
  let cfg := match world? with
    | some world => world.toShardConfig .interleaved true
    | none => { shardIndex := 0, numShards := 1, mode := .interleaved, dropRemainder := true }
  let indices : Array Nat := TinyGrad4.Data.allShardIndices cfg numBatches
  let slices := indices.map fun idx => baseSlices[idx]!

  -- Create loader and consume all batches
  let (totalMs, _) ← timeMs do
    let loader ← GPUDataLoader.fromSlices slices device batchSize [784] .uint8 bufferSize
    let mut count := 0
    for _lease in loader do
      count := count + 1
    pure count

  let sliceCount := slices.size
  let batchPerSec := sliceCount.toFloat * 1000.0 / totalMs
  let mbPerSec := sliceCount.toFloat * batchBytes.toFloat / totalMs / 1000.0
  IO.println s!"  {sliceCount} batches in {totalMs.toString.take 6}ms"
  IO.println s!"  {batchPerSec.toString.take 6} batch/s, {mbPerSec.toString.take 6} MB/s"
  IO.println ""

/-- Benchmark MultiGPULoader throughput across devices. -/
def benchMultiGPULoader (devices : Array DeviceId) (imageData : ByteArray)
    (imageSize : Nat) (batchSize : Nat) (bufferSize : Nat) (poolSlots : Nat)
    (world? : Option WorldConfig := none) : IO Unit := do
  if devices.size < 2 then
    IO.println "MultiGPULoader: need at least 2 devices"
    return

  IO.println s!"=== MultiGPULoader Throughput (devices={devices.size}, batch={batchSize}, buffer={bufferSize}, pool={poolSlots}) ==="

  let numImages := imageData.size / imageSize
  let baseDs : ByteImageDataset := { data := imageData, imageSize, numImages }
  let batchBytes := batchSize * imageSize

  let (totalMs, totalBatches) ← timeMs do
    let pool ← MultiGPULoader.create baseDs devices batchSize [imageSize] .uint8 bufferSize poolSlots
      (world? := world?)
    let rounds := pool.numBatchesPerDevice
    let mut count := 0
    if rounds == 0 then
      pool.stop
      pure count
    else
      for _ in [:rounds] do
        let batches ← pool.nextAll
        for (_, lease?) in batches do
          match lease? with
          | some lease =>
              count := count + 1
              lease.release
          | none => pure ()
      pool.stop
      pure count

  if totalBatches == 0 then
    IO.println "  skipped (dataset too small for sharding)"
    IO.println ""
    return

  let batchPerSec := totalBatches.toFloat * 1000.0 / totalMs
  let mbPerSec := totalBatches.toFloat * batchBytes.toFloat / totalMs / 1000.0
  IO.println s!"  {totalBatches} batches in {totalMs.toString.take 6}ms"
  IO.println s!"  {batchPerSec.toString.take 6} batch/s, {mbPerSec.toString.take 6} MB/s"
  IO.println ""

/-- Main benchmark runner -/
def main : IO Unit := do
  IO.println "╔═══════════════════════════════════════════════════════════════╗"
  IO.println "║          GPU Data Loader Benchmark (TinyGrad4 Lean)           ║"
  IO.println "╚═══════════════════════════════════════════════════════════════╝"
  IO.println ""

  let quick ← getEnvBool "TINYGRAD4_BENCH_QUICK" false
  let defaultImages := if quick then 2000 else 10000
  let defaultBatch := if quick then 32 else 64
  let defaultBuffer := if quick then 2 else 4
  let defaultIters := if quick then 20 else 100

  let deviceEnv ← getEnvString "TG4_GPU_DEVICE" "auto"
  let allDevices ← discoverDevices
  if allDevices.isEmpty then
    IO.println "No GPU devices available!"
    return

  let world ← WorldConfig.fromEnv
  let worldShard := if world.worldSize > 1 then some world else none
  if world.worldSize > 1 then
    IO.println s!"World shard: rank={world.rank} world={world.worldSize}"

  let device ← match parseDevice deviceEnv with
    | some dev => do
        match dev with
        | .cuda idx =>
            let count ← TinyGrad4.Backend.Cuda.deviceCount
            if idx >= count then
              throw (IO.userError s!"TG4_GPU_DEVICE={deviceEnv} not in [0, {count})")
        | _ => pure ()
        if !(← dev.isAvailable) then
          throw (IO.userError s!"TG4_GPU_DEVICE={deviceEnv} is not available")
        pure dev
    | none => pure allDevices[0]!

  let deviceName ← device.name
  IO.println s!"Using device: {deviceName}"
  IO.println ""

  -- Generate synthetic data (similar to MNIST: 10k images of 784 bytes)
  let numImages ← getEnvNat "TG4_GPU_NUM_IMAGES" defaultImages
  let imageSize ← getEnvNat "TG4_GPU_IMAGE_SIZE" 784
  let batchSize ← getEnvNat "TG4_GPU_BATCH" defaultBatch
  let bufferSize ← getEnvNat "TG4_GPU_BUFFER" defaultBuffer
  let poolSlots ← getEnvNat "TG4_GPU_POOL_SLOTS" 0
  let iters ← getEnvNat "TG4_GPU_ITERS" defaultIters
  if numImages < batchSize then
    throw (IO.userError s!"TG4_GPU_NUM_IMAGES={numImages} must be >= batch size {batchSize}")
  let totalBytes := numImages * imageSize
  IO.println s!"Generating synthetic data ({numImages} x {imageSize} bytes = {totalBytes / 1024}KB)..."
  let imageData := ByteArray.mk (Array.replicate totalBytes 42)
  IO.println s!"Generated {numImages} images ({totalBytes / 1024}KB)"
  IO.println ""

  -- Run benchmarks
  benchBufferAlloc device #[1024, 4096, 16384, 65536, 262144] iters
  benchZeroCopy device imageData batchSize iters
  benchGPULoader device imageData batchSize bufferSize poolSlots worldShard
  benchGPULoaderPrefetch device imageData batchSize 8 bufferSize poolSlots worldShard
  benchSliceLoader device imageData batchSize bufferSize worldShard

  let multiEnabled ← getEnvBool "TG4_GPU_MULTI" true
  let multiDevices := allDevices.filter fun d =>
    match d with
    | .cuda _ => true
    | .metal => true
    | _ => false
  if multiEnabled && multiDevices.size >= 2 then
    benchMultiGPULoader multiDevices imageData imageSize batchSize bufferSize poolSlots worldShard

  -- Larger batch sizes
  IO.println "=== Batch Size Scaling ==="
  for bs in [32, 64, 128, 256] do
    let numBatches := numImages / bs
    let batchBytes := bs * 784

    let slices := Array.range numBatches |>.map fun i =>
      ByteSlice.mk' imageData (i * batchBytes) batchBytes

    let (ms, _) ← timeMs do
      let loader ← GPUDataLoader.fromSlices slices device batchSize [imageSize] .uint8 bufferSize
      for _lease in loader do
        pure ()

    let batchPerSec := numBatches.toFloat * 1000.0 / ms
    IO.println s!"  batch={bs}: {batchPerSec.toString.take 6} batch/s ({numBatches} batches in {ms.toString.take 5}ms)"

  IO.println ""
  IO.println "✓ Benchmark complete"

end TinyGrad4.Test

-- Export main at top level for executable
def main := TinyGrad4.Test.main
