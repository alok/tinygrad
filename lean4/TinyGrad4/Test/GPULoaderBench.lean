import TinyGrad4.Data.GPULoader

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
      let buf ← GPUBuffer.alloc device size
      buf.copyIn data
      buf.free

    -- Measure allocation + copy
    let (allocMs, _) ← timeMs do
      for _ in [:iters] do
        let buf ← GPUBuffer.alloc device size
        buf.copyIn data
        buf.free

    let avgUs := allocMs * 1000.0 / iters.toFloat
    let mbPerSec := size.toFloat * iters.toFloat / allocMs / 1000.0
    IO.println s!"  {size / 1024}KB: {avgUs.toString.take 6}μs/op, {mbPerSec.toString.take 6} MB/s"

  device.sync
  IO.println ""

/-- Benchmark zero-copy vs regular copy on Metal -/
def benchZeroCopy (imageData : ByteArray) (batchSize : Nat := 64) (iters : Nat := 100) : IO Unit := do
  IO.println "=== Zero-Copy vs Regular Copy (Metal) ==="

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
        let buf ← ByteSlice.toGPUBuffer slice .metal .uint8
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
        let buf ← ByteSlice.toGPUBufferZeroCopy slice .metal .uint8
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

/-- Benchmark GPUDataLoader throughput -/
def benchGPULoader (imageData : ByteArray) (batchSize : Nat := 64) (bufferSize : Nat := 4) : IO Unit := do
  IO.println s!"=== GPUDataLoader Throughput (batch={batchSize}, buffer={bufferSize}) ==="

  let numImages := imageData.size / 784
  let numBatches := numImages / batchSize
  let batchBytes := batchSize * 784

  -- Create chunked dataset
  let chunks := Array.range numBatches |>.map fun i =>
    let start := i * batchBytes
    imageData.extract start (start + batchBytes)

  let ds : ByteChunkDataset := { chunks }

  -- Create loader and consume all batches
  let (totalMs, _) ← timeMs do
    let loader ← GPUDataLoader.create ds .metal batchSize bufferSize .uint8
    let mut count := 0
    for buf in loader do
      count := count + 1
      buf.free
    pure count

  let batchPerSec := numBatches.toFloat * 1000.0 / totalMs
  let mbPerSec := numBatches.toFloat * batchBytes.toFloat / totalMs / 1000.0
  IO.println s!"  {numBatches} batches in {totalMs.toString.take 6}ms"
  IO.println s!"  {batchPerSec.toString.take 6} batch/s, {mbPerSec.toString.take 6} MB/s"
  IO.println ""

/-- Benchmark ByteSlice loader with zero-copy -/
def benchSliceLoader (imageData : ByteArray) (batchSize : Nat := 64) (bufferSize : Nat := 4) : IO Unit := do
  IO.println s!"=== GPUDataLoader from Slices (zero-copy, batch={batchSize}) ==="

  let numImages := imageData.size / 784
  let numBatches := numImages / batchSize
  let batchBytes := batchSize * 784

  -- Create slices (zero-copy views)
  let slices := Array.range numBatches |>.map fun i =>
    ByteSlice.mk' imageData (i * batchBytes) batchBytes

  -- Create loader and consume all batches
  let (totalMs, _) ← timeMs do
    let loader ← GPUDataLoader.fromSlices slices .metal bufferSize .uint8
    let mut count := 0
    for buf in loader do
      count := count + 1
      buf.free
    pure count

  let batchPerSec := numBatches.toFloat * 1000.0 / totalMs
  let mbPerSec := numBatches.toFloat * batchBytes.toFloat / totalMs / 1000.0
  IO.println s!"  {numBatches} batches in {totalMs.toString.take 6}ms"
  IO.println s!"  {batchPerSec.toString.take 6} batch/s, {mbPerSec.toString.take 6} MB/s"
  IO.println ""

/-- Main benchmark runner -/
def main : IO Unit := do
  IO.println "╔═══════════════════════════════════════════════════════════════╗"
  IO.println "║          GPU Data Loader Benchmark (TinyGrad4 Lean)           ║"
  IO.println "╚═══════════════════════════════════════════════════════════════╝"
  IO.println ""

  -- Discover devices
  let devices ← discoverDevices
  if devices.isEmpty then
    IO.println "No GPU devices available!"
    return

  for dev in devices do
    let name ← dev.name
    IO.println s!"Found device: {name}"
  IO.println ""

  -- Generate synthetic data (similar to MNIST: 10k images of 784 bytes)
  IO.println "Generating synthetic data (10k x 784 bytes = 7.84MB)..."
  let numImages := 10000
  let imageSize := 784
  let totalBytes := numImages * imageSize
  let imageData := ByteArray.mk (Array.replicate totalBytes 42)
  IO.println s!"Generated {numImages} images ({totalBytes / 1024}KB)"
  IO.println ""

  -- Run benchmarks
  benchBufferAlloc .metal #[1024, 4096, 16384, 65536, 262144] 100
  benchZeroCopy imageData 64 100
  benchGPULoader imageData 64 4
  benchSliceLoader imageData 64 4

  -- Larger batch sizes
  IO.println "=== Batch Size Scaling ==="
  for bs in [32, 64, 128, 256] do
    let numBatches := numImages / bs
    let batchBytes := bs * 784

    let slices := Array.range numBatches |>.map fun i =>
      ByteSlice.mk' imageData (i * batchBytes) batchBytes

    let (ms, _) ← timeMs do
      let loader ← GPUDataLoader.fromSlices slices .metal 4 .uint8
      for buf in loader do
        buf.free

    let batchPerSec := numBatches.toFloat * 1000.0 / ms
    IO.println s!"  batch={bs}: {batchPerSec.toString.take 6} batch/s ({numBatches} batches in {ms.toString.take 5}ms)"

  IO.println ""
  IO.println "✓ Benchmark complete"

end TinyGrad4.Test

-- Export main at top level for executable
def main := TinyGrad4.Test.main
