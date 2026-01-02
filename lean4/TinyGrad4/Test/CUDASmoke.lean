import TinyGrad4.Backend.Cuda
import TinyGrad4.Data.GPULoader
-- Disable IO.monoNanosNow linter: benchmark timing uses raw monotonic clocks.
set_option linter.monoNanosNow false

/-!
# CUDA Smoke Test

Basic verification that CUDA FFI works:
1. Device detection
2. Buffer allocation/copy
-/

open TinyGrad4.Backend.Cuda
open TinyGrad4.Data.GPULoader

namespace TinyGrad4.Test.CUDASmoke

/-- Measure time in milliseconds -/
def timeMs (action : IO α) : IO (Float × α) := do
  let start ← IO.monoNanosNow
  let result ← action
  let stop ← IO.monoNanosNow
  let ms := (stop - start).toFloat / 1_000_000.0
  pure (ms, result)

/-- Test CUDA device detection -/
def testDeviceDetection : IO Bool := do
  IO.println "Testing CUDA device detection..."
  let result ← try
    let name ← cudaDeviceName
    IO.println s!"  ✓ Found CUDA device: {name}"
    pure (some true)
  catch e =>
    IO.println s!"  ✗ CUDA not available: {e}"
    pure none
  match result with
  | some b => pure b
  | none => pure false

/-- Test CUDA buffer operations -/
def testBufferOps : IO Bool := do
  IO.println "Testing CUDA buffer operations..."
  let result ← try
    -- Allocate 1024 bytes
    let buf ← cudaAllocBytes 1024
    IO.println "  ✓ Allocated 1024-byte buffer"

    -- Create test data
    let testData := ByteArray.mk (Array.replicate 1024 42)
    cudaCopyInBytes buf testData
    IO.println "  ✓ Copied data to GPU"

    -- Copy back
    let resultData ← cudaCopyOutBytes buf 1024
    IO.println s!"  ✓ Copied data back ({resultData.size} bytes)"

    -- Verify
    let correct := resultData.data.all (· == 42)
    if correct then
      IO.println "  ✓ Data verification passed"
    else
      IO.println "  ✗ Data verification failed"

    cudaFree buf
    IO.println "  ✓ Buffer freed"
    pure (some correct)
  catch e =>
    IO.println s!"  ✗ Buffer test failed: {e}"
    pure none
  match result with
  | some b => pure b
  | none => pure false

/-- Benchmark CUDA buffer throughput -/
def benchThroughput : IO Unit := do
  IO.println "Benchmarking CUDA throughput..."
  let result ← try
    let sizes := #[1024, 4096, 16384, 65536, 262144, 1048576]  -- 1KB to 1MB
    for size in sizes do
      let testData := ByteArray.mk (Array.replicate size 42)
      let iters := 100

      let (ms, _) ← timeMs do
        for _ in [:iters] do
          let buf ← cudaAllocBytes size
          cudaCopyInBytes buf testData
          let _ ← cudaCopyOutBytes buf size
          cudaFree buf

      let avgUs := ms * 1000.0 / iters.toFloat
      let mbPerSec := size.toFloat * 2 * iters.toFloat / ms / 1000.0  -- 2x for round-trip
      IO.println s!"  {size / 1024}KB: {avgUs.toString.take 6}μs/op, {mbPerSec.toString.take 6} MB/s"
    pure (some ())
  catch e =>
    IO.println s!"  ✗ Benchmark failed: {e}"
    pure none
  let _ := result  -- suppress warning

/-- Test GPULoader with CUDA device -/
def testGPULoader : IO Bool := do
  IO.println "Testing GPULoader with CUDA..."
  let result ← try
    -- Check if CUDA is available
    let devices ← discoverDevices
    let hasCuda := devices.any fun d => match d with
      | .cuda _ => true
      | _ => false

    if !hasCuda then
      IO.println "  ⊘ No CUDA device in GPULoader (expected on macOS)"
      pure (some true)
    else
      -- Allocate buffer via GPULoader API
      let buf ← GPUBuffer.alloc (.cuda 0) 1024 .uint8
      IO.println s!"  ✓ Allocated via GPULoader ({buf.byteSize} bytes)"

      let testData := ByteArray.mk (Array.replicate 1024 42)
      buf.copyIn testData
      IO.println "  ✓ Copied via GPULoader"

      buf.free
      IO.println "  ✓ Freed via GPULoader"
      pure (some true)
  catch e =>
    IO.println s!"  ✗ GPULoader test failed: {e}"
    pure none
  match result with
  | some b => pure b
  | none => pure false

def main : IO UInt32 := do
  IO.println "╔═══════════════════════════════════════════════════════════════╗"
  IO.println "║                  CUDA Smoke Test (TinyGrad4)                  ║"
  IO.println "╚═══════════════════════════════════════════════════════════════╝"
  IO.println ""

  let deviceOk ← testDeviceDetection
  if !deviceOk then
    IO.println ""
    IO.println "CUDA not available - skipping remaining tests"
    IO.println "This is expected on macOS. Run on Linux with CUDA."
    return 0

  IO.println ""
  let bufferOk ← testBufferOps
  IO.println ""
  let loaderOk ← testGPULoader
  IO.println ""
  benchThroughput

  IO.println ""
  if deviceOk && bufferOk && loaderOk then
    IO.println "═══════════════════════════════════════════════════════════════"
    IO.println "✓ All CUDA tests passed!"
    return 0
  else
    IO.println "═══════════════════════════════════════════════════════════════"
    IO.println "✗ Some CUDA tests failed"
    return 1

end TinyGrad4.Test.CUDASmoke

def main := TinyGrad4.Test.CUDASmoke.main
