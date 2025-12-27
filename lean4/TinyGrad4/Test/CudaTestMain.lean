import TinyGrad4.Backend.Device
import TinyGrad4.Backend.Cuda

/-!
# CUDA Backend End-to-End Test

Tests the complete CUDA backend pipeline:
1. Device abstraction with typeclasses
2. Buffer allocation/copy via FFI
3. Shader compilation via NVRTC
4. Kernel execution
5. Result verification
-/

namespace TinyGrad4.Test.CudaTestMain

open TinyGrad4.Backend
open TinyGrad4.Backend.Cuda

/-- Test CUDA device info -/
def testDeviceInfo : IO Unit := do
  IO.println "=== CUDA Device Info ==="
  let name ← cudaDeviceName
  IO.println s!"Device: {name}"
  IO.println ""

/-- Test buffer roundtrip -/
def testBufferRoundtrip : IO Unit := do
  IO.println "=== Buffer Roundtrip Test ==="

  -- Create test data
  let testData : FloatArray := ⟨#[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]⟩
  IO.println s!"Input:  {testData.data.toList}"

  -- Allocate, copy in, copy out
  let buf ← cudaAlloc testData.size
  cudaCopyIn buf testData
  let result ← cudaCopyOut buf
  cudaFree buf

  IO.println s!"Output: {result.data.toList}"

  -- Verify
  let mut maxDiff : Float := 0.0
  for i in [:testData.size] do
    if _h : i < testData.size ∧ i < result.size then
      let diff := (testData.data[i]! - result.data[i]!).abs
      maxDiff := max maxDiff diff

  if maxDiff < 0.0001 then
    IO.println s!"✓ Roundtrip passed (max diff: {maxDiff})"
  else
    IO.println s!"✗ Roundtrip failed (max diff: {maxDiff})"
  IO.println ""

/-- Test simple add kernel -/
def testAddKernel : IO Unit := do
  IO.println "=== Add Kernel Test ==="

  let addSource := "extern \"C\" __global__ void test_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    out[gid] = a[gid] + b[gid];
}
"

  let size := 8
  let a : FloatArray := ⟨#[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]⟩
  let b : FloatArray := ⟨#[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]⟩

  IO.println s!"a = {a.data.toList}"
  IO.println s!"b = {b.data.toList}"

  -- Allocate buffers
  let bufA ← cudaAlloc size
  let bufB ← cudaAlloc size
  let bufOut ← cudaAlloc size

  cudaCopyIn bufA a
  cudaCopyIn bufB b

  -- Compile kernel
  let prog ← cudaCompile "test_add" addSource

  -- Launch
  let bufs := #[bufA, bufB, bufOut]
  cudaLaunch prog bufs size 1 1 256 1 1
  cudaSync

  -- Get result
  let result ← cudaCopyOut bufOut

  -- Cleanup
  cudaFree bufA
  cudaFree bufB
  cudaFree bufOut

  IO.println s!"a + b = {result.data.toList}"

  -- Verify
  let expected : List Float := [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0]
  let mut passed := true
  for i in [:size] do
    if _h : i < result.size then
      let diff := (result.data[i]! - expected[i]!).abs
      if diff > 0.001 then
        passed := false

  if passed then
    IO.println "✓ Add kernel test passed"
  else
    IO.println "✗ Add kernel test failed"
  IO.println ""

/-- Test multiply kernel -/
def testMulKernel : IO Unit := do
  IO.println "=== Multiply Kernel Test ==="

  let mulSource := "extern \"C\" __global__ void test_mul(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    out[gid] = a[gid] * b[gid];
}
"

  let size := 4
  let a : FloatArray := ⟨#[2.0, 3.0, 4.0, 5.0]⟩
  let b : FloatArray := ⟨#[10.0, 10.0, 10.0, 10.0]⟩

  IO.println s!"a = {a.data.toList}"
  IO.println s!"b = {b.data.toList}"

  let bufA ← cudaAlloc size
  let bufB ← cudaAlloc size
  let bufOut ← cudaAlloc size

  cudaCopyIn bufA a
  cudaCopyIn bufB b

  let prog ← cudaCompile "test_mul" mulSource
  cudaLaunch prog #[bufA, bufB, bufOut] size 1 1 256 1 1
  cudaSync

  let result ← cudaCopyOut bufOut

  cudaFree bufA
  cudaFree bufB
  cudaFree bufOut

  IO.println s!"a * b = {result.data.toList}"

  let expected : List Float := [20.0, 30.0, 40.0, 50.0]
  let mut passed := true
  for i in [:size] do
    if _h : i < result.size then
      let diff := (result.data[i]! - expected[i]!).abs
      if diff > 0.001 then
        passed := false

  if passed then
    IO.println "✓ Multiply kernel test passed"
  else
    IO.println "✗ Multiply kernel test failed"
  IO.println ""

/-- Benchmark: large vector add -/
def benchmarkVectorAdd : IO Unit := do
  IO.println "=== Vector Add Benchmark ==="

  let size := 1000000  -- 1M elements
  IO.println s!"Size: {size} elements ({size * 4 / 1000000} MB)"

  let addSource := "extern \"C\" __global__ void bench_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < 1000000) {
        out[gid] = a[gid] + b[gid];
    }
}
"

  -- Create data
  let mut aData : Array Float := #[]
  let mut bData : Array Float := #[]
  for i in [:size] do
    aData := aData.push ((i % 1000).toFloat / 1000.0)
    bData := bData.push (((i + 500) % 1000).toFloat / 1000.0)

  let a : FloatArray := ⟨aData⟩
  let b : FloatArray := ⟨bData⟩

  -- Allocate
  let bufA ← cudaAlloc size
  let bufB ← cudaAlloc size
  let bufOut ← cudaAlloc size

  cudaCopyIn bufA a
  cudaCopyIn bufB b

  -- Compile
  let prog ← cudaCompile "bench_add" addSource

  -- Warmup
  for _ in [:3] do
    cudaLaunch prog #[bufA, bufB, bufOut] size 1 1 256 1 1
    cudaSync

  -- Benchmark (async - only sync at end)
  let iterations := 100
  let startTime ← IO.monoNanosNow
  for _ in [:iterations] do
    cudaLaunch prog #[bufA, bufB, bufOut] size 1 1 256 1 1
  cudaSync  -- Only sync once at the end
  let endTime ← IO.monoNanosNow

  let totalMs := (endTime - startTime).toFloat / 1e6
  let avgUs := totalMs * 1000.0 / iterations.toFloat
  let gflops := (size.toFloat / avgUs) * 1e6 / 1e9
  let bandwidth := (3.0 * size.toFloat * 4.0 / avgUs) * 1e6 / 1e9  -- read 2, write 1

  IO.println s!"Time: {avgUs} μs"
  IO.println s!"Throughput: {gflops} GFLOP/s"
  IO.println s!"Bandwidth: {bandwidth} GB/s"

  cudaFree bufA
  cudaFree bufB
  cudaFree bufOut
  IO.println ""

end TinyGrad4.Test.CudaTestMain

/-- Main entry point (must be at top level for Lean to generate C main) -/
def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║     TinyGrad4 CUDA Backend End-to-End Test               ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"
  IO.println ""

  TinyGrad4.Test.CudaTestMain.testDeviceInfo
  TinyGrad4.Test.CudaTestMain.testBufferRoundtrip
  TinyGrad4.Test.CudaTestMain.testAddKernel
  TinyGrad4.Test.CudaTestMain.testMulKernel
  TinyGrad4.Test.CudaTestMain.benchmarkVectorAdd

  IO.println "=== All Tests Complete ==="
