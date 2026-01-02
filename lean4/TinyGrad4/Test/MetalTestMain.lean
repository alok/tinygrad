import TinyGrad4.Backend.Device
import TinyGrad4.Backend.Metal
import TinyGrad4.Backend.Engine
import TinyGrad4.Backend.MetalRenderer
import TinyGrad4.UOp.UOp

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false
-- Disable IO.monoNanosNow linter: benchmark timing uses raw monotonic clocks.
set_option linter.monoNanosNow false

/-!
# Metal Backend End-to-End Test

Tests the complete Metal backend pipeline:
1. Device abstraction with typeclasses
2. Buffer allocation/copy via FFI
3. Shader compilation
4. Kernel execution
5. Result verification
-/

namespace TinyGrad4.Test.MetalTestMain

open TinyGrad4.Backend
open TinyGrad4.Backend.Metal

/-- Test Metal device info -/
def testDeviceInfo : IO Unit := do
  IO.println "=== Metal Device Info ==="
  let name ← metalDeviceName
  IO.println s!"Device: {name}"
  IO.println ""

/-- Test buffer roundtrip -/
def testBufferRoundtrip : IO Unit := do
  IO.println "=== Buffer Roundtrip Test ==="

  -- Create test data
  let testData : FloatArray := ⟨#[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]⟩
  IO.println s!"Input:  {testData.data.toList}"

  -- Allocate, copy in, copy out
  let buf ← metalAlloc testData.size
  metalCopyIn buf testData
  let result ← metalCopyOut buf
  metalFree buf

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

  let addSource := "#include <metal_stdlib>
using namespace metal;

kernel void test_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] + b[gid];
}
"

  let size := 8
  let a : FloatArray := ⟨#[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]⟩
  let b : FloatArray := ⟨#[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]⟩

  IO.println s!"a = {a.data.toList}"
  IO.println s!"b = {b.data.toList}"

  -- Allocate buffers
  let bufA ← metalAlloc size
  let bufB ← metalAlloc size
  let bufOut ← metalAlloc size

  metalCopyIn bufA a
  metalCopyIn bufB b

  -- Compile kernel
  let prog ← metalCompile "test_add" addSource

  -- Launch
  let bufs := #[bufA, bufB, bufOut]
  metalLaunch prog bufs size 1 1 256 1 1
  metalSync

  -- Get result
  let result ← metalCopyOut bufOut

  -- Cleanup
  metalFree bufA
  metalFree bufB
  metalFree bufOut

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

  let mulSource := "#include <metal_stdlib>
using namespace metal;

kernel void test_mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] * b[gid];
}
"

  let size := 4
  let a : FloatArray := ⟨#[2.0, 3.0, 4.0, 5.0]⟩
  let b : FloatArray := ⟨#[10.0, 10.0, 10.0, 10.0]⟩

  IO.println s!"a = {a.data.toList}"
  IO.println s!"b = {b.data.toList}"

  let bufA ← metalAlloc size
  let bufB ← metalAlloc size
  let bufOut ← metalAlloc size

  metalCopyIn bufA a
  metalCopyIn bufB b

  let prog ← metalCompile "test_mul" mulSource
  metalLaunch prog #[bufA, bufB, bufOut] size 1 1 256 1 1
  metalSync

  let result ← metalCopyOut bufOut

  metalFree bufA
  metalFree bufB
  metalFree bufOut

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

/-- Test fused add+mul kernel -/
def testFusedKernel : IO Unit := do
  IO.println "=== Fused Add+Mul Kernel Test ==="

  let fusedSource := "#include <metal_stdlib>
using namespace metal;

kernel void fused_add_mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* c [[buffer(2)]],
    device float* out [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = (a[gid] + b[gid]) * c[gid];
}
"

  let size := 4
  let a : FloatArray := ⟨#[1.0, 2.0, 3.0, 4.0]⟩
  let b : FloatArray := ⟨#[1.0, 1.0, 1.0, 1.0]⟩
  let c : FloatArray := ⟨#[2.0, 2.0, 2.0, 2.0]⟩

  IO.println s!"a = {a.data.toList}"
  IO.println s!"b = {b.data.toList}"
  IO.println s!"c = {c.data.toList}"

  let bufA ← metalAlloc size
  let bufB ← metalAlloc size
  let bufC ← metalAlloc size
  let bufOut ← metalAlloc size

  metalCopyIn bufA a
  metalCopyIn bufB b
  metalCopyIn bufC c

  let prog ← metalCompile "fused_add_mul" fusedSource
  metalLaunch prog #[bufA, bufB, bufC, bufOut] size 1 1 256 1 1
  metalSync

  let result ← metalCopyOut bufOut

  metalFree bufA
  metalFree bufB
  metalFree bufC
  metalFree bufOut

  IO.println s!"(a + b) * c = {result.data.toList}"

  -- Expected: (1+1)*2=4, (2+1)*2=6, (3+1)*2=8, (4+1)*2=10
  let expected : List Float := [4.0, 6.0, 8.0, 10.0]
  let mut passed := true
  for i in [:size] do
    if _h : i < result.size then
      let diff := (result.data[i]! - expected[i]!).abs
      if diff > 0.001 then
        passed := false

  if passed then
    IO.println "✓ Fused kernel test passed"
  else
    IO.println "✗ Fused kernel test failed"
  IO.println ""

/-- Benchmark: large vector add -/
def benchmarkVectorAdd : IO Unit := do
  IO.println "=== Vector Add Benchmark ==="

  let size := 1000000  -- 1M elements
  IO.println s!"Size: {size} elements ({size * 4 / 1000000} MB)"

  let addSource := "#include <metal_stdlib>
using namespace metal;

kernel void bench_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] + b[gid];
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
  let bufA ← metalAlloc size
  let bufB ← metalAlloc size
  let bufOut ← metalAlloc size

  metalCopyIn bufA a
  metalCopyIn bufB b

  -- Compile
  let prog ← metalCompile "bench_add" addSource

  -- Warmup
  for _ in [:3] do
    metalLaunch prog #[bufA, bufB, bufOut] size 1 1 256 1 1
    metalSync

  -- Benchmark (async - only sync at end, like Python tinygrad)
  let iterations := 100
  let startTime ← IO.monoNanosNow
  for _ in [:iterations] do
    metalLaunch prog #[bufA, bufB, bufOut] size 1 1 256 1 1
  metalSync  -- Only sync once at the end
  let endTime ← IO.monoNanosNow

  let totalMs := (endTime - startTime).toFloat / 1e6
  let avgUs := totalMs * 1000.0 / iterations.toFloat
  let gflops := (size.toFloat / avgUs) * 1e6 / 1e9
  let bandwidth := (3.0 * size.toFloat * 4.0 / avgUs) * 1e6 / 1e9  -- read 2, write 1

  IO.println s!"Time: {avgUs} μs"
  IO.println s!"Throughput: {gflops} GFLOP/s"
  IO.println s!"Bandwidth: {bandwidth} GB/s"

  metalFree bufA
  metalFree bufB
  metalFree bufOut
  IO.println ""

/-- Benchmark: vectorized (float4) add -/
def benchmarkVectorAddFloat4 : IO Unit := do
  IO.println "=== Vector Add float4 Benchmark ==="

  let size := 1000000  -- 1M elements
  let vecSize := size / 4
  IO.println s!"Size: {size} elements (vec4), {size * 4 / 1000000} MB"

  let addSource := "#include <metal_stdlib>
using namespace metal;

kernel void bench_add_float4(
    device const float4* a [[buffer(0)]],
    device const float4* b [[buffer(1)]],
    device float4* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] + b[gid];
}
"

  let mut aData : Array Float := #[]
  let mut bData : Array Float := #[]
  for i in [:size] do
    aData := aData.push ((i % 1000).toFloat / 1000.0)
    bData := bData.push (((i + 500) % 1000).toFloat / 1000.0)

  let a : FloatArray := ⟨aData⟩
  let b : FloatArray := ⟨bData⟩

  let bufA ← metalAlloc size
  let bufB ← metalAlloc size
  let bufOut ← metalAlloc size

  metalCopyIn bufA a
  metalCopyIn bufB b

  let prog ← metalCompile "bench_add_float4" addSource

  -- Warmup
  for _ in [:3] do
    metalLaunch prog #[bufA, bufB, bufOut] vecSize 1 1 256 1 1
    metalSync

  let iterations := 100
  let startTime ← IO.monoNanosNow
  for _ in [:iterations] do
    metalLaunch prog #[bufA, bufB, bufOut] vecSize 1 1 256 1 1
  metalSync
  let endTime ← IO.monoNanosNow

  let totalMs := (endTime - startTime).toFloat / 1e6
  let avgUs := totalMs * 1000.0 / iterations.toFloat
  let gflops := (size.toFloat / avgUs) * 1e6 / 1e9
  let bandwidth := (3.0 * size.toFloat * 4.0 / avgUs) * 1e6 / 1e9

  IO.println s!"Time: {avgUs} μs"
  IO.println s!"Throughput: {gflops} GFLOP/s"
  IO.println s!"Bandwidth: {bandwidth} GB/s"

  metalFree bufA
  metalFree bufB
  metalFree bufOut
  IO.println ""

/-- Benchmark: reduction -/
def benchmarkReduce : IO Unit := do
  IO.println "=== Reduction Benchmark ==="

  let size := 1000000
  IO.println s!"Size: {size} elements"

  let reduceSource := "#include <metal_stdlib>
using namespace metal;

kernel void reduce_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    shared[lid] = (gid < 1000000) ? input[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) output[group_id] = shared[0];
}
"

  let mut data : Array Float := #[]
  for i in [:size] do
    data := data.push ((i % 100).toFloat / 100.0)

  let input : FloatArray := ⟨data⟩
  let threadgroupSize := 256
  let numGroups := (size + threadgroupSize - 1) / threadgroupSize

  let bufIn ← metalAlloc size
  let bufOut ← metalAlloc numGroups

  metalCopyIn bufIn input

  let prog ← metalCompile "reduce_sum" reduceSource

  -- Warmup
  for _ in [:3] do
    metalLaunch prog #[bufIn, bufOut] size 1 1 threadgroupSize 1 1
    metalSync

  let iterations := 100
  let startTime ← IO.monoNanosNow
  for _ in [:iterations] do
    metalLaunch prog #[bufIn, bufOut] size 1 1 threadgroupSize 1 1
  metalSync
  let endTime ← IO.monoNanosNow

  let totalMs := (endTime - startTime).toFloat / 1e6
  let avgUs := totalMs * 1000.0 / iterations.toFloat
  let bandwidth := (size.toFloat * 4.0 / avgUs) * 1e6 / 1e9

  IO.println s!"Time: {avgUs} μs"
  IO.println s!"Bandwidth: {bandwidth} GB/s"

  metalFree bufIn
  metalFree bufOut
  IO.println ""

/-- Benchmark: fused ReLU+Mul+Add -/
def benchmarkFusedEwise : IO Unit := do
  IO.println "=== Fused ReLU+Mul+Add Benchmark ==="

  let size := 1000000
  IO.println s!"Size: {size} elements"

  let fusedSource := "#include <metal_stdlib>
using namespace metal;

kernel void fused_relu_mul_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* c [[buffer(2)]],
    device float* out [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    float relu_a = max(a[gid], 0.0f);
    out[gid] = relu_a * b[gid] + c[gid];
}
"

  let mut aData : Array Float := #[]
  let mut bData : Array Float := #[]
  let mut cData : Array Float := #[]
  for i in [:size] do
    aData := aData.push (Float.ofInt ((i : Int) - (size / 2 : Int)) / 1000.0)
    bData := bData.push 2.0
    cData := cData.push 1.0

  let a : FloatArray := ⟨aData⟩
  let b : FloatArray := ⟨bData⟩
  let c : FloatArray := ⟨cData⟩

  let bufA ← metalAlloc size
  let bufB ← metalAlloc size
  let bufC ← metalAlloc size
  let bufOut ← metalAlloc size

  metalCopyIn bufA a
  metalCopyIn bufB b
  metalCopyIn bufC c

  let prog ← metalCompile "fused_relu_mul_add" fusedSource

  -- Warmup
  for _ in [:3] do
    metalLaunch prog #[bufA, bufB, bufC, bufOut] size 1 1 256 1 1
    metalSync

  let iterations := 100
  let startTime ← IO.monoNanosNow
  for _ in [:iterations] do
    metalLaunch prog #[bufA, bufB, bufC, bufOut] size 1 1 256 1 1
  metalSync
  let endTime ← IO.monoNanosNow

  let totalMs := (endTime - startTime).toFloat / 1e6
  let avgUs := totalMs * 1000.0 / iterations.toFloat
  let bandwidth := (4.0 * size.toFloat * 4.0 / avgUs) * 1e6 / 1e9  -- 3 read + 1 write
  let gflops := (3.0 * size.toFloat / avgUs) * 1e6 / 1e9  -- relu + mul + add

  IO.println s!"Time: {avgUs} μs"
  IO.println s!"Throughput: {gflops} GFLOP/s"
  IO.println s!"Bandwidth: {bandwidth} GB/s"

  metalFree bufA
  metalFree bufB
  metalFree bufC
  metalFree bufOut
  IO.println ""

end TinyGrad4.Test.MetalTestMain

/-- Main entry point (must be at top level for Lean to generate C main) -/
def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║     TinyGrad4 Metal Backend Benchmark Suite              ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"
  IO.println ""

  TinyGrad4.Test.MetalTestMain.testDeviceInfo
  TinyGrad4.Test.MetalTestMain.testBufferRoundtrip
  TinyGrad4.Test.MetalTestMain.testAddKernel
  TinyGrad4.Test.MetalTestMain.testMulKernel
  TinyGrad4.Test.MetalTestMain.testFusedKernel

  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║                    BENCHMARKS                            ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"
  IO.println ""

  TinyGrad4.Test.MetalTestMain.benchmarkVectorAdd
  TinyGrad4.Test.MetalTestMain.benchmarkVectorAddFloat4
  TinyGrad4.Test.MetalTestMain.benchmarkReduce
  TinyGrad4.Test.MetalTestMain.benchmarkFusedEwise

  IO.println "=== All Tests Complete ==="
