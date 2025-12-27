import TinyGrad4.Benchmark.Framework
import TinyGrad4.Backend.Metal
import TinyGrad4.Backend.Vectorization

/-!
# Metal Benchmark Kernels

Implements `BenchmarkKernel` for Metal backend.
-/

namespace TinyGrad4.Benchmark.Metal

open TinyGrad4.Benchmark
open TinyGrad4.Backend.Metal
open TinyGrad4.Backend.Vectorization

/-! ## Vector Add Benchmark -/

/-- Metal kernel source for vector add (scalar) -/
def vectorAddSource : String :=
"#include <metal_stdlib>
using namespace metal;

kernel void bench_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] + b[gid];
}"

/-- Metal kernel source for vector add (float4 vectorized) -/
def vectorAddFloat4Source (size : Nat) : String :=
  let numVecIters := size / 4
  "#include <metal_stdlib>
using namespace metal;

kernel void bench_add_vec4(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < " ++ toString numVecIters ++ ") {
        uint i = gid * 4;
        float4 va = *((device const float4*)(a + i));
        float4 vb = *((device const float4*)(b + i));
        *((device float4*)(out + i)) = va + vb;
    }
}"

/-- State for vector add benchmark -/
structure VectorAddState where
  bufA : MetalBuffer
  bufB : MetalBuffer
  bufOut : MetalBuffer
  prog : MetalProgram
  size : Nat
  hostA : FloatArray
  hostB : FloatArray

/-- Create vector add benchmark kernel for given size -/
def makeVectorAddKernel (size : Nat) : IO BenchmarkKernel := do
  -- Mutable state holder
  let stateRef : IO.Ref (Option VectorAddState) ← IO.mkRef none

  return {
    setup := do
      -- Allocate buffers
      let bufA ← metalAlloc size
      let bufB ← metalAlloc size
      let bufOut ← metalAlloc size

      -- Initialize host data
      let hostA := FloatArray.mk ((Array.range size).map fun i =>
        (i % 1000).toFloat / 1000.0)
      let hostB := FloatArray.mk ((Array.range size).map fun i =>
        ((i + 500) % 1000).toFloat / 1000.0)

      -- Copy to GPU
      metalCopyIn bufA hostA
      metalCopyIn bufB hostB

      -- Compile kernel
      let prog ← metalCompile "bench_add" vectorAddSource

      -- Store state
      stateRef.set (some { bufA, bufB, bufOut, prog, size, hostA, hostB })

    runOnce := do
      match ← stateRef.get with
      | none => throw (IO.Error.userError "benchmark not set up")
      | some s =>
        let bufs := #[s.bufA, s.bufB, s.bufOut]
        -- Use 256 threads per threadgroup
        metalLaunch s.prog bufs s.size 1 1 256 1 1

    sync := metalSync

    verify := do
      match ← stateRef.get with
      | none => return false
      | some s =>
        let result ← metalCopyOut s.bufOut
        -- Check a few samples
        let mut maxDiff : Float := 0.0
        for i in [:min 1000 s.size] do
          let expected := s.hostA.get! i + s.hostB.get! i
          let actual := result.get! i
          let diff := (actual - expected).abs
          if diff > maxDiff then maxDiff := diff
        return maxDiff < 0.0001

    cleanup := do
      match ← stateRef.get with
      | none => pure ()
      | some s =>
        metalFree s.bufA
        metalFree s.bufB
        metalFree s.bufOut
        stateRef.set none

    backendName := "METAL"

    deviceName := metalDeviceName
  }

/-! ## Vectorized Vector Add Benchmark -/

/-- Create float4-vectorized vector add benchmark kernel.
    The alignment proof `h` guarantees `size % 4 = 0` at compile time via `native_decide`.
    Example: `makeVectorAddFloat4Kernel 1_000_000` works because 1M is divisible by 4. -/
def makeVectorAddFloat4Kernel (size : Nat)
    (_h : VectorAligned .w4 size := by native_decide) : IO BenchmarkKernel := do
  -- No runtime check needed - _h proves alignment at compile time

  let stateRef : IO.Ref (Option VectorAddState) ← IO.mkRef none

  return {
    setup := do
      let bufA ← metalAlloc size
      let bufB ← metalAlloc size
      let bufOut ← metalAlloc size

      let hostA := FloatArray.mk ((Array.range size).map fun i =>
        (i % 1000).toFloat / 1000.0)
      let hostB := FloatArray.mk ((Array.range size).map fun i =>
        ((i + 500) % 1000).toFloat / 1000.0)

      metalCopyIn bufA hostA
      metalCopyIn bufB hostB

      let prog ← metalCompile "bench_add_vec4" (vectorAddFloat4Source size)

      stateRef.set (some { bufA, bufB, bufOut, prog, size, hostA, hostB })

    runOnce := do
      match ← stateRef.get with
      | none => throw (IO.Error.userError "benchmark not set up")
      | some s =>
        let bufs := #[s.bufA, s.bufB, s.bufOut]
        -- Each thread processes 4 elements
        let numVecIters := s.size / 4
        metalLaunch s.prog bufs numVecIters 1 1 256 1 1

    sync := metalSync

    verify := do
      match ← stateRef.get with
      | none => return false
      | some s =>
        let result ← metalCopyOut s.bufOut
        let mut maxDiff : Float := 0.0
        for i in [:min 1000 s.size] do
          let expected := s.hostA.get! i + s.hostB.get! i
          let actual := result.get! i
          let diff := (actual - expected).abs
          if diff > maxDiff then maxDiff := diff
        return maxDiff < 0.0001

    cleanup := do
      match ← stateRef.get with
      | none => pure ()
      | some s =>
        metalFree s.bufA
        metalFree s.bufB
        metalFree s.bufOut
        stateRef.set none

    backendName := "METAL (float4)"

    deviceName := metalDeviceName
  }

/-! ## Reduction Benchmark -/

/-- Metal kernel source for sum reduction -/
def sumReduceSource : String :=
"#include <metal_stdlib>
using namespace metal;

kernel void bench_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    uint gridSize [[threads_per_grid]]
) {
    // Simple serial reduction per thread
    // (Real implementation would use threadgroup memory)
    if (gid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < gridSize; i++) {
            sum += input[i];
        }
        output[0] = sum;
    }
}"

/-! ## Pre-configured Benchmark Runners -/

/-- Run vector add benchmark with standard spec -/
def runVectorAdd1M : IO BenchmarkResult := do
  let kernel ← makeVectorAddKernel 1_000_000
  runBenchmarkKernel vectorAdd1M kernel

/-- Run vectorized (float4) vector add benchmark -/
def runVectorAdd1MFloat4 : IO BenchmarkResult := do
  let kernel ← makeVectorAddFloat4Kernel 1_000_000
  runBenchmarkKernel vectorAdd1M kernel

/-- Run vector add benchmark with 10M elements -/
def runVectorAdd10M : IO BenchmarkResult := do
  let kernel ← makeVectorAddKernel 10_000_000
  runBenchmarkKernel vectorAdd10M kernel

/-- Run small vector add for dispatch overhead measurement -/
def runVectorAddSmall : IO BenchmarkResult := do
  let kernel ← makeVectorAddKernel 10_000
  runBenchmarkKernel vectorAddSmall kernel

/-- Run all standard Metal benchmarks -/
def runAllBenchmarks : IO (Array BenchmarkResult) := do
  let mut results := #[]

  IO.println "Running Metal benchmarks..."
  IO.println ""

  IO.println "  [1/4] vector_add_1m (scalar)..."
  results := results.push (← runVectorAdd1M)
  IO.println s!"        Done: {results[results.size - 1]!.stats.mean.toMicros} μs mean"

  IO.println "  [2/4] vector_add_1m (float4)..."
  results := results.push (← runVectorAdd1MFloat4)
  IO.println s!"        Done: {results[results.size - 1]!.stats.mean.toMicros} μs mean"

  IO.println "  [3/4] vector_add_10m (scalar)..."
  results := results.push (← runVectorAdd10M)
  IO.println s!"        Done: {results[results.size - 1]!.stats.mean.toMicros} μs mean"

  IO.println "  [4/4] vector_add_small..."
  results := results.push (← runVectorAddSmall)
  IO.println s!"        Done: {results[results.size - 1]!.stats.mean.toMicros} μs mean"

  IO.println ""
  return results

end TinyGrad4.Benchmark.Metal
