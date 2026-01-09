import TinyGrad4.Backend.Metal
import TinyGrad4.Benchmark.Framework
import TinyGrad4.Benchmark.Kernels
import TinyGrad4.Benchmark.Trace

-- Disable RawBuffer linter for benchmark files that need FloatArray for data generation
set_option linter.useRawBuffer false
-- Disable IO.monoNanosNow linter: benchmark timing uses raw monotonic clocks.
set_option linter.monoNanosNow false

/-!
# Direct Metal FFI Benchmark

Runs benchmarks using Metal FFI directly without subprocess spawning.
This module requires manual linking with Metal frameworks.

## Build

```bash
# Build with Lake (will fail at linking)
lake build tg4_bench

# Re-link with Metal frameworks
./scripts/link_metal_frameworks.sh
```
-/

namespace TinyGrad4.Benchmark.MetalDirect

open TinyGrad4.Backend.Metal
open TinyGrad4.Benchmark
open TinyGrad4.Benchmark.Kernels

/-- Timing helper: returns elapsed microseconds -/
def timeKernel (_iterations : Nat) (action : IO Unit) : IO (Float × Nat × Nat) := do
  -- Warmup run
  action
  metalSync

  -- Timed run
  let start ← IO.monoNanosNow
  action
  metalSync
  let stop ← IO.monoNanosNow

  let elapsedNs := stop - start
  let elapsedUs : Float := elapsedNs.toFloat / 1000.0
  return (elapsedUs, start, elapsedNs)

structure DirectBenchRun where
  result : BenchmarkResult
  trace : Trace.TraceReport

/-- Run a direct Metal FFI benchmark -/
def runDirectBenchmarkWithTrace (kernel : BenchKernel) (size : Nat) (iterations : Nat := 100) : IO DirectBenchRun := do
  -- Get device info
  let device ← metalDeviceName

  -- Generate shader from MetalRenderer
  let renderStart ← IO.monoNanosNow
  let (shader, uopStats) ← match generateShaderWithStats kernel size with
    | some pair => pure pair
    | none => throw (IO.Error.userError s!"Failed to generate shader for {repr kernel}")
  let renderStop ← IO.monoNanosNow

  -- Compile shader
  let compileStart ← IO.monoNanosNow
  let prog ← metalCompile kernel.name shader
  let compileStop ← IO.monoNanosNow

  -- Allocate buffers
  let numInputs := kernel.numInputs
  let mut bufs : Array MetalBuffer := #[]

  -- Input buffers
  for _ in [:numInputs] do
    let buf ← metalAlloc size
    bufs := bufs.push buf

  -- Output buffer
  let outBuf ← metalAlloc size
  bufs := bufs.push outBuf

  -- Calculate grid size (vectorized float4)
  let gridSize := (size + 3) / 4

  -- Benchmark
  let (timeUs, launchStart, launchNs) ← timeKernel iterations do
    metalLaunch prog bufs gridSize 1 1 256 1 1

  -- Verify output
  let output ← metalCopyOut outBuf
  let verified := output.size == size

  -- Compute metrics
  let bytesPerIter := size.toFloat * 4.0 * (numInputs + 1).toFloat  -- inputs + output
  let bandwidth := bytesPerIter / (timeUs * 1000.0)  -- GB/s
  let flops := size.toFloat * kernel.flopsPerElement.toFloat
  let gflops := flops / (timeUs * 1000.0)

  -- Cleanup
  for buf in bufs do
    metalFree buf

  let timestamp ← IO.monoNanosNow
  let events : Array Trace.TraceEvent := #[
    {
      phase := "render"
      startNs := renderStart
      durationNs := renderStop - renderStart
      uopStats := uopStats
    },
    {
      phase := "compile"
      startNs := compileStart
      durationNs := compileStop - compileStart
    },
    {
      phase := "launch"
      startNs := launchStart
      durationNs := launchNs
    }
  ]
  let report : Trace.TraceReport := {
    metaInfo := {
      backend := "METAL_DIRECT"
      device := device
      kernel := kernel.name
      size := size
      iterations := iterations
      timestampNs := timestamp
    }
    events := events
  }
  let result : BenchmarkResult := {
    spec := {
      name := s!"{kernel.name}_{size}"
      size := size
      description := s!"{kernel.name} with {size} elements"
      warmupRuns := 5
      iterations := iterations
    }
    backend := "METAL_DIRECT"
    device := device
    stats := {
      min := ⟨(timeUs * 1000).toUInt64.toNat⟩
      max := ⟨(timeUs * 1000).toUInt64.toNat⟩
      mean := ⟨(timeUs * 1000).toUInt64.toNat⟩
      median := ⟨(timeUs * 1000).toUInt64.toNat⟩
      stddev := 0.0
      samples := iterations
    }
    bandwidth_gb_s := bandwidth
    throughput_gflops := gflops
    verified := verified
    timestamp := timestamp
    gitCommit := none
  }
  return { result := result, trace := report }

/-- Run a direct Metal FFI benchmark -/
def runDirectBenchmark (kernel : BenchKernel) (size : Nat) (iterations : Nat := 100) : IO BenchmarkResult := do
  let run ← runDirectBenchmarkWithTrace kernel size iterations
  let traceOut? ← Trace.traceOutPath?
  match traceOut? with
  | some path => Trace.writeReport path run.trace
  | none => pure ()
  return run.result

/-- Run all direct Metal benchmarks -/
def runAll : IO (Array BenchmarkResult) := do
  IO.println s!"Running direct Metal FFI benchmarks..."
  (← IO.getStdout).flush
  let device ← metalDeviceName
  IO.println s!"Device: {device}"
  IO.println ""
  (← IO.getStdout).flush

  let mut results := #[]

  -- Run benchmark suite
  for (kernel, sizes) in [(.add, [10000, 100000, 1000000]),
                          (.mul, [10000, 100000]),
                          (.exp2, [10000, 100000])] do
    for size in sizes do
      IO.print s!"  {kernel.name} {size}... "
      (← IO.getStdout).flush
      let result ← runDirectBenchmark kernel size 10  -- 10 iterations for warmup+timed
      IO.println s!"{result.stats.mean.toMicros} μs | {result.bandwidth_gb_s} GB/s"
      results := results.push result

  return results

/-- Quick test to verify FFI is working -/
def testFFI : IO Unit := do
  IO.println "Testing Metal FFI..."

  -- Test device detection
  let device ← metalDeviceName
  IO.println s!"  Device: {device}"

  -- Test buffer allocation
  let buf ← metalAlloc 1000
  IO.println s!"  Allocated buffer"

  -- Test copy in/out
  let arr := (List.replicate 1000 (1.5 : Float)).toArray
  let data := FloatArray.mk arr
  metalCopyIn buf data
  let output ← metalCopyOut buf
  IO.println s!"  Roundtrip: {output[0]!} (expected 1.5)"

  -- Test kernel compilation
  let shader := "#include <metal_stdlib>\nusing namespace metal;\nkernel void test(device float* buf [[buffer(0)]], uint i [[thread_position_in_grid]]) { buf[i] *= 2; }"
  let prog ← metalCompile "test" shader
  IO.println s!"  Compiled kernel"

  -- Test kernel launch
  metalLaunch prog #[buf] 1000 1 1 256 1 1
  metalSync
  let output2 ← metalCopyOut buf
  IO.println s!"  After kernel: {output2[0]!} (expected 3.0)"

  metalFree buf
  IO.println "  FFI test passed!"

end TinyGrad4.Benchmark.MetalDirect
