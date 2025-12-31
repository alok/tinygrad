import TinyGrad4.Benchmark.CudaBenchmark
import TinyGrad4.Benchmark.Framework

/-!
# CUDA Benchmark Runner

Runs CUDA kernel benchmarks on NVIDIA GPUs.
-/

open TinyGrad4.Benchmark
open TinyGrad4.Benchmark.Cuda

def main : IO UInt32 := do
  IO.println "╔═══════════════════════════════════════════════════════════════╗"
  IO.println "║               CUDA Kernel Benchmark (TinyGrad4)               ║"
  IO.println "╚═══════════════════════════════════════════════════════════════╝"
  IO.println ""

  let results ← runAllBenchmarks

  IO.println "═══════════════════════════════════════════════════════════════"
  IO.println "Summary:"
  IO.println ""

  for result in results do
    -- stddev is already a Float in nanoseconds, convert to microseconds
    let stddevUs := result.stats.stddev / 1000.0
    IO.println s!"  {result.spec.name}:"
    IO.println s!"    Time: {result.stats.mean.toMicros} μs (±{stddevUs} μs)"
    IO.println s!"    GFLOP/s: {result.throughput_gflops}"
    IO.println s!"    Memory BW: {result.bandwidth_gb_s} GB/s"
    IO.println s!"    Verified: {result.verified}"
    IO.println ""

  return 0
