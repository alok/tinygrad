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
    let gflops := result.spec.operations.toFloat / result.stats.mean / 1e9
    let bw := result.spec.memoryBytes.toFloat / result.stats.mean / 1e9
    IO.println s!"  {result.spec.name}:"
    IO.println s!"    Time: {result.stats.mean.toMicros} μs (±{result.stats.stddev.toMicros} μs)"
    IO.println s!"    GFLOP/s: {gflops}"
    IO.println s!"    Memory BW: {bw} GB/s"
    IO.println ""

  return 0
