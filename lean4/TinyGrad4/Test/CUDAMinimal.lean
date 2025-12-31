import TinyGrad4.Benchmark.CudaBenchmark
import TinyGrad4.Benchmark.Framework

/-!
# Minimal CUDA Benchmark Test

Just imports CudaBenchmark to test if the issue is in module initialization.
-/

open TinyGrad4.Benchmark
open TinyGrad4.Benchmark.Cuda

def main : IO UInt32 := do
  IO.println "Starting minimal test..."
  IO.println "Module loading worked!"
  IO.println ""
  IO.println "Calling runAllBenchmarks..."
  let _results ← runAllBenchmarks
  IO.println "Done!"
  return 0
