import TinyGrad4.Benchmark.CudaBenchmark
import TinyGrad4.Benchmark.Framework

/-!
# Minimal CUDA Benchmark Test

Just imports CudaBenchmark to test if the issue is in module initialization.
-/

open TinyGrad4.Benchmark
open TinyGrad4.Benchmark.Cuda

def main : IO UInt32 := do
  IO.println "Step 1: Starting..."

  IO.println "Step 2: Calling runVectorAdd1M..."
  let result ← runVectorAdd1M
  IO.println s!"Step 3: Got result: {result.spec.name}"

  IO.println "Done!"
  return 0
