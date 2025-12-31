import TinyGrad4.Benchmark.CudaBenchmark

/-!
# Minimal CUDA Benchmark Test

Just imports CudaBenchmark to test if the issue is in module initialization.
-/

def main : IO UInt32 := do
  IO.println "Starting minimal test..."
  IO.println "If you see this, module loading worked!"
  return 0
