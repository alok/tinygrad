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
  IO.println s!"Kernel source length: {(vectorAddSource 1000).length}"

  IO.println "Step 2: Creating kernel..."
  let kernel ← makeVectorAddKernel 1000
  IO.println s!"Kernel created! Backend: {kernel.backendName}"

  IO.println "Step 3: Calling setup..."
  kernel.setup
  IO.println "Setup complete!"

  IO.println "Step 4: Running kernel..."
  kernel.runOnce
  IO.println "RunOnce complete!"

  IO.println "Step 5: Syncing..."
  kernel.sync
  IO.println "Sync complete!"

  IO.println "Done!"
  return 0
