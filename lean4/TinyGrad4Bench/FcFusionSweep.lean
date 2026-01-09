import TinyGrad4Bench.FcFusionBench

/-!
# FcFusionSweep

Sweep a few FC sizes to compare fused vs node performance.
-/

namespace TinyGrad4Bench.FcFusionSweep

def run : IO Unit := do
  let cfgs : List (Nat × Nat × Nat × Nat × Nat) := [
    (32, 784, 256, 10, 10),
    (64, 784, 512, 10, 5),
    (128, 784, 512, 10, 5),
    (128, 784, 1024, 10, 5)
  ]
  for (batch, inDim, hidden, outDim, iters) in cfgs do
    TinyGrad4Bench.FcFusionBench.runWith batch inDim hidden outDim iters

end TinyGrad4Bench.FcFusionSweep

#eval! TinyGrad4Bench.FcFusionSweep.run
