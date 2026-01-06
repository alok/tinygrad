import TinyGrad4Bench.MNISTFusionBench

/-!
# MNISTFusionBenchBig

Bigger synthetic MLP benchmark to stress fusion and kernel selection.
-/

namespace TinyGrad4Bench.MNISTFusionBenchBig

def run : IO Unit := do
  TinyGrad4Bench.MNISTFusionBench.runWith 128 512 5

end TinyGrad4Bench.MNISTFusionBenchBig

#eval! TinyGrad4Bench.MNISTFusionBenchBig.run
