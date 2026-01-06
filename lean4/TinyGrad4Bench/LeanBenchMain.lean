import LeanBench.Runner
import TinyGrad4Bench.LeanBenchBenches
import TinyGrad4Bench.DataLoaderLeanBench
import TinyGrad4Bench.GPUTraceLeanBench

def main (args : List String) : IO UInt32 :=
  LeanBench.runMain args
