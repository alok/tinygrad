import LeanBenchNew.Benchmark

namespace TinyGrad4Bench

open LeanBenchNew

def microBenchConfig : Config := {
  warmupIterations := 10
  timedIterations := 100
  printProgress := false
}

def mediumBenchConfig : Config := {
  warmupIterations := 5
  timedIterations := 50
  printProgress := false
}

end TinyGrad4Bench
