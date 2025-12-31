import TinyGrad4.Data.MNISTDataset

def main : IO Unit := do
  IO.println "=== MNIST Data Loader Benchmark (Lean) ==="
  IO.println ""
  TinyGrad4.Data.benchmarkComparison "../data" 64 10000
