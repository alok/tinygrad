import TinyGrad4.Data.MNISTRaw

def main : IO Unit := do
  TinyGrad4.Data.MNISTRaw.benchmarkZeroCopy "../data" 64 10000
