import TinyGrad4.Benchmark.MetalDirect

/-!
# Direct Metal FFI Benchmark Entry Point

Build and link manually:
```bash
lake build metal_direct_bench
./scripts/link_metal_frameworks.sh metal_direct_bench metal_direct
.lake/build/metal/metal_direct
```
-/

open TinyGrad4.Benchmark.MetalDirect

def main : IO Unit := do
  IO.println "TinyGrad4 Direct Metal FFI Benchmark"
  IO.println "===================================="
  IO.println ""
  (← IO.getStdout).flush

  IO.println "Starting FFI test..."
  (← IO.getStdout).flush

  -- Test FFI first
  testFFI
  IO.println ""
  (← IO.getStdout).flush

  IO.println "Starting benchmarks..."
  (← IO.getStdout).flush

  -- Run benchmarks
  let results ← runAll
  IO.println ""
  IO.println s!"Completed {results.size} benchmarks"
