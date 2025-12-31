import TinyGrad4.Benchmark.Framework
import TinyGrad4.Benchmark.Kernels
-- Backend-specific imports are conditional on availability
-- import TinyGrad4.Benchmark.MetalBenchmark
-- import TinyGrad4.Benchmark.CudaBenchmark
import TinyGrad4.Benchmark.Runner
-- Direct FFI benchmark (requires manual Metal framework linking)
import TinyGrad4.Benchmark.MetalDirect
-- Fine-grained timing instrumentation
import TinyGrad4.Benchmark.Instrumentation

/-!
# TinyGrad4 Benchmark Library

Reusable infrastructure for GPU backend benchmarking with:
- Statistical timing analysis
- JSON/Markdown reporting
- Multi-backend comparison
- CI integration

## Quick Start

```lean
import TinyGrad4.Benchmark

-- Run Metal benchmarks
#eval TinyGrad4.Benchmark.Metal.runAllBenchmarks

-- Run CUDA benchmarks (on NVIDIA GPU)
#eval TinyGrad4.Benchmark.Cuda.runAllBenchmarks
```

## Architecture

```
Framework.lean      -- Core types: Timing, Stats, BenchmarkSpec, BenchmarkResult
MetalBenchmark.lean -- Metal-specific kernel implementations
CudaBenchmark.lean  -- CUDA-specific kernel implementations
Runner.lean         -- Multi-backend orchestration and reporting
```
-/

namespace TinyGrad4.Benchmark

-- Re-exports are handled by the imports - all types are available via
-- TinyGrad4.Benchmark.Timing, TinyGrad4.Benchmark.Runner, etc.

end TinyGrad4.Benchmark
