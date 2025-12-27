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

import TinyGrad4.Benchmark.Framework
-- Backend-specific imports are conditional on availability
-- import TinyGrad4.Benchmark.MetalBenchmark
-- import TinyGrad4.Benchmark.CudaBenchmark
import TinyGrad4.Benchmark.Runner

namespace TinyGrad4.Benchmark

-- Re-export main types
export Framework (
  Timing
  TimingStats
  BenchmarkSpec
  BenchmarkResult
  BenchmarkKernel
  runBenchmarkKernel
  vectorAdd1M
  vectorAdd10M
  vectorAddSmall
  formatResult
  formatResultJson
  formatComparison
)

export Runner (
  BackendRunner
  BenchmarkSuite
  BenchmarkComparison
  BenchmarkReport
  RunConfig
  defaultConfig
  parseArgs
  printUsage
  printBackendStatus
  detectBackends
  getMachineInfo
  getGitCommit
  formatReportMarkdown
  writeJsonReport
  writeMarkdownReport
)

end TinyGrad4.Benchmark
