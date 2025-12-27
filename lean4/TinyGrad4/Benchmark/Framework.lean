import Lean.Data.Json

/-!
# Benchmark Framework

Reusable infrastructure for GPU backend benchmarking.

## Design Goals
1. **Reproducible**: Fixed parameters, multiple runs, statistical analysis
2. **Comparable**: Structured output for cross-backend comparison
3. **Extensible**: Easy to add new benchmarks and backends
4. **CI-friendly**: JSON output, exit codes, regression detection

## Usage
```lean
def myBench : BenchmarkSpec := {
  name := "vector_add"
  size := 1_000_000
  iterations := 100
  warmupRuns := 3
}

#eval runBenchmark myBench metalBackend
```
-/

namespace TinyGrad4.Benchmark

/-! ## Core Types -/

/-- Single timing measurement -/
structure Timing where
  /-- Time in nanoseconds -/
  nanos : Nat
  deriving Repr, Inhabited

namespace Timing

def toMicros (t : Timing) : Float := t.nanos.toFloat / 1000.0
def toMillis (t : Timing) : Float := t.nanos.toFloat / 1_000_000.0
def toSeconds (t : Timing) : Float := t.nanos.toFloat / 1_000_000_000.0

end Timing

/-- Statistical summary of multiple runs -/
structure TimingStats where
  min : Timing
  max : Timing
  mean : Timing
  median : Timing
  stddev : Float
  samples : Nat
  deriving Repr, Inhabited

namespace TimingStats

def compute (timings : Array Timing) : TimingStats :=
  if timings.isEmpty then
    { min := ⟨0⟩, max := ⟨0⟩, mean := ⟨0⟩, median := ⟨0⟩, stddev := 0.0, samples := 0 }
  else
    let sorted := timings.qsort (·.nanos < ·.nanos)
    let n := sorted.size
    let sum := sorted.foldl (· + ·.nanos) 0
    let mean := sum / n
    let variance := sorted.foldl (fun acc t =>
      let diff := (t.nanos : Int) - (mean : Int)
      let diffF := Float.ofInt diff
      acc + diffF * diffF) 0.0 / n.toFloat
    {
      min := sorted[0]!
      max := sorted[n - 1]!
      mean := ⟨mean⟩
      median := sorted[n / 2]!
      stddev := variance.sqrt
      samples := n
    }

end TimingStats

/-- Benchmark specification -/
structure BenchmarkSpec where
  /-- Unique identifier -/
  name : String
  /-- Number of elements to process -/
  size : Nat := 1_000_000
  /-- Number of timed iterations -/
  iterations : Nat := 100
  /-- Warmup runs (not timed) -/
  warmupRuns : Nat := 3
  /-- Description for reports -/
  description : String := ""
  deriving Repr, Inhabited

/-- Benchmark result with metadata -/
structure BenchmarkResult where
  /-- Spec that was run -/
  spec : BenchmarkSpec
  /-- Backend name (e.g., "METAL", "CUDA") -/
  backend : String
  /-- Device name -/
  device : String
  /-- Timing statistics -/
  stats : TimingStats
  /-- Computed metrics -/
  bandwidth_gb_s : Float
  throughput_gflops : Float
  /-- Whether results were verified correct -/
  verified : Bool
  /-- Timestamp (Unix epoch seconds) -/
  timestamp : Nat
  /-- Git commit hash if available -/
  gitCommit : Option String := none
  deriving Repr, Inhabited

namespace BenchmarkResult

/-- Compute bandwidth for elementwise ops (read 2 + write 1 = 3 arrays) -/
def computeBandwidth (size : Nat) (timeUs : Float) : Float :=
  let bytesTransferred := 3.0 * size.toFloat * 4.0  -- 3 arrays * size * sizeof(float)
  (bytesTransferred / timeUs) * 1e6 / 1e9  -- GB/s

/-- Compute throughput for elementwise ops (1 FLOP per element) -/
def computeThroughput (size : Nat) (timeUs : Float) : Float :=
  (size.toFloat / timeUs) * 1e6 / 1e9  -- GFLOP/s

end BenchmarkResult

/-! ## JSON Serialization -/

open Lean Json in
instance : ToJson Timing where
  toJson t := Json.mkObj [("nanos", toJson t.nanos)]

open Lean Json in
instance : ToJson TimingStats where
  toJson s := Json.mkObj [
    ("min_us", toJson s.min.toMicros),
    ("max_us", toJson s.max.toMicros),
    ("mean_us", toJson s.mean.toMicros),
    ("median_us", toJson s.median.toMicros),
    ("stddev_us", toJson (s.stddev / 1000.0)),
    ("samples", toJson s.samples)
  ]

open Lean Json in
instance : ToJson BenchmarkSpec where
  toJson s := Json.mkObj [
    ("name", toJson s.name),
    ("size", toJson s.size),
    ("iterations", toJson s.iterations),
    ("warmup_runs", toJson s.warmupRuns),
    ("description", toJson s.description)
  ]

open Lean Json in
instance : ToJson BenchmarkResult where
  toJson r := Json.mkObj [
    ("spec", toJson r.spec),
    ("backend", toJson r.backend),
    ("device", toJson r.device),
    ("stats", toJson r.stats),
    ("bandwidth_gb_s", toJson r.bandwidth_gb_s),
    ("throughput_gflops", toJson r.throughput_gflops),
    ("verified", toJson r.verified),
    ("timestamp", toJson r.timestamp),
    ("git_commit", match r.gitCommit with | some c => Json.str c | none => Json.null)
  ]

/-! ## Benchmark Runner Infrastructure -/

/-- Abstract benchmark kernel that can be run on any backend -/
structure BenchmarkKernel where
  /-- Setup: allocate buffers, compile kernel -/
  setup : IO Unit
  /-- Run one iteration -/
  runOnce : IO Unit
  /-- Sync and get results -/
  sync : IO Unit
  /-- Verify results are correct -/
  verify : IO Bool
  /-- Cleanup resources -/
  cleanup : IO Unit
  /-- Get backend name -/
  backendName : String
  /-- Get device name -/
  deviceName : IO String

/-- Run a benchmark with the given kernel -/
def runBenchmarkKernel (spec : BenchmarkSpec) (kernel : BenchmarkKernel) : IO BenchmarkResult := do
  -- Setup
  kernel.setup

  -- Warmup
  for _ in [:spec.warmupRuns] do
    kernel.runOnce
    kernel.sync

  -- Timed runs
  let mut timings : Array Timing := #[]
  for _ in [:spec.iterations] do
    let start ← IO.monoNanosNow
    kernel.runOnce
    kernel.sync
    let stop ← IO.monoNanosNow
    timings := timings.push ⟨stop - start⟩

  -- Verify
  let verified ← kernel.verify

  -- Compute stats
  let stats := TimingStats.compute timings
  let timeUs := stats.min.toMicros
  let bandwidth := BenchmarkResult.computeBandwidth spec.size timeUs
  let throughput := BenchmarkResult.computeThroughput spec.size timeUs

  -- Get metadata
  let device ← kernel.deviceName
  let timestamp ← IO.monoMsNow  -- Approximate Unix timestamp

  -- Cleanup
  kernel.cleanup

  return {
    spec := spec
    backend := kernel.backendName
    device := device
    stats := stats
    bandwidth_gb_s := bandwidth
    throughput_gflops := throughput
    verified := verified
    timestamp := timestamp
    gitCommit := none
  }

/-! ## Standard Benchmark Specs -/

/-- Vector add benchmark (1M elements) -/
def vectorAdd1M : BenchmarkSpec := {
  name := "vector_add_1m"
  size := 1_000_000
  iterations := 100
  warmupRuns := 3
  description := "Element-wise addition of two 1M float32 vectors"
}

/-- Vector add benchmark (10M elements) -/
def vectorAdd10M : BenchmarkSpec := {
  name := "vector_add_10m"
  size := 10_000_000
  iterations := 50
  warmupRuns := 3
  description := "Element-wise addition of two 10M float32 vectors"
}

/-- Small vector add for quick smoke tests -/
def vectorAddSmall : BenchmarkSpec := {
  name := "vector_add_small"
  size := 10_000
  iterations := 1000
  warmupRuns := 10
  description := "Small vector add for measuring dispatch overhead"
}

/-! ## Output Formatting -/

/-- Format result as human-readable string -/
def formatResult (r : BenchmarkResult) : String :=
  let header := s!"=== {r.spec.name} on {r.backend} ({r.device}) ==="
  let timing := s!"Time: {r.stats.min.toMicros} μs (min), {r.stats.mean.toMicros} μs (mean)"
  let perf := s!"Bandwidth: {r.bandwidth_gb_s} GB/s | Throughput: {r.throughput_gflops} GFLOP/s"
  let status := if r.verified then "✓ Verified" else "✗ FAILED"
  s!"{header}\n{timing}\n{perf}\n{status}"

/-- Format result as JSON string -/
def formatResultJson (r : BenchmarkResult) : String :=
  toString (Lean.toJson r)

/-- Format multiple results as comparison table -/
def formatComparison (results : Array BenchmarkResult) : String :=
  let header := "| Backend | Device | Min (μs) | Mean (μs) | BW (GB/s) | Status |"
  let sep := "|---------|--------|----------|-----------|-----------|--------|"
  let rows := results.map fun r =>
    let status := if r.verified then "✓" else "✗"
    s!"| {r.backend} | {r.device} | {r.stats.min.toMicros} | {r.stats.mean.toMicros} | {r.bandwidth_gb_s} | {status} |"
  header ++ "\n" ++ sep ++ "\n" ++ String.intercalate "\n" rows.toList

end TinyGrad4.Benchmark
