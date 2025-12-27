import TinyGrad4.Benchmark.Framework
import Lean.Data.Json

/-!
# Benchmark Runner

Unified multi-backend benchmark orchestration with comparison reporting.

## Usage

```lean
-- Run all benchmarks on available backends
#eval runAllBackends

-- Compare specific benchmark across backends
#eval compareBenchmark vectorAdd1M #["METAL", "CUDA"]

-- Generate JSON report
#eval generateJsonReport "benchmark_results.json"
```
-/

namespace TinyGrad4.Benchmark.Runner

open TinyGrad4.Benchmark

/-! ## Backend Registry -/

/-- Backend availability checker and benchmark runner -/
structure BackendRunner where
  /-- Backend identifier -/
  name : String
  /-- Check if backend is available -/
  isAvailable : IO Bool
  /-- Run a benchmark spec and return result -/
  runSpec : BenchmarkSpec → IO BenchmarkResult
  /-- Run all standard benchmarks -/
  runAll : IO (Array BenchmarkResult)

/-- Registry of available backend runners -/
def backendRegistry : IO (Array BackendRunner) := do
  -- We return stubs here; actual implementations link at compile time
  -- based on what backends are compiled in
  return #[
    -- Metal backend (macOS only)
    {
      name := "METAL"
      isAvailable := do
        -- Check if we're on macOS by trying to import Metal
        try
          let _ ← IO.Process.run { cmd := "sw_vers", args := #["-productName"] }
          return true
        catch _ =>
          return false
      runSpec := fun _spec => throw (IO.Error.userError "Metal runner not linked")
      runAll := throw (IO.Error.userError "Metal runner not linked")
    },
    -- CUDA backend (NVIDIA GPU required)
    {
      name := "CUDA"
      isAvailable := do
        try
          let _ ← IO.Process.run { cmd := "nvidia-smi" }
          return true
        catch _ =>
          return false
      runSpec := fun _spec => throw (IO.Error.userError "CUDA runner not linked")
      runAll := throw (IO.Error.userError "CUDA runner not linked")
    }
  ]

/-! ## Benchmark Suite -/

/-- Full benchmark suite with all standard tests -/
structure BenchmarkSuite where
  /-- Suite name -/
  name : String
  /-- Description -/
  description : String
  /-- Specs to run -/
  specs : Array BenchmarkSpec
  deriving Repr

/-- Standard elementwise benchmark suite -/
def ewiseSuite : BenchmarkSuite := {
  name := "elementwise"
  description := "Element-wise operations (add, mul, etc.)"
  specs := #[vectorAddSmall, vectorAdd1M, vectorAdd10M]
}

/-- All standard suites -/
def allSuites : Array BenchmarkSuite := #[ewiseSuite]

/-! ## Report Types -/

/-- Comparison of a single benchmark across backends -/
structure BenchmarkComparison where
  spec : BenchmarkSpec
  results : Array BenchmarkResult
  /-- Fastest backend name -/
  fastestBackend : String
  /-- Speedup vs slowest -/
  maxSpeedup : Float
  deriving Repr

namespace BenchmarkComparison

/-- Compute comparison from results -/
def fromResults (spec : BenchmarkSpec) (results : Array BenchmarkResult) : BenchmarkComparison :=
  if results.isEmpty then
    { spec, results, fastestBackend := "none", maxSpeedup := 1.0 }
  else
    let sorted := results.qsort (·.stats.min.nanos < ·.stats.min.nanos)
    let fastest := sorted[0]!
    let slowest := sorted[sorted.size - 1]!
    let speedup := if fastest.stats.min.nanos == 0 then 1.0
                   else slowest.stats.min.nanos.toFloat / fastest.stats.min.nanos.toFloat
    { spec, results, fastestBackend := fastest.backend, maxSpeedup := speedup }

end BenchmarkComparison

/-- Full benchmark report -/
structure BenchmarkReport where
  /-- Report timestamp -/
  timestamp : Nat
  /-- Git commit if available -/
  gitCommit : Option String
  /-- Machine info -/
  machineInfo : String
  /-- All results by backend -/
  resultsByBackend : List (String × Array BenchmarkResult)
  /-- Comparisons -/
  comparisons : Array BenchmarkComparison
  deriving Repr

open Lean Json in
instance : ToJson BenchmarkComparison where
  toJson c := Json.mkObj [
    ("spec", toJson c.spec),
    ("results", Json.arr (c.results.map toJson)),
    ("fastest_backend", toJson c.fastestBackend),
    ("max_speedup", toJson c.maxSpeedup)
  ]

open Lean Json in
instance : ToJson BenchmarkReport where
  toJson r := Json.mkObj [
    ("timestamp", toJson r.timestamp),
    ("git_commit", match r.gitCommit with | some c => Json.str c | none => Json.null),
    ("machine_info", toJson r.machineInfo),
    ("results_by_backend", Json.mkObj (r.resultsByBackend.map fun (name, results) =>
      (name, Json.arr (results.map toJson)))),
    ("comparisons", Json.arr (r.comparisons.map toJson))
  ]

/-! ## Runner Functions -/

/-- Get machine info string -/
def getMachineInfo : IO String := do
  try
    -- Try to get OS info
    let os ← IO.Process.run { cmd := "uname", args := #["-s", "-r", "-m"] }
    return os.trimAscii.toString
  catch _ =>
    return "unknown"

/-- Get git commit hash -/
def getGitCommit : IO (Option String) := do
  try
    let hash ← IO.Process.run { cmd := "git", args := #["rev-parse", "--short", "HEAD"] }
    return some hash.trimAscii.toString
  catch _ =>
    return none

/-- Detect available backends -/
def detectBackends : IO (Array String) := do
  let backends ← backendRegistry
  let mut available := #[]
  for backend in backends do
    if ← backend.isAvailable then
      available := available.push backend.name
  return available

/-- Print backend availability -/
def printBackendStatus : IO Unit := do
  IO.println "Backend Availability:"
  let backends ← backendRegistry
  for backend in backends do
    let status := if ← backend.isAvailable then "available" else "not available"
    IO.println s!"  {backend.name}: {status}"
  IO.println ""

/-! ## Output Formatting -/

/-- Format comparison as markdown table -/
def formatComparisonMarkdown (c : BenchmarkComparison) : String :=
  let header := s!"### {c.spec.name}\n\n"
  let tableHeader := "| Backend | Device | Min (μs) | Mean (μs) | BW (GB/s) | Status |\n"
  let tableSep := "|---------|--------|----------|-----------|-----------|--------|\n"
  let rows := c.results.map fun r =>
    let status := if r.verified then "OK" else "FAIL"
    s!"| {r.backend} | {r.device} | {r.stats.min.toMicros} | {r.stats.mean.toMicros} | {r.bandwidth_gb_s} | {status} |"
  let summary := s!"\n**Fastest**: {c.fastestBackend} ({c.maxSpeedup}x faster than slowest)\n"
  header ++ tableHeader ++ tableSep ++ String.intercalate "\n" rows.toList ++ summary

/-- Format full report as markdown -/
def formatReportMarkdown (r : BenchmarkReport) : String :=
  let header := "# TinyGrad4 Benchmark Report\n\n"
  let metadata := s!"- **Machine**: {r.machineInfo}\n" ++
                  s!"- **Git Commit**: {r.gitCommit.getD "unknown"}\n" ++
                  s!"- **Timestamp**: {r.timestamp}\n\n"
  let comparisons := r.comparisons.map formatComparisonMarkdown
  header ++ metadata ++ "## Results\n\n" ++ String.intercalate "\n\n" comparisons.toList

/-- Write report to JSON file -/
def writeJsonReport (path : String) (report : BenchmarkReport) : IO Unit := do
  let json := Lean.toJson report
  IO.FS.writeFile path (toString json)

/-- Write report to markdown file -/
def writeMarkdownReport (path : String) (report : BenchmarkReport) : IO Unit := do
  let md := formatReportMarkdown report
  IO.FS.writeFile path md

/-! ## Main Entry Points -/

/-- Configuration for benchmark run -/
structure RunConfig where
  /-- Backends to run (empty = all available) -/
  backends : Array String := #[]
  /-- Suites to run (empty = all) -/
  suites : Array String := #[]
  /-- JSON output path -/
  jsonOutput : Option String := none
  /-- Markdown output path -/
  markdownOutput : Option String := none
  /-- Verbose output -/
  verbose : Bool := true
  deriving Repr, Inhabited

/-- Default configuration -/
def defaultConfig : RunConfig := {}

/-- Parse command line args into config -/
def parseArgs (args : List String) : RunConfig := Id.run do
  let mut config := defaultConfig
  let mut i := 0
  let argArray := args.toArray
  while i < argArray.size do
    let arg := argArray[i]!
    if arg == "--json" && i + 1 < argArray.size then
      config := { config with jsonOutput := some argArray[i + 1]! }
      i := i + 2
    else if arg == "--markdown" && i + 1 < argArray.size then
      config := { config with markdownOutput := some argArray[i + 1]! }
      i := i + 2
    else if arg == "--backend" && i + 1 < argArray.size then
      config := { config with backends := config.backends.push argArray[i + 1]! }
      i := i + 2
    else if arg == "--quiet" then
      config := { config with verbose := false }
      i := i + 1
    else
      i := i + 1
  return config

/-- Print usage -/
def printUsage : IO Unit := do
  IO.println "TinyGrad4 Benchmark Runner"
  IO.println ""
  IO.println "Usage: benchmark [options]"
  IO.println ""
  IO.println "Options:"
  IO.println "  --json <path>      Write JSON report to file"
  IO.println "  --markdown <path>  Write Markdown report to file"
  IO.println "  --backend <name>   Run only specified backend (can repeat)"
  IO.println "  --quiet            Suppress progress output"
  IO.println ""
  IO.println "Examples:"
  IO.println "  benchmark                        # Run all backends, print results"
  IO.println "  benchmark --json results.json    # Save JSON report"
  IO.println "  benchmark --backend METAL        # Run only Metal backend"

end TinyGrad4.Benchmark.Runner
