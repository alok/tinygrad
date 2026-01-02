import TinyGrad4.Benchmark.Framework
import TinyGrad4.Benchmark.Kernels
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
open TinyGrad4.Benchmark.Kernels

/-! ## Metal Runner Integration -/

/-- Generate Metal shader for a benchmark kernel using MetalRenderer -/
def generateBenchShader (kernel : BenchKernel) (size : Nat) : IO String := do
  match generateShader kernel size with
  | some shader => pure shader
  | none =>
    -- Fallback to hardcoded shader if generation fails
    IO.eprintln s!"Warning: Using fallback shader for {repr kernel}"
    pure "#include <metal_stdlib>
using namespace metal;

kernel void add(device const float* buf0 [[buffer(0)]],
                device const float* buf1 [[buffer(1)]],
                device float* buf2 [[buffer(2)]],
                uint gid [[thread_position_in_grid]]) {
    buf2[gid] = buf0[gid] + buf1[gid];
}"

/-- Parse metal_runner output via shell awk to extract float values -/
def parseMetalOutput (output : String) : IO (Float × Float × Float × String) := do
  -- Use awk to extract values from metal_runner output
  -- Device: Apple M4 Max
  -- Time: 123.45 μs
  -- Throughput: 9.06 GFLOP/s
  -- Bandwidth: 108.77 GB/s
  -- Use string concat to avoid interpolation issues with awk $2
  let awkPrint := "'{print $2}'"
  let deviceLine ← IO.Process.run {
    cmd := "sh"
    args := #["-c", "echo " ++ output.quote ++ " | grep 'Device:' | cut -d: -f2 | xargs"]
  }
  let timeLine ← IO.Process.run {
    cmd := "sh"
    args := #["-c", "echo " ++ output.quote ++ " | grep 'Time:' | awk " ++ awkPrint]
  }
  let tputLine ← IO.Process.run {
    cmd := "sh"
    args := #["-c", "echo " ++ output.quote ++ " | grep 'Throughput:' | awk " ++ awkPrint]
  }
  let bwLine ← IO.Process.run {
    cmd := "sh"
    args := #["-c", "echo " ++ output.quote ++ " | grep 'Bandwidth:' | awk " ++ awkPrint]
  }

  -- Parse Float from the decimal string (Lean 4 Float.ofScientific trick)
  let parseF (s : String) : Float := Id.run do
    let s' := s.trimAscii.toString
    -- Manual decimal parsing
    let parts := s'.splitOn "."
    let intPart := (parts.getD 0 "0").toNat!
    if parts.length > 1 then
      let fracStr := parts.getD 1 "0"
      let fracPart := fracStr.toNat!
      let denom := (10 : Float) ^ fracStr.length.toFloat
      intPart.toFloat + fracPart.toFloat / denom
    else
      intPart.toFloat

  return (parseF timeLine, parseF tputLine, parseF bwLine, deviceLine.trimAscii.toString)

/-- Locate metal_runner from the current working directory. -/
def findMetalRunner? : IO (Option String) := do
  let candidates := #[
    ".lake/build/metal/metal_runner",
    "lean4/.lake/build/metal/metal_runner"
  ]
  for path in candidates do
    if (← System.FilePath.pathExists path) then
      return some path
  return none

/-- Get metal_runner path or raise with a setup hint. -/
def findMetalRunner : IO String := do
  match ← findMetalRunner? with
  | some path => pure path
  | none =>
    throw (IO.Error.userError "metal_runner not found; run lean4/scripts/build_metal_ffi.sh")

/-- Run Metal benchmark for a spec using standalone runner -/
def runMetalSpec (spec : BenchmarkSpec) : IO BenchmarkResult := do
  let runner ← findMetalRunner

  -- Generate shader via MetalRenderer (uses actual codegen path!)
  let shader ← generateBenchShader .add spec.size
  let shaderPath := "/tmp/tg4_bench_add.metal"
  IO.FS.writeFile shaderPath shader

  -- Run metal_runner
  let output ← IO.Process.run {
    cmd := runner
    args := #[shaderPath, "add", toString spec.size]
  }

  -- Parse output
  let (timeUs, gflops, gbps, device) ← parseMetalOutput output
  let timestamp ← IO.monoMsNow
  return {
    spec := spec
    backend := "METAL"
    device := device
    stats := {
      min := ⟨(timeUs * 1000).toUInt64.toNat⟩  -- μs to ns
      max := ⟨(timeUs * 1000).toUInt64.toNat⟩
      mean := ⟨(timeUs * 1000).toUInt64.toNat⟩
      median := ⟨(timeUs * 1000).toUInt64.toNat⟩
      stddev := 0.0
      samples := spec.iterations
    }
    bandwidth_gb_s := gbps
    throughput_gflops := gflops
    verified := true  -- metal_runner verifies output
    timestamp := timestamp
    gitCommit := none
  }

/-- Run all standard Metal benchmarks -/
def runMetalAll : IO (Array BenchmarkResult) := do
  let specs := #[vectorAddSmall, vectorAdd1M, vectorAdd10M]
  let mut results := #[]
  for spec in specs do
    let result ← runMetalSpec spec
    results := results.push result
  return results

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
    -- Metal backend (macOS only) - uses standalone metal_runner process
    {
      name := "METAL"
      isAvailable := do
        -- Check if metal_runner exists and we're on macOS
        try
          let _ ← IO.Process.run { cmd := "sw_vers", args := #["-productName"] }
          let runner? ← findMetalRunner?
          return runner?.isSome
        catch _ =>
          return false
      runSpec := fun spec => runMetalSpec spec
      runAll := runMetalAll
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
