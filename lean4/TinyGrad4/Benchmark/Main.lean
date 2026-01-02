import Cli
import TinyGrad4.Benchmark.Framework
import TinyGrad4.Benchmark.Runner

/-!
# TinyGrad4 Benchmark CLI

GPU backend benchmark tool with structured output.

## Usage

```bash
# Run all benchmarks on available backends
lake exe tg4_bench run

# Run specific suite
lake exe tg4_bench run --suite elementwise

# Output JSON report
lake exe tg4_bench run --json results.json

# Compare backends
lake exe tg4_bench compare --backend METAL --backend CUDA

# List available backends
lake exe tg4_bench backends
```
-/

open Cli
open TinyGrad4.Benchmark

/-! ## Version -/

def version : String := "0.1.0"

/-! ## Output Helpers -/

/-- Styled output for terminal -/
structure Output where
  /-- Whether to use colors/formatting -/
  useColor : Bool := true
  /-- Verbose mode -/
  verbose : Bool := false

namespace Output

def bold (o : Output) (s : String) : String :=
  if o.useColor then s!"\x1b[1m{s}\x1b[0m" else s

def green (o : Output) (s : String) : String :=
  if o.useColor then s!"\x1b[32m{s}\x1b[0m" else s

def red (o : Output) (s : String) : String :=
  if o.useColor then s!"\x1b[31m{s}\x1b[0m" else s

def yellow (o : Output) (s : String) : String :=
  if o.useColor then s!"\x1b[33m{s}\x1b[0m" else s

def cyan (o : Output) (s : String) : String :=
  if o.useColor then s!"\x1b[36m{s}\x1b[0m" else s

def dim (o : Output) (s : String) : String :=
  if o.useColor then s!"\x1b[2m{s}\x1b[0m" else s

def header (o : Output) (title : String) : IO Unit := do
  let line := String.ofList (List.replicate (title.length + 4) '─')
  IO.println s!"┌{line}┐"
  IO.println s!"│  {o.bold title}  │"
  IO.println s!"└{line}┘"

def success (o : Output) (msg : String) : IO Unit := do
  IO.println s!"{o.green "✓"} {msg}"

def failure (o : Output) (msg : String) : IO Unit := do
  IO.println s!"{o.red "✗"} {msg}"

def info (o : Output) (msg : String) : IO Unit := do
  IO.println s!"{o.cyan "→"} {msg}"

def progress (o : Output) (current : Nat) (total : Nat) (label : String) : IO Unit := do
  IO.println s!"{o.dim s!"[{current}/{total}]"} {label}"

end Output

/-! ## Command Handlers -/

/-- Run benchmarks -/
def runBenchmarks (p : Parsed) : IO UInt32 := do
  let out : Output := {
    useColor := !p.hasFlag "no-color"
    verbose := p.hasFlag "verbose"
  }

  out.header "TinyGrad4 Benchmark"

  -- Get machine info
  let machineInfo ← Runner.getMachineInfo
  let gitCommit ← Runner.getGitCommit
  out.info s!"Machine: {machineInfo}"
  if let some commit := gitCommit then
    out.info s!"Git: {commit}"
  IO.println ""

  -- Detect backends
  out.info "Detecting backends..."
  let available ← Runner.detectBackends
  if available.isEmpty then
    out.failure "No GPU backends available"
    return 1

  for backend in available do
    out.success s!"{backend} available"
  IO.println ""

  -- Run benchmarks
  out.info "Running benchmarks..."
  IO.println ""

  let mut allResults : Array BenchmarkResult := #[]
  let mut resultsByBackend : Array (String × Array BenchmarkResult) := #[]

  -- Get backend runners
  let backends ← Runner.backendRegistry
  for backend in backends do
    if ← backend.isAvailable then
      IO.println s!"{out.bold backend.name}:"
      let mut backendResults : Array BenchmarkResult := #[]

      for suite in Runner.allSuites do
        IO.println s!"  Suite: {suite.name}"
        let specs := suite.specs
        for h : idx in [:specs.size] do
          let spec := specs[idx]
          out.progress (idx + 1) specs.size s!"  {spec.name}"
          try
            let result ← backend.runSpec spec
            allResults := allResults.push result
            backendResults := backendResults.push result
            IO.println s!"    {out.green "OK"} {result.stats.mean.toMicros} μs | {result.bandwidth_gb_s} GB/s | {result.throughput_gflops} GFLOP/s"
          catch e =>
            IO.println s!"    {out.red "ERR"} {e.toString}"
      IO.println ""
      resultsByBackend := resultsByBackend.push (backend.name, backendResults)

  -- Output results
  let jsonPath := p.flag? "json" |>.map (·.as! String)
  let mdPath := p.flag? "markdown" |>.map (·.as! String)

  if jsonPath.isSome || mdPath.isSome then
    let timestamp ← Runner.getUnixTimestamp
    let allSpecs := Runner.allSuites.foldl (fun acc suite => acc ++ suite.specs) #[]
    let comparisons := allSpecs.map fun spec =>
      let results := allResults.filter (fun r => r.spec.name == spec.name)
      Runner.BenchmarkComparison.fromResults spec results
    let report : Runner.BenchmarkReport := {
      timestamp := timestamp
      gitCommit := gitCommit
      machineInfo := machineInfo
      resultsByBackend := resultsByBackend.toList
      comparisons := comparisons
    }

    if let some path := jsonPath then
      out.info s!"Writing JSON report to {path}"
      Runner.writeJsonReport path report

    if let some path := mdPath then
      out.info s!"Writing Markdown report to {path}"
      Runner.writeMarkdownReport path report

  out.success "Benchmarks complete"
  return 0

/-- List available backends -/
def listBackends (p : Parsed) : IO UInt32 := do
  let out : Output := { useColor := !p.hasFlag "no-color" }

  out.header "Available Backends"
  IO.println ""

  let backends ← Runner.backendRegistry
  for backend in backends do
    let available ← backend.isAvailable
    if available then
      out.success s!"{backend.name}"
    else
      IO.println s!"{out.dim "○"} {out.dim backend.name} {out.dim "(not available)"}"

  IO.println ""
  return 0

/-- Show result formatting examples -/
def showFormats (p : Parsed) : IO UInt32 := do
  let out : Output := { useColor := !p.hasFlag "no-color" }

  out.header "Output Formats"
  IO.println ""

  -- Create example result
  let exampleResult : BenchmarkResult := {
    spec := vectorAdd1M
    backend := "METAL"
    device := "Apple M3 Pro"
    stats := {
      min := ⟨70000⟩
      max := ⟨150000⟩
      mean := ⟨100000⟩
      median := ⟨95000⟩
      stddev := 25000.0
      samples := 100
    }
    bandwidth_gb_s := 171.7
    throughput_gflops := 14.3
    verified := true
    timestamp := 1703577600
  }

  IO.println s!"{out.bold "Human-readable:"}"
  IO.println ""
  IO.println (formatResult exampleResult)
  IO.println ""

  IO.println s!"{out.bold "JSON:"}"
  IO.println ""
  IO.println (formatResultJson exampleResult)
  IO.println ""

  return 0

/-! ## Command Definitions -/

def runCmd : Cmd := `[Cli|
  run VIA runBenchmarks; [version]
  "Run GPU benchmarks on available backends."

  FLAGS:
    v, verbose;                    "Enable verbose output."
    "no-color";                    "Disable colored output."
    json : String;                 "Write JSON report to file."
    markdown : String;             "Write Markdown report to file."
    backend : Array String;        "Backends to run (default: all available)."
    suite : Array String;          "Suites to run (default: all)."
    quick;                         "Run quick benchmarks (fewer iterations)."
]

def backendsCmd : Cmd := `[Cli|
  backends VIA listBackends; [version]
  "List available GPU backends."

  FLAGS:
    "no-color";    "Disable colored output."
]

def formatsCmd : Cmd := `[Cli|
  formats VIA showFormats; [version]
  "Show output format examples."

  FLAGS:
    "no-color";    "Disable colored output."
]

def mainCmd : Cmd := `[Cli|
  tg4_bench NOOP; [version]
  "TinyGrad4 GPU backend benchmark tool.

Compare performance across Metal, CUDA, and other backends.
Outputs structured JSON and Markdown reports for CI integration."

  SUBCOMMANDS:
    runCmd;
    backendsCmd;
    formatsCmd
]

/-! ## Main Entry Point -/

def main (args : List String) : IO UInt32 := do
  mainCmd.validate args
