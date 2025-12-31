import TinyGrad4.Backend.Metal

/-!
# Benchmark Instrumentation

Provides fine-grained timing for Metal GPU operations:
- Compile time: shader compilation (Metal library + pipeline creation)
- Dispatch time: CPU-side kernel setup and submission
- Kernel time: GPU execution time
- Sync time: wait/fence overhead

## Usage

```lean
open TinyGrad4.Benchmark

-- Compile with timing
let (prog, compileNs) ← timedCompile "kernel" source

-- Time any IO action
let (result, timeNs) ← timed (metalLaunch prog bufs ...)
```
-/

namespace TinyGrad4.Benchmark

open TinyGrad4.Backend
open TinyGrad4.Backend.Metal

/-- Kernel timing breakdown -/
structure KernelTiming where
  /-- CPU-side dispatch time (setup + submit) in nanoseconds -/
  dispatchNs : Nat
  /-- Total GPU time (commit to completion) in nanoseconds -/
  kernelNs : Nat
  /-- Wait/sync overhead in nanoseconds -/
  syncNs : Nat
  deriving Repr, Inhabited

namespace KernelTiming

/-- Total time in nanoseconds -/
def totalNs (t : KernelTiming) : Nat := t.dispatchNs + t.kernelNs

/-- Dispatch time in microseconds -/
def dispatchUs (t : KernelTiming) : Float := t.dispatchNs.toFloat / 1000.0

/-- Kernel time in microseconds -/
def kernelUs (t : KernelTiming) : Float := t.kernelNs.toFloat / 1000.0

/-- Sync time in microseconds -/
def syncUs (t : KernelTiming) : Float := t.syncNs.toFloat / 1000.0

/-- Total time in milliseconds -/
def totalMs (t : KernelTiming) : Float := t.totalNs.toFloat / 1000000.0

/-- Format as human-readable string -/
def format (t : KernelTiming) : String :=
  s!"dispatch={t.dispatchUs}μs kernel={t.kernelUs}μs sync={t.syncUs}μs"

end KernelTiming

/-- Full operation timing breakdown -/
structure OpTiming where
  /-- Compile time in nanoseconds -/
  compileNs : Nat
  /-- Kernel timing breakdown -/
  kernel : KernelTiming
  deriving Repr, Inhabited

namespace OpTiming

/-- Total time in nanoseconds -/
def totalNs (t : OpTiming) : Nat := t.compileNs + t.kernel.totalNs

/-- Compile time in milliseconds -/
def compileMs (t : OpTiming) : Float := t.compileNs.toFloat / 1000000.0

/-- Total time in milliseconds -/
def totalMs (t : OpTiming) : Float := t.totalNs.toFloat / 1000000.0

/-- Compile fraction (0-1) -/
def compileFrac (t : OpTiming) : Float :=
  if t.totalNs == 0 then 0.0
  else t.compileNs.toFloat / t.totalNs.toFloat

/-- Kernel fraction (0-1) -/
def kernelFrac (t : OpTiming) : Float :=
  if t.totalNs == 0 then 0.0
  else t.kernel.kernelNs.toFloat / t.totalNs.toFloat

/-- Format as human-readable string with percentages -/
def format (t : OpTiming) : String :=
  let total := t.totalMs
  let compPct := t.compileFrac * 100.0
  let kernPct := t.kernelFrac * 100.0
  let dispPct := if t.totalNs == 0 then 0.0 else t.kernel.dispatchNs.toFloat / t.totalNs.toFloat * 100.0
  s!"total={total}ms compile={compPct}% dispatch={dispPct}% kernel={kernPct}%"

end OpTiming

/-- High-resolution timestamp in nanoseconds -/
def getTimeNs : IO Nat := do
  let ms ← IO.monoMsNow
  pure (ms * 1000000)

/-- Measure execution time of an IO action in nanoseconds -/
def timed (action : IO α) : IO (α × Nat) := do
  let start ← IO.monoNanosNow
  let result ← action
  let stop ← IO.monoNanosNow
  pure (result, stop - start)

/-- Measure execution time of an IO action in milliseconds -/
def timedMs (action : IO α) : IO (α × Float) := do
  let (result, ns) ← timed action
  pure (result, ns.toFloat / 1000000.0)

/-- Compile Metal shader with timing -/
def timedCompile (name : String) (source : String) : IO (MetalProgram × Nat) := do
  timed (metalCompile name source)

/-- Launch Metal kernel with timing (includes sync) -/
def timedLaunch (prog : MetalProgram) (bufs : Array MetalBuffer)
    (globalX globalY globalZ : Nat) (localX localY localZ : Nat)
    : IO (Unit × Nat) := do
  let (_, launchNs) ← timed (metalLaunch prog bufs globalX globalY globalZ localX localY localZ)
  let (_, syncNs) ← timed metalSync
  pure ((), launchNs + syncNs)

/-- Compile and launch with full timing breakdown -/
def timedOp (name : String) (source : String) (bufs : Array MetalBuffer)
    (globalX globalY globalZ : Nat) (localX localY localZ : Nat)
    : IO OpTiming := do
  -- Compile with timing
  let (prog, compileNs) ← timedCompile name source

  -- Launch with timing
  let dispatchStart ← IO.monoNanosNow
  metalLaunch prog bufs globalX globalY globalZ localX localY localZ
  let dispatchEnd ← IO.monoNanosNow
  let dispatchNs := dispatchEnd - dispatchStart

  -- Sync with timing
  let syncStart ← IO.monoNanosNow
  metalSync
  let syncEnd ← IO.monoNanosNow
  let syncNs := syncEnd - syncStart

  -- Total kernel time = dispatch + sync
  let kernelNs := dispatchNs + syncNs

  pure {
    compileNs,
    kernel := { dispatchNs, kernelNs, syncNs }
  }

/-- Print timing summary for an operation -/
def printTiming (label : String) (timing : OpTiming) : IO Unit := do
  IO.println s!"{label}: {timing.format}"

/-- Aggregate statistics for multiple timed runs (distinct from Framework.TimingStats) -/
structure MultiRunStats where
  /-- Minimum time in nanoseconds -/
  minNs : Nat
  /-- Maximum time in nanoseconds -/
  maxNs : Nat
  /-- Mean time in nanoseconds -/
  meanNs : Float
  /-- Number of samples -/
  count : Nat
  deriving Repr

namespace MultiRunStats

/-- Create from list of times -/
def fromTimes (times : List Nat) : MultiRunStats :=
  if times.isEmpty then
    { minNs := 0, maxNs := 0, meanNs := 0.0, count := 0 }
  else
    let minT := times.foldl min (times.head!)
    let maxT := times.foldl max (times.head!)
    let sum := times.foldl (· + ·) 0
    let mean := sum.toFloat / times.length.toFloat
    { minNs := minT, maxNs := maxT, meanNs := mean, count := times.length }

/-- Mean time in milliseconds -/
def meanMs (s : MultiRunStats) : Float := s.meanNs / 1000000.0

/-- Min time in milliseconds -/
def minMs (s : MultiRunStats) : Float := s.minNs.toFloat / 1000000.0

/-- Max time in milliseconds -/
def maxMs (s : MultiRunStats) : Float := s.maxNs.toFloat / 1000000.0

/-- Format as human-readable string -/
def format (s : MultiRunStats) : String :=
  s!"min={s.minMs}ms mean={s.meanMs}ms max={s.maxMs}ms (n={s.count})"

end MultiRunStats

/-- Run an action multiple times and collect timing statistics -/
def benchmarkN (n : Nat) (action : IO α) : IO (MultiRunStats × α) := do
  let mut times : List Nat := []
  let mut result : Option α := none
  for _ in [:n] do
    let (r, ns) ← timed action
    times := ns :: times
    result := some r
  let stats := MultiRunStats.fromTimes times
  match result with
  | some r => pure (stats, r)
  | none => throw (IO.userError "benchmarkN: n must be > 0")

/-- Benchmark an action with warmup runs -/
def benchmarkWithWarmup (warmup n : Nat) (action : IO α) : IO (MultiRunStats × α) := do
  -- Warmup runs
  for _ in [:warmup] do
    let _ ← action
  -- Timed runs
  benchmarkN n action

end TinyGrad4.Benchmark
