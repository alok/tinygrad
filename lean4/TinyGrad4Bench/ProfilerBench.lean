import TinyGrad4.Data.Profile
import TinyGrad4Bench.BenchUtil

/-!
# Profiler Bench

Microbenchmarks for profiler overhead + a small pipeline demo with wait ratio output.
-/

open TinyGrad4.Data
open LeanBenchNew
open TinyGrad4Bench

/-! ## Microbenchmarks -/

def benchProfilerRecord : IO (Array Result) := do
  let mut results := #[]
  let profiler ← Profiler.new

  let r1 ← run "Profiler.recordSample" microBenchConfig fun () => do
    profiler.recordSample "stage" 100 10
  results := results.push r1

  let r2 ← run "Profiler.record" microBenchConfig fun () => do
    let _ ← profiler.record "stage" (pure ())
    pure ()
  results := results.push r2

  pure results

/-! ## Pipeline Demo -/

private def sleepMs (ms : Nat) : IO Unit :=
  IO.sleep (UInt32.ofNat ms)

private def runPipeline (label : String) (bufferSize : Option Nat) (n : Nat := 128) (delayMs : Nat := 1) : IO String := do
  let profiler ← Profiler.new
  let base := ofArray (Array.range n)
  let staged := profileMapIODs profiler "mapIO" (fun x => do
    sleepMs delayMs
    pure (x + 1)) base

  match bufferSize with
  | none =>
    let iter ← Dataset.toIterator staged
    let iter := profileIter profiler "iter" iter
    repeat do
      match ← iter.next with
      | some _ => pure ()
      | none => break
  | some bufSize =>
    let prefetched := prefetchDs bufSize staged
    let prefetcher ← PrefetchedDataset.toPrefetcher prefetched
    repeat do
      match ← profilePrefetcherNext profiler "prefetch" prefetcher with
      | some _ => pure ()
      | none => break
    prefetcher.wait

  let summary ← profiler.summaryByWaitRatio
  pure s!"{label}\n{summary}"

/-! ## Run All -/

def main : IO Unit := do
  IO.println "═══════════════════════════════════════════════════════════════"
  IO.println "                     PROFILER BENCH"
  IO.println "═══════════════════════════════════════════════════════════════"
  IO.println ""

  IO.println "▶ Profiler Microbenchmarks"
  IO.println "──────────────────────────"
  let profResults ← benchProfilerRecord
  printComparison profResults
  IO.println ""

  IO.println "▶ Pipeline Demo (wait ratio)"
  IO.println "───────────────────────────"
  let baseline ← runPipeline "baseline (no prefetch)" none
  IO.println baseline
  IO.println ""
  let prefetched ← runPipeline "prefetch (buffer=4)" (some 4)
  IO.println prefetched
  IO.println ""

  IO.println "═══════════════════════════════════════════════════════════════"
  IO.println "                     BENCH COMPLETE"
  IO.println "═══════════════════════════════════════════════════════════════"
