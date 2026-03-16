import Float64
import TinyGrad4.Test.MacEndToEndTrainCore

/-!
# MacEndToEndTrainBench

Multi-trial benchmark wrapper for the compiled macOS end-to-end training smoke.

Environment knobs:
- `TG4_E2E_STEPS` (default: 80)
- `TG4_E2E_TRIALS` (default: 5)
- `TG4_E2E_WARMUP` (default: 1)
-/

namespace TinyGrad4.Test.MacEndToEndTrainBench

open TinyGrad4.Test.MacEndToEndTrainCore

private def envNat (key : String) (default : Nat) : IO Nat := do
  match (← IO.getEnv key) with
  | none => pure default
  | some s =>
    match s.toNat? with
    | some n => pure n
    | none =>
      IO.println s!"warning: invalid {key}={s}, using default {default}"
      pure default

private def insertSorted (x : Float64) : List Float64 → List Float64
  | [] => [x]
  | y :: ys =>
    if x <= y then
      x :: y :: ys
    else
      y :: insertSorted x ys

private def sortFloats (xs : List Float64) : List Float64 :=
  xs.foldl (fun acc x => insertSorted x acc) []

private def mean (xs : List Float64) : Float64 :=
  if xs.isEmpty then
    0.0
  else
    xs.foldl (· + ·) 0.0 / (Float64.ofNat xs.length)

private def percentile (sorted : Array Float64) (p : Nat) : Float64 :=
  if sorted.isEmpty then
    0.0
  else
    let idx := (p * (sorted.size - 1)) / 100
    sorted[idx]!

def runBench : IO Unit := do
  let steps ← envNat "TG4_E2E_STEPS" 80
  let trials ← envNat "TG4_E2E_TRIALS" 5
  let warmup ← envNat "TG4_E2E_WARMUP" 1
  let lr : Float64 := 0.5

  IO.println "=== MacEndToEndTrainBench ==="
  IO.println s!"config: steps={steps}, trials={trials}, warmup={warmup}, lr={lr}"

  for i in [:warmup] do
    let _ ← runOnce steps lr (verbose := false)
    IO.println s!"warmup {i + 1}/{warmup} complete"

  let mut times : List Float64 := []
  let mut finals : List Float64 := []
  for i in [:trials] do
    let m ← runOnce steps lr (verbose := false)
    times := m.elapsedMs :: times
    finals := m.finalLoss :: finals
    IO.println s!"trial {i + 1}/{trials}: final_loss={m.finalLoss}, elapsed_ms={m.elapsedMs}"

  let timesSorted := sortFloats times
  let finalsSorted := sortFloats finals
  let timesArr := timesSorted.toArray
  let finalsArr := finalsSorted.toArray

  let p50 := percentile timesArr 50
  let p95 := percentile timesArr 95
  let minT := percentile timesArr 0
  let maxT := percentile timesArr 100
  let meanT := mean times
  let finalMed := percentile finalsArr 50

  IO.println s!"time_ms: min={minT}, p50={p50}, p95={p95}, max={maxT}, mean={meanT}"
  IO.println s!"final_loss_median={finalMed}"
  IO.println "=== MacEndToEndTrainBench OK ==="

end TinyGrad4.Test.MacEndToEndTrainBench

def main : IO Unit := TinyGrad4.Test.MacEndToEndTrainBench.runBench
