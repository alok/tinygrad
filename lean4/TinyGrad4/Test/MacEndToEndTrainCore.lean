import Float64
import TinyGrad4.Test.Training

/-!
# MacEndToEndTrainCore

Shared training runner for macOS end-to-end smoke/bench executables.
-/

namespace TinyGrad4.Test.MacEndToEndTrainCore

open TinyGrad4
open TinyGrad4.Test.Training

structure RunMetrics where
  steps : Nat
  lr : Float64
  initialLoss : Float64
  finalLoss : Float64
  elapsedMs : Float64
  deriving Repr

private def lossLooksFinite (x : Float64) : Bool :=
  x == x && Float64.abs x < 1.0e12

private def assertIO (cond : Bool) (msg : String) : IO Unit := do
  if !cond then
    throw (IO.userError msg)

def runOnce (steps : Nat := 80) (lr : Float64 := 0.5) (verbose : Bool := true) : IO RunMetrics := do
  if verbose then
    IO.println "=== MacEndToEndTrainSmoke ==="
    IO.println s!"steps={steps}, lr={lr}"

  let t0 ← IO.monoNanosNow
  let program ← buildProgram lr

  let init := initWeights
  let mut state := init
  let mut firstLoss : Option Float64 := none
  let mut lastLoss : Float64 := 0.0

  for i in [:steps] do
    let (loss, nextState) ← trainStep program state
    state := nextState
    if firstLoss.isNone then
      firstLoss := some loss
    lastLoss := loss

    assertIO (lossLooksFinite loss) s!"loss became invalid at step {i}: {loss}"
    if verbose && (i % 20 == 0 || i + 1 == steps) then
      IO.println s!"step {i + 1}: loss={loss}"

  let t1 ← IO.monoNanosNow
  let elapsedMs : Float64 := (Float64.ofNat (t1 - t0)) / 1000000.0

  let first := firstLoss.getD lastLoss
  assertIO (lastLoss < first) s!"loss did not decrease: first={first}, last={lastLoss}"
  assertIO (state.w1Data.data != init.w1Data.data) "w1 bytes unchanged after training"
  assertIO (state.w2Data.data != init.w2Data.data) "w2 bytes unchanged after training"

  if verbose then
    IO.println s!"initial_loss={first}"
    IO.println s!"final_loss={lastLoss}"
    IO.println s!"elapsed_ms={elapsedMs}"
    IO.println "=== MacEndToEndTrainSmoke OK ==="

  pure {
    steps := steps,
    lr := lr,
    initialLoss := first,
    finalLoss := lastLoss,
    elapsedMs := elapsedMs
  }

end TinyGrad4.Test.MacEndToEndTrainCore
