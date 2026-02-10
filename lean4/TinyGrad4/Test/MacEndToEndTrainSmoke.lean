import Float64
import TinyGrad4.Test.Training

/-!
# MacEndToEndTrainSmoke

A compiled end-to-end training baseline that runs on macOS.

This is intended as a stable optimization target:
- build once, run on a real machine
- run multiple training steps with weight updates
- assert basic health invariants (finite loss, loss reduction, weight mutation)
-/

namespace TinyGrad4.Test.MacEndToEndTrainSmoke

open TinyGrad4
open TinyGrad4.Test.Training

private def lossLooksFinite (x : Float64) : Bool :=
  x == x && Float64.abs x < 1.0e12

private def assertIO (cond : Bool) (msg : String) : IO Unit := do
  if !cond then
    throw (IO.userError msg)

def run (steps : Nat := 80) (lr : Float64 := 0.5) : IO Unit := do
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
    if i % 20 == 0 || i + 1 == steps then
      IO.println s!"step {i + 1}: loss={loss}"

  let t1 ← IO.monoNanosNow
  let elapsedMs : Float64 := (Float64.ofNat (t1 - t0)) / 1000000.0

  let first := firstLoss.getD lastLoss
  assertIO (lastLoss < first) s!"loss did not decrease: first={first}, last={lastLoss}"
  assertIO (state.w1Data.data != init.w1Data.data) "w1 bytes unchanged after training"
  assertIO (state.w2Data.data != init.w2Data.data) "w2 bytes unchanged after training"

  IO.println s!"initial_loss={first}"
  IO.println s!"final_loss={lastLoss}"
  IO.println s!"elapsed_ms={elapsedMs}"
  IO.println "=== MacEndToEndTrainSmoke OK ==="

def runMain : IO Unit := run

end TinyGrad4.Test.MacEndToEndTrainSmoke

def main : IO Unit := TinyGrad4.Test.MacEndToEndTrainSmoke.runMain
