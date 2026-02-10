import Float64
import TinyGrad4.Test.MacEndToEndTrainCore

/-!
# MacEndToEndTrainSmoke

A compiled end-to-end training baseline that runs on macOS.

This is intended as a stable optimization target:
- build once, run on a real machine
- run multiple training steps with weight updates
- assert basic health invariants (finite loss, loss reduction, weight mutation)
-/

namespace TinyGrad4.Test.MacEndToEndTrainSmoke

open TinyGrad4.Test.MacEndToEndTrainCore

def run (steps : Nat := 80) (lr : Float64 := 0.5) : IO Unit := do
  let _ ← runOnce steps lr (verbose := true)
  pure ()

def runMain : IO Unit := run

end TinyGrad4.Test.MacEndToEndTrainSmoke

def main : IO Unit := TinyGrad4.Test.MacEndToEndTrainSmoke.runMain
