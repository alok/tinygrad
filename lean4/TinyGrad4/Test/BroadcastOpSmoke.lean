import Float64
import TinyGrad4

/-!
# BroadcastOpSmoke

Smoke tests for the broadcast-aware elementwise operators (`+.`, `*.`, etc).

These are Lean-first ergonomics: the output shape is computed at compile time via `Shape.broadcastOut`,
while the runtime still checks broadcastability.
-/

namespace TinyGrad4.Test.BroadcastOpSmoke

-- Disable RawBuffer linter for test files that need Array Float64 literals
set_option linter.useRawBuffer false

open TinyGrad4
open TinyGrad4.Interpreter
open TinyGrad4.StaticTensor

private def assertEqList (got expected : List Float64) (msg : String) : IO Unit := do
  if got != expected then
    throw (IO.userError s!"{msg}: got={got} expected={expected}")

private def testScalarBroadcastAdd : IO Unit := do
  let (xId, yId, outU) := Id.run do
    runTensorM do
      let x ← Tensor.buffer [2, 3] .float32
      let y ← Tensor.buffer [] .float32
      let z ← addBroadcast x y (by native_decide)
      pure (x.uop.uid, y.uop.uid, z.uop)

  let xVals : Array Float64 := #[1.0, 2.0, 3.0,  4.0, 5.0, 6.0]
  let yVal : Array Float64 := #[10.0]
  let env : Env := (∅ : Env)
    |>.insert xId (RawBuffer.ofFloats xVals)
    |>.insert yId (RawBuffer.ofFloats yVal)

  let out := (eval outU env).decode.data.toList
  assertEqList out [11.0, 12.0, 13.0, 14.0, 15.0, 16.0] "scalar broadcast add"

private def testRankBroadcastAdd : IO Unit := do
  let (xId, yId, outU) := Id.run do
    runTensorM do
      let x ← Tensor.buffer [1, 3] .float32
      let y ← Tensor.buffer [2, 1] .float32
      let z ← addBroadcast x y (by native_decide)
      pure (x.uop.uid, y.uop.uid, z.uop)

  let xVals : Array Float64 := #[1.0, 2.0, 3.0]
  let yVals : Array Float64 := #[10.0, 20.0]
  let env : Env := (∅ : Env)
    |>.insert xId (RawBuffer.ofFloats xVals)
    |>.insert yId (RawBuffer.ofFloats yVals)

  let out := (eval outU env).decode.data.toList
  -- Broadcasting: x repeats along dim0, y repeats along dim1.
  assertEqList out [11.0, 12.0, 13.0,  21.0, 22.0, 23.0] "rank broadcast add"

def runAll : IO Unit := do
  IO.println "=== BroadcastOpSmoke Tests ==="
  testScalarBroadcastAdd
  IO.println "✓ scalar broadcast add"
  testRankBroadcastAdd
  IO.println "✓ rank broadcast add"
  IO.println "=== BroadcastOpSmoke OK ==="

end TinyGrad4.Test.BroadcastOpSmoke

#eval! TinyGrad4.Test.BroadcastOpSmoke.runAll
