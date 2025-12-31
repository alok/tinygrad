import TinyGrad4

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

/-!
# ReduceAxis Fin API Smoke Test

This is the dependent-typing version of `ReduceAxis`:
we reduce with `Fin rank` axes so out-of-bounds axes are unrepresentable.
-/

namespace TinyGrad4.Test.ReduceAxisFinSmoke

open TinyGrad4
open StaticTensor
open Interpreter

private def assertSize (arr : FloatArray) (expected : Nat) (label : String) : IO Unit := do
  if arr.size != expected then
    throw (IO.userError s!"{label}: size {arr.size} != {expected}")

private def assertAllClose (arr : FloatArray) (expected : Float) (tol : Float) (label : String) : IO Unit := do
  for i in [:arr.size] do
    let v := arr[i]!
    let diff := Float.abs (v - expected)
    if diff > tol then
      throw (IO.userError s!"{label}: idx {i} value {v} expected {expected} diff {diff} > {tol}")

def testSumAxisFin : IO Unit := do
  let res := runTensorM do
    let x ← Tensor.full [2, 3] .float32 2.0
    let axisLast : Fin 2 := ⟨1, by decide⟩
    let y ← sumAxisF x axisLast true
    pure (eval y.uop (∅ : Env))
  assertSize res 2 "sumAxisF"
  assertAllClose res 6.0 0.01 "sumAxisF"

def testMaxAxisFin : IO Unit := do
  let res := runTensorM do
    let x ← Tensor.full [2, 3] .float32 2.0
    let axisFirst : Fin 2 := ⟨0, by decide⟩
    let y ← maxAxisF x axisFirst true
    pure (eval y.uop (∅ : Env))
  assertSize res 3 "maxAxisF"
  assertAllClose res 2.0 0.01 "maxAxisF"

def runAll : IO Unit := do
  IO.println "=== ReduceAxisFinSmoke Tests ==="
  testSumAxisFin
  IO.println "✓ sumAxisF"
  testMaxAxisFin
  IO.println "✓ maxAxisF"
  IO.println "=== ReduceAxisFinSmoke OK ==="

end TinyGrad4.Test.ReduceAxisFinSmoke

#eval! TinyGrad4.Test.ReduceAxisFinSmoke.runAll

