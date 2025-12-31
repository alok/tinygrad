import TinyGrad4

/-!
# CostExprSmoke

Checks the symbolic cost expression layer (`Backend.CostExpr`) lines up with the numeric cost model:

- Lifting a `CostProg` to `CostProgExpr` and evaluating it matches the numeric time.
- Hand-written symbolic programs evaluate as expected under an environment.
-/

namespace TinyGrad4.Test.CostExprSmoke

open TinyGrad4
open TinyGrad4.Backend
open Std

private def assertEqNat (got expected : Nat) (msg : String) : IO Unit := do
  if got != expected then
    throw (IO.userError s!"{msg}: got={got} expected={expected}")

private def testLiftMatchesNumeric : IO Unit := do
  let cm : CostModel := {
    kernelOverhead := 100
    elem := 2
    moveElem := 3
    memReadByte := 4
    memWriteByte := 5
    memReadViewByte := 7
    memWriteViewByte := 11
    reduceElem := 13
    matmulMulAdd := 17
    matmulViewMulAdd := 19
  }

  let p : CostProg := #[
    .launch 2,
    .mem 10 5,
    .memView 3 7,
    .elemwise 9 4,
    .move 6,
    .reduce 8,
    .matmul 12 (view := true)
  ]

  let num := CostProg.time cm p
  let expr := CostProgExpr.time cm (CostProg.toExpr p)
  let got := CostExpr.eval (∅ : HashMap String Nat) expr
  assertEqNat got num "lifted CostProgExpr should match CostProg.time"

private def testSymbolicEval : IO Unit := do
  let cm : CostModel := {
    kernelOverhead := 1000
    elem := 2
    memReadByte := 3
    memWriteByte := 5
    matmulMulAdd := 7
  }

  let p : CostProgExpr := #[
    .launch (.var "L"),
    .mem (.var "r") (.var "w"),
    .elemwise (.var "n") 4,
    .matmul (.var "mm") (view := false)
  ]

  let expr := CostProgExpr.time cm p
  let env : HashMap String Nat := (∅ : HashMap String Nat)
    |>.insert "L" 2
    |>.insert "r" 10
    |>.insert "w" 4
    |>.insert "n" 6
    |>.insert "mm" 3

  let got := CostExpr.eval env expr
  let expected :=
    (cm.kernelOverhead * 2) +
    (cm.memReadByte * 10) + (cm.memWriteByte * 4) +
    (cm.elem * 6 * 4) +
    (cm.matmulMulAdd * 3)

  assertEqNat got expected "symbolic CostProgExpr evaluation mismatch"

def runAll : IO Unit := do
  IO.println "=== CostExprSmoke Tests ==="
  testLiftMatchesNumeric
  IO.println "✓ lift numeric -> symbolic preserves time"
  testSymbolicEval
  IO.println "✓ symbolic eval matches expected"
  IO.println "=== CostExprSmoke OK ==="

end TinyGrad4.Test.CostExprSmoke

#eval! TinyGrad4.Test.CostExprSmoke.runAll

