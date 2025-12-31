import TinyGrad4
import TinyGrad4.Backend.CostExprMeta

/-!
# CostExprMetaSmoke

Checks `costExpr!` can reify a `Nat` cost computation into a `CostExpr` AST, including variables.
-/

namespace TinyGrad4.Test.CostExprMetaSmoke

open TinyGrad4
open TinyGrad4.Backend
open Std

private def assertEqNat (got expected : Nat) (msg : String) : IO Unit := do
  if got != expected then
    throw (IO.userError s!"{msg}: got={got} expected={expected}")

private def basicExpr (m n : Nat) : CostExpr :=
  costExpr! (m * n + 2 * m + 3)

private def minMaxExpr (m n : Nat) : CostExpr :=
  costExpr! (Nat.min m n + Nat.max m n)

private def costProgExpr (n : Nat) : CostExpr := by
  let cm : CostModel := {
    kernelOverhead := 10
    memReadByte := 2
    memWriteByte := 3
    elem := 4
  }
  let p : CostProg := #[
    .launch 2,
    .mem (n * 4) 8,
    .elemwise (n * 2) 3
  ]
  exact costExpr! (CostProg.time cm p)

private def testBasic : IO Unit := do
  let env : HashMap String Nat := (∅ : HashMap String Nat)
    |>.insert "m" 5
    |>.insert "n" 7
  let got := CostExpr.eval env (basicExpr 0 0)
  assertEqNat got (5 * 7 + 2 * 5 + 3) "basicExpr"

private def testMinMax : IO Unit := do
  let env : HashMap String Nat := (∅ : HashMap String Nat)
    |>.insert "m" 5
    |>.insert "n" 7
  let got := CostExpr.eval env (minMaxExpr 0 0)
  assertEqNat got (Nat.min 5 7 + Nat.max 5 7) "minMaxExpr"

private def testCostProg : IO Unit := do
  let env : HashMap String Nat := (∅ : HashMap String Nat)
    |>.insert "n" 5
  let got := CostExpr.eval env (costProgExpr 0)
  let expected := (10 * 2) + (2 * (5 * 4) + 3 * 8) + (4 * (5 * 2) * 3)
  assertEqNat got expected "costProgExpr"

def runAll : IO Unit := do
  IO.println "=== CostExprMetaSmoke Tests ==="
  testBasic
  IO.println "✓ costExpr! basic arithmetic"
  testMinMax
  IO.println "✓ costExpr! min/max"
  testCostProg
  IO.println "✓ costExpr! CostProg.time"
  IO.println "=== CostExprMetaSmoke OK ==="

end TinyGrad4.Test.CostExprMetaSmoke

#eval! TinyGrad4.Test.CostExprMetaSmoke.runAll

