import TinyGrad4.Kernel.Spec

/-!
# KernelExprSmoke

Sanity checks for `TinyGrad4.Kernel.Spec.Expr`:
- typed `input` (both `.f32` and `.bool`)
- `truthy` float-to-bool conversion
- `constBool`
- `evalExpr` execution
-/

namespace TinyGrad4.Test.KernelExprSmoke

open TinyGrad4
open Kernel

private def f32OpsTest : ScalarOps Float32 :=
  { neg := fun x => (-x.toFloat).toFloat32
    sqrt := fun x => x
    reciprocal := fun x => (1.0 / x.toFloat).toFloat32
    exp2 := fun x => x
    log2 := fun x => x
    sin := fun x => x
    add := fun a b => (a.toFloat + b.toFloat).toFloat32
    sub := fun a b => (a.toFloat - b.toFloat).toFloat32
    mul := fun a b => (a.toFloat * b.toFloat).toFloat32
    div := fun a b => (a.toFloat / b.toFloat).toFloat32
    max := fun a b => if a.toFloat <= b.toFloat then b else a
    cmplt := fun a b => a.toFloat < b.toFloat
    where_ := fun c x y => if c then x else y
    zero := (0.0 : Float).toFloat32
    negInf := (-1.0e30 : Float).toFloat32 }

private def readTest : (t : Ty) → Nat → Ty.denote t
  | .f32, i =>
    match i with
    | 0 => (1.0 : Float).toFloat32
    | 1 => (2.0 : Float).toFloat32
    | 2 => (0.0 : Float).toFloat32
    | _ => (0.0 : Float).toFloat32
  | .bool, i =>
    match i with
    | 0 => true
    | _ => false

private def assertEqF32 (got expected : Float32) (msg : String) : IO Unit := do
  if got.toBits != expected.toBits then
    throw (IO.userError s!"{msg}: got={got.toFloat} expected={expected.toFloat}")

private def testBoolInput : IO Unit := do
  let expr : Expr .f32 := .where_ (.input .bool 0) (.input .f32 0) (.input .f32 1)
  let got := evalExpr f32OpsTest readTest expr
  let expected := (1.0 : Float).toFloat32
  assertEqF32 got expected "bool input"

private def testTruthy : IO Unit := do
  let expr : Expr .f32 := .where_ (.truthy (.input .f32 2)) (.input .f32 0) (.input .f32 1)
  let got := evalExpr f32OpsTest readTest expr
  let expected := (2.0 : Float).toFloat32
  assertEqF32 got expected "truthy (0.0 -> false)"

private def testConstBool : IO Unit := do
  let expr : Expr .f32 := .where_ (.constBool false) (.input .f32 0) (.input .f32 1)
  let got := evalExpr f32OpsTest readTest expr
  let expected := (2.0 : Float).toFloat32
  assertEqF32 got expected "constBool false"

def runAll : IO Unit := do
  IO.println "=== KernelExprSmoke Tests ==="
  testBoolInput
  IO.println "✓ typed input (.bool)"
  testTruthy
  IO.println "✓ truthy (.f32 -> .bool)"
  testConstBool
  IO.println "✓ constBool"
  IO.println "=== KernelExprSmoke OK ==="

end TinyGrad4.Test.KernelExprSmoke

