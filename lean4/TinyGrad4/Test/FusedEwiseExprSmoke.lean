import TinyGrad4

/-!
# FusedEwiseExprSmoke

Checks we can:
- compile a simple scalar elementwise graph into a `Backend.FusedEwise.Plan`
- decode it into a proof-friendly `Kernel.Spec.Expr`
- evaluate both paths to the same scalar result
-/

namespace TinyGrad4.Test.FusedEwiseExprSmoke

open Std
open TinyGrad4
open TinyGrad4.Backend
open Interpreter
open StaticTensor
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

private def assertEqF32 (got expected : Float32) (msg : String) : IO Unit := do
  if got.toBits != expected.toBits then
    throw (IO.userError s!"{msg}: got={got.toFloat} expected={expected.toFloat}")

private def mkRefCnt (nodes : List UOp) : HashMap UOpId Nat :=
  nodes.foldl (init := (∅ : HashMap UOpId Nat)) fun m u =>
    u.src.foldl (fun m' s => m'.insert s.uid (m'.getD s.uid 0 + 1)) m

private def readFromEnv (plan : Backend.FusedEwise.Plan) (env : Env) : (t : Ty) → Nat → Ty.denote t
  | .f32, i =>
    if i < plan.leaves.size then
      let uid := plan.leaves[i]!
      let buf := env.getD uid (RawBuffer.zeros .float32 1)
      ((RawBuffer.decodeScalarF32 buf).toFloat32)
    else
      (0.0 : Float).toFloat32
  | .bool, i =>
    if i < plan.leaves.size then
      let uid := plan.leaves[i]!
      let buf := env.getD uid (RawBuffer.zeros .bool 1)
      buf.data.get! 0 != 0
    else
      false

private def testScalarEwise : IO Unit := do
  let (aId, bId, cId, outU) := Id.run do
    runTensorM do
      let a ← Tensor.buffer [] .float32
      let b ← Tensor.buffer [] .float32
      let c ← Tensor.buffer [] .float32
      let bc ← mul b c
      let y ← add a bc
      pure (a.uop.uid, b.uop.uid, c.uop.uid, y.uop)

  let nodes := UOp.toposort outU
  let refCnt := mkRefCnt nodes
  let keep : UOpIdSet := UOpIdSet.mkEmpty
  let plan ←
    match Backend.FusedEwise.compile outU keep refCnt with
    | some p => pure p
    | none => throw (IO.userError "expected FusedEwise.compile to succeed")

  let expr : Expr .f32 ←
    match Backend.FusedEwiseExpr.toKernelExpr? plan with
    | some e => pure e
    | none => throw (IO.userError "expected toKernelExpr? to succeed")

  let env : Env := (∅ : Env)
    |>.insert aId (RawBuffer.ofF32 ⟨#[1.0]⟩)
    |>.insert bId (RawBuffer.ofF32 ⟨#[2.0]⟩)
    |>.insert cId (RawBuffer.ofF32 ⟨#[3.0]⟩)

  let gotUop := (RawBuffer.decodeScalarF32 (← evalRawCached outU env)).toFloat32
  let gotExpr := evalExpr f32OpsTest (readFromEnv plan env) expr
  let expected := (7.0 : Float).toFloat32

  assertEqF32 gotUop expected "uop eval"
  assertEqF32 gotExpr expected "expr eval"

def runAll : IO Unit := do
  IO.println "=== FusedEwiseExprSmoke Tests ==="
  testScalarEwise
  IO.println "✓ plan -> expr matches scalar eval"
  IO.println "=== FusedEwiseExprSmoke OK ==="

end TinyGrad4.Test.FusedEwiseExprSmoke

