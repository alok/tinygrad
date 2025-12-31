import TinyGrad4

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

/-!
# FusedReduceExprSmoke

Checks we can:
- compile a `FusedReduce.Plan` for a simple reduce graph
- decode the fused map program into `Kernel.Spec.Expr`
- evaluate the `Kernel.Spec` map-reduce semantics and match the runtime interpreter output
-/

namespace TinyGrad4.Test.FusedReduceExprSmoke

open Std
open TinyGrad4
open TinyGrad4.Backend
open Interpreter
open StaticTensor
open Kernel

private def f32OpsTest : ScalarOps Float32 :=
  { neg := fun x => -x
    sqrt := fun x => x.sqrt
    reciprocal := fun x => (1.0 : Float32) / x
    exp2 := fun x => x.exp2
    log2 := fun x => x.log2
    sin := fun x => x.sin
    add := fun a b => a + b
    sub := fun a b => a - b
    mul := fun a b => a * b
    div := fun a b => a / b
    max := fun a b => max a b
    cmplt := fun a b => decide (a < b)
    where_ := fun c x y => if c then x else y
    zero := (Float32.ofBits 0)
    negInf := (Float32.ofBits 0xFF800000) }

private def assertEqF32 (got expected : Float32) (msg : String) : IO Unit := do
  if got.toBits != expected.toBits then
    throw (IO.userError s!"{msg}: got={got.toFloat} expected={expected.toFloat}")

private def mkRefCnt (nodes : List UOp) : HashMap UOpId Nat :=
  nodes.foldl (init := (∅ : HashMap UOpId Nat)) fun m u =>
    u.src.foldl (fun m' s => m'.insert s.uid (m'.getD s.uid 0 + 1)) m

private def getF32At (vals : FloatArray) (shape : Shape) (idx : List Nat) : Float32 :=
  let flat := Interpreter.flattenIndex idx shape
  (vals[flat]!).toFloat32

private def testReduceSumAxis : IO Unit := do
  let (xId, outU) := Id.run do
    runTensorM do
      let x ← Tensor.buffer [2, 3] .float32
      let xx ← mul x x
      let y ← sumAxis xx 1 true
      pure (x.uop.uid, y.uop)

  let nodes := UOp.toposort outU
  let refCnt := mkRefCnt nodes
  let keep : UOpIdSet := UOpIdSet.add UOpIdSet.mkEmpty outU.uid

  let plan ←
    match Backend.FusedReduce.compile outU keep refCnt with
    | some p => pure p
    | none => throw (IO.userError "expected FusedReduce.compile to succeed")

  let planExpr ←
    match Backend.FusedReduceExpr.ofPlan? outU plan with
    | some pe => pure pe
    | none => throw (IO.userError "expected FusedReduceExpr.ofPlan? to succeed")

  -- A tiny known input.
  let xVals : FloatArray := ⟨#[1.0, 2.0, 3.0,  4.0, 5.0, 6.0]⟩
  let env : Env := (∅ : Env)
    |>.insert xId (RawBuffer.ofF32 xVals)

  let outBuf ← Interpreter.evalRawCached outU env
  let outArr := outBuf.decode

  -- Sanity: this should be a single-leaf map program over `x`.
  if plan.ewise.leaves.size != 1 then
    throw (IO.userError s!"expected 1 leaf, got {plan.ewise.leaves.size}")
  if plan.ewise.leaves[0]! != xId then
    throw (IO.userError s!"expected leaf[0]={xId}, got {plan.ewise.leaves[0]!}")

  let fullShape := planExpr.fullShape
  let axes := planExpr.axes
  let outShape := Shape.reduce fullShape axes true

  let readAt : (t : Ty) → Nat → Index fullShape → Ty.denote t
    | .f32, i, outIdx =>
      if i == 0 then
        getF32At xVals fullShape outIdx
      else
        (Float32.ofBits 0)
    | .bool, _i, _outIdx =>
      false

  let sem := mapReduceKeepdimF32 f32OpsTest planExpr.reduceOp planExpr.map axes readAt

  let got0 := (outArr[Interpreter.flattenIndex [0, 0] outShape]!).toFloat32
  let got1 := (outArr[Interpreter.flattenIndex [1, 0] outShape]!).toFloat32
  let exp0 := sem [0, 0]
  let exp1 := sem [1, 0]

  assertEqF32 got0 exp0 "out[0]"
  assertEqF32 got1 exp1 "out[1]"

def runAll : IO Unit := do
  IO.println "=== FusedReduceExprSmoke Tests ==="
  testReduceSumAxis
  IO.println "✓ reduce plan -> map-reduce semantics matches runtime"
  IO.println "=== FusedReduceExprSmoke OK ==="

end TinyGrad4.Test.FusedReduceExprSmoke

#eval! TinyGrad4.Test.FusedReduceExprSmoke.runAll
