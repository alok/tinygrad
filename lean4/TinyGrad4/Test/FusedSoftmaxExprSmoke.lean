import TinyGrad4

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

/-!
# FusedSoftmaxExprSmoke

Checks that the specialized fused-softmax runtime kernel matches the common semantic core:
two map-reduces (max + sum(exp)) plus a final map.
-/

namespace TinyGrad4.Test.FusedSoftmaxExprSmoke

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

private def assertAllCloseF32 (got expected : Float32) (tol : Float) (msg : String) : IO Unit := do
  let diff := Float.abs (got.toFloat - expected.toFloat)
  if diff > tol then
    throw (IO.userError s!"{msg}: got={got.toFloat} expected={expected.toFloat} diff={diff}")

private def getF32At (vals : FloatArray) (shape : Shape) (idx : List Nat) : Float32 :=
  let flat := Interpreter.flattenIndex idx shape
  (vals[flat]!).toFloat32

private def testSoftmax : IO Unit := do
  let (xId, xShape, outU) := Id.run do
    runTensorM do
      let x ← Tensor.buffer [2, 4] .float32
      let y ← softmax x
      pure (x.uop.uid, x.uop.shape, y.uop)

  let compiled ← Interpreter.compileManyCached [outU]
  let plan ←
    match compiled.implMap[outU.uid]? with
    | some (.fusedSoftmax p) => pure p
    | _ => throw (IO.userError "expected fusion selector to pick fusedSoftmax for softmax graph")

  let planExpr ←
    match Backend.FusedSoftmaxExpr.ofPlan? xShape plan with
    | some pe => pure pe
    | none => throw (IO.userError "expected FusedSoftmaxExpr.ofPlan? to succeed")

  let xVals : FloatArray := ⟨#[1.0, 2.0, 3.0, 4.0,  0.0, 0.5, 1.0, 1.5]⟩
  let env : Env := (∅ : Env) |>.insert xId (RawBuffer.ofF32 xVals)

  let cache := Interpreter.evalCompiledRaw compiled env
  let outBuf := cache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))
  let outArr := outBuf.decode

  let fullShape := xShape
  let axis := planExpr.axis
  let reduceShape := Shape.reduce fullShape [axis] true

  let readAtMax : (t : Ty) → Nat → Index fullShape → Ty.denote t
    | .f32, i, outIdx =>
      if i == 0 then getF32At xVals fullShape outIdx else (Float32.ofBits 0)
    | .bool, _i, _outIdx => false

  let maxVal := mapReduceKeepdimF32 f32OpsTest .max planExpr.maxProg [axis] readAtMax

  let readAtSum : (t : Ty) → Nat → Index fullShape → Ty.denote t
    | .f32, i, outIdx =>
      if i == 0 then
        getF32At xVals fullShape outIdx
      else if i == 1 then
        maxVal (broadcastIndex reduceShape fullShape outIdx)
      else
        (Float32.ofBits 0)
    | .bool, _i, _outIdx => false

  let sumExp := mapReduceKeepdimF32 f32OpsTest .sum planExpr.sumProg [axis] readAtSum

  let readAtOut : (t : Ty) → Nat → Index fullShape → Ty.denote t
    | .f32, i, outIdx =>
      if i == 0 then
        getF32At xVals fullShape outIdx
      else if i == 1 then
        maxVal (broadcastIndex reduceShape fullShape outIdx)
      else if i == 2 then
        sumExp (broadcastIndex reduceShape fullShape outIdx)
      else
        (Float32.ofBits 0)
    | .bool, _i, _outIdx => false

  let sem := mapExprF32 f32OpsTest planExpr.outProg readAtOut

  let n := listProd fullShape
  if outArr.size != n then
    throw (IO.userError s!"softmax: expected out size {n}, got {outArr.size}")

  for i in [:n] do
    let idx := Interpreter.unflattenIndex i fullShape
    let got := (outArr[i]!).toFloat32
    let expected := sem idx
    assertAllCloseF32 got expected 1.0e-4 s!"out[{i}]"

def runAll : IO Unit := do
  IO.println "=== FusedSoftmaxExprSmoke Tests ==="
  testSoftmax
  IO.println "✓ fused softmax matches map-reduce semantics"
  IO.println "=== FusedSoftmaxExprSmoke OK ==="

end TinyGrad4.Test.FusedSoftmaxExprSmoke

#eval! TinyGrad4.Test.FusedSoftmaxExprSmoke.runAll
