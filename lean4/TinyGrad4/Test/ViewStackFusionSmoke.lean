import Float64
import TinyGrad4

-- Disable RawBuffer linter for test files that need Array Float64 literals
set_option linter.useRawBuffer false

/-!
# ViewStackFusionSmoke

Ensures we can virtualize non-affine movement chains (e.g. `PERMUTE → RESHAPE`) inside fused kernels.

This requires a ShapeTracker-style view stackUnsafe, since `permuteUnsafe→reshapeUnsafe` cannot be collapsed into a single affine
`(shape,strides,offset,mask)` view.
-/

namespace TinyGrad4.Test.ViewStackFusionSmoke

open Std
open TinyGrad4
open Interpreter
open StaticTensor
open Backend

private def assertAllClose (arr : FloatArray) (expected : Array Float64) (tol : Float64) (label : String) : IO Unit := do
  if arr.size != expected.size then
    throw (IO.userError s!"{label}: size {arr.size} != {expected.size}")
  for i in [:arr.size] do
    let v := arr[i]!
    let e := expected[i]!
    let diff := Float64.abs (v - e)
    if diff > tol then
      throw (IO.userError s!"{label}: idx {i} value {v} expected {e} diff {diff} > {tol}")

private def getFusedEwise (compiled : Interpreter.Compiled) (root : UOp) : IO Backend.FusedEwise.Plan := do
  match compiled.implMap[root.uid]? with
  | some (.fusedEwise plan) => pure plan
  | _ => throw (IO.userError "expected root to select fusedEwise")

private def assertNoMovement (compiled : Interpreter.Compiled) (ops : List Ops) (label : String) : IO Unit := do
  let bad := compiled.nodes.filter fun u => ops.contains u.op
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"{label}: expected movement virtualized, found {bad.length}\n{msg}")

private def testPermuteReshapeIntoFusedEwise : IO Unit := do
  let xData : FloatArray := ⟨#[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]⟩

  let (xU, outU) := runTensorM do
    let x ← Tensor.buffer [2, 3] .float32
    let xp ← StaticTensor.permuteUnsafe x [1, 0]
    let xr ← StaticTensor.reshapeUnsafe xp [6]
    let c1 ← UOp.const .float32 1.0
    let c2 ← UOp.const .float32 2.0
    let y ← UOp.add xr.uop c1
    let out ← UOp.mul y c2
    pure (x.uop, out)

  let env : Env :=
    Interpreter.setBuffer (∅ : Env) xU (RawBuffer.ofF32 xData)

  let compiled := Interpreter.compile outU
  let plan ← getFusedEwise compiled outU
  if !plan.needsStack then
    throw (IO.userError "expected fused ewise to require a view stackUnsafe (permuteUnsafe→reshapeUnsafe)")

  assertNoMovement compiled [.PERMUTE, .RESHAPE] "permuteUnsafe→reshapeUnsafe"

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outArr := (outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))).decode
  assertAllClose outArr #[2.0, 8.0, 4.0, 10.0, 6.0, 12.0] 1.0e-6 "permuteUnsafe→reshapeUnsafe fused ewise"

private def testShrinkReshapeFoldFusedEwise : IO Unit := do
  let xData : FloatArray := ⟨#[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                             12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]⟩

  let (xU, outU) := runTensorM do
    let x ← Tensor.buffer [2, 3, 4] .float32
    let xs ← StaticTensor.shrinkUnsafe x [(0, 2), (0, 2), (0, 4)]
    let xr ← StaticTensor.reshapeUnsafe xs [2, 8]
    let c1 ← UOp.const .float32 1.0
    let c2 ← UOp.const .float32 2.0
    let y ← UOp.add xr.uop c1
    let out ← UOp.mul y c2
    pure (x.uop, out)

  let env : Env :=
    Interpreter.setBuffer (∅ : Env) xU (RawBuffer.ofF32 xData)

  let compiled := Interpreter.compile outU
  let plan ← getFusedEwise compiled outU
  if plan.needsStack then
    throw (IO.userError "expected shrinkUnsafe→reshapeUnsafe fold to avoid a view stackUnsafe")

  assertNoMovement compiled [.RESHAPE, .SHRINK] "shrinkUnsafe→reshapeUnsafe fold"

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outArr := (outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))).decode
  assertAllClose outArr #[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0]
    1.0e-6 "shrinkUnsafe→reshapeUnsafe fold fused ewise"

private def testShrinkReshapeGapFusedEwise : IO Unit := do
  let xData : FloatArray := ⟨#[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                             12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]⟩

  let (xU, outU) := runTensorM do
    let x ← Tensor.buffer [2, 3, 4] .float32
    let xs ← StaticTensor.shrinkUnsafe x [(0, 2), (0, 2), (0, 4)]
    let xr ← StaticTensor.reshapeUnsafe xs [4, 4]
    let c1 ← UOp.const .float32 1.0
    let c2 ← UOp.const .float32 2.0
    let y ← UOp.add xr.uop c1
    let out ← UOp.mul y c2
    pure (x.uop, out)

  let env : Env :=
    Interpreter.setBuffer (∅ : Env) xU (RawBuffer.ofF32 xData)

  let compiled := Interpreter.compile outU
  let plan ← getFusedEwise compiled outU
  if !plan.needsStack then
    throw (IO.userError "expected shrinkUnsafe→reshapeUnsafe gap to require a view stackUnsafe")

  assertNoMovement compiled [.RESHAPE, .SHRINK] "shrinkUnsafe→reshapeUnsafe gap"

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outArr := (outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))).decode
  assertAllClose outArr #[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0]
    1.0e-6 "shrinkUnsafe→reshapeUnsafe gap fused ewise"

private def testPermuteReshapeBackNoStack : IO Unit := do
  let xData : FloatArray := ⟨#[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                             12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]⟩

  let (xU, outU) := runTensorM do
    let x ← Tensor.buffer [2, 3, 4] .float32
    let xp ← StaticTensor.permuteUnsafe x [1, 0, 2]
    let xr0 ← StaticTensor.reshapeUnsafe xp [6, 4]
    let xr1 ← StaticTensor.reshapeUnsafe xr0 [3, 2, 4]
    let c1 ← UOp.const .float32 1.0
    let c2 ← UOp.const .float32 2.0
    let y ← UOp.add xr1.uop c1
    let out ← UOp.mul y c2
    pure (x.uop, out)

  let env : Env :=
    Interpreter.setBuffer (∅ : Env) xU (RawBuffer.ofF32 xData)

  let compiled := Interpreter.compile outU
  let plan ← getFusedEwise compiled outU
  if plan.needsStack then
    throw (IO.userError "expected permuteUnsafe→reshapeUnsafe→reshapeUnsafe to avoid a view stackUnsafe")

  assertNoMovement compiled [.PERMUTE, .RESHAPE] "permuteUnsafe→reshapeUnsafe→reshapeUnsafe"

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outArr := (outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))).decode
  assertAllClose outArr
    #[2.0, 4.0, 6.0, 8.0, 26.0, 28.0, 30.0, 32.0, 10.0, 12.0, 14.0, 16.0, 34.0, 36.0, 38.0, 40.0,
      18.0, 20.0, 22.0, 24.0, 42.0, 44.0, 46.0, 48.0]
    1.0e-6 "permuteUnsafe→reshapeUnsafe→reshapeUnsafe fused ewise"

def runAll : IO Unit := do
  IO.println "=== ViewStackFusionSmoke Tests ==="
  testPermuteReshapeIntoFusedEwise
  IO.println "✓ permuteUnsafe→reshapeUnsafe virtualized in fused ewise"
  testShrinkReshapeFoldFusedEwise
  IO.println "✓ shrinkUnsafe→reshapeUnsafe fold fused ewise"
  testShrinkReshapeGapFusedEwise
  IO.println "✓ shrinkUnsafe→reshapeUnsafe gap fused ewise"
  testPermuteReshapeBackNoStack
  IO.println "✓ permuteUnsafe→reshapeUnsafe→reshapeUnsafe fused ewise"
  IO.println "=== ViewStackFusionSmoke OK ==="

end TinyGrad4.Test.ViewStackFusionSmoke

#eval! TinyGrad4.Test.ViewStackFusionSmoke.runAll
