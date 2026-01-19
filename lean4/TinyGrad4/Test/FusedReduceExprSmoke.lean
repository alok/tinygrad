import Float64
import TinyGrad4

-- Disable RawBuffer linter for test files that need Array Float64 literals
set_option linter.useRawBuffer false

/-!
# FusedReduceExprSmoke

Checks we can:
- compile a `FusedReduce.Plan` for a simple reduce graph
- execute the compiled graph and match expected numeric output
- verify the fused-reduce plan is wired to the input
-/

namespace TinyGrad4.Test.FusedReduceExprSmoke

open Std
open TinyGrad4
open TinyGrad4.Backend
open Interpreter
open StaticTensor

private def assertEqF32 (got expected : Float32) (msg : String) : IO Unit := do
  if got.toBits != expected.toBits then
    throw (IO.userError s!"{msg}: got={got.toFloat} expected={expected.toFloat}")

private def mkRefCnt (nodes : List UOp) : HashMap UOpId Nat :=
  nodes.foldl (init := (∅ : HashMap UOpId Nat)) fun m u =>
    u.src.foldl (fun m' s => m'.insert s.uid (m'.getD s.uid 0 + 1)) m

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

  -- A tiny known input.
  let xVals : FloatArray := ⟨#[1.0, 2.0, 3.0,  4.0, 5.0, 6.0]⟩
  let env : Env := (∅ : Env)
    |>.insert xId (RawBuffer.ofF32 xVals)

  let compiled ← Interpreter.compileManyCached [outU]
  let cache := Interpreter.evalCompiledRaw compiled env
  let outBuf := cache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))
  let outArr := outBuf.decode

  -- Sanity: this should be a single-leaf map program over `x`.
  if plan.ewise.leafBases.size != 1 then
    throw (IO.userError s!"expected 1 leaf, got {plan.ewise.leafBases.size}")
  if plan.ewise.leafBases[0]! != xId then
    throw (IO.userError s!"expected leaf[0]={xId}, got {plan.ewise.leafBases[0]!}")

  let fullShape := plan.fullShape.toList
  let axes := plan.axes.toList
  let outShape := Shape.reduce fullShape axes true

  let got0 := (outArr[Interpreter.flattenIndex [0, 0] outShape]!).toFloat32
  let got1 := (outArr[Interpreter.flattenIndex [1, 0] outShape]!).toFloat32
  let exp0 : Float32 := 14.0
  let exp1 : Float32 := 77.0

  assertEqF32 got0 exp0 "out[0]"
  assertEqF32 got1 exp1 "out[1]"

def runAll : IO Unit := do
  IO.println "=== FusedReduceExprSmoke Tests ==="
  testReduceSumAxis
  IO.println "✓ reduce plan matches numeric output"
  IO.println "=== FusedReduceExprSmoke OK ==="

end TinyGrad4.Test.FusedReduceExprSmoke

#eval! TinyGrad4.Test.FusedReduceExprSmoke.runAll
