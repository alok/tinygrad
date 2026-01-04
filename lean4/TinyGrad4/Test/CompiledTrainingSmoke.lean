import TinyGrad4

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

/-!
# Compiled Training Smoke Test

Checks we can:
- build forward+backward graph once
- compile it into `.KERNEL` boundaries once
- run multiple SGD steps by only swapping the input buffers (env)

This is the intended "compile once, run many" execution model.
-/

namespace TinyGrad4.Test.CompiledTrainingSmoke

open TinyGrad4
open StaticTensor
open Interpreter

private structure Program where
  w1Id : UOpId
  w2Id : UOpId
  xId : UOpId
  yId : UOpId
  loss : UOp
  newW1 : UOp
  newW2 : UOp
  compiled : Interpreter.Compiled

private def trainingData : RawBuffer × RawBuffer :=
  let x : FloatArray := ⟨#[0.0, 0.0,  0.0, 1.0,  1.0, 0.0,  1.0, 1.0]⟩
  let y : FloatArray := ⟨#[0.0, 0.5, 0.5, 1.0]⟩
  (RawBuffer.ofF32 x, RawBuffer.ofF32 y)

private def initW1 : RawBuffer :=
  RawBuffer.ofF32 ⟨#[0.1, 0.2, -0.1, 0.15,  0.05, -0.15, 0.2, -0.05]⟩

private def initW2 : RawBuffer :=
  RawBuffer.ofF32 ⟨#[0.3, -0.2, 0.1, 0.25]⟩

private def buildProgram (lr : Float) : IO Program := do
  let (w1Id, w2Id, xId, yId, lossUop, newW1Uop, newW2Uop) := runTensorM do
    let w1Buf ← Tensor.buffer [2, 4] .float32
    let w2Buf ← Tensor.buffer [4, 1] .float32
    let xBuf ← Tensor.buffer [4, 2] .float32
    let yBuf ← Tensor.buffer [4, 1] .float32

    let h ← matmul xBuf w1Buf
    let hRelu ← relu h
    let pred ← matmul hRelu w2Buf

    let diff ← sub pred yBuf
    let sq ← mul diff diff
    let loss ← mean sq

    let gradMap ← backward loss [w1Buf.uop, w2Buf.uop]
    let gradW1 := gradMap.getD w1Buf.uop.uid w1Buf.uop
    let gradW2 := gradMap.getD w2Buf.uop.uid w2Buf.uop

    let lrConst ← UOp.const .float32 lr.toFloat32
    let stepW1 ← UOp.sub w1Buf.uop (← UOp.mul gradW1 lrConst)
    let stepW2 ← UOp.sub w2Buf.uop (← UOp.mul gradW2 lrConst)

    pure (w1Buf.uop.uid, w2Buf.uop.uid, xBuf.uop.uid, yBuf.uop.uid, loss.uop, stepW1, stepW2)

  let roots : List UOp := [lossUop, newW1Uop, newW2Uop]
  let compiled ← Interpreter.compileManyCached roots
  pure { w1Id, w2Id, xId, yId, loss := lossUop, newW1 := newW1Uop, newW2 := newW2Uop, compiled }

private def step (p : Program) (w1 w2 : RawBuffer) : Float × RawBuffer × RawBuffer :=
  let (xData, yData) := trainingData
  let env : Env := (∅ : Env)
    |>.insert p.xId xData
    |>.insert p.yId yData
    |>.insert p.w1Id w1
    |>.insert p.w2Id w2

  let cache := Interpreter.evalCompiledRaw p.compiled env
  let lossBuf := cache.getD p.loss.uid (RawBuffer.zeros p.loss.dtype (listProd p.loss.shape))
  let lossVal := RawBuffer.decodeScalarF32 lossBuf
  let newW1 := cache.getD p.newW1.uid (RawBuffer.zeros p.newW1.dtype (listProd p.newW1.shape))
  let newW2 := cache.getD p.newW2.uid (RawBuffer.zeros p.newW2.dtype (listProd p.newW2.shape))
  (lossVal, newW1, newW2)

private def testCompiledTraining : IO Unit := do
  let lr : Float := 0.5
  let p ← buildProgram lr

  match p.compiled.implMap[p.newW1.uid]? with
  | some (.fusedSGD _) => pure ()
  | _ => throw (IO.userError "expected fusion selector to pick fusedSGD for w1 update")

  match p.compiled.implMap[p.newW2.uid]? with
  | some (.fusedSGD _) => pure ()
  | _ => throw (IO.userError "expected fusion selector to pick fusedSGD for w2 update")

  let mut w1 := initW1
  let mut w2 := initW2
  let (loss0, w1', w2') := step p w1 w2
  w1 := w1'
  w2 := w2'

  for _ in [:20] do
    let (_, w1', w2') := step p w1 w2
    w1 := w1'
    w2 := w2'

  let (lossN, _, _) := step p w1 w2
  if !(lossN < loss0) then
    throw (IO.userError s!"compiled training: expected loss to drop, got loss0={loss0} lossN={lossN}")

def runAll : IO Unit := do
  IO.println "=== CompiledTrainingSmoke Tests ==="
  testCompiledTraining
  IO.println "✓ compiled training (loss decreases)"
  IO.println "=== CompiledTrainingSmoke OK ==="

end TinyGrad4.Test.CompiledTrainingSmoke

