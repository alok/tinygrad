import TinyGrad4
import TinyGrad4.Data.MNIST

/-!
# Compiled MNIST Training Loop

Runs a small MNIST MLP training loop using `Interpreter.compileManyCached` once and
reuses the compiled program for all batches (RawBuffer hot path).
-/

namespace TinyGrad4.Test.MNISTCompiledTrain

open TinyGrad4
open TinyGrad4.Data.MNIST
open StaticTensor
open Interpreter
open Backend
open Std

private def emptyBuf : RawBuffer := RawBuffer.zeros .float32 0

instance : Inhabited (RawBuffer × RawBuffer) where
  default := (emptyBuf, emptyBuf)

private structure Program where
  w1Id : UOpId
  w2Id : UOpId
  xId : UOpId
  yId : UOpId
  loss : UOp
  newW1 : UOp
  newW2 : UOp
  compiled : Interpreter.Compiled

private def buildProgram (batchSize hidden : Nat) (lr : Float32) : IO Program := do
  let (w1Id, w2Id, xId, yId, lossUop, newW1Uop, newW2Uop) := runTensorM do
    let xBuf ← Tensor.buffer [batchSize, 784] .float32
    let yBuf ← Tensor.buffer [batchSize, 10] .float32
    let w1Buf ← Tensor.buffer [784, hidden] .float32
    let w2Buf ← Tensor.buffer [hidden, 10] .float32

    let h ← matmul xBuf w1Buf
    let hRelu ← relu h
    let logits ← matmul hRelu w2Buf

    let loss ← crossEntropyOneHot logits yBuf

    let gradMap ← backward loss [w1Buf.uop, w2Buf.uop]
    let gradW1 := gradMap.getD w1Buf.uop.uid w1Buf.uop
    let gradW2 := gradMap.getD w2Buf.uop.uid w2Buf.uop

    let lrConst ← UOp.const .float32 lr
    let stepW1 ← UOp.sub w1Buf.uop (← UOp.mul gradW1 lrConst)
    let stepW2 ← UOp.sub w2Buf.uop (← UOp.mul gradW2 lrConst)

    pure (w1Buf.uop.uid, w2Buf.uop.uid, xBuf.uop.uid, yBuf.uop.uid, loss.uop, stepW1, stepW2)

  let roots : List UOp := [lossUop, newW1Uop, newW2Uop]
  let compiled ← Interpreter.compileManyCached roots
  pure { w1Id, w2Id, xId, yId, loss := lossUop, newW1 := newW1Uop, newW2 := newW2Uop, compiled }

private def initWeights (hidden : Nat) : RawBuffer × RawBuffer :=
  let w1 := { dtype := .float32, data := Native.fullF32Bits (784 * hidden) ((0.01 : Float32).toBits) }
  let w2 := { dtype := .float32, data := Native.fullF32Bits (hidden * 10) ((0.01 : Float32).toBits) }
  (w1, w2)

private def fileExists (path : String) : IO Bool := do
  try
    IO.FS.withFile path .read (fun _ => pure ())
    pure true
  catch _ =>
    pure false

private def tagOf (c : Interpreter.Compiled) (u : UOp) : String :=
  match c.implMap[u.uid]? with
  | some impl => impl.tag
  | none => "node"

def run (epochs : Nat := 1) (batchSize : Nat := 32) (hidden : Nat := 128) (numBatches : Nat := 50) (lr : Float32 := 0.01) : IO Unit := do
  if batchSize == 0 || hidden == 0 || numBatches == 0 || epochs == 0 then
    throw (IO.userError "epochs/batchSize/hidden/numBatches must be > 0")

  let trainImagesPath := "data/train-images-idx3-ubyte"
  let trainLabelsPath := "data/train-labels-idx1-ubyte"
  if !(← fileExists trainImagesPath) || !(← fileExists trainLabelsPath) then
    throw (IO.userError s!"missing MNIST files in `data/`.\nexpected:\n  {trainImagesPath}\n  {trainLabelsPath}")

  let maxImages := batchSize * numBatches
  let (images, labels) ← loadTrain "data" (maxImages? := some maxImages)

  let mut batches : Array (RawBuffer × RawBuffer) := #[]
  for bi in [:numBatches] do
    let startIdx := bi * batchSize
    let xData := getBatch images startIdx batchSize
    let yLabels := getBatchLabels labels startIdx batchSize
    let yOneHot := toOneHot yLabels
    let xBuf := RawBuffer.ofF32 ⟨xData⟩
    let yBuf := RawBuffer.ofF32 ⟨yOneHot⟩
    batches := batches.push (xBuf, yBuf)

  let p ← buildProgram batchSize hidden lr
  match p.compiled.implMap[p.newW1.uid]? with
  | some (.fusedSGD _) => pure ()
  | _ => throw (IO.userError "expected fusedSGD for w1 update")
  match p.compiled.implMap[p.newW2.uid]? with
  | some (.fusedSGD _) => pure ()
  | _ => throw (IO.userError "expected fusedSGD for w2 update")

  IO.println s!"=== MNIST Compiled Train epochs={epochs} batch={batchSize} hidden={hidden} batches={numBatches} ==="
  IO.println s!"root tags: loss={tagOf p.compiled p.loss}, newW1={tagOf p.compiled p.newW1}, newW2={tagOf p.compiled p.newW2}"

  let (w1Init, w2Init) := initWeights hidden
  let w1Ref ← IO.mkRef w1Init
  let w2Ref ← IO.mkRef w2Init

  for epoch in [:epochs] do
    let mut totalLoss : Float := 0.0
    for bi in [:numBatches] do
      let (xBuf, yBuf) := batches[bi]!
      let env : Env := (∅ : Env)
        |>.insert p.xId xBuf
        |>.insert p.yId yBuf
        |>.insert p.w1Id (← w1Ref.get)
        |>.insert p.w2Id (← w2Ref.get)
      let cache := Interpreter.evalCompiledRaw p.compiled env
      let w1' := cache.getD p.newW1.uid (RawBuffer.zeros p.newW1.dtype (listProd p.newW1.shape))
      let w2' := cache.getD p.newW2.uid (RawBuffer.zeros p.newW2.dtype (listProd p.newW2.shape))
      w1Ref.set w1'
      w2Ref.set w2'
      let lossBuf := cache.getD p.loss.uid (RawBuffer.zeros p.loss.dtype (listProd p.loss.shape))
      let lossVal := RawBuffer.decodeScalarF32 lossBuf
      totalLoss := totalLoss + lossVal
      if bi % 10 == 0 then
        IO.print s!"\r  epoch {epoch + 1}/{epochs} batch {bi + 1}/{numBatches} loss={lossVal}"
        (← IO.getStdout).flush
    IO.println ""
    let avgLoss := totalLoss / (Float.ofNat numBatches)
    IO.println s!"  avg loss: {avgLoss}"

end TinyGrad4.Test.MNISTCompiledTrain

