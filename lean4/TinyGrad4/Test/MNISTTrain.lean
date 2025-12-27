import TinyGrad4
import TinyGrad4.Data.MNIST
import TinyGrad4.Optim.SGD

/-!
# Real MNIST Training

Train a simple MLP on actual MNIST data.
Architecture: 784 -> 128 -> 10
-/

namespace TinyGrad4.Test.MNISTTrain

open TinyGrad4
open TinyGrad4.Data.MNIST
open TinyGrad4.Optim
open StaticTensor
open Interpreter

/-- Training state with weights as RawBuffer -/
structure Model where
  w1Data : RawBuffer  -- [784, 128]
  w2Data : RawBuffer  -- [128, 10]

/-- Initialize weights with small values -/
def initModel : IO Model := do
  -- Xavier-ish initialization: scale by sqrt(2/fan_in)
  let w1Size := 784 * 128
  let w2Size := 128 * 10

  -- Simple pseudo-random init using a linear congruential generator
  let mut w1 : ByteArray := ByteArray.emptyWithCapacity (w1Size * 4)
  let mut seed : UInt64 := 42
  for _ in [:w1Size] do
    seed := seed * 1103515245 + 12345
    let val := ((seed >>> 16).toNat % 1000).toFloat / 1000.0 - 0.5
    -- Pack as little-endian float32
    let v := (val * 0.1).toFloat32
    w1 := w1.push (v.toBits.toNat.toUInt8)
    w1 := w1.push ((v.toBits.toNat >>> 8).toUInt8)
    w1 := w1.push ((v.toBits.toNat >>> 16).toUInt8)
    w1 := w1.push ((v.toBits.toNat >>> 24).toUInt8)

  let mut w2 : ByteArray := ByteArray.emptyWithCapacity (w2Size * 4)
  for _ in [:w2Size] do
    seed := seed * 1103515245 + 12345
    let val := ((seed >>> 16).toNat % 1000).toFloat / 1000.0 - 0.5
    let v := (val * 0.1).toFloat32
    w2 := w2.push (v.toBits.toNat.toUInt8)
    w2 := w2.push ((v.toBits.toNat >>> 8).toUInt8)
    w2 := w2.push ((v.toBits.toNat >>> 16).toUInt8)
    w2 := w2.push ((v.toBits.toNat >>> 24).toUInt8)

  pure { w1Data := { dtype := .float32, data := w1 },
         w2Data := { dtype := .float32, data := w2 } }

/-- Run forward pass and compute loss -/
def forwardLoss (model : Model) (xData : Array Float) (yOneHot : Array Float)
    (batchSize : Nat) : IO (Float × RawBuffer × RawBuffer) := do
  -- Build graph
  let result := runTensorM do
    let xBuf ← Tensor.buffer [batchSize, 784] .float32
    let w1Buf ← Tensor.buffer [784, 128] .float32
    let w2Buf ← Tensor.buffer [128, 10] .float32
    let yBuf ← Tensor.buffer [batchSize, 10] .float32

    -- Forward: x @ w1 -> layerNorm -> relu -> @ w2
    let h ← matmul xBuf w1Buf        -- [batch, 128]
    let hNorm ← layerNorm h
    let hRelu ← relu hNorm
    let logits ← matmul hRelu w2Buf  -- [batch, 10]

    let loss ← crossEntropyOneHot logits yBuf
    let gradMap ← backward loss [w1Buf.uop, w2Buf.uop]

    pure (loss, xBuf.uop.uid, w1Buf.uop.uid, w2Buf.uop.uid, yBuf.uop.uid,
      loss.uop, w1Buf.uop, w2Buf.uop, gradMap)

  let (_loss, xId, w1Id, w2Id, yId, lossUop, _w1Uop, _w2Uop, gradResult) := result

  -- Set up env with RawBuffer
  let xRaw := RawBuffer.ofFloats xData
  let yRaw := RawBuffer.ofFloats yOneHot

  let env : Env := (∅ : Env)
    |>.insert xId xRaw
    |>.insert w1Id model.w1Data
    |>.insert w2Id model.w2Data
    |>.insert yId yRaw

  let gradW1Uop? := gradResult[w1Id]?
  let gradW2Uop? := gradResult[w2Id]?

  let roots := [lossUop] ++ gradW1Uop?.toList ++ gradW2Uop?.toList
  let vals ← evalManyCached roots env

  let lossRaw := vals.getD lossUop.uid (RawBuffer.zeros .float32 1)
  let lossVal := RawBuffer.decodeScalarF32 lossRaw

  let gradW1 := match gradW1Uop? with
    | some g => vals.getD g.uid model.w1Data
    | none => model.w1Data

  let gradW2 := match gradW2Uop? with
    | some g => vals.getD g.uid model.w2Data
    | none => model.w2Data

  pure (lossVal, gradW1, gradW2)

/-- Single training step using native SGD update -/
def trainStep (model : Model) (xData : Array Float) (yOneHot : Array Float)
    (batchSize : Nat) (lr : Float) : IO (Float × Model) := do
  let (loss, gradW1, gradW2) ← forwardLoss model xData yOneHot batchSize

  -- SGD update using native float32 kernel
  let newW1 := SGD.updateRawF32 model.w1Data gradW1 lr
  let newW2 := SGD.updateRawF32 model.w2Data gradW2 lr

  pure (loss, { w1Data := newW1, w2Data := newW2 })

/-- Compute accuracy on test set -/
def computeAccuracy (model : Model) (images : ImageData) (labels : LabelData)
    (numSamples : Nat) : IO Float := do
  let mut correct := 0

  -- Process in batches
  let batchSize := min 100 numSamples
  for batchStart in List.range (numSamples / batchSize) do
    let startIdx := batchStart * batchSize
    let xData := getBatch images startIdx batchSize
    let yLabels := getBatchLabels labels startIdx batchSize

    -- Forward pass
    let result := runTensorM do
      let xBuf ← Tensor.buffer [batchSize, 784] .float32
      let w1Buf ← Tensor.buffer [784, 128] .float32
      let w2Buf ← Tensor.buffer [128, 10] .float32

      let h ← matmul xBuf w1Buf
      let hNorm ← layerNorm h
      let hRelu ← relu hNorm
      let logits ← matmul hRelu w2Buf

      pure (logits, xBuf.uop.uid, w1Buf.uop.uid, w2Buf.uop.uid, logits.uop)

    let (_, xId, w1Id, w2Id, logitsUop) := result

    let xRaw := RawBuffer.ofFloats xData

    let env : Env := (∅ : Env)
      |>.insert xId xRaw
      |>.insert w1Id model.w1Data
      |>.insert w2Id model.w2Data

    let logitsRaw ← evalCached logitsUop env

    -- Compute argmax for each sample
    for i in [:batchSize] do
      let mut maxVal := logitsRaw.getF32 (i * 10)
      let mut maxIdx := 0
      for j in [1:10] do
        let val := logitsRaw.getF32 (i * 10 + j)
        if val > maxVal then
          maxVal := val
          maxIdx := j
      let trueLabel := yLabels[i]!.toUInt64.toNat
      if maxIdx == trueLabel then
        correct := correct + 1

  pure (correct.toFloat / numSamples.toFloat * 100.0)

/-- Main training loop -/
def train (numEpochs : Nat := 3) (batchSize : Nat := 32) (lr : Float := 0.01) : IO Unit := do
  IO.println "=== MNIST Training ==="
  IO.println s!"Epochs: {numEpochs}, Batch size: {batchSize}, Learning rate: {lr}"
  IO.println ""

  -- Load data
  IO.println "Loading MNIST data..."
  let (trainImages, trainLabels) ← loadTrain "data"
  let (testImages, testLabels) ← loadTest "data"
  IO.println s!"Train: {trainImages.numImages} images, Test: {testImages.numImages} images"
  IO.println ""

  -- Initialize model
  IO.println "Initializing model (784 -> 128 -> 10)..."
  let mut model ← initModel

  let numBatches := trainImages.numImages / batchSize

  for epoch in [:numEpochs] do
    IO.println s!"Epoch {epoch + 1}/{numEpochs}"
    let mut totalLoss := 0.0

    for batchIdx in [:min numBatches 100] do  -- Limit batches for speed
      let startIdx := batchIdx * batchSize
      let xData := getBatch trainImages startIdx batchSize
      let yLabels := getBatchLabels trainLabels startIdx batchSize
      let yOneHot := toOneHot yLabels

      let (loss, newModel) ← trainStep model xData yOneHot batchSize lr
      model := newModel
      totalLoss := totalLoss + loss

      if batchIdx % 20 == 0 then
        IO.print s!"\r  Batch {batchIdx}/{min numBatches 100}, Loss: {loss}"
        (← IO.getStdout).flush

    IO.println ""
    let avgLoss := totalLoss / (min numBatches 100).toFloat
    IO.println s!"  Average loss: {avgLoss}"

    -- Compute test accuracy (on subset for speed)
    let testAcc ← computeAccuracy model testImages testLabels 1000
    IO.println s!"  Test accuracy: {testAcc}%"
    IO.println ""

  IO.println "Training complete!"

end TinyGrad4.Test.MNISTTrain

def main : IO Unit := TinyGrad4.Test.MNISTTrain.train 3 32 0.01
