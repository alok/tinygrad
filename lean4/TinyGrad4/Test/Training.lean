import TinyGrad4

/-!
# Training Loop Example

Demonstrates actual training with weight updates using SGD.
Uses BUFFER ops so weights can be updated between iterations.
-/

namespace TinyGrad4.Test.Training

open TinyGrad4
open StaticTensor
open Interpreter

/-- Precompiled forward+backward program so we can "compile once, run many". -/
structure Program where
  w1Id : UOpId
  w2Id : UOpId
  xId : UOpId
  yId : UOpId
  predUop : UOp
  lossUop : UOp
  newW1Uop : UOp
  newW2Uop : UOp
  compiled : Interpreter.Compiled

/-- Build the training graph once (forward + backward), then lower it into `.KERNEL` nodes once. -/
def buildProgram (lr : Float) : IO Program := do
  let (w1Id, w2Id, xId, yId, predUop, lossUop, newW1Uop, newW2Uop) := runTensorM do
    let w1Buf ← Tensor.buffer [2, 4] .float32
    let w2Buf ← Tensor.buffer [4, 1] .float32
    let xBuf ← Tensor.buffer [4, 2] .float32
    let yBuf ← Tensor.buffer [4, 1] .float32

    let h ← matmul xBuf w1Buf
    let hNorm ← layerNorm h
    let hAct ← silu hNorm
    let pred ← matmul hAct w2Buf

    let loss ← binaryCrossEntropyWithLogits pred yBuf

    let gradMap ← backward loss [w1Buf.uop, w2Buf.uop]
    let gradW1 := gradMap.getD w1Buf.uop.uid w1Buf.uop
    let gradW2 := gradMap.getD w2Buf.uop.uid w2Buf.uop

    let lrConst ← UOp.const .float32 lr.toFloat32
    let newW1 ← UOp.sub w1Buf.uop (← UOp.mul gradW1 lrConst)
    let newW2 ← UOp.sub w2Buf.uop (← UOp.mul gradW2 lrConst)

    pure (w1Buf.uop.uid, w2Buf.uop.uid, xBuf.uop.uid, yBuf.uop.uid, pred.uop, loss.uop, newW1, newW2)

  let roots : List UOp := [predUop, lossUop, newW1Uop, newW2Uop]
  let compiled ← Interpreter.compileManyCached roots
  pure { w1Id, w2Id, xId, yId, predUop, lossUop, newW1Uop, newW2Uop, compiled }

/-- Simple XOR-like problem: learn to predict output from 2D input -/
structure TrainState where
  w1Data : RawBuffer  -- [2, 4] weights (float32 bytes)
  w2Data : RawBuffer  -- [4, 1] weights (float32 bytes)
  step : Nat := 0

/-- Initialize weights with small random-ish values -/
def initWeights : TrainState :=
  -- Pseudo-random initialization (deterministic for reproducibility)
  let w1 := #[0.1, 0.2, -0.1, 0.15, 0.05, -0.15, 0.2, -0.05]  -- [2, 4]
  let w2 := #[0.3, -0.2, 0.1, 0.25]  -- [4, 1]
  { w1Data := RawBuffer.ofFloats w1, w2Data := RawBuffer.ofFloats w2 }

/-- Create training data: XOR pattern
    Input: [0,0] -> 0, [0,1] -> 1, [1,0] -> 1, [1,1] -> 0 -/
def trainingData : Array Float × Array Float :=
  -- 4 samples, 2 features each: [4, 2]
  let x := #[0.0, 0.0,   -- sample 0
             0.0, 1.0,   -- sample 1
             1.0, 0.0,   -- sample 2
             1.0, 1.0]   -- sample 3
  -- Targets: XOR -> [4, 1]
  let y := #[0.0, 1.0, 1.0, 0.0]
  (x, y)

/-- Run one training step, return (loss, updated state) -/
def trainStep (p : Program) (state : TrainState) : IO (Float × TrainState) := do
  let (xData, yData) := trainingData

  -- Set up environment with current values
  let env : Env := (∅ : Env)
    |>.insert p.xId (RawBuffer.ofFloats xData)
    |>.insert p.yId (RawBuffer.ofFloats yData)
    |>.insert p.w1Id state.w1Data
    |>.insert p.w2Id state.w2Data

  let cache := Interpreter.evalCompiledRaw p.compiled env
  let lossBuf := cache.getD p.lossUop.uid (RawBuffer.zeros p.lossUop.dtype (listProd p.lossUop.shape))
  let lossVal := RawBuffer.decodeScalarF32 lossBuf
  let newW1 := cache.getD p.newW1Uop.uid (RawBuffer.zeros p.newW1Uop.dtype (listProd p.newW1Uop.shape))
  let newW2 := cache.getD p.newW2Uop.uid (RawBuffer.zeros p.newW2Uop.dtype (listProd p.newW2Uop.shape))

  let newState := { state with
    w1Data := newW1
    w2Data := newW2
    step := state.step + 1
  }

  pure (lossVal, newState)

/-- Train for multiple steps -/
def train (numSteps : Nat) (lr : Float := 0.1) : IO Unit := do
  IO.println "=== Training Loop Demo ==="
  IO.println s!"Learning rate: {lr}, Steps: {numSteps}"
  IO.println ""

  let p ← buildProgram lr
  let mut state := initWeights
  IO.println s!"Initial weights W1: {state.w1Data}"
  IO.println s!"Initial weights W2: {state.w2Data}"
  IO.println ""

  for i in [:numSteps] do
    let (loss, newState) ← trainStep p state
    state := newState
    if i % 10 == 0 || i == numSteps - 1 then
      IO.println s!"Step {i}: loss = {loss}"

  IO.println ""
  IO.println s!"Final weights W1: {state.w1Data}"
  IO.println s!"Final weights W2: {state.w2Data}"

  -- Final predictions
  let (xData, yData) := trainingData
  let envPred : Env := (∅ : Env)
    |>.insert p.xId (RawBuffer.ofFloats xData)
    |>.insert p.yId (RawBuffer.ofFloats yData)
    |>.insert p.w1Id state.w1Data
    |>.insert p.w2Id state.w2Data

  let cachePred := Interpreter.evalCompiledRaw p.compiled envPred
  let predBuf := cachePred.getD p.predUop.uid (RawBuffer.zeros p.predUop.dtype (listProd p.predUop.shape))
  let lossBuf := cachePred.getD p.lossUop.uid (RawBuffer.zeros p.lossUop.dtype (listProd p.lossUop.shape))
  let finalLoss := RawBuffer.decodeScalarF32 lossBuf

  IO.println ""
  IO.println "Final predictions vs targets:"
  IO.println s!"  Predictions: {predBuf}"
  IO.println s!"  Final loss:  {finalLoss}"
  IO.println s!"  Targets:     {yData}"

end TinyGrad4.Test.Training

def main : IO Unit := TinyGrad4.Test.Training.train 100 0.5
