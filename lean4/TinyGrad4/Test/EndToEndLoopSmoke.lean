import TinyGrad4

/-!
# EndToEndLoopSmoke

End-to-end training loop using:
- DataLoader (DataArrayN batches)
- Weight init (Init)
- Optimizer (SGD w/ momentum)
- Autodiff + compiled execution
-/

namespace TinyGrad4.Test.EndToEndLoopSmoke

open TinyGrad4
open StaticTensor
open TinyGrad4.Optim

abbrev BatchSize : Nat := 4
abbrev InDim : Nat := 2
abbrev Hidden : Nat := 4
abbrev OutDim : Nat := 1

structure Program where
  w1Id : UOpId
  w2Id : UOpId
  b1Id : UOpId
  b2Id : UOpId
  v1Id : UOpId
  v2Id : UOpId
  vb1Id : UOpId
  vb2Id : UOpId
  xId : UOpId
  yId : UOpId
  lossUop : UOp
  newW1Uop : UOp
  newW2Uop : UOp
  newB1Uop : UOp
  newB2Uop : UOp
  newV1Uop : UOp
  newV2Uop : UOp
  newVB1Uop : UOp
  newVB2Uop : UOp
  compiled : Interpreter.Compiled

structure TrainState where
  w1 : RawBuffer
  w2 : RawBuffer
  b1 : RawBuffer
  b2 : RawBuffer
  v1 : RawBuffer
  v2 : RawBuffer
  vb1 : RawBuffer
  vb2 : RawBuffer
  step : Nat := 0

structure ToyLoader (batch : Nat) where
  xs : Array (DataArrayN (batch :: [InDim]) .float32)
  ys : Array (DataArrayN (batch :: [OutDim]) .float32)

instance : DataLoader (ToyLoader batch) batch [InDim] [OutDim] .float32 .float32 where
  numBatches := fun l => min l.xs.size l.ys.size
  getBatch := fun l i => do
    if i < l.xs.size && i < l.ys.size then
      pure { x := l.xs[i]!, y := l.ys[i]! }
    else
      throw (IO.userError s!"ToyLoader: batch {i} out of range")

private def buildProgram (cfg : SGD) : IO Program := do
  let (w1Id, w2Id, b1Id, b2Id, v1Id, v2Id, vb1Id, vb2Id, xId, yId,
      lossU, newW1U, newW2U, newB1U, newB2U, newV1U, newV2U, newVB1U, newVB2U) := runTensorM do
    let w1Buf : Matrix InDim Hidden .float32 ← Tensor.buffer [InDim, Hidden] .float32
    let w2Buf : Matrix Hidden OutDim .float32 ← Tensor.buffer [Hidden, OutDim] .float32
    let b1Buf : Vector Hidden .float32 ← Tensor.buffer [Hidden] .float32
    let b2Buf : Vector OutDim .float32 ← Tensor.buffer [OutDim] .float32
    let xBuf : Matrix BatchSize InDim .float32 ← Tensor.buffer [BatchSize, InDim] .float32
    let yBuf : Matrix BatchSize OutDim .float32 ← Tensor.buffer [BatchSize, OutDim] .float32

    let s1 ← SGD.optimizer.initState (s := [InDim, Hidden]) (d := .float32) cfg
    let v1Buf : Matrix InDim Hidden .float32 ←
      match s1 with
      | [v] => pure v
      | _ => Tensor.buffer [InDim, Hidden] .float32
    let s2 ← SGD.optimizer.initState (s := [Hidden, OutDim]) (d := .float32) cfg
    let v2Buf : Matrix Hidden OutDim .float32 ←
      match s2 with
      | [v] => pure v
      | _ => Tensor.buffer [Hidden, OutDim] .float32
    let sb1 ← SGD.optimizer.initState (s := [Hidden]) (d := .float32) cfg
    let vb1Buf : Vector Hidden .float32 ←
      match sb1 with
      | [v] => pure v
      | _ => Tensor.buffer [Hidden] .float32
    let sb2 ← SGD.optimizer.initState (s := [OutDim]) (d := .float32) cfg
    let vb2Buf : Vector OutDim .float32 ←
      match sb2 with
      | [v] => pure v
      | _ => Tensor.buffer [OutDim] .float32

    let h ← linearBias xBuf w1Buf b1Buf
    let hRelu ← relu h
    let pred ← linearBias hRelu w2Buf b2Buf
    let diff ← sub pred yBuf
    let sq ← mul diff diff
    let loss ← mean sq

    let gradMap ← backward loss [w1Buf.uop, w2Buf.uop, b1Buf.uop, b2Buf.uop]
    let gradW1U := gradMap.getD w1Buf.uop.uid w1Buf.uop
    let gradW2U := gradMap.getD w2Buf.uop.uid w2Buf.uop
    let gradB1U := gradMap.getD b1Buf.uop.uid b1Buf.uop
    let gradB2U := gradMap.getD b2Buf.uop.uid b2Buf.uop
    let gradW1 : Matrix InDim Hidden .float32 := { uop := gradW1U, h_shape := sorry_proof }
    let gradW2 : Matrix Hidden OutDim .float32 := { uop := gradW2U, h_shape := sorry_proof }
    let gradB1 : Vector Hidden .float32 := { uop := gradB1U, h_shape := sorry_proof }
    let gradB2 : Vector OutDim .float32 := { uop := gradB2U, h_shape := sorry_proof }

    let stepRes1 ← SGD.optimizer.step w1Buf gradW1 [v1Buf] cfg
    let stepRes2 ← SGD.optimizer.step w2Buf gradW2 [v2Buf] cfg
    let stepResB1 ← SGD.optimizer.step b1Buf gradB1 [vb1Buf] cfg
    let stepResB2 ← SGD.optimizer.step b2Buf gradB2 [vb2Buf] cfg
    let newV1 ←
      match stepRes1.state with
      | [v] => pure v
      | _ => Tensor.buffer [InDim, Hidden] .float32
    let newV2 ←
      match stepRes2.state with
      | [v] => pure v
      | _ => Tensor.buffer [Hidden, OutDim] .float32
    let newVB1 ←
      match stepResB1.state with
      | [v] => pure v
      | _ => Tensor.buffer [Hidden] .float32
    let newVB2 ←
      match stepResB2.state with
      | [v] => pure v
      | _ => Tensor.buffer [OutDim] .float32

    pure (w1Buf.uop.uid, w2Buf.uop.uid, b1Buf.uop.uid, b2Buf.uop.uid, v1Buf.uop.uid, v2Buf.uop.uid,
      vb1Buf.uop.uid, vb2Buf.uop.uid, xBuf.uop.uid, yBuf.uop.uid,
      loss.uop, stepRes1.param.uop, stepRes2.param.uop, stepResB1.param.uop, stepResB2.param.uop,
      newV1.uop, newV2.uop, newVB1.uop, newVB2.uop)

  let roots : List UOp := [lossU, newW1U, newW2U, newB1U, newB2U, newV1U, newV2U, newVB1U, newVB2U]
  let compiled ← Interpreter.compileManyCached roots
  let program : Program :=
    { w1Id, w2Id, b1Id, b2Id, v1Id, v2Id, vb1Id, vb2Id, xId, yId, lossUop := lossU,
      newW1Uop := newW1U, newW2Uop := newW2U, newB1Uop := newB1U, newB2Uop := newB2U,
      newV1Uop := newV1U, newV2Uop := newV2U, newVB1Uop := newVB1U, newVB2Uop := newVB2U, compiled }
  pure program

private def initState : TrainState :=
  let w1 := Init.kaimingUniformF32 [InDim, Hidden] 123 |>.toRawBuffer
  let w2 := Init.xavierUniformF32 [Hidden, OutDim] 321 |>.toRawBuffer
  let b1 := RawBuffer.zeros .float32 (Shape.numel [Hidden])
  let b2 := RawBuffer.zeros .float32 (Shape.numel [OutDim])
  let v1 := RawBuffer.zeros .float32 (Shape.numel [InDim, Hidden])
  let v2 := RawBuffer.zeros .float32 (Shape.numel [Hidden, OutDim])
  let vb1 := RawBuffer.zeros .float32 (Shape.numel [Hidden])
  let vb2 := RawBuffer.zeros .float32 (Shape.numel [OutDim])
  { w1, w2, b1, b2, v1, v2, vb1, vb2 }

private def step (p : Program) (state : TrainState)
    (b : Batch BatchSize [InDim] [OutDim] .float32 .float32) : IO (Float × TrainState) := do
  let env : Env := (∅ : Env)
    |>.insert p.xId b.x.toRawBuffer
    |>.insert p.yId b.y.toRawBuffer
    |>.insert p.w1Id state.w1
    |>.insert p.w2Id state.w2
    |>.insert p.b1Id state.b1
    |>.insert p.b2Id state.b2
    |>.insert p.v1Id state.v1
    |>.insert p.v2Id state.v2
    |>.insert p.vb1Id state.vb1
    |>.insert p.vb2Id state.vb2

  let cache := Interpreter.evalCompiledRaw p.compiled env
  let lossBuf := cache.getD p.lossUop.uid (RawBuffer.zeros p.lossUop.dtype (listProd p.lossUop.shape))
  let lossVal := RawBuffer.decodeScalarF32 lossBuf
  let newW1 := cache.getD p.newW1Uop.uid (RawBuffer.zeros p.newW1Uop.dtype (listProd p.newW1Uop.shape))
  let newW2 := cache.getD p.newW2Uop.uid (RawBuffer.zeros p.newW2Uop.dtype (listProd p.newW2Uop.shape))
  let newB1 := cache.getD p.newB1Uop.uid (RawBuffer.zeros p.newB1Uop.dtype (listProd p.newB1Uop.shape))
  let newB2 := cache.getD p.newB2Uop.uid (RawBuffer.zeros p.newB2Uop.dtype (listProd p.newB2Uop.shape))
  let newV1 := cache.getD p.newV1Uop.uid (RawBuffer.zeros p.newV1Uop.dtype (listProd p.newV1Uop.shape))
  let newV2 := cache.getD p.newV2Uop.uid (RawBuffer.zeros p.newV2Uop.dtype (listProd p.newV2Uop.shape))
  let newVB1 := cache.getD p.newVB1Uop.uid (RawBuffer.zeros p.newVB1Uop.dtype (listProd p.newVB1Uop.shape))
  let newVB2 := cache.getD p.newVB2Uop.uid (RawBuffer.zeros p.newVB2Uop.dtype (listProd p.newVB2Uop.shape))

  let newState := {
    w1 := newW1, w2 := newW2, b1 := newB1, b2 := newB2,
    v1 := newV1, v2 := newV2, vb1 := newVB1, vb2 := newVB2,
    step := state.step + 1
  }
  pure (lossVal, newState)

def runAll : IO Unit := do
  IO.println "=== EndToEndLoopSmoke Tests ==="
  let cfg : SGD := { learningRate := 0.01, momentum := 0.5 }
  let p ← buildProgram cfg
  let loader : ToyLoader BatchSize := {
    xs := #[
      DataArrayN.ofArrayF32 [BatchSize, InDim] #[
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
      ],
      DataArrayN.ofArrayF32 [BatchSize, InDim] #[
        2.0, 0.0,
        0.0, 2.0,
        2.0, 2.0,
        1.0, 2.0
      ]
    ],
    ys := #[
      DataArrayN.ofArrayF32 [BatchSize, OutDim] #[
        0.0, 0.5, 0.5, 1.0
      ],
      DataArrayN.ofArrayF32 [BatchSize, OutDim] #[
        1.0, 1.0, 2.0, 1.5
      ]
    ]
  }

  let mut state := initState
  let mut firstLoss : Option Float := none
  let mut lastLoss : Float := 0.0
  for _ in [:10] do
    let numBatches :=
      DataLoader.numBatches (L := ToyLoader BatchSize) (batch := BatchSize) (xShape := [InDim]) (yShape := [OutDim])
        (xD := .float32) (yD := .float32) loader
    for i in [:numBatches] do
      let b ← DataLoader.getBatch (L := ToyLoader BatchSize) (batch := BatchSize) (xShape := [InDim]) (yShape := [OutDim])
        (xD := .float32) (yD := .float32) loader i
      let (loss, newState) ← step p state b
      if firstLoss.isNone then
        firstLoss := some loss
      lastLoss := loss
      state := newState

  let start := firstLoss.getD lastLoss
  if lastLoss >= start then
    throw (IO.userError s!"loss did not decrease: start={start} last={lastLoss}")
  IO.println "=== EndToEndLoopSmoke OK ==="

end TinyGrad4.Test.EndToEndLoopSmoke

