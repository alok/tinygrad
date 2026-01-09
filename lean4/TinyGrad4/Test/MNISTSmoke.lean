import TinyGrad4

/-!
# MNIST Smoke Test

Runs a single forward + backward pass on MNIST-shaped synthetic data.
This is meant to be fast enough to run under `lake build`.
-/

namespace TinyGrad4.Test.MNISTSmoke

open TinyGrad4
open StaticTensor

def smoke : IO Unit := do
  IO.println "=== MNIST Smoke Test ==="

  let batchSize := 8

  let result := runTensorM do
    let xBuf ← Tensor.buffer [batchSize, 784] .float32
    let w1Buf ← Tensor.buffer [784, 16] .float32
    let w2Buf ← Tensor.buffer [16, 10] .float32
    let yBuf ← Tensor.buffer [batchSize, 10] .float32

    let h ← matmul xBuf w1Buf
    let hRelu ← relu h
    let logits ← matmul hRelu w2Buf

    let loss ← crossEntropyOneHot logits yBuf

    pure (loss, w1Buf.uop.uid, w2Buf.uop.uid, loss.uop, w1Buf.uop, w2Buf.uop)

  let (loss, w1Id, w2Id, lossUop, w1Uop, w2Uop) := result

  -- Backward and gradient shape checks (no interpreter eval here).
  let gradMap := runTensorM do
    backward loss [w1Uop, w2Uop]

  if lossUop.shape != [] then
    throw (IO.userError s!"loss shape {lossUop.shape} != []")

  match gradMap[w1Id]? with
  | some g =>
    if g.shape != [784, 16] then
      throw (IO.userError s!"gradW1 shape {g.shape} != [784, 16]")
  | none => throw (IO.userError "Missing gradW1")

  match gradMap[w2Id]? with
  | some g =>
    if g.shape != [16, 10] then
      throw (IO.userError s!"gradW2 shape {g.shape} != [16, 10]")
  | none => throw (IO.userError "Missing gradW2")

  IO.println "✓ Gradients computed with correct shapes"

end TinyGrad4.Test.MNISTSmoke

