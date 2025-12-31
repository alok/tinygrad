import TinyGrad4
import TinyGrad4.Data.MNIST
import TinyGrad4.Backend.Metal

/-!
# MNIST GPU vs CPU Benchmark

Compares training performance between:
- CPU path: `evalCompiledRaw` (pure, no GPU)
- GPU path: `evalCompiledRawIO` (GPU matmul + ewise)
-/

namespace TinyGrad4.Test.MNISTGPUBench

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

/-- Run training on CPU path -/
def runCPU (p : Program) (batches : Array (RawBuffer × RawBuffer)) (numBatches : Nat) : IO (Float × Nat) := do
  let (w1Init, w2Init) := initWeights (p.newW1.shape[1]!)
  let w1Ref ← IO.mkRef w1Init
  let w2Ref ← IO.mkRef w2Init

  let startTime ← IO.monoNanosNow
  let mut totalLoss : Float := 0.0

  for bi in [:numBatches] do
    let (xBuf, yBuf) := batches[bi]!
    let env : Env := (∅ : Env)
      |>.insert p.xId xBuf
      |>.insert p.yId yBuf
      |>.insert p.w1Id (← w1Ref.get)
      |>.insert p.w2Id (← w2Ref.get)

    -- CPU path
    let cache := Interpreter.evalCompiledRaw p.compiled env

    let w1' := cache.getD p.newW1.uid (RawBuffer.zeros p.newW1.dtype (listProd p.newW1.shape))
    let w2' := cache.getD p.newW2.uid (RawBuffer.zeros p.newW2.dtype (listProd p.newW2.shape))
    w1Ref.set w1'
    w2Ref.set w2'
    let lossBuf := cache.getD p.loss.uid (RawBuffer.zeros p.loss.dtype (listProd p.loss.shape))
    totalLoss := totalLoss + RawBuffer.decodeScalarF32 lossBuf

  let endTime ← IO.monoNanosNow
  let elapsedNs := endTime - startTime
  let avgLoss := totalLoss / (Float.ofNat numBatches)
  pure (avgLoss, elapsedNs)

/-- Run training on GPU path -/
def runGPU (p : Program) (batches : Array (RawBuffer × RawBuffer)) (numBatches : Nat) : IO (Float × Nat) := do
  let (w1Init, w2Init) := initWeights (p.newW1.shape[1]!)
  let w1Ref ← IO.mkRef w1Init
  let w2Ref ← IO.mkRef w2Init

  let startTime ← IO.monoNanosNow
  let mut totalLoss : Float := 0.0

  for bi in [:numBatches] do
    let (xBuf, yBuf) := batches[bi]!
    let env : Env := (∅ : Env)
      |>.insert p.xId xBuf
      |>.insert p.yId yBuf
      |>.insert p.w1Id (← w1Ref.get)
      |>.insert p.w2Id (← w2Ref.get)

    -- GPU path (matmul + ewise on GPU)
    let cache ← Interpreter.evalCompiledRawIO p.compiled env

    let w1' := cache.getD p.newW1.uid (RawBuffer.zeros p.newW1.dtype (listProd p.newW1.shape))
    let w2' := cache.getD p.newW2.uid (RawBuffer.zeros p.newW2.dtype (listProd p.newW2.shape))
    w1Ref.set w1'
    w2Ref.set w2'
    let lossBuf := cache.getD p.loss.uid (RawBuffer.zeros p.loss.dtype (listProd p.loss.shape))
    totalLoss := totalLoss + RawBuffer.decodeScalarF32 lossBuf

  let endTime ← IO.monoNanosNow
  let elapsedNs := endTime - startTime
  let avgLoss := totalLoss / (Float.ofNat numBatches)
  pure (avgLoss, elapsedNs)

def run (batchSize : Nat := 64) (hidden : Nat := 256) (numBatches : Nat := 20) (lr : Float32 := 0.01) : IO Unit := do
  IO.println "=== MNIST GPU vs CPU Benchmark ==="

  -- Check Metal availability
  let metalAvail ← Metal.isAvailable
  IO.println s!"Metal available: {metalAvail}"

  let trainImagesPath := "data/train-images-idx3-ubyte"
  let trainLabelsPath := "data/train-labels-idx1-ubyte"
  if !(← fileExists trainImagesPath) || !(← fileExists trainLabelsPath) then
    throw (IO.userError s!"missing MNIST files in `data/`")

  let maxImages := batchSize * numBatches
  let (images, labels) ← loadTrain "data" (maxImages? := some maxImages)

  -- Prepare batches
  let mut batches : Array (RawBuffer × RawBuffer) := #[]
  for bi in [:numBatches] do
    let startIdx := bi * batchSize
    let xData := getBatch images startIdx batchSize
    let yLabels := getBatchLabels labels startIdx batchSize
    let yOneHot := toOneHot yLabels
    let xBuf := RawBuffer.ofF32 ⟨xData⟩
    let yBuf := RawBuffer.ofF32 ⟨yOneHot⟩
    batches := batches.push (xBuf, yBuf)

  IO.println s!"batch_size={batchSize} hidden={hidden} batches={numBatches}"
  IO.println s!"matmul sizes: [{batchSize}, 784] @ [784, {hidden}], [{batchSize}, {hidden}] @ [{hidden}, 10]"

  let p ← buildProgram batchSize hidden lr

  -- Warmup run (compile shaders, etc)
  IO.println "\nWarmup run..."
  let _ ← runGPU p batches 1

  -- CPU benchmark
  IO.println "\nRunning CPU benchmark..."
  let (cpuLoss, cpuNs) ← runCPU p batches numBatches
  let cpuMs := Float.ofNat cpuNs / 1000000.0
  let cpuPerBatch := cpuMs / Float.ofNat numBatches

  -- GPU benchmark
  IO.println "Running GPU benchmark..."
  let (gpuLoss, gpuNs) ← runGPU p batches numBatches
  let gpuMs := Float.ofNat gpuNs / 1000000.0
  let gpuPerBatch := gpuMs / Float.ofNat numBatches

  -- Results
  IO.println "\n=== Results ==="
  IO.println s!"CPU: {cpuMs} ms total, {cpuPerBatch} ms/batch, loss={cpuLoss}"
  IO.println s!"GPU: {gpuMs} ms total, {gpuPerBatch} ms/batch, loss={gpuLoss}"

  let speedup := cpuMs / gpuMs
  if speedup > 1.0 then
    IO.println s!"GPU speedup: {speedup}x faster"
  else
    IO.println s!"CPU faster by: {1.0 / speedup}x (GPU overhead dominates at this size)"

  -- Verify correctness (losses should be similar)
  let lossDiff := Float.abs (cpuLoss - gpuLoss)
  if lossDiff > 0.1 then
    IO.println s!"WARNING: Loss mismatch! CPU={cpuLoss} GPU={gpuLoss} diff={lossDiff}"
  else
    IO.println s!"✓ Losses match (diff={lossDiff})"

end TinyGrad4.Test.MNISTGPUBench

def main : IO Unit := TinyGrad4.Test.MNISTGPUBench.run (batchSize := 256) (hidden := 256) (numBatches := 10)
