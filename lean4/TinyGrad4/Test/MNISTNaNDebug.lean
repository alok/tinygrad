import TinyGrad4
import TinyGrad4.Data.MNIST
import TinyGrad4.Backend.Metal

/-!
Debug NaN in MNIST GPU training at batch=256.
Traces each step to find where NaN appears.
-/

set_option linter.useRawBuffer false

namespace TinyGrad4.Test.MNISTNaNDebug

open TinyGrad4
open TinyGrad4.Data.MNIST
open StaticTensor
open Interpreter
open Backend
open Std

/-- Check if buffer has NaN -/
def checkNaN (name : String) (buf : RawBuffer) : IO Bool := do
  let decoded := buf.decode.data
  let mut nanCount := 0
  for v in decoded do
    if v != v then nanCount := nanCount + 1
  if nanCount > 0 then
    IO.println s!"  ❌ {name}: {nanCount}/{decoded.size} NaN values"
    return true
  else
    IO.println s!"  ✓ {name}: OK (size={decoded.size})"
    return false

private def fileExists (path : String) : IO Bool := do
  try
    IO.FS.withFile path .read (fun _ => pure ())
    pure true
  catch _ =>
    pure false

def run (batchSize : Nat := 256) (hidden : Nat := 256) : IO Unit := do
  IO.println s!"=== MNIST NaN Debug: batch={batchSize} hidden={hidden} ==="

  let trainImagesPath := "data/train-images-idx3-ubyte"
  let trainLabelsPath := "data/train-labels-idx1-ubyte"
  if !(← fileExists trainImagesPath) || !(← fileExists trainLabelsPath) then
    throw (IO.userError s!"missing MNIST files in `data/`")

  -- Load one batch
  let (images, labels) ← loadTrain "data" (maxImages? := some batchSize)
  let xData := getBatch images 0 batchSize
  let yLabels := getBatchLabels labels 0 batchSize
  let yOneHot := toOneHot yLabels
  let xBuf := RawBuffer.ofF32 ⟨xData⟩
  let yBuf := RawBuffer.ofF32 ⟨yOneHot⟩

  IO.println s!"\n--- Input Data ---"
  let _ ← checkNaN "X (input images)" xBuf
  let _ ← checkNaN "Y (one-hot labels)" yBuf

  -- Build forward pass only (no backward)
  IO.println s!"\n--- Building Forward Pass ---"
  let (w1Id, w2Id, xId, yId, hUop, hReluUop, logitsUop, lossUop) := runTensorM do
    let xBuf ← Tensor.buffer [batchSize, 784] .float32
    let yBuf ← Tensor.buffer [batchSize, 10] .float32
    let w1Buf ← Tensor.buffer [784, hidden] .float32
    let w2Buf ← Tensor.buffer [hidden, 10] .float32

    let h ← matmul xBuf w1Buf
    let hRelu ← relu h
    let logits ← matmul hRelu w2Buf
    let loss ← crossEntropyOneHot logits yBuf

    pure (w1Buf.uop.uid, w2Buf.uop.uid, xBuf.uop.uid, yBuf.uop.uid, h.uop, hRelu.uop, logits.uop, loss.uop)

  -- Initialize weights
  let w1Init : RawBuffer := { dtype := .float32, data := Native.fullF32Bits (784 * hidden) ((0.01 : Float32).toBits) }
  let w2Init : RawBuffer := { dtype := .float32, data := Native.fullF32Bits (hidden * 10) ((0.01 : Float32).toBits) }

  IO.println s!"\n--- Weights ---"
  let _ ← checkNaN "W1 init" w1Init
  let _ ← checkNaN "W2 init" w2Init

  -- Compile each step separately
  IO.println s!"\n--- Step 1: h = X @ W1 (matmul) ---"
  let hCompiled ← Interpreter.compileManyCached [hUop]
  let hEnv : Env := (∅ : Env) |>.insert xId xBuf |>.insert w1Id w1Init
  let hCacheCPU := Interpreter.evalCompiledRaw hCompiled hEnv
  let hBufCPU := hCacheCPU.getD hUop.uid (RawBuffer.zeros .float32 0)
  let _ ← checkNaN "h (CPU)" hBufCPU

  let hCacheGPU ← Interpreter.evalCompiledRawIO hCompiled hEnv
  let hBufGPU := hCacheGPU.getD hUop.uid (RawBuffer.zeros .float32 0)
  let hadNaN ← checkNaN "h (GPU)" hBufGPU

  if hadNaN then
    IO.println "  Stopping at first NaN"
    return

  IO.println s!"\n--- Step 2: hRelu = relu(h) ---"
  let hReluCompiled ← Interpreter.compileManyCached [hReluUop]
  let hReluEnv : Env := (∅ : Env) |>.insert xId xBuf |>.insert w1Id w1Init
  let hReluCacheCPU := Interpreter.evalCompiledRaw hReluCompiled hReluEnv
  let hReluBufCPU := hReluCacheCPU.getD hReluUop.uid (RawBuffer.zeros .float32 0)
  let _ ← checkNaN "hRelu (CPU)" hReluBufCPU

  let hReluCacheGPU ← Interpreter.evalCompiledRawIO hReluCompiled hReluEnv
  let hReluBufGPU := hReluCacheGPU.getD hReluUop.uid (RawBuffer.zeros .float32 0)
  let hadNaN ← checkNaN "hRelu (GPU)" hReluBufGPU

  if hadNaN then
    IO.println "  Stopping at first NaN"
    return

  IO.println s!"\n--- Step 3: logits = hRelu @ W2 ---"
  let logitsCompiled ← Interpreter.compileManyCached [logitsUop]
  let logitsEnv : Env := (∅ : Env) |>.insert xId xBuf |>.insert yId yBuf |>.insert w1Id w1Init |>.insert w2Id w2Init
  let logitsCacheCPU := Interpreter.evalCompiledRaw logitsCompiled logitsEnv
  let logitsBufCPU := logitsCacheCPU.getD logitsUop.uid (RawBuffer.zeros .float32 0)
  let _ ← checkNaN "logits (CPU)" logitsBufCPU

  let logitsCacheGPU ← Interpreter.evalCompiledRawIO logitsCompiled logitsEnv
  let logitsBufGPU := logitsCacheGPU.getD logitsUop.uid (RawBuffer.zeros .float32 0)
  let hadNaN ← checkNaN "logits (GPU)" logitsBufGPU

  if hadNaN then
    IO.println "  Stopping at first NaN"
    return

  IO.println s!"\n--- Step 4: loss = crossEntropy(logits, y) ---"
  let lossCompiled ← Interpreter.compileManyCached [lossUop]
  let lossEnv : Env := (∅ : Env) |>.insert xId xBuf |>.insert yId yBuf |>.insert w1Id w1Init |>.insert w2Id w2Init
  let lossCacheCPU := Interpreter.evalCompiledRaw lossCompiled lossEnv
  let lossBufCPU := lossCacheCPU.getD lossUop.uid (RawBuffer.zeros .float32 0)
  let cpuNaN ← checkNaN "loss (CPU)" lossBufCPU
  if !cpuNaN then
    IO.println s!"  CPU loss value: {RawBuffer.decodeScalarF32 lossBufCPU}"

  let lossCacheGPU ← Interpreter.evalCompiledRawIO lossCompiled lossEnv
  let lossBufGPU := lossCacheGPU.getD lossUop.uid (RawBuffer.zeros .float32 0)
  let gpuNaN ← checkNaN "loss (GPU)" lossBufGPU
  if !gpuNaN then
    IO.println s!"  GPU loss value: {RawBuffer.decodeScalarF32 lossBufGPU}"

  -- Now test with backward pass
  IO.println s!"\n--- Step 5: Backward Pass (gradW1, gradW2) ---"
  let (w1Id2, w2Id2, xId2, yId2, newW1Uop, newW2Uop) := runTensorM do
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

    let lrConst ← UOp.const .float32 (0.01 : Float32)
    let stepW1 ← UOp.sub w1Buf.uop (← UOp.mul gradW1 lrConst)
    let stepW2 ← UOp.sub w2Buf.uop (← UOp.mul gradW2 lrConst)

    pure (w1Buf.uop.uid, w2Buf.uop.uid, xBuf.uop.uid, yBuf.uop.uid, stepW1, stepW2)

  -- Compile full training step
  let fullCompiled ← Interpreter.compileManyCached [newW1Uop, newW2Uop]
  let fullEnv : Env := (∅ : Env) |>.insert xId2 xBuf |>.insert yId2 yBuf |>.insert w1Id2 w1Init |>.insert w2Id2 w2Init

  IO.println "  Compiling full training step..."

  -- CPU path
  let fullCacheCPU := Interpreter.evalCompiledRaw fullCompiled fullEnv
  let newW1CPU := fullCacheCPU.getD newW1Uop.uid (RawBuffer.zeros .float32 0)
  let newW2CPU := fullCacheCPU.getD newW2Uop.uid (RawBuffer.zeros .float32 0)
  let _ ← checkNaN "newW1 (CPU)" newW1CPU
  let _ ← checkNaN "newW2 (CPU)" newW2CPU

  -- GPU path
  IO.println "  Running GPU path..."
  let fullCacheGPU ← Interpreter.evalCompiledRawIO fullCompiled fullEnv
  let newW1GPU := fullCacheGPU.getD newW1Uop.uid (RawBuffer.zeros .float32 0)
  let newW2GPU := fullCacheGPU.getD newW2Uop.uid (RawBuffer.zeros .float32 0)
  let hadNaN1 ← checkNaN "newW1 (GPU)" newW1GPU
  let hadNaN2 ← checkNaN "newW2 (GPU)" newW2GPU

  if hadNaN1 || hadNaN2 then
    -- Show first few values
    let gpuData := newW1GPU.decode.data
    let cpuData := newW1CPU.decode.data
    let gpu10 := gpuData[:Nat.min 10 gpuData.size]
    let cpu10 := cpuData[:Nat.min 10 cpuData.size]
    IO.println s!"  newW1 GPU first 10: {gpu10}"
    IO.println s!"  newW1 CPU first 10: {cpu10}"

  IO.println "\n=== Done ==="

end TinyGrad4.Test.MNISTNaNDebug

def main : IO Unit := TinyGrad4.Test.MNISTNaNDebug.run
