import TinyGrad4
import TinyGrad4.Data.MNIST
import TinyGrad4.Backend.Metal

/-!
Debug NaN in MNIST backward pass by running CPU vs GPU on full graph
and comparing intermediate results.
-/

set_option linter.useRawBuffer false

namespace TinyGrad4.Test.BackwardStepDebug

open TinyGrad4
open TinyGrad4.Data.MNIST
open StaticTensor
open Interpreter
open Backend
open Std

/-- Check if buffer has NaN and report -/
def checkNaN (name : String) (buf : RawBuffer) : IO Bool := do
  let decoded := buf.decode.data
  let mut nanCount := 0
  let mut minVal : Float := 1e30
  let mut maxVal : Float := -1e30
  for v in decoded do
    if v != v then
      nanCount := nanCount + 1
    else
      if v < minVal then minVal := v
      if v > maxVal then maxVal := v
  if nanCount > 0 then
    IO.println s!"  ❌ {name}: {nanCount}/{decoded.size} NaN, range=[{minVal}, {maxVal}]"
    return true
  else
    IO.println s!"  ✓ {name}: range=[{minVal}, {maxVal}] (n={decoded.size})"
    return false

/-- Compare CPU and GPU results for a UOp -/
def compareCPUGPU (name : String) (cpuCache gpuCache : HashMap UOpId RawBuffer) (uid : UOpId) : IO Bool := do
  let cpuBuf := cpuCache.getD uid (RawBuffer.zeros .float32 0)
  let gpuBuf := gpuCache.getD uid (RawBuffer.zeros .float32 0)

  if cpuBuf.data.size == 0 then
    IO.println s!"  {name}: not in cache"
    return false

  let cpuDecoded := cpuBuf.decode.data
  let gpuDecoded := gpuBuf.decode.data

  let mut cpuNaN := 0
  let mut gpuNaN := 0
  let mut maxDiff : Float := 0.0

  for i in [:Nat.min cpuDecoded.size gpuDecoded.size] do
    let cv := cpuDecoded[i]!
    let gv := gpuDecoded[i]!
    if cv != cv then cpuNaN := cpuNaN + 1
    if gv != gv then gpuNaN := gpuNaN + 1
    if cv == cv && gv == gv then
      let diff := Float.abs (cv - gv)
      if diff > maxDiff then maxDiff := diff

  if cpuNaN > 0 || gpuNaN > 0 then
    IO.println s!"  ❌ {name}: CPU NaN={cpuNaN}, GPU NaN={gpuNaN}, maxDiff={maxDiff}"
    return true
  else if maxDiff > 0.01 then
    IO.println s!"  ⚠️ {name}: Large diff={maxDiff} (no NaN)"
    return false
  else
    IO.println s!"  ✓ {name}: OK, maxDiff={maxDiff}"
    return false

private def fileExists (path : String) : IO Bool := do
  try
    IO.FS.withFile path .read (fun _ => pure ())
    pure true
  catch _ =>
    pure false

def run (batchSize : Nat := 256) (hidden : Nat := 256) : IO Unit := do
  IO.println s!"=== Backward Pass CPU vs GPU Debug: batch={batchSize} hidden={hidden} ==="

  let trainImagesPath := "data/train-images-idx3-ubyte"
  let trainLabelsPath := "data/train-labels-idx1-ubyte"
  if !(← fileExists trainImagesPath) || !(← fileExists trainLabelsPath) then
    throw (IO.userError "missing MNIST files in `data/`")

  -- Load one batch
  let (images, labels) ← loadTrain "data" (maxImages? := some batchSize)
  let xData := getBatch images 0 batchSize
  let yLabels := getBatchLabels labels 0 batchSize
  let yOneHot := toOneHot yLabels
  let xBuf := RawBuffer.ofF32 ⟨xData⟩
  let yBuf := RawBuffer.ofF32 ⟨yOneHot⟩

  IO.println "\n--- Building full training graph ---"

  let (w1Id, w2Id, xId, yId, hUop, hReluUop, logitsUop, lossUop, gradW1Uop, gradW2Uop, newW1Uop, newW2Uop) := runTensorM do
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

    pure (w1Buf.uop.uid, w2Buf.uop.uid, xBuf.uop.uid, yBuf.uop.uid,
          h.uop, hRelu.uop, logits.uop, loss.uop, gradW1, gradW2, stepW1, stepW2)

  -- Initialize weights
  let w1Init : RawBuffer := { dtype := .float32, data := Native.fullF32Bits (784 * hidden) ((0.01 : Float32).toBits) }
  let w2Init : RawBuffer := { dtype := .float32, data := Native.fullF32Bits (hidden * 10) ((0.01 : Float32).toBits) }
  let env : Env := (∅ : Env) |>.insert xId xBuf |>.insert yId yBuf |>.insert w1Id w1Init |>.insert w2Id w2Init

  -- Compile - include intermediate UOps as roots so they stay in cache
  let compiled ← Interpreter.compileManyCached [hUop, hReluUop, logitsUop, lossUop, gradW1Uop, gradW2Uop, newW1Uop, newW2Uop]

  IO.println s!"  Graph has {compiled.nodes.length} nodes"
  IO.println s!"  Schedule has {compiled.schedule.length} items"

  -- Run CPU
  IO.println "\n--- Running CPU path ---"
  let cpuCache := Interpreter.evalCompiledRaw compiled env

  -- Run GPU
  IO.println "--- Running GPU path ---"
  let gpuCache ← Interpreter.evalCompiledRawIO compiled env

  IO.println "\n--- Checking forward pass ---"
  let _ ← compareCPUGPU "h (X @ W1)" cpuCache gpuCache hUop.uid
  let _ ← compareCPUGPU "hRelu" cpuCache gpuCache hReluUop.uid
  let _ ← compareCPUGPU "logits (hRelu @ W2)" cpuCache gpuCache logitsUop.uid
  let _ ← compareCPUGPU "loss" cpuCache gpuCache lossUop.uid

  IO.println "\n--- Checking gradients ---"
  let _ ← compareCPUGPU "gradW1" cpuCache gpuCache gradW1Uop.uid
  let _ ← compareCPUGPU "gradW2" cpuCache gpuCache gradW2Uop.uid

  IO.println "\n--- Checking final weight updates ---"
  let _ ← compareCPUGPU "newW1" cpuCache gpuCache newW1Uop.uid
  let _ ← compareCPUGPU "newW2" cpuCache gpuCache newW2Uop.uid

  -- Find where NaN first appears
  IO.println "\n--- Scanning all cached values for NaN ---"
  let mut nanNodes : Array UOpId := #[]
  for (uid, buf) in gpuCache.toList do
    let decoded := buf.decode.data
    let hasNaN := decoded.any (fun v => v != v)
    if hasNaN then
      nanNodes := nanNodes.push uid

  if nanNodes.isEmpty then
    IO.println "  No NaN found in GPU cache"
  else
    IO.println s!"  Found {nanNodes.size} nodes with NaN in GPU cache"
    for uid in nanNodes[:Nat.min 10 nanNodes.size] do
      let gpuBuf := gpuCache.getD uid (RawBuffer.zeros .float32 0)
      let cpuBuf := cpuCache.getD uid (RawBuffer.zeros .float32 0)
      let gpuNaN := (gpuBuf.decode.data.filter (fun v => v != v)).size
      let cpuNaN := (cpuBuf.decode.data.filter (fun v => v != v)).size
      IO.println s!"    UID {uid}: GPU NaN={gpuNaN}, CPU NaN={cpuNaN}, size={gpuBuf.decode.data.size}"

  IO.println "\n=== Done ==="

end TinyGrad4.Test.BackwardStepDebug

def main : IO Unit := TinyGrad4.Test.BackwardStepDebug.run
