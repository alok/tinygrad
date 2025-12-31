import TinyGrad4
import TinyGrad4.NN

/-!
# BatchNorm Smoke Tests

Tests for BatchNorm1d and BatchNorm2d implementations.
-/

namespace TinyGrad4.Test.BatchNormSmoke

set_option linter.useRawBuffer false

open TinyGrad4
open TinyGrad4.NN
open Interpreter
open StaticTensor

private def assertClose (got expected : Float) (tol : Float := 0.01) (label : String) : IO Unit := do
  let diff := (got - expected).abs
  if diff > tol then
    throw (IO.userError s!"{label}: {got} != {expected} (diff={diff})")

private def assertShape (got expected : Shape) (label : String) : IO Unit := do
  if got != expected then
    throw (IO.userError s!"{label}: shape {got} != {expected}")

def testBatchNorm2dShape : IO Unit := do
  -- Test that BatchNorm2d produces correct output shape
  let y := runTensorM do
    let bn ← batchNorm2d 16
    let x ← Tensor.buffer [4, 16, 8, 8] .float32
    BatchNormParams.forward2d bn x
  assertShape y.uop.shape [4, 16, 8, 8] "batchnorm2d shape"

def testBatchNorm1dShape : IO Unit := do
  -- Test that BatchNorm1d produces correct output shape
  let y := runTensorM do
    let bn ← batchNorm1d 64
    let x ← Tensor.buffer [32, 64] .float32
    BatchNormParams.forward1d bn x
  assertShape y.uop.shape [32, 64] "batchnorm1d shape"

def testBatchNorm2dTrainValues : IO Unit := do
  -- Test BatchNorm2d in training mode normalizes correctly
  -- Input: [2, 2, 2, 2] with known values
  let (y, xU, bnWeight, bnBias, bnMean, bnVar) := runTensorM do
    let bn ← batchNorm2d 2  -- 2 channels
    let x ← Tensor.buffer [2, 2, 2, 2] .float32
    let out ← BatchNormParams.forward2d bn x
    pure (out, x.uop, bn.weight.uop, bn.bias.uop, bn.runningMean.uop, bn.runningVar.uop)

  -- Create input with distinct values per channel
  -- Channel 0: all 1.0, Channel 1: all 2.0
  let xData : FloatArray := ⟨#[
    -- batch 0, channel 0, 2x2
    1.0, 1.0, 1.0, 1.0,
    -- batch 0, channel 1, 2x2
    2.0, 2.0, 2.0, 2.0,
    -- batch 1, channel 0, 2x2
    1.0, 1.0, 1.0, 1.0,
    -- batch 1, channel 1, 2x2
    2.0, 2.0, 2.0, 2.0
  ]⟩

  let weightData : FloatArray := ⟨#[1.0, 1.0]⟩  -- gamma = 1
  let biasData : FloatArray := ⟨#[0.0, 0.0]⟩    -- beta = 0
  let meanData : FloatArray := ⟨#[0.0, 0.0]⟩    -- running mean (not used in train)
  let varData : FloatArray := ⟨#[1.0, 1.0]⟩     -- running var (not used in train)

  let env : Env :=
    let e := Interpreter.setBuffer ∅ xU (RawBuffer.ofF32 xData)
    let e := Interpreter.setBuffer e bnWeight (RawBuffer.ofF32 weightData)
    let e := Interpreter.setBuffer e bnBias (RawBuffer.ofF32 biasData)
    let e := Interpreter.setBuffer e bnMean (RawBuffer.ofF32 meanData)
    Interpreter.setBuffer e bnVar (RawBuffer.ofF32 varData)

  let compiled := compileMany [y.uop]
  let result := evalCompiledRaw compiled env
  let out := result.getD y.uop.uid (RawBuffer.zeros .float32 0)
  let outF32 := out.decode

  -- After BatchNorm with constant input per channel:
  -- mean = value, var = 0, so normalized = 0 (or close to it)
  -- Then scaled by weight=1 and bias=0, result should be near 0
  for i in [:outF32.size] do
    assertClose outF32[i]! 0.0 0.1 s!"batchnorm2d train output[{i}]"

def testBatchNorm2dEvalValues : IO Unit := do
  -- Test BatchNorm2d in eval mode uses running stats
  let (y, xU, bnWeight, bnBias, bnMean, bnVar) := runTensorM do
    let bn ← batchNorm2d 2
    let bn := bn.eval  -- Switch to eval mode
    let x ← Tensor.buffer [2, 2, 2, 2] .float32
    let out ← BatchNormParams.forward2d bn x
    pure (out, x.uop, bn.weight.uop, bn.bias.uop, bn.runningMean.uop, bn.runningVar.uop)

  let xData : FloatArray := ⟨Array.replicate 16 1.0⟩
  let weightData : FloatArray := ⟨#[1.0, 1.0]⟩
  let biasData : FloatArray := ⟨#[0.0, 0.0]⟩
  let meanData : FloatArray := ⟨#[0.0, 0.0]⟩  -- running mean = 0
  let varData : FloatArray := ⟨#[1.0, 1.0]⟩   -- running var = 1

  let env : Env :=
    let e := Interpreter.setBuffer ∅ xU (RawBuffer.ofF32 xData)
    let e := Interpreter.setBuffer e bnWeight (RawBuffer.ofF32 weightData)
    let e := Interpreter.setBuffer e bnBias (RawBuffer.ofF32 biasData)
    let e := Interpreter.setBuffer e bnMean (RawBuffer.ofF32 meanData)
    Interpreter.setBuffer e bnVar (RawBuffer.ofF32 varData)

  let compiled := compileMany [y.uop]
  let result := evalCompiledRaw compiled env
  let out := result.getD y.uop.uid (RawBuffer.zeros .float32 0)
  let outF32 := out.decode

  -- Eval mode: uses running_mean=0, running_var=1
  -- So: (1.0 - 0) / sqrt(1 + eps) * 1 + 0 ≈ 1.0
  for i in [:outF32.size] do
    assertClose outF32[i]! 1.0 0.01 s!"batchnorm2d eval output[{i}]"

def testBatchNormParams : IO Unit := do
  -- Test parameter creation
  let bn := runTensorM do
    batchNorm2d 64
  if bn.weight.uop.shape != [64] then
    throw (IO.userError s!"weight shape: {bn.weight.uop.shape} != [64]")
  if bn.bias.uop.shape != [64] then
    throw (IO.userError s!"bias shape: {bn.bias.uop.shape} != [64]")
  if bn.runningMean.uop.shape != [64] then
    throw (IO.userError s!"running_mean shape: {bn.runningMean.uop.shape} != [64]")
  if bn.runningVar.uop.shape != [64] then
    throw (IO.userError s!"running_var shape: {bn.runningVar.uop.shape} != [64]")
  if !bn.training then
    throw (IO.userError "should start in training mode")

def testBatchNormTrainEvalSwitch : IO Unit := do
  let bn := runTensorM do batchNorm2d 16
  if !bn.training then
    throw (IO.userError "should start in training mode")
  let bnEval := bn.eval
  if bnEval.training then
    throw (IO.userError "eval should set training=false")
  let bnTrain := bnEval.train
  if !bnTrain.training then
    throw (IO.userError "train should set training=true")

def benchBatchNorm2d : IO Unit := do
  IO.println "\n=== BatchNorm2d Benchmark ==="
  let sizes : List (Nat × Nat × Nat × Nat) := [
    (32, 64, 32, 32),   -- Typical CNN layer
    (64, 128, 16, 16),  -- Deeper layer
    (128, 256, 8, 8),   -- Late stage
  ]

  for (batch, channels, h, w) in sizes do
    let (y, xU, bnWeight, bnBias, bnMean, bnVar) := runTensorM do
      let bn ← batchNorm2d channels
      let x ← Tensor.buffer [batch, channels, h, w] .float32
      let out ← BatchNormParams.forward2d bn x
      pure (out, x.uop, bn.weight.uop, bn.bias.uop, bn.runningMean.uop, bn.runningVar.uop)

    let numel := batch * channels * h * w
    let xData : FloatArray := ⟨Array.replicate numel 1.0⟩
    let weightData : FloatArray := ⟨Array.replicate channels 1.0⟩
    let biasData : FloatArray := ⟨Array.replicate channels 0.0⟩
    let meanData : FloatArray := ⟨Array.replicate channels 0.0⟩
    let varData : FloatArray := ⟨Array.replicate channels 1.0⟩

    let env : Env :=
      let e := Interpreter.setBuffer ∅ xU (RawBuffer.ofF32 xData)
      let e := Interpreter.setBuffer e bnWeight (RawBuffer.ofF32 weightData)
      let e := Interpreter.setBuffer e bnBias (RawBuffer.ofF32 biasData)
      let e := Interpreter.setBuffer e bnMean (RawBuffer.ofF32 meanData)
      Interpreter.setBuffer e bnVar (RawBuffer.ofF32 varData)

    let compiled := compileMany [y.uop]

    -- Warmup
    let _ := evalCompiledRaw compiled env

    -- Time 5 iterations
    let startTime ← IO.monoMsNow
    for _ in [:5] do
      let _ := evalCompiledRaw compiled env
    let endTime ← IO.monoMsNow
    let avgMs := (endTime - startTime).toFloat / 5.0

    IO.println s!"  [{batch}, {channels}, {h}, {w}] ({numel} elements): {avgMs} ms"

def runAll : IO Unit := do
  IO.println "=== BatchNorm Smoke Tests ==="
  testBatchNorm2dShape
  IO.println "  batchnorm2d shape ✓"
  testBatchNorm1dShape
  IO.println "  batchnorm1d shape ✓"
  testBatchNormParams
  IO.println "  batchnorm params ✓"
  testBatchNormTrainEvalSwitch
  IO.println "  train/eval switch ✓"
  testBatchNorm2dTrainValues
  IO.println "  batchnorm2d train values ✓"
  testBatchNorm2dEvalValues
  IO.println "  batchnorm2d eval values ✓"
  benchBatchNorm2d
  IO.println "=== BatchNorm Smoke OK ==="

end TinyGrad4.Test.BatchNormSmoke

def main : IO Unit := TinyGrad4.Test.BatchNormSmoke.runAll
