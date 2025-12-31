import TinyGrad4
import TinyGrad4.NN

/-!
# Dropout Smoke Tests

Tests for Dropout layer implementation.
-/

namespace TinyGrad4.Test.DropoutSmoke

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

def testDropoutCreate : IO Unit := do
  -- Test default creation
  let d := dropout 0.5
  if d.p != 0.5 then
    throw (IO.userError s!"expected p=0.5, got {d.p}")
  if !d.training then
    throw (IO.userError "should start in training mode")

def testDropoutTrainEvalSwitch : IO Unit := do
  let d := dropout 0.3
  if !d.training then
    throw (IO.userError "should start in training mode")
  let dEval := d.eval
  if dEval.training then
    throw (IO.userError "eval should set training=false")
  let dTrain := dEval.train
  if !dTrain.training then
    throw (IO.userError "train should set training=true")

def testDropoutEvalPassthrough : IO Unit := do
  -- In eval mode, dropout should be identity function
  let (y, xU) := runTensorM do
    let d := (dropout 0.5).eval  -- eval mode
    let x ← Tensor.buffer [4, 4] .float32
    let out ← DropoutParams.forward d x 42
    pure (out, x.uop)

  -- Input: all 1.0
  let xData : FloatArray := ⟨Array.replicate 16 1.0⟩

  let env := Interpreter.setBuffer ∅ xU (RawBuffer.ofF32 xData)
  let compiled := compileMany [y.uop]
  let result := evalCompiledRaw compiled env
  let out := result.getD y.uop.uid (RawBuffer.zeros .float32 0)
  let outF32 := out.decode

  -- In eval mode, output should equal input (all 1.0)
  for i in [:outF32.size] do
    assertClose outF32[i]! 1.0 0.001 s!"dropout eval output[{i}]"

def testDropoutP0Passthrough : IO Unit := do
  -- With p=0, dropout should be identity even in training
  let (y, xU) := runTensorM do
    let d := dropout 0.0  -- p=0, no dropout
    let x ← Tensor.buffer [4, 4] .float32
    let out ← DropoutParams.forward d x 42
    pure (out, x.uop)

  let xData : FloatArray := ⟨Array.replicate 16 2.0⟩

  let env := Interpreter.setBuffer ∅ xU (RawBuffer.ofF32 xData)
  let compiled := compileMany [y.uop]
  let result := evalCompiledRaw compiled env
  let out := result.getD y.uop.uid (RawBuffer.zeros .float32 0)
  let outF32 := out.decode

  -- With p=0, output should equal input
  for i in [:outF32.size] do
    assertClose outF32[i]! 2.0 0.001 s!"dropout p=0 output[{i}]"

def testDropoutP1AllZeros : IO Unit := do
  -- With p=1, dropout should zero everything
  let (y, xU) := runTensorM do
    let d := dropout 1.0  -- p=1, drop all
    let x ← Tensor.buffer [4, 4] .float32
    let out ← DropoutParams.forward d x 42
    pure (out, x.uop)

  let xData : FloatArray := ⟨Array.replicate 16 5.0⟩

  let env := Interpreter.setBuffer ∅ xU (RawBuffer.ofF32 xData)
  let compiled := compileMany [y.uop]
  let result := evalCompiledRaw compiled env
  let out := result.getD y.uop.uid (RawBuffer.zeros .float32 0)
  let outF32 := out.decode

  -- With p=1, output should be all zeros
  for i in [:outF32.size] do
    assertClose outF32[i]! 0.0 0.001 s!"dropout p=1 output[{i}]"

def testDropoutTrainingMask : IO Unit := do
  -- Test that training mode applies dropout and scaling
  let (y, xU) := runTensorM do
    let d := dropout 0.5  -- 50% dropout
    let x ← Tensor.buffer [100] .float32  -- larger size for statistical test
    let out ← DropoutParams.forward d x 123  -- fixed seed
    pure (out, x.uop)

  -- Input: all 1.0
  let xData : FloatArray := ⟨Array.replicate 100 1.0⟩

  let env := Interpreter.setBuffer ∅ xU (RawBuffer.ofF32 xData)
  let compiled := compileMany [y.uop]
  let result := evalCompiledRaw compiled env
  let out := result.getD y.uop.uid (RawBuffer.zeros .float32 0)
  let outF32 := out.decode

  -- Count zeros and non-zeros
  let mut numZeros := 0
  let mut numNonZeros := 0
  let mut sum := 0.0
  for i in [:outF32.size] do
    let v := outF32[i]!
    sum := sum + v
    if v == 0.0 then
      numZeros := numZeros + 1
    else
      numNonZeros := numNonZeros + 1
      -- Non-zero values should be scaled by 1/(1-0.5) = 2.0
      assertClose v 2.0 0.001 s!"dropout scaled value[{i}]"

  -- With p=0.5, expect roughly half zeros (allow 30% variance for randomness)
  let zeroFrac := (numZeros.toFloat / 100.0)
  if zeroFrac < 0.2 || zeroFrac > 0.8 then
    throw (IO.userError s!"expected ~50% zeros, got {numZeros}% ({numZeros}/100)")

  -- Mean should be close to 1.0 (expected value preservation)
  let mean := sum / 100.0
  assertClose mean 1.0 0.3 "dropout mean (expected value)"

def testDropoutDifferentSeeds : IO Unit := do
  -- Different seeds should produce different masks
  let (y1, y2, xU) := runTensorM do
    let d := dropout 0.5
    let x ← Tensor.buffer [20] .float32
    let out1 ← DropoutParams.forward d x 111
    let out2 ← DropoutParams.forward d x 222
    pure (out1, out2, x.uop)

  let xData : FloatArray := ⟨Array.replicate 20 1.0⟩

  let env := Interpreter.setBuffer ∅ xU (RawBuffer.ofF32 xData)

  let compiled1 := compileMany [y1.uop]
  let result1 := evalCompiledRaw compiled1 env
  let out1 := result1.getD y1.uop.uid (RawBuffer.zeros .float32 0)
  let outF32_1 := out1.decode

  let compiled2 := compileMany [y2.uop]
  let result2 := evalCompiledRaw compiled2 env
  let out2 := result2.getD y2.uop.uid (RawBuffer.zeros .float32 0)
  let outF32_2 := out2.decode

  -- Count differences
  let mut diffs := 0
  for i in [:20] do
    if outF32_1[i]! != outF32_2[i]! then
      diffs := diffs + 1

  -- With different seeds, masks should differ (expect at least some differences)
  if diffs < 2 then
    throw (IO.userError s!"different seeds should produce different masks, got only {diffs} differences")

def testDropoutShape : IO Unit := do
  -- Test that output shape matches input shape
  let y := runTensorM do
    let d := dropout 0.3
    let x ← Tensor.buffer [8, 16, 32] .float32
    DropoutParams.forward d x 42
  assertShape y.uop.shape [8, 16, 32] "dropout shape"

def testDropoutNoParams : IO Unit := do
  -- Dropout should have no trainable parameters
  let d := dropout 0.5
  let params := DropoutParams.parameters d
  if params.length != 0 then
    throw (IO.userError s!"dropout should have 0 parameters, got {params.length}")
  if DropoutParams.numParams d != 0 then
    throw (IO.userError s!"numParams should be 0, got {DropoutParams.numParams d}")

def benchDropout : IO Unit := do
  IO.println "\n=== Dropout Benchmark ==="
  let sizes : List (List Nat) := [
    [64, 256],        -- Small
    [128, 512],       -- Medium
    [256, 1024],      -- Large
  ]

  for shape in sizes do
    let numel := listProd shape
    let (y, xU) := runTensorM do
      let d := dropout 0.5
      let x ← Tensor.buffer shape .float32
      let out ← DropoutParams.forward d x 42
      pure (out, x.uop)

    let xData : FloatArray := ⟨Array.replicate numel 1.0⟩

    let env := Interpreter.setBuffer ∅ xU (RawBuffer.ofF32 xData)
    let compiled := compileMany [y.uop]

    -- Warmup
    let _ := evalCompiledRaw compiled env

    -- Time 10 iterations
    let startTime ← IO.monoMsNow
    for _ in [:10] do
      let _ := evalCompiledRaw compiled env
    let endTime ← IO.monoMsNow
    let avgMs := (endTime - startTime).toFloat / 10.0

    IO.println s!"  {shape} ({numel} elements): {avgMs} ms"

def runAll : IO Unit := do
  IO.println "=== Dropout Smoke Tests ==="
  testDropoutCreate
  IO.println "  dropout create ✓"
  testDropoutTrainEvalSwitch
  IO.println "  train/eval switch ✓"
  testDropoutEvalPassthrough
  IO.println "  eval passthrough ✓"
  testDropoutP0Passthrough
  IO.println "  p=0 passthrough ✓"
  testDropoutP1AllZeros
  IO.println "  p=1 all zeros ✓"
  testDropoutTrainingMask
  IO.println "  training mask & scaling ✓"
  testDropoutDifferentSeeds
  IO.println "  different seeds ✓"
  testDropoutShape
  IO.println "  shape preserved ✓"
  testDropoutNoParams
  IO.println "  no trainable params ✓"
  benchDropout
  IO.println "=== Dropout Smoke OK ==="

end TinyGrad4.Test.DropoutSmoke

def main : IO Unit := TinyGrad4.Test.DropoutSmoke.runAll
