import TinyGrad4
import TinyGrad4.Backend.Metal

/-!
Test GPU reduce operations for correctness and performance.
-/

set_option linter.useRawBuffer false

namespace TinyGrad4.Test.GPUReduceTest

open TinyGrad4
open StaticTensor
open Interpreter
open Backend
open Std

/-- Simple sum reduce test -/
def testSumReduce (n : Nat) : IO Unit := do
  IO.println s!"\n=== Sum reduce test: n={n} ==="

  -- Create tensor and sum it
  let (xId, sumUop) := runTensorM do
    let xBuf ← Tensor.buffer [n] .float32
    let result ← UOp.sum xBuf.uop [0] false
    pure (xBuf.uop.uid, result)

  -- Input: values 1, 2, 3, ..., n
  let xData := Array.range n |>.map (fun i => Float.ofNat (i + 1))
  let xBuf := RawBuffer.ofF32 ⟨xData⟩

  let compiled ← Interpreter.compileManyCached [sumUop]
  let env : Env := (∅ : Env) |>.insert xId xBuf

  -- CPU path
  let cpuStart ← IO.monoMsNow
  let cpuCache := Interpreter.evalCompiledRaw compiled env
  let cpuEnd ← IO.monoMsNow
  let cpuResult := cpuCache.getD sumUop.uid (RawBuffer.zeros .float32 0)
  let cpuData := cpuResult.decode.data

  -- GPU path
  let gpuStart ← IO.monoMsNow
  let gpuCache ← Interpreter.evalCompiledRawIO compiled env
  let gpuEnd ← IO.monoMsNow
  let gpuResult := gpuCache.getD sumUop.uid (RawBuffer.zeros .float32 0)
  let gpuData := gpuResult.decode.data

  -- Expected: n * (n+1) / 2
  let expected := (n.toFloat * (n + 1).toFloat) / 2.0

  IO.println s!"  CPU result: {cpuData} ({cpuEnd - cpuStart} ms)"
  IO.println s!"  GPU result: {gpuData} ({gpuEnd - gpuStart} ms)"
  IO.println s!"  Expected: {expected}"

  let cpuDiff := Float.abs (cpuData[0]! - expected)
  let gpuDiff := Float.abs (gpuData[0]! - expected)

  if cpuDiff < 0.1 then
    IO.println "  ✓ CPU correct"
  else
    IO.println s!"  ❌ CPU wrong (diff={cpuDiff})"

  if gpuDiff < 0.1 then
    IO.println "  ✓ GPU correct"
  else
    IO.println s!"  ❌ GPU wrong (diff={gpuDiff})"

/-- Batch sum reduce (like softmax sum step) -/
def testBatchSumReduce (batch k : Nat) : IO Unit := do
  IO.println s!"\n=== Batch sum reduce: [{batch}, {k}] -> [{batch}] ==="

  -- Create tensor and reduce over last axis
  let (xId, sumUop) := runTensorM do
    let xBuf ← Tensor.buffer [batch, k] .float32
    let result ← UOp.sum xBuf.uop [1] false
    pure (xBuf.uop.uid, result)

  -- Input: all 1s (so sum = k for each row)
  let xData := Array.replicate (batch * k) 1.0
  let xBuf := RawBuffer.ofF32 ⟨xData⟩

  let compiled ← Interpreter.compileManyCached [sumUop]
  let env : Env := (∅ : Env) |>.insert xId xBuf

  -- CPU path
  let cpuStart ← IO.monoMsNow
  let cpuCache := Interpreter.evalCompiledRaw compiled env
  let cpuEnd ← IO.monoMsNow
  let cpuResult := cpuCache.getD sumUop.uid (RawBuffer.zeros .float32 0)
  let cpuData := cpuResult.decode.data

  -- GPU path
  let gpuStart ← IO.monoMsNow
  let gpuCache ← Interpreter.evalCompiledRawIO compiled env
  let gpuEnd ← IO.monoMsNow
  let gpuResult := gpuCache.getD sumUop.uid (RawBuffer.zeros .float32 0)
  let gpuData := gpuResult.decode.data

  IO.println s!"  CPU time: {cpuEnd - cpuStart} ms"
  IO.println s!"  GPU time: {gpuEnd - gpuStart} ms"
  IO.println s!"  CPU first 5: {cpuData[:Nat.min 5 cpuData.size]}"
  IO.println s!"  GPU first 5: {gpuData[:Nat.min 5 gpuData.size]}"

  -- Each row should sum to k
  let expected := Float.ofNat k
  let mut cpuOk := true
  let mut gpuOk := true
  for i in [:batch] do
    if Float.abs (cpuData[i]! - expected) > 0.1 then
      cpuOk := false
    if Float.abs (gpuData[i]! - expected) > 0.1 then
      gpuOk := false

  if cpuOk then
    IO.println "  ✓ CPU correct"
  else
    IO.println "  ❌ CPU wrong"

  if gpuOk then
    IO.println "  ✓ GPU correct"
  else
    IO.println "  ❌ GPU wrong"

/-- Max reduce (like softmax max step) -/
def testBatchMaxReduce (batch k : Nat) : IO Unit := do
  IO.println s!"\n=== Batch max reduce: [{batch}, {k}] -> [{batch}] ==="

  -- Create tensor and reduce over last axis
  let (xId, maxUop) := runTensorM do
    let xBuf ← Tensor.buffer [batch, k] .float32
    let result ← UOp.max_ xBuf.uop [1] false
    pure (xBuf.uop.uid, result)

  -- Input: column index as value (so max = k-1 for each row)
  let mut xData : Array Float := #[]
  for _ in [:batch] do
    for j in [:k] do
      xData := xData.push (Float.ofNat j)
  let xBuf := RawBuffer.ofF32 ⟨xData⟩

  let compiled ← Interpreter.compileManyCached [maxUop]
  let env : Env := (∅ : Env) |>.insert xId xBuf

  -- CPU path
  let cpuStart ← IO.monoMsNow
  let cpuCache := Interpreter.evalCompiledRaw compiled env
  let cpuEnd ← IO.monoMsNow
  let cpuResult := cpuCache.getD maxUop.uid (RawBuffer.zeros .float32 0)
  let cpuData := cpuResult.decode.data

  -- GPU path
  let gpuStart ← IO.monoMsNow
  let gpuCache ← Interpreter.evalCompiledRawIO compiled env
  let gpuEnd ← IO.monoMsNow
  let gpuResult := gpuCache.getD maxUop.uid (RawBuffer.zeros .float32 0)
  let gpuData := gpuResult.decode.data

  IO.println s!"  CPU time: {cpuEnd - cpuStart} ms"
  IO.println s!"  GPU time: {gpuEnd - gpuStart} ms"
  IO.println s!"  CPU first 5: {cpuData[:Nat.min 5 cpuData.size]}"
  IO.println s!"  GPU first 5: {gpuData[:Nat.min 5 gpuData.size]}"

  -- Each row should have max = k-1
  let expected := Float.ofNat (k - 1)
  let mut cpuOk := true
  let mut gpuOk := true
  for i in [:batch] do
    if Float.abs (cpuData[i]! - expected) > 0.1 then
      cpuOk := false
    if Float.abs (gpuData[i]! - expected) > 0.1 then
      gpuOk := false

  if cpuOk then
    IO.println "  ✓ CPU correct"
  else
    IO.println "  ❌ CPU wrong"

  if gpuOk then
    IO.println "  ✓ GPU correct"
  else
    IO.println "  ❌ GPU wrong"

def run : IO Unit := do
  IO.println "=== GPU Reduce Test ==="

  let gpuAvail ← Metal.isAvailable
  IO.println s!"Metal available: {gpuAvail}"

  -- Simple 1D reduce tests
  testSumReduce 100
  testSumReduce 1000
  testSumReduce 10000

  -- Batch reduce tests (like softmax patterns)
  testBatchSumReduce 64 100    -- small
  testBatchSumReduce 256 784   -- MNIST hidden layer size
  testBatchSumReduce 256 10    -- MNIST output layer size

  testBatchMaxReduce 64 100
  testBatchMaxReduce 256 784
  testBatchMaxReduce 256 10

  IO.println "\n=== Done ==="

end TinyGrad4.Test.GPUReduceTest

def main : IO Unit := TinyGrad4.Test.GPUReduceTest.run
