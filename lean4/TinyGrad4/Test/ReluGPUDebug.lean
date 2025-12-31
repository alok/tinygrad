import TinyGrad4
import TinyGrad4.Backend.Metal

/-!
Debug ReLU operation on GPU vs CPU.
-/

set_option linter.useRawBuffer false

namespace TinyGrad4.Test.ReluGPUDebug

open TinyGrad4
open StaticTensor
open Interpreter
open Backend
open Std

def testRelu (n : Nat) (name : String) : IO Unit := do
  IO.println s!"\n=== Testing ReLU: {name}, n={n} ==="

  -- Create tensor with positive and negative values
  let (xId, reluUop) := runTensorM do
    let xBuf ← Tensor.buffer [n] .float32
    let reluResult ← relu xBuf
    pure (xBuf.uop.uid, reluResult.uop)

  -- Input data: alternating positive/negative
  let mut xData : Array Float := #[]
  for i in [:n] do
    let v := if i % 2 == 0 then Float.ofNat (i % 100) else -(Float.ofNat (i % 100))
    xData := xData.push v
  let xBuf := RawBuffer.ofF32 ⟨xData⟩

  IO.println s!"  Input first 10: {xData[:Nat.min 10 n]}"

  let compiled ← Interpreter.compileManyCached [reluUop]
  let env : Env := (∅ : Env) |>.insert xId xBuf

  -- CPU path
  let cpuCache := Interpreter.evalCompiledRaw compiled env
  let cpuResult := cpuCache.getD reluUop.uid (RawBuffer.zeros .float32 0)
  let cpuData := cpuResult.decode.data

  -- GPU path
  let gpuCache ← Interpreter.evalCompiledRawIO compiled env
  let gpuResult := gpuCache.getD reluUop.uid (RawBuffer.zeros .float32 0)
  let gpuData := gpuResult.decode.data

  IO.println s!"  CPU first 10: {cpuData[:Nat.min 10 n]}"
  IO.println s!"  GPU first 10: {gpuData[:Nat.min 10 n]}"

  -- Expected: max(input, 0), so negative values become 0
  let mut expectedData : Array Float := #[]
  for v in xData do
    expectedData := expectedData.push (if v > 0 then v else 0)
  IO.println s!"  Expected first 10: {expectedData[:Nat.min 10 n]}"

  -- Compare
  let mut maxDiffCPU : Float := 0.0
  let mut maxDiffGPU : Float := 0.0
  let mut cpuNaN := 0
  let mut gpuNaN := 0

  for i in [:n] do
    let exp := expectedData[i]!
    let cpuV := cpuData[i]!
    let gpuV := gpuData[i]!

    if cpuV != cpuV then cpuNaN := cpuNaN + 1
    if gpuV != gpuV then gpuNaN := gpuNaN + 1

    if cpuV == cpuV then
      let diff := Float.abs (cpuV - exp)
      if diff > maxDiffCPU then maxDiffCPU := diff

    if gpuV == gpuV then
      let diff := Float.abs (gpuV - exp)
      if diff > maxDiffGPU then maxDiffGPU := diff

  IO.println s!"  CPU NaN: {cpuNaN}, maxDiff from expected: {maxDiffCPU}"
  IO.println s!"  GPU NaN: {gpuNaN}, maxDiff from expected: {maxDiffGPU}"

  if gpuNaN > 0 || maxDiffGPU > 0.001 then
    IO.println "  ❌ GPU FAIL"
  else
    IO.println "  ✓ GPU OK"

def testReluAfterMatmul (batch m n : Nat) (name : String) : IO Unit := do
  IO.println s!"\n=== Testing ReLU after matmul: {name} ==="
  IO.println s!"  X[{batch}, {m}] @ W[{m}, {n}] then ReLU"

  -- Create forward pass like in MNIST
  let (xId, wId, reluUop) := runTensorM do
    let xBuf ← Tensor.buffer [batch, m] .float32
    let wBuf ← Tensor.buffer [m, n] .float32
    let h ← matmul xBuf wBuf
    let hRelu ← relu h
    pure (xBuf.uop.uid, wBuf.uop.uid, hRelu.uop)

  -- Initialize with small values
  let xBuf : RawBuffer := { dtype := .float32, data := Native.fullF32Bits (batch * m) ((0.01 : Float32).toBits) }
  let wBuf : RawBuffer := { dtype := .float32, data := Native.fullF32Bits (m * n) ((0.01 : Float32).toBits) }

  let compiled ← Interpreter.compileManyCached [reluUop]
  let env : Env := (∅ : Env) |>.insert xId xBuf |>.insert wId wBuf

  -- CPU path
  let cpuCache := Interpreter.evalCompiledRaw compiled env
  let cpuResult := cpuCache.getD reluUop.uid (RawBuffer.zeros .float32 0)
  let cpuData := cpuResult.decode.data

  -- GPU path
  let gpuCache ← Interpreter.evalCompiledRawIO compiled env
  let gpuResult := gpuCache.getD reluUop.uid (RawBuffer.zeros .float32 0)
  let gpuData := gpuResult.decode.data

  IO.println s!"  Output size: {cpuData.size}"
  IO.println s!"  CPU first 10: {cpuData[:Nat.min 10 cpuData.size]}"
  IO.println s!"  GPU first 10: {gpuData[:Nat.min 10 gpuData.size]}"

  -- Compare
  let mut maxDiff : Float := 0.0
  let mut cpuNaN := 0
  let mut gpuNaN := 0

  for i in [:cpuData.size] do
    let cpuV := cpuData[i]!
    let gpuV := gpuData[i]!

    if cpuV != cpuV then cpuNaN := cpuNaN + 1
    if gpuV != gpuV then gpuNaN := gpuNaN + 1

    if cpuV == cpuV && gpuV == gpuV then
      let diff := Float.abs (cpuV - gpuV)
      if diff > maxDiff then maxDiff := diff

  IO.println s!"  CPU NaN: {cpuNaN}, GPU NaN: {gpuNaN}, maxDiff: {maxDiff}"

  if gpuNaN > 0 || maxDiff > 0.001 then
    IO.println "  ❌ GPU FAIL"
    -- Show where first big diff is
    for i in [:Nat.min 100 cpuData.size] do
      let cpuV := cpuData[i]!
      let gpuV := gpuData[i]!
      let diff := Float.abs (cpuV - gpuV)
      if diff > 0.001 then
        IO.println s!"  First diff at i={i}: CPU={cpuV}, GPU={gpuV}"
        break
  else
    IO.println "  ✓ GPU OK"

def run : IO Unit := do
  IO.println "=== ReLU GPU Debug ==="

  let gpuAvail ← Metal.isAvailable
  IO.println s!"Metal available: {gpuAvail}"

  -- Standalone ReLU tests
  testRelu 100 "small"
  testRelu 65536 "medium (64K)"
  testRelu 200704 "large (same as h in MNIST)"

  -- ReLU after matmul (like in MNIST forward pass)
  testReluAfterMatmul 64 784 256 "batch=64"
  testReluAfterMatmul 128 784 256 "batch=128"
  testReluAfterMatmul 256 784 256 "batch=256 (failing case)"

  IO.println "\n=== Done ==="

end TinyGrad4.Test.ReluGPUDebug

def main : IO Unit := TinyGrad4.Test.ReluGPUDebug.run
