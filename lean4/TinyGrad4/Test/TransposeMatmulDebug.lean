import TinyGrad4
import TinyGrad4.Backend.Metal

/-!
Debug NaN in transpose + matmul (backward pass scenario).
Tests: X.T @ Y where X.T is a PERMUTE.
-/

set_option linter.useRawBuffer false

namespace TinyGrad4.Test.TransposeMatmulDebug

open TinyGrad4
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

def testTransposeMatmul (batch m n : Nat) (desc : String) : IO Unit := do
  IO.println s!"\n=== Testing {desc} ==="
  IO.println s!"  X[{batch}, {m}].T @ Y[{batch}, {n}] = result[{m}, {n}]"

  -- Create test tensors
  let (xId, yId, resultUop) := runTensorM do
    let xBuf ← Tensor.buffer [batch, m] .float32
    let yBuf ← Tensor.buffer [batch, n] .float32

    -- Transpose X: [batch, m] -> [m, batch]
    let xT ← permute xBuf [1, 0]

    -- Matmul: [m, batch] @ [batch, n] = [m, n]
    let result ← matmul xT yBuf

    pure (xBuf.uop.uid, yBuf.uop.uid, result.uop)

  -- Initialize inputs with small values
  let xBuf : RawBuffer := { dtype := .float32, data := Native.fullF32Bits (batch * m) ((0.01 : Float32).toBits) }
  let yBuf : RawBuffer := { dtype := .float32, data := Native.fullF32Bits (batch * n) ((0.01 : Float32).toBits) }

  -- Compile
  let compiled ← Interpreter.compileManyCached [resultUop]
  let env : Env := (∅ : Env) |>.insert xId xBuf |>.insert yId yBuf

  -- CPU path
  IO.println "  Running CPU path..."
  let cpuCache := Interpreter.evalCompiledRaw compiled env
  let cpuResult := cpuCache.getD resultUop.uid (RawBuffer.zeros .float32 0)
  let cpuNaN ← checkNaN "CPU result" cpuResult

  -- GPU path
  IO.println "  Running GPU path..."
  let gpuCache ← Interpreter.evalCompiledRawIO compiled env
  let gpuResult := gpuCache.getD resultUop.uid (RawBuffer.zeros .float32 0)
  let gpuNaN ← checkNaN "GPU result" gpuResult

  -- Compare first few values
  let cpuData := cpuResult.decode.data
  let gpuData := gpuResult.decode.data
  let sz := Nat.min 10 cpuData.size
  IO.println s!"  CPU first {sz}: {cpuData[:sz]}"
  IO.println s!"  GPU first {sz}: {gpuData[:sz]}"

  -- Expected value
  let expected := Float.ofNat batch * 0.01 * 0.01
  IO.println s!"  Expected per element: batch * 0.01 * 0.01 = {expected}"

  if cpuNaN || gpuNaN then
    IO.println s!"  ⚠️ NaN detected!"
  else
    -- Check accuracy
    let mut maxDiff : Float := 0.0
    for i in [:Nat.min 100 cpuData.size] do
      let diff := Float.abs (gpuData[i]! - cpuData[i]!)
      if diff > maxDiff then maxDiff := diff
    IO.println s!"  Max diff (first 100): {maxDiff}"
    if maxDiff < 0.001 then
      IO.println "  ✓ PASS"
    else
      IO.println "  ⚠️ Accuracy issue"

def testDirectMatmul (m k n : Nat) (desc : String) : IO Unit := do
  IO.println s!"\n=== Testing {desc} (direct, no transpose) ==="
  IO.println s!"  A[{m}, {k}] @ B[{k}, {n}] = result[{m}, {n}]"

  -- Create test tensors
  let (aId, bId, resultUop) := runTensorM do
    let aBuf ← Tensor.buffer [m, k] .float32
    let bBuf ← Tensor.buffer [k, n] .float32
    let result ← matmul aBuf bBuf
    pure (aBuf.uop.uid, bBuf.uop.uid, result.uop)

  -- Initialize inputs with small values
  let aBuf : RawBuffer := { dtype := .float32, data := Native.fullF32Bits (m * k) ((0.01 : Float32).toBits) }
  let bBuf : RawBuffer := { dtype := .float32, data := Native.fullF32Bits (k * n) ((0.01 : Float32).toBits) }

  -- Compile
  let compiled ← Interpreter.compileManyCached [resultUop]
  let env : Env := (∅ : Env) |>.insert aId aBuf |>.insert bId bBuf

  -- CPU path
  let cpuCache := Interpreter.evalCompiledRaw compiled env
  let cpuResult := cpuCache.getD resultUop.uid (RawBuffer.zeros .float32 0)
  let _ ← checkNaN "CPU result" cpuResult

  -- GPU path
  let gpuCache ← Interpreter.evalCompiledRawIO compiled env
  let gpuResult := gpuCache.getD resultUop.uid (RawBuffer.zeros .float32 0)
  let _ ← checkNaN "GPU result" gpuResult

  let cpuData := cpuResult.decode.data
  let gpuData := gpuResult.decode.data
  IO.println s!"  CPU first 5: {cpuData[:Nat.min 5 cpuData.size]}"
  IO.println s!"  GPU first 5: {gpuData[:Nat.min 5 gpuData.size]}"

def run : IO Unit := do
  IO.println "=== Transpose + Matmul NaN Debug ==="

  let gpuAvail ← Metal.isAvailable
  IO.println s!"Metal available: {gpuAvail}"

  -- Direct matmul (no transpose) - baseline
  testDirectMatmul 784 256 256 "baseline (no transpose)"

  -- Transpose + matmul (like backward pass for gradW1)
  -- In backward: X.T @ gradH where X is [batch, 784], so X.T is [784, batch]
  -- Result: [784, batch] @ [batch, hidden] = [784, hidden]
  testTransposeMatmul 256 784 256 "backward gradW1 (batch=256, m=784, n=256)"

  -- Smaller sizes
  testTransposeMatmul 64 784 256 "backward gradW1 (batch=64)"
  testTransposeMatmul 128 784 256 "backward gradW1 (batch=128)"

  -- Other backward matmuls
  testTransposeMatmul 256 256 10 "backward gradW2 (batch=256, m=256, n=10)"

  IO.println "\n=== Done ==="

end TinyGrad4.Test.TransposeMatmulDebug

def main : IO Unit := TinyGrad4.Test.TransposeMatmulDebug.run
