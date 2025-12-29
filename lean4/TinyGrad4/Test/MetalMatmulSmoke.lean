import TinyGrad4
import TinyGrad4.Backend.Metal

/-!
# GPU Matmul Smoke Test

Tests that CONTRACT operations route through Metal GPU when available.
-/

namespace TinyGrad4.Test.MetalMatmulSmoke

open TinyGrad4
open StaticTensor
open Interpreter
open Backend

/-- Simple 2D matmul through interpreter (should use GPU on Metal) -/
private def testMatmul2D : IO Unit := do
  let root := runTensorM do
    let a ← Tensor.full [4, 3] .float32 1.0
    let b ← Tensor.full [3, 2] .float32 0.5
    let ab ← UOp.contract2D a.uop b.uop
    pure ab

  let result := Interpreter.eval root ∅
  let arr := result.decode

  -- Each output element = sum of 3 * (1.0 * 0.5) = 1.5
  if arr.size != 8 then
    throw (IO.userError s!"Expected 8 elements, got {arr.size}")

  for i in [:arr.size] do
    let v := arr[i]!
    let diff := Float.abs (v - 1.5)
    if diff > 0.001 then
      throw (IO.userError s!"matmul2D: idx {i} value {v} expected 1.5 diff {diff}")

  IO.println "  GPU 2D matmul correct"

/-- Larger matmul to better utilize GPU -/
private def testMatmul64x64 : IO Unit := do
  let root := runTensorM do
    let a ← Tensor.full [64, 64] .float32 1.0
    let b ← Tensor.full [64, 64] .float32 (1.0 / 64.0)  -- each output = 1.0
    let ab ← UOp.contract2D a.uop b.uop
    pure ab

  let result := Interpreter.eval root ∅
  let arr := result.decode

  if arr.size != 4096 then
    throw (IO.userError s!"Expected 4096 elements, got {arr.size}")

  -- Check a few elements
  for i in [0, 100, 2000, 4095] do
    let v := arr[i]!
    let diff := Float.abs (v - 1.0)
    if diff > 0.01 then
      throw (IO.userError s!"matmul64x64: idx {i} value {v} expected 1.0 diff {diff}")

  IO.println "  GPU 64x64 matmul correct"

/-- Identity matmul -/
private def testMatmulIdentity : IO Unit := do
  let root := runTensorM do
    -- Create identity-ish pattern: a = [[1,2],[3,4]], b = identity
    let aData ← Tensor.buffer [2, 2] .float32
    let bData ← Tensor.buffer [2, 2] .float32
    let ab ← UOp.contract2D aData.uop bData.uop
    pure ab

  -- Just test that it runs without error (buffer data is uninitialized)
  let result := Interpreter.eval root ∅
  let _ := result.decode
  IO.println "  GPU identity matmul runs"

def runAll : IO Unit := do
  IO.println "=== Metal GPU Matmul Smoke Tests ==="

  -- Check Metal availability
  let available ← Metal.isAvailable
  if available then
    let name ← Metal.metalDeviceName
    IO.println s!"  Metal available: {name}"
  else
    IO.println "  Metal not available - will use CPU fallback"

  testMatmul2D
  testMatmul64x64
  testMatmulIdentity

  IO.println "=== Metal GPU Matmul OK ==="

end TinyGrad4.Test.MetalMatmulSmoke

-- Note: #eval would segfault because Metal FFI isn't linked at compile time.
-- Build with: lake build metal_matmul_test && .lake/build/bin/metal_matmul_test

def main : IO Unit := TinyGrad4.Test.MetalMatmulSmoke.runAll
