import TinyGrad4

/-!
# IO Evaluator Smoke Test

Tests the IO-based evaluation with GPU preference.
-/

namespace TinyGrad4.Test.IOEvalSmoke

open TinyGrad4
open Interpreter
open Backend

/-- Test that IO evaluator matches pure evaluator for simple matmul -/
def testMatmulIOvsPure : IO Unit := do
  IO.println "Testing IO-based GPU evaluator vs pure evaluator..."

  -- Create simple 4x3 @ 3x2 = 4x2 matmul
  let (aU, bU, outU) := runTensorM do
    let a ← Tensor.buffer [4, 3] .float32
    let b ← Tensor.buffer [3, 2] .float32
    let out ← UOp.contract2D a.uop b.uop
    pure (a.uop, b.uop, out)

  -- Create test data: A = all 1.0f, B = all 2.0f
  -- Result should be 4x2 matrix of 6.0f (3 * 1.0 * 2.0 = 6.0)
  let aBytes := Native.fullF32Bits 12 1065353216  -- 1.0f bits
  let bBytes := Native.fullF32Bits 6 1073741824   -- 2.0f bits

  let env : Env :=
    let e0 : Env := ∅
    let e1 := setBuffer e0 aU { dtype := .float32, data := aBytes }
    setBuffer e1 bU { dtype := .float32, data := bBytes }

  -- Test pure evaluator (uses metalMatmulSync)
  let resultPure := eval outU env
  IO.println s!"  Pure eval: {resultPure.data.size / 4} floats"

  -- Test IO evaluator (uses MetalMatmul.runMatmul2D with GPU preference)
  let resultIO ← evalIO outU env
  IO.println s!"  IO eval: {resultIO.data.size / 4} floats"

  -- Compare results
  let pure_decoded := resultPure.decode
  let io_decoded := resultIO.decode

  if pure_decoded.size != io_decoded.size then
    throw (IO.userError s!"Size mismatch: pure={pure_decoded.size}, io={io_decoded.size}")

  let mut maxDiff : Float := 0.0
  for i in [:pure_decoded.size] do
    let diff := Float.abs (pure_decoded[i]! - io_decoded[i]!)
    if diff > maxDiff then
      maxDiff := diff

  IO.println s!"  Max difference: {maxDiff}"

  if maxDiff > 0.001 then
    throw (IO.userError s!"Results differ by {maxDiff} > 0.001")

  -- Also verify the actual value is correct (should be 6.0)
  let expectedValue : Float := 6.0
  let actualValue := pure_decoded[0]!
  if Float.abs (actualValue - expectedValue) > 0.001 then
    throw (IO.userError s!"Expected {expectedValue}, got {actualValue}")

  IO.println "✓ IO evaluator matches pure evaluator"
  IO.println s!"✓ Matmul result correct: {actualValue} ≈ {expectedValue}"

def runAll : IO Unit := do
  IO.println "=== IO Eval Smoke Tests ==="
  testMatmulIOvsPure
  IO.println "=== IO Eval Smoke OK ==="

end TinyGrad4.Test.IOEvalSmoke

def main : IO Unit := TinyGrad4.Test.IOEvalSmoke.runAll
