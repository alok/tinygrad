import TinyGrad4

/-!
# Test: Matrix Multiplication + Gradient

This file demonstrates the end-to-end pipeline:
1. Create tensors with shapes verified at compile time
2. Perform matmul (shapes checked statically)
3. Compute gradients via autodiff
4. Evaluate with pure Lean interpreter
-/

namespace TinyGrad4.Test

open TinyGrad4
open StaticTensor
open Interpreter

/-- Simple test: matmul [2,3] @ [3,4] -> [2,4] -/
def testMatmul : TensorM (Matrix 2 4 .float32) := do
  -- Create input buffers
  let a ← Tensor.buffer [2, 3] .float32
  let b ← Tensor.buffer [3, 4] .float32
  -- Matmul: shape checked at compile time!
  matmul a b

/-- Test gradient computation (returns grad UIDs) -/
def testGradient : TensorM (Option UOpId × Option UOpId) := do
  -- Create a simple computation: c = a * b, loss = sum(c)
  let a ← Tensor.buffer [2, 2] .float32
  let b ← Tensor.buffer [2, 2] .float32
  let c ← mul a b
  let loss ← sum c

  -- Compute gradients
  let gradMap ← backward loss [a.uop, b.uop]

  -- Get individual gradients
  let gradAId := gradMap[a.uop.uid]?.map (·.uid)
  let gradBId := gradMap[b.uop.uid]?.map (·.uid)

  pure (gradAId, gradBId)

/-- Full test with evaluation -/
def testEval : IO Unit := do
  let result := runTensorM do
    -- Create tensors
    let a ← Tensor.full [2, 2] .float32 2.0
    let b ← Tensor.full [2, 2] .float32 3.0

    -- Multiply
    let c ← mul a b

    -- Sum
    let loss ← sum c

    -- Evaluate
    let env : Env := {}
    let lossVal := eval loss.uop env
    pure (loss, lossVal)

  let (loss, lossVal) := result
  IO.println s!"Loss UOp: {repr loss.uop.op}"
  IO.println s!"Loss value: {lossVal}"
  IO.println s!"Expected: #[24.0] (2*3*4 elements)"

-- Demonstrate type-safe shape errors caught at compile time
-- Uncomment to see compile error:
-- def shapeError : TensorM (Matrix 2 4 .float32) := do
--   let a ← Tensor.buffer [2, 3] .float32  -- [2, 3]
--   let b ← Tensor.buffer [5, 4] .float32  -- [5, 4] - WRONG! k=3 vs k=5
--   matmul a b  -- This would fail to compile!

end TinyGrad4.Test

-- Run the test
