import TinyGrad4

/-!
# Multi-Layer Perceptron (MLP) Example

A simple 2-layer neural network demonstrating:
- Forward pass with type-safe matrix operations
- Loss computation
- Backward pass with gradient computation
- Parameter updates (gradient descent step)
-/

namespace TinyGrad4.Test.MLP

open TinyGrad4
open StaticTensor
open Interpreter

/-!
## Network Architecture

Input:  [batch, 2]   -- 2 features
Hidden: [batch, 4]   -- 4 hidden units
Output: [batch, 1]   -- 1 output

Layers:
- Linear1: [2, 4] weights, no bias for simplicity
- Linear2: [4, 1] weights
-/

-- Simple forward pass: x @ W1 @ W2
def mlpForward {batch : Nat}
    (x : Matrix batch 2 .float32)
    (w1 : Matrix 2 4 .float32)
    (w2 : Matrix 4 1 .float32)
    : TensorM (Matrix batch 1 .float32) := do
  -- Layer 1: [batch, 2] @ [2, 4] = [batch, 4]
  let h ← matmul x w1
  -- Layer 2: [batch, 4] @ [4, 1] = [batch, 1]
  matmul h w2

-- Mean squared error loss
def mseLoss {batch : Nat}
    (pred : Matrix batch 1 .float32)
    (target : Matrix batch 1 .float32)
    : TensorM (Scalar .float32) := do
  -- (pred - target)^2
  let diff ← sub pred target
  let sq ← mul diff diff
  mean sq

-- Full training step: forward, loss, backward
def trainingStep : IO Unit := do
  let result := runTensorM do
    -- Create dummy data: 3 samples, 2 features
    -- Input: all 1s
    let x ← Tensor.full [3, 2] .float32 1.0

    -- Target: all 0.5s
    let target ← Tensor.full [3, 1] .float32 0.5

    -- Initialize weights
    -- W1: [2, 4] - all 0.1
    let w1 ← Tensor.full [2, 4] .float32 0.1
    -- W2: [4, 1] - all 0.2
    let w2 ← Tensor.full [4, 1] .float32 0.2

    -- Forward pass
    let pred ← mlpForward x w1 w2

    -- Compute loss
    let loss ← mseLoss pred target

    -- Compute gradients
    let gradMap ← backward loss [w1.uop, w2.uop]

    -- Evaluate everything
    let env : Env := ∅
    let predVal := eval pred.uop env
    let lossVal := eval loss.uop env

    let gradW1 := gradMap[w1.uop.uid]?.map (eval · env)
    let gradW2 := gradMap[w2.uop.uid]?.map (eval · env)

    pure (predVal, lossVal, gradW1, gradW2)

  let (predVal, lossVal, gradW1, gradW2) := result

  IO.println "=== MLP Training Step ==="
  IO.println ""
  IO.println "Architecture: Input[3,2] -> Linear[2,4] -> Linear[4,1] -> Output[3,1]"
  IO.println ""
  IO.println s!"Predictions (should be ~0.16 for W1=0.1, W2=0.2):"
  IO.println s!"  {predVal}"
  -- Expected: x @ W1 @ W2 = [1,1] @ [[0.1]*4] @ [[0.2]] = 2*0.1*4*0.2 = 0.16
  IO.println ""
  IO.println s!"Loss (MSE against target=0.5):"
  IO.println s!"  {lossVal}"
  IO.println ""
  IO.println s!"Gradient w.r.t. W1 [2,4]:"
  IO.println s!"  {gradW1}"
  IO.println ""
  IO.println s!"Gradient w.r.t. W2 [4,1]:"
  IO.println s!"  {gradW2}"
  IO.println ""
  IO.println "Note: Negative gradients indicate we should increase weights to reduce loss"
  IO.println "(since pred < target, we need larger weights)"

-- Demonstrate shape safety: mismatched dimensions won't compile
-- Uncomment to see compile error:
-- def shapeError : TensorM (Matrix 3 1 .float32) := do
--   let x ← Tensor.full [3, 2] .float32 1.0
--   let w1 ← Tensor.full [5, 4] .float32 0.1  -- WRONG! 2 != 5
--   matmul x w1  -- Type error: Matrix 3 2 @ Matrix 5 4 is invalid

-- Simple matmul gradient test
def simpleMatmulGrad : IO Unit := do
  let result := runTensorM do
    -- Simple: [2, 3] @ [3, 2] = [2, 2]
    let a ← Tensor.full [2, 3] .float32 1.0
    let b ← Tensor.full [3, 2] .float32 0.5

    -- Matmul
    let c ← matmul a b

    -- Sum to scalar
    let loss ← sum c

    -- Get gradient w.r.t. a and b
    let gradMap ← backward loss [a.uop, b.uop]

    let env : Env := ∅
    let lossVal := eval loss.uop env
    let gradA := gradMap[a.uop.uid]?.map (eval · env)
    let gradB := gradMap[b.uop.uid]?.map (eval · env)

    pure (lossVal, gradA, gradB)

  let (lossVal, gradA, gradB) := result
  IO.println "=== Simple Matmul Gradient Test ==="
  IO.println s!"A = [2,3] all 1.0, B = [3,2] all 0.5"
  IO.println s!"C = A @ B = [2,2] all 1.5 (3 * 1.0 * 0.5)"
  IO.println s!"loss = sum(C) = {lossVal}"
  IO.println s!"Expected loss: 6.0 (4 elements * 1.5)"
  IO.println ""
  IO.println s!"dL/dA = {gradA}"
  IO.println "Expected dL/dA: [2,3] where each = sum of B's row = 2*0.5 = 1.0"
  IO.println ""
  IO.println s!"dL/dB = {gradB}"
  IO.println "Expected dL/dB: [3,2] where each = sum of A's col = 2*1.0 = 2.0"

-- Test forward pass only for chained matmuls
def chainedMatmulForward : IO Unit := do
  let result := runTensorM do
    -- [3, 2] @ [2, 4] @ [4, 1] = [3, 1]
    let x ← Tensor.full [3, 2] .float32 1.0
    let w1 ← Tensor.full [2, 4] .float32 0.1
    let w2 ← Tensor.full [4, 1] .float32 0.2

    -- First matmul: [3,2] @ [2,4] = [3,4]
    let h ← matmul x w1
    -- Second matmul: [3,4] @ [4,1] = [3,1]
    let out ← matmul h w2

    -- Sum to scalar
    let loss ← sum out

    let env : Env := ∅
    let hVal := eval h.uop env
    let outVal := eval out.uop env
    let lossVal := eval loss.uop env

    pure (hVal, outVal, lossVal)

  let (hVal, outVal, lossVal) := result
  IO.println "=== Chained Matmul Forward Test ==="
  IO.println s!"h = X @ W1 = {hVal}"
  IO.println s!"out = h @ W2 = {outVal}"
  IO.println s!"loss = {lossVal}"

-- Test gradients for just the SECOND matmul
def secondMatmulGrad : IO Unit := do
  let result := runTensorM do
    -- h is fixed (pretend it's constant), just get grad of w2
    let h ← Tensor.full [3, 4] .float32 0.2  -- Simulate output of first matmul
    let w2 ← Tensor.full [4, 1] .float32 0.2

    let out ← matmul h w2
    let loss ← sum out

    let gradMap ← backward loss [w2.uop]

    let env : Env := ∅
    let lossVal := eval loss.uop env
    let gradW2 := gradMap[w2.uop.uid]?.map (eval · env)

    pure (lossVal, gradW2)

  let (lossVal, gradW2) := result
  IO.println "=== Second Matmul Gradient Only ==="
  IO.println s!"loss = {lossVal}"
  IO.println s!"dL/dW2 = {gradW2}"

-- Test chained matmuls with gradient for W1 only
def chainedMatmulGradW1Only : IO Unit := do
  let result := runTensorM do
    let x ← Tensor.full [3, 2] .float32 1.0
    let w1 ← Tensor.full [2, 4] .float32 0.1
    let w2 ← Tensor.full [4, 1] .float32 0.2

    let h ← matmul x w1
    let out ← matmul h w2
    let loss ← sum out

    -- Only get gradient for W1 (which requires backprop through second matmul)
    let gradMap ← backward loss [w1.uop]

    let env : Env := ∅
    let lossVal := eval loss.uop env
    let gradW1 := gradMap[w1.uop.uid]?.map (eval · env)

    pure (lossVal, gradW1)

  let (lossVal, gradW1) := result
  IO.println "=== Chained Matmul Gradient (W1 only) ==="
  IO.println s!"loss = {lossVal}"
  IO.println s!"dL/dW1 = {gradW1}"

end TinyGrad4.Test.MLP

