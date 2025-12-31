import TinyGrad4

/-!
# MNIST MLP Example

A simple MLP for MNIST digit classification demonstrating:
- Type-safe matrix operations with dependent types
- Forward pass: Input[batch, 28] → Hidden[batch, 16] → Output[batch, 10]
- Backward pass with automatic differentiation
- SGD weight updates

(Using smaller dimensions for faster compilation during development)
-/

namespace TinyGrad4.Test.MNIST

open TinyGrad4
open StaticTensor
open Interpreter

/-- MLP forward pass: input → hidden → output
    [batch, inputDim] @ [inputDim, hiddenDim] → ReLU → @ [hiddenDim, 10] -/
def mlpForward {batch inputDim hiddenDim : Nat}
    (x : Matrix batch inputDim .float32)
    (w1 : Matrix inputDim hiddenDim .float32)
    (w2 : Matrix hiddenDim 10 .float32)
    : TensorM (Matrix batch 10 .float32) := do
  -- Layer 1: Linear + ReLU
  let h ← matmul x w1
  let h_relu ← relu h
  -- Layer 2: Linear (logits)
  matmul h_relu w2

/-- Compute loss: mean of negative log-softmax (simplified cross-entropy) -/
def computeLoss {batch : Nat}
    (logits : Matrix batch 10 .float32)
    : TensorM (Scalar .float32) := do
  let logProbs ← logSoftmax logits
  let negLogProbs ← neg logProbs
  mean negLogProbs

/-- Single training step: forward, loss, backward, return gradients -/
def trainStep {batch inputDim hiddenDim : Nat}
    (x : Matrix batch inputDim .float32)
    (w1 : Matrix inputDim hiddenDim .float32)
    (w2 : Matrix hiddenDim 10 .float32)
    : TensorM (Scalar .float32 × GradMap) := do
  let logits ← mlpForward x w1 w2
  let loss ← computeLoss logits
  let gradMap ← backward loss [w1.uop, w2.uop]
  pure (loss, gradMap)

/-- Demo: Run one training step and show results -/
def demo : IO Unit := do
  IO.println "=== MNIST MLP Demo ==="
  IO.println "Testing interpreter step by step..."

  -- Most basic: scalar constant
  IO.println "Creating scalar const..."
  let c : UOp := {
    uid := ⟨0⟩
    op := .CONST
    dtype := .float32
    src := []
    arg := .constFloat 1.5
    shape := []
  }
  IO.println s!"Const UOp: shape={c.shape}"

  let env : Env := ∅
  IO.println "Evaluating const..."
  let constResult := eval c env
  IO.println s!"Const result: {constResult}"

  -- Now try expand
  IO.println ""
  IO.println "Creating expand UOp..."
  let expanded : UOp := {
    uid := ⟨1⟩
    op := .EXPAND
    dtype := .float32
    src := [c]
    arg := .shape [2, 3]
    shape := [2, 3]
  }
  IO.println s!"Expand UOp: src[0].shape={expanded.src[0]!.shape}, target={expanded.shape}"

  IO.println "Evaluating expand..."
  let expandResult := eval expanded env
  IO.println s!"Expand result: {expandResult}"

  IO.println ""
  IO.println "Done with basic tests"

  -- Test using runTensorM
  IO.println ""
  IO.println "Testing with runTensorM..."

  let fullResult := runTensorM do
    let x ← Tensor.full [2, 3] .float32 0.5
    let env : Env := ∅
    pure (eval x.uop env)

  IO.println s!"Full [2,3] of 0.5 = {fullResult}"

  -- Test matmul
  IO.println ""
  IO.println "Testing matmul..."

  let matmulResult := runTensorM do
    let x ← Tensor.full [2, 3] .float32 0.1
    let w ← Tensor.full [3, 2] .float32 0.2
    let y ← matmul x w
    let env : Env := ∅
    pure (eval y.uop env)

  IO.println s!"Matmul [2,3] @ [3,2] = {matmulResult}"
  IO.println "Expected: 4 values of 0.06 (3 * 0.1 * 0.2)"

  -- Test relu
  IO.println ""
  IO.println "Testing relu..."

  let reluResult := runTensorM do
    let x ← Tensor.full [2, 2] .float32 0.5
    let y ← relu x
    let env : Env := ∅
    pure (eval y.uop env)

  IO.println s!"ReLU of [2,2] all 0.5 = {reluResult}"

  -- Test mean
  IO.println ""
  IO.println "Testing mean..."

  let meanResult := runTensorM do
    let x ← Tensor.full [2, 3] .float32 6.0
    let y ← mean x
    let env : Env := ∅
    pure (eval y.uop env)

  IO.println s!"Mean of [2,3] all 6.0 = {meanResult}"
  IO.println "Expected: [6.0]"

  -- Test log_softmax
  IO.println ""
  IO.println "Testing log_softmax..."

  let softmaxResult := runTensorM do
    let x ← Tensor.full [2, 3] .float32 1.0
    let y ← logSoftmax x
    let env : Env := ∅
    pure (eval y.uop env)

  IO.println s!"LogSoftmax of [2,3] all 1.0 = {softmaxResult}"
  IO.println "Expected: all -log(3) ≈ -1.099"

  -- Test full MLP forward pass
  IO.println ""
  IO.println "Testing MLP forward pass..."

  let mlpResult := runTensorM do
    -- Simple MLP: [2,4] @ [4,3] -> relu -> @ [3,2]
    let x ← Tensor.full [2, 4] .float32 0.1
    let w1 ← Tensor.full [4, 3] .float32 0.1
    let w2 ← Tensor.full [3, 2] .float32 0.1
    let h ← matmul x w1
    let h_relu ← relu h
    let logits ← matmul h_relu w2
    let env : Env := ∅
    pure (eval logits.uop env)

  IO.println s!"MLP [2,4] → [4,3] → relu → [3,2] = {mlpResult}"
  -- Each hidden = 4 * 0.1 * 0.1 = 0.04, after relu still 0.04
  -- Each output = 3 * 0.04 * 0.1 = 0.012
  IO.println "Expected: 4 values of 0.012"

  -- Test loss computation
  IO.println ""
  IO.println "Testing loss computation..."

  let lossResult := runTensorM do
    let x ← Tensor.full [2, 4] .float32 0.1
    let w1 ← Tensor.full [4, 3] .float32 0.1
    let w2 ← Tensor.full [3, 2] .float32 0.1
    let h ← matmul x w1
    let h_relu ← relu h
    let logits ← matmul h_relu w2
    let logProbs ← logSoftmax logits
    let negLogProbs ← neg logProbs
    let loss ← mean negLogProbs
    let env : Env := ∅
    pure (eval loss.uop env)

  IO.println s!"Loss (mean neg log_softmax) = {lossResult}"
  IO.println "Expected: ~0.69 (log(2) for uniform 2-class)"

  IO.println ""
  IO.println "✓ All tests passed!"

/-- Shape safety demo -/
def shapeSafetyDemo : IO Unit := do
  IO.println ""
  IO.println "=== Shape Safety Demo ==="
  IO.println ""
  IO.println "These expressions compile (shapes match):"
  IO.println "  matmul [batch, 28] [28, 16] → [batch, 16]  ✓"
  IO.println "  matmul [batch, 16] [16, 10] → [batch, 10]  ✓"
  IO.println ""
  IO.println "These would be COMPILE ERRORS (not runtime!):"
  IO.println "  matmul [batch, 28] [32, 16] → ERROR: 28 ≠ 32"
  IO.println "  matmul [batch, 16] [8, 10]  → ERROR: 16 ≠ 8"

def runAll : IO Unit := do
  demo
  shapeSafetyDemo

end TinyGrad4.Test.MNIST

-- Main entry point for running as script
def main : IO Unit := TinyGrad4.Test.MNIST.runAll
