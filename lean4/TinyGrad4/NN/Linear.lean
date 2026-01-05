import TinyGrad4.Tensor.Tensor
import TinyGrad4.Tensor.Math
import TinyGrad4.Tensor.Movement

/-!
# Linear Layer

Applies a linear transformation: y = x @ W^T + b

Mirrors tinygrad's `nn.Linear` with Kaiming uniform initialization.
-/

namespace TinyGrad4.NN

open TinyGrad4
open StaticTensor

/-- Linear layer parameters -/
structure LinearParams (inFeatures outFeatures : Nat) (dt : DType) where
  /-- Weight matrix {lit}`[outFeatures, inFeatures]` (stored transposed for efficient matmul). -/
  weight : Matrix outFeatures inFeatures dt
  /-- Optional bias {lit}`[outFeatures]`. -/
  bias : Option (Vector outFeatures dt)

namespace LinearParams

/-- Create linear layer with Kaiming uniform initialization.
    bound = 1 / sqrt({lit}`in_features`), matching tinygrad's nn.Linear -/
def create (inFeatures outFeatures : Nat) (dt : DType := .float32) (useBias : Bool := true)
    (seed : Nat := 42) : TensorM (LinearParams inFeatures outFeatures dt) := do
  -- Kaiming bound = 1 / sqrt(in_features)
  let bound := (1.0 / Float.sqrt (Float.ofNat inFeatures)).toFloat32

  -- Weight: uniform(-bound, bound) shaped [outFeatures, inFeatures]
  let weight ← uniformInit [outFeatures, inFeatures] dt (-bound) bound seed

  -- Bias: uniform(-bound, bound) shaped [outFeatures]
  let bias ← if useBias then
    let b ← uniformInit [outFeatures] dt (-bound) bound (seed + 1)
    pure (some b)
  else
    pure none

  pure { weight, bias }

/-- Forward pass: x @ W^T + b
    Input:  {lit}`[batch, inFeatures]`
    Output: {lit}`[batch, outFeatures]` -/
def forward {batch : Nat} (params : LinearParams inFeatures outFeatures dt)
    (x : Matrix batch inFeatures dt) : TensorM (Matrix batch outFeatures dt) := do
  -- x @ W^T: [batch, in] @ [in, out] = [batch, out]
  -- We store W as [out, in], so we need W^T = [in, out]
  let wT ← T params.weight
  let y ← matmul x wT

  match params.bias with
  | none => pure y
  | some b =>
    -- Add bias with broadcasting
    let yb ← addB y b
    pure { uop := yb.uop, h_shape := sorry_proof, requiresGrad := yb.requiresGrad }

/-- Get all trainable parameters -/
def parameters (params : LinearParams inFeatures outFeatures dt)
    : List UOp :=
  match params.bias with
  | none => [params.weight.uop]
  | some b => [params.weight.uop, b.uop]

end LinearParams

/-- Convenience: Create and apply linear layer in one step -/
def linear' {batch inFeatures outFeatures : Nat} {dt : DType}
    (x : Matrix batch inFeatures dt)
    (weight : Matrix outFeatures inFeatures dt)
    (bias : Option (Vector outFeatures dt) := none)
    : TensorM (Matrix batch outFeatures dt) := do
  let wT ← T weight
  let y ← matmul x wT
  match bias with
  | none => pure y
  | some b =>
    let yb ← addB y b
    pure { uop := yb.uop, h_shape := sorry_proof, requiresGrad := yb.requiresGrad }

end TinyGrad4.NN
