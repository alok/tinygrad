import TinyGrad4.Tensor.Tensor
import TinyGrad4.Tensor.Math
import TinyGrad4.Tensor.Movement

/-!
# Normalization Layers

Implements:
- RMSNorm: Root Mean Square Layer Normalization
- LayerNorm: Layer Normalization

Mirrors tinygrad's nn.RMSNorm and nn.LayerNorm.
-/

namespace TinyGrad4.NN

open TinyGrad4
open StaticTensor

/-! ## RMSNorm -/

/-- RMSNorm parameters -/
structure RMSNormParams (dim : Nat) (dt : DType) where
  /-- Learnable scale parameter [dim] -/
  weight : Option (Vector dim dt)
  /-- Epsilon for numerical stability -/
  eps : Float32

namespace RMSNormParams

/-- Create RMSNorm layer -/
def create (dim : Nat) (dt : DType := .float32) (eps : Float32 := 1e-6)
    (elementwiseAffine : Bool := true) : TensorM (RMSNormParams dim dt) := do
  let weight ← if elementwiseAffine then
    let w ← Tensor.ones [dim] dt
    pure (some w)
  else
    pure none
  pure { weight, eps }

/-- Coerce tensor to target shape (uses sorry_proof for shape equality) -/
private def coerceShape {s1 s2 : List Nat} {d : DType}
    (t : StaticTensor s1 d) : StaticTensor s2 d :=
  { uop := t.uop, h_shape := sorry_proof, requiresGrad := t.requiresGrad }

/-- Compute RMS norm: x / sqrt(mean(x^2) + eps) -/
private def rmsNormInternal {s : List Nat} {d : DType} (x : StaticTensor s d) (eps : Float32)
    : TensorM (StaticTensor s d) := do
  -- x^2
  let xSq ← mul x x
  -- mean(x^2) along last axis with keepdim
  let axis := s.length - 1
  let meanSq ← meanAxis xSq axis true
  -- mean + eps
  let epsT ← Tensor.full (Shape.reduce s [axis] true) d eps
  let meanEps ← addB meanSq epsT
  -- rsqrt(mean + eps)
  let scale ← rsqrt meanEps
  -- x * scale (broadcast) - coerce back to original shape
  let result ← mulB x scale
  pure (coerceShape result)

/-- Forward pass -/
def forward {s : List Nat} (params : RMSNormParams dim dt)
    (x : StaticTensor s dt)
    : TensorM (StaticTensor s dt) := do
  -- Normalize
  let normalized ← rmsNormInternal x params.eps

  -- Apply weight if present
  match params.weight with
  | none => pure normalized
  | some w =>
    -- Broadcast weight [dim] to match input shape
    let result ← mulB normalized w
    pure (coerceShape result)

/-- Get trainable parameters -/
def parameters (params : RMSNormParams dim dt) : List UOp :=
  match params.weight with
  | none => []
  | some w => [w.uop]

end RMSNormParams

/-! ## LayerNorm -/

/-- LayerNorm parameters -/
structure LayerNormParams (normalizedShape : List Nat) (dt : DType) where
  /-- Learnable scale [normalizedShape] -/
  weight : Option (StaticTensor normalizedShape dt)
  /-- Learnable bias [normalizedShape] -/
  bias : Option (StaticTensor normalizedShape dt)
  /-- Epsilon for numerical stability -/
  eps : Float32

namespace LayerNormParams

/-- Create LayerNorm layer -/
def create (normalizedShape : List Nat) (dt : DType := .float32) (eps : Float32 := 1e-5)
    (elementwiseAffine : Bool := true) : TensorM (LayerNormParams normalizedShape dt) := do
  let (weight, bias) ← if elementwiseAffine then do
    let w ← Tensor.ones normalizedShape dt
    let b ← Tensor.zeros normalizedShape dt
    pure (some w, some b)
  else
    pure (none, none)
  pure { weight, bias, eps }

/-- Coerce tensor to target shape (uses sorry_proof for shape equality) -/
private def coerceShape {s1 s2 : List Nat} {d : DType}
    (t : StaticTensor s1 d) : StaticTensor s2 d :=
  { uop := t.uop, h_shape := sorry_proof, requiresGrad := t.requiresGrad }

/-- Compute layer norm: (x - mean) / sqrt(var + eps)
    Simplified: normalizes over the last axis -/
private def layerNormInternal {s : List Nat} {d : DType} (x : StaticTensor s d)
    (eps : Float32) : TensorM (StaticTensor s d) := do
  let axis := s.length - 1
  -- mean
  let m ← meanAxis x axis true
  -- x - mean
  let diff ← subB x m
  let diff : StaticTensor s d := coerceShape diff
  -- variance = mean((x - mean)^2)
  let diffSq ← mul diff diff
  let v ← meanAxis diffSq axis true
  -- add epsilon
  let epsT ← Tensor.full (Shape.reduce s [axis] true) d eps
  let vEps ← addB v epsT
  -- 1 / sqrt(var + eps)
  let invStd ← rsqrt vEps
  -- normalize
  let result ← mulB diff invStd
  pure (coerceShape result)

/-- Forward pass -/
def forward {s : List Nat} (params : LayerNormParams normalizedShape dt)
    (x : StaticTensor s dt)
    : TensorM (StaticTensor s dt) := do
  let normalized ← layerNormInternal x params.eps

  -- Apply affine transform if present
  match params.weight, params.bias with
  | none, none => pure normalized
  | some w, none =>
    let result ← mulB normalized w
    pure (coerceShape result)
  | none, some b =>
    let result ← addB normalized b
    pure (coerceShape result)
  | some w, some b =>
    let scaled ← mulB normalized w
    let scaled : StaticTensor s dt := coerceShape scaled
    let result ← addB scaled b
    pure (coerceShape result)

/-- Get trainable parameters -/
def parameters (params : LayerNormParams normalizedShape dt) : List UOp :=
  match params.weight, params.bias with
  | none, none => []
  | some w, none => [w.uop]
  | none, some b => [b.uop]
  | some w, some b => [w.uop, b.uop]

end LayerNormParams

/-! ## Convenience constructors -/

/-- Create RMSNorm for a given dimension -/
def rmsNorm (dim : Nat) (dt : DType := .float32) (eps : Float32 := 1e-6)
    : TensorM (RMSNormParams dim dt) :=
  RMSNormParams.create dim dt eps

/-- Create LayerNorm for a given normalized shape -/
def layerNorm (normalizedShape : List Nat) (dt : DType := .float32) (eps : Float32 := 1e-5)
    : TensorM (LayerNormParams normalizedShape dt) :=
  LayerNormParams.create normalizedShape dt eps

end TinyGrad4.NN
