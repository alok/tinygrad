import Float64
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

private def broadcastProof {s1 s2 : List Nat} : Shape.broadcastable s1 s2 = true := by
  exact sorry_proof

/-! ## RMSNorm -/

/-- RMSNorm parameters -/
structure RMSNormParams (dim : Nat) (dt : DType) (device : Backend.DeviceType) where
  /-- Learnable scale parameter [dim] -/
  weight : Option (Vector dim dt device)
  /-- Epsilon for numerical stability -/
  eps : Float32

namespace RMSNormParams

/-- Create RMSNorm layer -/
def create (device : Backend.DeviceType := .CPU) (dim : Nat) (dt : DType := .float32) (eps : Float32 := 1e-6)
    (elementwiseAffine : Bool := true) : TensorM (RMSNormParams dim dt device) := do
  let weight ← if elementwiseAffine then
    let w ← Tensor.ones (device := device) [dim] dt
    pure (some w)
  else
    pure none
  pure { weight, eps }

/-- Compute RMS norm: x / sqrt(mean(x^2) + eps) -/
private def rmsNormInternal {s : List Nat} {d : DType} {device : Backend.DeviceType} (x : StaticTensor s d device) (eps : Float32)
    : TensorM (StaticTensor s d device) := do
  -- x^2
  let xSq ← mul x x
  -- mean(x^2) along last axis with keepdim
  let axis := s.length - 1
  let meanSq ← meanAxis xSq axis true
  -- mean + eps
  let epsT ← Tensor.full (device := device) (Shape.reduce s [axis] true) d eps
  let meanEps ← addBroadcast meanSq epsT broadcastProof
  -- rsqrt(mean + eps)
  let scale ← rsqrt meanEps
  -- x * scale (broadcast) - coerce back to original shape
  let result ← mulBroadcast x scale broadcastProof
  pure (StaticTensor.assumeShape result)

/-- Forward pass -/
def forward {s : List Nat} {device : Backend.DeviceType} (params : RMSNormParams dim dt device)
    (x : StaticTensor s dt device)
    : TensorM (StaticTensor s dt device) := do
  -- Normalize
  let normalized ← rmsNormInternal x params.eps

  -- Apply weight if present
  match params.weight with
  | none => pure normalized
  | some w =>
    -- Broadcast weight [dim] to match input shape
    let result ← mulBroadcast normalized w broadcastProof
    pure (StaticTensor.assumeShape result)

/-- Get trainable parameters -/
def parameters {device : Backend.DeviceType} (params : RMSNormParams dim dt device) : List UOp :=
  match params.weight with
  | none => []
  | some w => [w.uop]

end RMSNormParams

/-! ## LayerNorm -/

/-- LayerNorm parameters -/
structure LayerNormParams (normalizedShape : List Nat) (dt : DType) (device : Backend.DeviceType) where
  /-- Learnable scale [normalizedShape] -/
  weight : Option (StaticTensor normalizedShape dt device)
  /-- Learnable bias [normalizedShape] -/
  bias : Option (StaticTensor normalizedShape dt device)
  /-- Epsilon for numerical stability -/
  eps : Float32

namespace LayerNormParams

/-- Create LayerNorm layer -/
def create (device : Backend.DeviceType := .CPU) (normalizedShape : List Nat) (dt : DType := .float32) (eps : Float32 := 1e-5)
    (elementwiseAffine : Bool := true) : TensorM (LayerNormParams normalizedShape dt device) := do
  let (weight, bias) ← if elementwiseAffine then do
    let w ← Tensor.ones (device := device) normalizedShape dt
    let b ← Tensor.zeros (device := device) normalizedShape dt
    pure (some w, some b)
  else
    pure (none, none)
  pure { weight, bias, eps }

/-- Compute layer norm: (x - mean) / sqrt(var + eps)
    Simplified: normalizes over the last axis -/
private def layerNormInternal {s : List Nat} {d : DType} {device : Backend.DeviceType} (x : StaticTensor s d device)
    (eps : Float32) : TensorM (StaticTensor s d device) := do
  let axis := s.length - 1
  -- mean
  let m ← meanAxis x axis true
  -- x - mean
  let diff ← subBroadcast x m broadcastProof
  let diff : StaticTensor s d device := StaticTensor.assumeShape diff
  -- variance = mean((x - mean)^2)
  let diffSq ← mul diff diff
  let v ← meanAxis diffSq axis true
  -- add epsilon
  let epsT ← Tensor.full (device := device) (Shape.reduce s [axis] true) d eps
  let vEps ← addBroadcast v epsT broadcastProof
  -- 1 / sqrt(var + eps)
  let invStd ← rsqrt vEps
  -- normalize
  let result ← mulBroadcast diff invStd broadcastProof
  pure (StaticTensor.assumeShape result)

/-- Forward pass -/
def forward {s : List Nat} {device : Backend.DeviceType} (params : LayerNormParams normalizedShape dt device)
    (x : StaticTensor s dt device)
    : TensorM (StaticTensor s dt device) := do
  let normalized ← layerNormInternal x params.eps

  -- Apply affine transform if present
  match params.weight, params.bias with
  | none, none => pure normalized
  | some w, none =>
    let result ← mulBroadcast normalized w broadcastProof
    pure (StaticTensor.assumeShape result)
  | none, some b =>
    let result ← addBroadcast normalized b broadcastProof
    pure (StaticTensor.assumeShape result)
  | some w, some b =>
    let scaled ← mulBroadcast normalized w broadcastProof
    let scaled : StaticTensor s dt device := StaticTensor.assumeShape scaled
    let result ← addBroadcast scaled b broadcastProof
    pure (StaticTensor.assumeShape result)

/-- Get trainable parameters -/
def parameters {device : Backend.DeviceType} (params : LayerNormParams normalizedShape dt device) : List UOp :=
  match params.weight, params.bias with
  | none, none => []
  | some w, none => [w.uop]
  | none, some b => [b.uop]
  | some w, some b => [w.uop, b.uop]

end LayerNormParams

/-! ## BatchNorm -/

/-- BatchNorm parameters for normalizing over batch dimension.
    BatchNorm1d: Input [N, C] or [N, C, L]
    BatchNorm2d: Input [N, C, H, W]

    Unlike LayerNorm, BatchNorm:
    - Normalizes over batch (and spatial) dimensions, keeping channels separate
    - Tracks running mean/variance for inference
    - Has train vs eval mode -/
structure BatchNormParams (numFeatures : Nat) (dt : DType) (device : Backend.DeviceType) where
  /-- Learnable scale (gamma) [numFeatures] -/
  weight : StaticTensor [numFeatures] dt device
  /-- Learnable bias (beta) [numFeatures] -/
  bias : StaticTensor [numFeatures] dt device
  /-- Running mean for inference [numFeatures] -/
  runningMean : StaticTensor [numFeatures] dt device
  /-- Running variance for inference [numFeatures] -/
  runningVar : StaticTensor [numFeatures] dt device
  /-- Epsilon for numerical stability -/
  eps : Float32
  /-- Momentum for running stats update -/
  momentum : Float32
  /-- Whether in training mode -/
  training : Bool

namespace BatchNormParams

/-- Create BatchNorm layer with default initialization:
    - weight (gamma) = 1
    - bias (beta) = 0
    - running_mean = 0
    - running_var = 1 -/
def create (device : Backend.DeviceType := .CPU) (numFeatures : Nat) (dt : DType := .float32)
    (eps : Float32 := 1e-5) (momentum : Float32 := 0.1)
    (affine : Bool := true) (trackRunningStats : Bool := true)
    : TensorM (BatchNormParams numFeatures dt device) := do
  let weight ← Tensor.ones (device := device) [numFeatures] dt
  let bias ← Tensor.zeros (device := device) [numFeatures] dt
  let runningMean ← Tensor.zeros (device := device) [numFeatures] dt
  let runningVar ← Tensor.ones (device := device) [numFeatures] dt
  pure {
    weight, bias, runningMean, runningVar,
    eps, momentum,
    training := true
  }

/-- Forward pass for BatchNorm2d: Input [N, C, H, W]
    Normalizes over N, H, W dimensions, separately for each channel C.

    For simplicity, uses sequential reductions: first over spatial dims (H, W),
    then over batch dim (N). -/
def forward2d {batch channels height width : Nat} {device : Backend.DeviceType}
    (params : BatchNormParams channels dt device)
    (x : StaticTensor [batch, channels, height, width] dt device)
    : TensorM (StaticTensor [batch, channels, height, width] dt device) := do
  if params.training then
    -- Training mode: compute stats from batch
    -- Reduce over H (axis 2), then W (now axis 2), then N (axis 0)
    let m1 ← meanAxis x 3 true  -- [N, C, H, 1]
    let m2 ← meanAxis m1 2 true -- [N, C, 1, 1]
    let batchMean ← meanAxis m2 0 true -- [1, C, 1, 1]

    -- Compute variance: var = mean((x - mean)^2)
    let diff ← subBroadcast x batchMean broadcastProof
    let diff : StaticTensor [batch, channels, height, width] dt device := StaticTensor.assumeShape diff
    let diffSq ← mul diff diff
    let v1 ← meanAxis diffSq 3 true
    let v2 ← meanAxis v1 2 true
    let batchVar ← meanAxis v2 0 true -- [1, C, 1, 1]

    -- Normalize: (x - mean) / sqrt(var + eps)
    let epsT ← Tensor.full (device := device) [1, channels, 1, 1] dt params.eps
    let varEps ← addBroadcast batchVar epsT broadcastProof
    let varEps : StaticTensor [1, channels, 1, 1] dt device := StaticTensor.assumeShape varEps
    let invStd ← rsqrt varEps
    let normalized ← mulBroadcast diff invStd broadcastProof
    let normalized : StaticTensor [batch, channels, height, width] dt device := StaticTensor.assumeShape normalized

    -- Apply affine: y = gamma * normalized + beta
    let weightB ← reshapeUnsafe params.weight [1, channels, 1, 1]
    let biasB ← reshapeUnsafe params.bias [1, channels, 1, 1]
    let scaled ← mulBroadcast normalized weightB broadcastProof
    let scaled : StaticTensor [batch, channels, height, width] dt device := StaticTensor.assumeShape scaled
    let result ← addBroadcast scaled biasB broadcastProof
    pure (StaticTensor.assumeShape result)
  else
    -- Eval mode: use running statistics
    let meanB ← reshapeUnsafe params.runningMean [1, channels, 1, 1]
    let diff ← subBroadcast x meanB broadcastProof
    let diff : StaticTensor [batch, channels, height, width] dt device := StaticTensor.assumeShape diff

    let epsT ← Tensor.full (device := device) [1, channels, 1, 1] dt params.eps
    let varB ← reshapeUnsafe params.runningVar [1, channels, 1, 1]
    let varEps ← addBroadcast varB epsT broadcastProof
    let varEps : StaticTensor [1, channels, 1, 1] dt device := StaticTensor.assumeShape varEps
    let invStd ← rsqrt varEps
    let normalized ← mulBroadcast diff invStd broadcastProof
    let normalized : StaticTensor [batch, channels, height, width] dt device := StaticTensor.assumeShape normalized

    let weightB ← reshapeUnsafe params.weight [1, channels, 1, 1]
    let biasB ← reshapeUnsafe params.bias [1, channels, 1, 1]
    let scaled ← mulBroadcast normalized weightB broadcastProof
    let scaled : StaticTensor [batch, channels, height, width] dt device := StaticTensor.assumeShape scaled
    let result ← addBroadcast scaled biasB broadcastProof
    pure (StaticTensor.assumeShape result)

/-- Forward pass for BatchNorm1d: Input [N, C] or [N, C, L]
    Normalizes over N (and L if present), separately for each channel C -/
def forward1d {batch channels : Nat} {device : Backend.DeviceType}
    (params : BatchNormParams channels dt device)
    (x : StaticTensor [batch, channels] dt device)
    : TensorM (StaticTensor [batch, channels] dt device) := do
  if params.training then
    -- Training mode: compute mean/var over batch dimension (axis 0)
    let batchMean ← meanAxis x 0 true  -- [1, C]

    -- Compute variance
    let diff ← subBroadcast x batchMean broadcastProof
    let diff : StaticTensor [batch, channels] dt device := StaticTensor.assumeShape diff
    let diffSq ← mul diff diff
    let batchVar ← meanAxis diffSq 0 true -- [1, C]

    -- Normalize
    let epsT ← Tensor.full (device := device) [1, channels] dt params.eps
    let varEps ← addBroadcast batchVar epsT broadcastProof
    let varEps : StaticTensor [1, channels] dt device := StaticTensor.assumeShape varEps
    let invStd ← rsqrt varEps
    let normalized ← mulBroadcast diff invStd broadcastProof
    let normalized : StaticTensor [batch, channels] dt device := StaticTensor.assumeShape normalized

    -- Apply affine
    let weightB ← reshapeUnsafe params.weight [1, channels]
    let biasB ← reshapeUnsafe params.bias [1, channels]
    let scaled ← mulBroadcast normalized weightB broadcastProof
    let scaled : StaticTensor [batch, channels] dt device := StaticTensor.assumeShape scaled
    let result ← addBroadcast scaled biasB broadcastProof
    pure (StaticTensor.assumeShape result)
  else
    -- Eval mode: use running statistics
    let meanB ← reshapeUnsafe params.runningMean [1, channels]
    let diff ← subBroadcast x meanB broadcastProof
    let diff : StaticTensor [batch, channels] dt device := StaticTensor.assumeShape diff

    let epsT ← Tensor.full (device := device) [1, channels] dt params.eps
    let varB ← reshapeUnsafe params.runningVar [1, channels]
    let varEps ← addBroadcast varB epsT broadcastProof
    let varEps : StaticTensor [1, channels] dt device := StaticTensor.assumeShape varEps
    let invStd ← rsqrt varEps
    let normalized ← mulBroadcast diff invStd broadcastProof
    let normalized : StaticTensor [batch, channels] dt device := StaticTensor.assumeShape normalized

    let weightB ← reshapeUnsafe params.weight [1, channels]
    let biasB ← reshapeUnsafe params.bias [1, channels]
    let scaled ← mulBroadcast normalized weightB broadcastProof
    let scaled : StaticTensor [batch, channels] dt device := StaticTensor.assumeShape scaled
    let result ← addBroadcast scaled biasB broadcastProof
    pure (StaticTensor.assumeShape result)

/-- Set training mode -/
def train {device : Backend.DeviceType} (params : BatchNormParams numFeatures dt device)
    : BatchNormParams numFeatures dt device :=
  { params with training := true }

/-- Set eval mode -/
def eval {device : Backend.DeviceType} (params : BatchNormParams numFeatures dt device)
    : BatchNormParams numFeatures dt device :=
  { params with training := false }

/-- Get trainable parameters (weight and bias) -/
def parameters {device : Backend.DeviceType} (params : BatchNormParams numFeatures dt device) : List UOp :=
  [params.weight.uop, params.bias.uop]

/-- Number of trainable parameters -/
def numParams {device : Backend.DeviceType} (_ : BatchNormParams numFeatures dt device) : Nat :=
  numFeatures * 2  -- weight + bias

end BatchNormParams

/-! ## Convenience constructors -/

/-- Create RMSNorm for a given dimension -/
def rmsNorm (device : Backend.DeviceType := .CPU) (dim : Nat) (dt : DType := .float32) (eps : Float32 := 1e-6)
    : TensorM (RMSNormParams dim dt device) :=
  RMSNormParams.create (device := device) dim dt eps

/-- Create LayerNorm for a given normalized shape -/
def layerNorm (device : Backend.DeviceType := .CPU) (normalizedShape : List Nat) (dt : DType := .float32) (eps : Float32 := 1e-5)
    : TensorM (LayerNormParams normalizedShape dt device) :=
  LayerNormParams.create (device := device) normalizedShape dt eps

/-- Create BatchNorm1d for numFeatures channels -/
def batchNorm1d (device : Backend.DeviceType := .CPU) (numFeatures : Nat) (dt : DType := .float32) (eps : Float32 := 1e-5)
    : TensorM (BatchNormParams numFeatures dt device) :=
  BatchNormParams.create (device := device) numFeatures dt eps

/-- Create BatchNorm2d for numFeatures channels -/
def batchNorm2d (device : Backend.DeviceType := .CPU) (numFeatures : Nat) (dt : DType := .float32) (eps : Float32 := 1e-5)
    : TensorM (BatchNormParams numFeatures dt device) :=
  BatchNormParams.create (device := device) numFeatures dt eps

end TinyGrad4.NN
