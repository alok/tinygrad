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
  /-- Learnable scale parameter {lit}`[dim]`. -/
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

/-- Coerce tensor to target shape (uses {lit}`sorry_proof` for shape equality). -/
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
  /-- Learnable scale {lit}`[normalizedShape]`. -/
  weight : Option (StaticTensor normalizedShape dt)
  /-- Learnable bias {lit}`[normalizedShape]`. -/
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

/-- Coerce tensor to target shape (uses {lit}`sorry_proof` for shape equality). -/
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

/-! ## BatchNorm -/

/-- BatchNorm parameters for normalizing over batch dimension.
    BatchNorm1d: Input {lit}`[N, C]` or {lit}`[N, C, L]`
    BatchNorm2d: Input {lit}`[N, C, H, W]`

    Unlike LayerNorm, BatchNorm:
    - Normalizes over batch (and spatial) dimensions, keeping channels separate
    - Tracks running mean/variance for inference
    - Has train vs eval mode -/
structure BatchNormParams (numFeatures : Nat) (dt : DType) where
  /-- Learnable scale (gamma) {lit}`[numFeatures]`. -/
  weight : StaticTensor [numFeatures] dt
  /-- Learnable bias (beta) {lit}`[numFeatures]`. -/
  bias : StaticTensor [numFeatures] dt
  /-- Running mean for inference {lit}`[numFeatures]`. -/
  runningMean : StaticTensor [numFeatures] dt
  /-- Running variance for inference {lit}`[numFeatures]`. -/
  runningVar : StaticTensor [numFeatures] dt
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
    - {lit}`running_mean = 0`
    - {lit}`running_var = 1` -/
def create (numFeatures : Nat) (dt : DType := .float32)
    (eps : Float32 := 1e-5) (momentum : Float32 := 0.1)
    (affine : Bool := true) (trackRunningStats : Bool := true)
    : TensorM (BatchNormParams numFeatures dt) := do
  let weight ← Tensor.ones [numFeatures] dt
  let bias ← Tensor.zeros [numFeatures] dt
  let runningMean ← Tensor.zeros [numFeatures] dt
  let runningVar ← Tensor.ones [numFeatures] dt
  pure {
    weight, bias, runningMean, runningVar,
    eps, momentum,
    training := true
  }

/-- Coerce tensor to target shape -/
private def coerceShape {s1 s2 : List Nat} {d : DType}
    (t : StaticTensor s1 d) : StaticTensor s2 d :=
  { uop := t.uop, h_shape := sorry_proof, requiresGrad := t.requiresGrad }

/-- Forward pass for BatchNorm2d: Input {lit}`[N, C, H, W]`
    Normalizes over N, H, W dimensions, separately for each channel C.

    For simplicity, uses sequential reductions: first over spatial dims (H, W),
    then over batch dim (N). -/
def forward2d {batch channels height width : Nat}
    (params : BatchNormParams channels dt)
    (x : StaticTensor [batch, channels, height, width] dt)
    : TensorM (StaticTensor [batch, channels, height, width] dt) := do
  if params.training then
    -- Training mode: compute stats from batch
    -- Reduce over H (axis 2), then W (now axis 2), then N (axis 0)
    let m1 ← meanAxis x 3 true  -- [N, C, H, 1]
    let m2 ← meanAxis m1 2 true -- [N, C, 1, 1]
    let batchMean ← meanAxis m2 0 true -- [1, C, 1, 1]

    -- Compute variance: var = mean((x - mean)^2)
    let diff ← subB x batchMean
    let diff : StaticTensor [batch, channels, height, width] dt := coerceShape diff
    let diffSq ← mul diff diff
    let v1 ← meanAxis diffSq 3 true
    let v2 ← meanAxis v1 2 true
    let batchVar ← meanAxis v2 0 true -- [1, C, 1, 1]

    -- Normalize: (x - mean) / sqrt(var + eps)
    let epsT ← Tensor.full [1, channels, 1, 1] dt params.eps
    let varEps ← addB batchVar epsT
    let varEps : StaticTensor [1, channels, 1, 1] dt := coerceShape varEps
    let invStd ← rsqrt varEps
    let normalized ← mulB diff invStd
    let normalized : StaticTensor [batch, channels, height, width] dt := coerceShape normalized

    -- Apply affine: y = gamma * normalized + beta
    let weightB ← reshape params.weight [1, channels, 1, 1]
    let biasB ← reshape params.bias [1, channels, 1, 1]
    let scaled ← mulB normalized weightB
    let scaled : StaticTensor [batch, channels, height, width] dt := coerceShape scaled
    let result ← addB scaled biasB
    pure (coerceShape result)
  else
    -- Eval mode: use running statistics
    let meanB ← reshape params.runningMean [1, channels, 1, 1]
    let diff ← subB x meanB
    let diff : StaticTensor [batch, channels, height, width] dt := coerceShape diff

    let epsT ← Tensor.full [1, channels, 1, 1] dt params.eps
    let varB ← reshape params.runningVar [1, channels, 1, 1]
    let varEps ← addB varB epsT
    let varEps : StaticTensor [1, channels, 1, 1] dt := coerceShape varEps
    let invStd ← rsqrt varEps
    let normalized ← mulB diff invStd
    let normalized : StaticTensor [batch, channels, height, width] dt := coerceShape normalized

    let weightB ← reshape params.weight [1, channels, 1, 1]
    let biasB ← reshape params.bias [1, channels, 1, 1]
    let scaled ← mulB normalized weightB
    let scaled : StaticTensor [batch, channels, height, width] dt := coerceShape scaled
    let result ← addB scaled biasB
    pure (coerceShape result)

/-- Forward pass for BatchNorm1d: Input {lit}`[N, C]` or {lit}`[N, C, L]`.
    Normalizes over N (and L if present), separately for each channel C. -/
def forward1d {batch channels : Nat}
    (params : BatchNormParams channels dt)
    (x : StaticTensor [batch, channels] dt)
    : TensorM (StaticTensor [batch, channels] dt) := do
  if params.training then
    -- Training mode: compute mean/var over batch dimension (axis 0)
    let batchMean ← meanAxis x 0 true  -- [1, C]

    -- Compute variance
    let diff ← subB x batchMean
    let diff : StaticTensor [batch, channels] dt := coerceShape diff
    let diffSq ← mul diff diff
    let batchVar ← meanAxis diffSq 0 true -- [1, C]

    -- Normalize
    let epsT ← Tensor.full [1, channels] dt params.eps
    let varEps ← addB batchVar epsT
    let varEps : StaticTensor [1, channels] dt := coerceShape varEps
    let invStd ← rsqrt varEps
    let normalized ← mulB diff invStd
    let normalized : StaticTensor [batch, channels] dt := coerceShape normalized

    -- Apply affine
    let weightB ← reshape params.weight [1, channels]
    let biasB ← reshape params.bias [1, channels]
    let scaled ← mulB normalized weightB
    let scaled : StaticTensor [batch, channels] dt := coerceShape scaled
    let result ← addB scaled biasB
    pure (coerceShape result)
  else
    -- Eval mode: use running statistics
    let meanB ← reshape params.runningMean [1, channels]
    let diff ← subB x meanB
    let diff : StaticTensor [batch, channels] dt := coerceShape diff

    let epsT ← Tensor.full [1, channels] dt params.eps
    let varB ← reshape params.runningVar [1, channels]
    let varEps ← addB varB epsT
    let varEps : StaticTensor [1, channels] dt := coerceShape varEps
    let invStd ← rsqrt varEps
    let normalized ← mulB diff invStd
    let normalized : StaticTensor [batch, channels] dt := coerceShape normalized

    let weightB ← reshape params.weight [1, channels]
    let biasB ← reshape params.bias [1, channels]
    let scaled ← mulB normalized weightB
    let scaled : StaticTensor [batch, channels] dt := coerceShape scaled
    let result ← addB scaled biasB
    pure (coerceShape result)

/-- Set training mode -/
def train (params : BatchNormParams numFeatures dt) : BatchNormParams numFeatures dt :=
  { params with training := true }

/-- Set eval mode -/
def eval (params : BatchNormParams numFeatures dt) : BatchNormParams numFeatures dt :=
  { params with training := false }

/-- Get trainable parameters (weight and bias) -/
def parameters (params : BatchNormParams numFeatures dt) : List UOp :=
  [params.weight.uop, params.bias.uop]

/-- Number of trainable parameters -/
def numParams (_ : BatchNormParams numFeatures dt) : Nat :=
  numFeatures * 2  -- weight + bias

end BatchNormParams

/-! ## Convenience constructors -/

/-- Create RMSNorm for a given dimension -/
def rmsNorm (dim : Nat) (dt : DType := .float32) (eps : Float32 := 1e-6)
    : TensorM (RMSNormParams dim dt) :=
  RMSNormParams.create dim dt eps

/-- Create LayerNorm for a given normalized shape -/
def layerNorm (normalizedShape : List Nat) (dt : DType := .float32) (eps : Float32 := 1e-5)
    : TensorM (LayerNormParams normalizedShape dt) :=
  LayerNormParams.create normalizedShape dt eps

/-- Create BatchNorm1d for numFeatures channels -/
def batchNorm1d (numFeatures : Nat) (dt : DType := .float32) (eps : Float32 := 1e-5)
    : TensorM (BatchNormParams numFeatures dt) :=
  BatchNormParams.create numFeatures dt eps

/-- Create BatchNorm2d for numFeatures channels -/
def batchNorm2d (numFeatures : Nat) (dt : DType := .float32) (eps : Float32 := 1e-5)
    : TensorM (BatchNormParams numFeatures dt) :=
  BatchNormParams.create numFeatures dt eps

end TinyGrad4.NN
