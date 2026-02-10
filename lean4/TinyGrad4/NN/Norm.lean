import Float64
import TinyGrad4.Tensor.Tensor
import TinyGrad4.Tensor.Math
import TinyGrad4.Tensor.Movement

/-!
# Normalization Layers

Implements:
- RMSNorm: Root Mean Square Layer Normalization
- LayerNorm: Layer Normalization
- BatchNorm: Batch Normalization for 1D/2D inputs

This module intentionally keeps two clear layers:
- ergonomic typed APIs (`Matrix`, `Vector`, `StaticTensor` with shapes in types)
- explicit low-level broadcast constraints at each callsite
-/

namespace TinyGrad4.NN

open TinyGrad4
open StaticTensor

/-! ## RMSNorm -/

/-- RMSNorm parameters -/
structure RMSNormParams (dim : Nat) (dt : DType) (device : Backend.DeviceType) where
  /-- Learnable scale parameter `[dim]` -/
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

/-- Compute RMS norm for matrix input:
    `x / sqrt(mean(x^2, axis=1, keepdim=true) + eps)` -/
private def rmsNormInternal {batch dim : Nat} {d : DType} {device : Backend.DeviceType}
    (x : Matrix batch dim d device) (eps : Float32)
    : TensorM (Matrix batch dim d device) := do
  let xSq ← mul x x
  let meanSqRaw ← meanAxis xSq 1 true
  let meanSq : Matrix batch 1 d device := StaticTensor.assumeShape meanSqRaw
  let epsT ← Tensor.full (device := device) [batch, 1] d eps
  let meanEps ← addBroadcast meanSq epsT (Shape.broadcastable_refl [batch, 1])
  let scaleRaw ← rsqrt meanEps
  let scale : Matrix batch 1 d device := StaticTensor.assumeShape scaleRaw
  let result ← mulBroadcast x scale (Shape.broadcastable_matrix_col batch dim)
  pure (StaticTensor.assumeShape result)

/-- Forward pass for matrix input `[batch, dim]`. -/
def forward {batch : Nat} {device : Backend.DeviceType} (params : RMSNormParams dim dt device)
    (x : Matrix batch dim dt device)
    : TensorM (Matrix batch dim dt device) := do
  let normalized ← rmsNormInternal x params.eps
  match params.weight with
  | none => pure normalized
  | some w =>
    let result ← mulBroadcast normalized w (Shape.broadcastable_matrix_vector batch dim)
    pure (StaticTensor.assumeShape result)

/-- Get trainable parameters -/
def parameters {device : Backend.DeviceType} (params : RMSNormParams dim dt device) : List UOp :=
  match params.weight with
  | none => []
  | some w => [w.uop]

end RMSNormParams

/-! ## LayerNorm -/

/-- LayerNorm parameters (normalizing over the feature axis). -/
structure LayerNormParams (dim : Nat) (dt : DType) (device : Backend.DeviceType) where
  /-- Learnable scale `[dim]` -/
  weight : Option (Vector dim dt device)
  /-- Learnable bias `[dim]` -/
  bias : Option (Vector dim dt device)
  /-- Epsilon for numerical stability -/
  eps : Float32

namespace LayerNormParams

/-- Create LayerNorm layer for matrix features of width `dim`. -/
def create (device : Backend.DeviceType := .CPU) (dim : Nat) (dt : DType := .float32) (eps : Float32 := 1e-5)
    (elementwiseAffine : Bool := true) : TensorM (LayerNormParams dim dt device) := do
  let (weight, bias) ← if elementwiseAffine then do
    let w ← Tensor.ones (device := device) [dim] dt
    let b ← Tensor.zeros (device := device) [dim] dt
    pure (some w, some b)
  else
    pure (none, none)
  pure { weight, bias, eps }

/-- Compute layer norm for matrix input:
    `(x - mean) / sqrt(var + eps)` along axis 1. -/
private def layerNormInternal {batch dim : Nat} {d : DType} {device : Backend.DeviceType}
    (x : Matrix batch dim d device) (eps : Float32)
    : TensorM (Matrix batch dim d device) := do
  let meanRaw ← meanAxis x 1 true
  let mean : Matrix batch 1 d device := StaticTensor.assumeShape meanRaw
  let diff ← subBroadcast x mean (Shape.broadcastable_matrix_col batch dim)
  let diff : Matrix batch dim d device := StaticTensor.assumeShape diff
  let diffSq ← mul diff diff
  let varianceRaw ← meanAxis diffSq 1 true
  let variance : Matrix batch 1 d device := StaticTensor.assumeShape varianceRaw
  let epsT ← Tensor.full (device := device) [batch, 1] d eps
  let varEps ← addBroadcast variance epsT (Shape.broadcastable_refl [batch, 1])
  let invStdRaw ← rsqrt varEps
  let invStd : Matrix batch 1 d device := StaticTensor.assumeShape invStdRaw
  let result ← mulBroadcast diff invStd (Shape.broadcastable_matrix_col batch dim)
  pure (StaticTensor.assumeShape result)

/-- Forward pass for matrix input `[batch, dim]`. -/
def forward {batch : Nat} {device : Backend.DeviceType} (params : LayerNormParams dim dt device)
    (x : Matrix batch dim dt device)
    : TensorM (Matrix batch dim dt device) := do
  let normalized ← layerNormInternal x params.eps
  match params.weight, params.bias with
  | none, none => pure normalized
  | some w, none =>
    let result ← mulBroadcast normalized w (Shape.broadcastable_matrix_vector batch dim)
    pure (StaticTensor.assumeShape result)
  | none, some b =>
    let result ← addBroadcast normalized b (Shape.broadcastable_matrix_vector batch dim)
    pure (StaticTensor.assumeShape result)
  | some w, some b =>
    let scaled ← mulBroadcast normalized w (Shape.broadcastable_matrix_vector batch dim)
    let scaled : Matrix batch dim dt device := StaticTensor.assumeShape scaled
    let result ← addBroadcast scaled b (Shape.broadcastable_matrix_vector batch dim)
    pure (StaticTensor.assumeShape result)

/-- Get trainable parameters -/
def parameters {device : Backend.DeviceType} (params : LayerNormParams dim dt device) : List UOp :=
  match params.weight, params.bias with
  | none, none => []
  | some w, none => [w.uop]
  | none, some b => [b.uop]
  | some w, some b => [w.uop, b.uop]

end LayerNormParams

/-! ## BatchNorm -/

/-- BatchNorm parameters for normalizing over batch dimensions.
    BatchNorm1d input: `[N, C]`
    BatchNorm2d input: `[N, C, H, W]` -/
structure BatchNormParams (numFeatures : Nat) (dt : DType) (device : Backend.DeviceType) where
  /-- Learnable scale (gamma) `[numFeatures]` -/
  weight : StaticTensor [numFeatures] dt device
  /-- Learnable bias (beta) `[numFeatures]` -/
  bias : StaticTensor [numFeatures] dt device
  /-- Running mean for inference `[numFeatures]` -/
  runningMean : StaticTensor [numFeatures] dt device
  /-- Running variance for inference `[numFeatures]` -/
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
    - running mean = 0
    - running var = 1 -/
def create (device : Backend.DeviceType := .CPU) (numFeatures : Nat) (dt : DType := .float32)
    (eps : Float32 := 1e-5) (momentum : Float32 := 0.1)
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

/-- Forward pass for BatchNorm2d: input `[N, C, H, W]`. -/
def forward2d {batch channels height width : Nat} {device : Backend.DeviceType}
    (params : BatchNormParams channels dt device)
    (x : StaticTensor [batch, channels, height, width] dt device)
    : TensorM (StaticTensor [batch, channels, height, width] dt device) := do
  if params.training then
    let m1Raw ← meanAxis x 3 true
    let m1 : StaticTensor [batch, channels, height, 1] dt device := StaticTensor.assumeShape m1Raw
    let m2Raw ← meanAxis m1 2 true
    let m2 : StaticTensor [batch, channels, 1, 1] dt device := StaticTensor.assumeShape m2Raw
    let batchMeanRaw ← meanAxis m2 0 true
    let batchMean : StaticTensor [1, channels, 1, 1] dt device := StaticTensor.assumeShape batchMeanRaw

    let diff ← subBroadcast x batchMean (Shape.broadcastable_nchw_channel batch channels height width)
    let diff : StaticTensor [batch, channels, height, width] dt device := StaticTensor.assumeShape diff
    let diffSq ← mul diff diff
    let v1Raw ← meanAxis diffSq 3 true
    let v1 : StaticTensor [batch, channels, height, 1] dt device := StaticTensor.assumeShape v1Raw
    let v2Raw ← meanAxis v1 2 true
    let v2 : StaticTensor [batch, channels, 1, 1] dt device := StaticTensor.assumeShape v2Raw
    let batchVarRaw ← meanAxis v2 0 true
    let batchVar : StaticTensor [1, channels, 1, 1] dt device := StaticTensor.assumeShape batchVarRaw

    let epsT ← Tensor.full (device := device) [1, channels, 1, 1] dt params.eps
    let varEps ← addBroadcast batchVar epsT (Shape.broadcastable_refl [1, channels, 1, 1])
    let varEps : StaticTensor [1, channels, 1, 1] dt device := StaticTensor.assumeShape varEps
    let invStd ← rsqrt varEps
    let normalized ← mulBroadcast diff invStd (Shape.broadcastable_nchw_channel batch channels height width)
    let normalized : StaticTensor [batch, channels, height, width] dt device := StaticTensor.assumeShape normalized

    let weightB ← reshapeUnsafe params.weight [1, channels, 1, 1]
    let biasB ← reshapeUnsafe params.bias [1, channels, 1, 1]
    let scaled ← mulBroadcast normalized weightB (Shape.broadcastable_nchw_channel batch channels height width)
    let scaled : StaticTensor [batch, channels, height, width] dt device := StaticTensor.assumeShape scaled
    let result ← addBroadcast scaled biasB (Shape.broadcastable_nchw_channel batch channels height width)
    pure (StaticTensor.assumeShape result)
  else
    let meanB ← reshapeUnsafe params.runningMean [1, channels, 1, 1]
    let diff ← subBroadcast x meanB (Shape.broadcastable_nchw_channel batch channels height width)
    let diff : StaticTensor [batch, channels, height, width] dt device := StaticTensor.assumeShape diff

    let epsT ← Tensor.full (device := device) [1, channels, 1, 1] dt params.eps
    let varB ← reshapeUnsafe params.runningVar [1, channels, 1, 1]
    let varEps ← addBroadcast varB epsT (Shape.broadcastable_refl [1, channels, 1, 1])
    let varEps : StaticTensor [1, channels, 1, 1] dt device := StaticTensor.assumeShape varEps
    let invStd ← rsqrt varEps
    let normalized ← mulBroadcast diff invStd (Shape.broadcastable_nchw_channel batch channels height width)
    let normalized : StaticTensor [batch, channels, height, width] dt device := StaticTensor.assumeShape normalized

    let weightB ← reshapeUnsafe params.weight [1, channels, 1, 1]
    let biasB ← reshapeUnsafe params.bias [1, channels, 1, 1]
    let scaled ← mulBroadcast normalized weightB (Shape.broadcastable_nchw_channel batch channels height width)
    let scaled : StaticTensor [batch, channels, height, width] dt device := StaticTensor.assumeShape scaled
    let result ← addBroadcast scaled biasB (Shape.broadcastable_nchw_channel batch channels height width)
    pure (StaticTensor.assumeShape result)

/-- Forward pass for BatchNorm1d: input `[N, C]`. -/
def forward1d {batch channels : Nat} {device : Backend.DeviceType}
    (params : BatchNormParams channels dt device)
    (x : StaticTensor [batch, channels] dt device)
    : TensorM (StaticTensor [batch, channels] dt device) := do
  if params.training then
    let batchMeanRaw ← meanAxis x 0 true
    let batchMean : StaticTensor [1, channels] dt device := StaticTensor.assumeShape batchMeanRaw
    let diff ← subBroadcast x batchMean (Shape.broadcastable_nc_channel batch channels)
    let diff : StaticTensor [batch, channels] dt device := StaticTensor.assumeShape diff
    let diffSq ← mul diff diff
    let batchVarRaw ← meanAxis diffSq 0 true
    let batchVar : StaticTensor [1, channels] dt device := StaticTensor.assumeShape batchVarRaw

    let epsT ← Tensor.full (device := device) [1, channels] dt params.eps
    let varEps ← addBroadcast batchVar epsT (Shape.broadcastable_refl [1, channels])
    let varEps : StaticTensor [1, channels] dt device := StaticTensor.assumeShape varEps
    let invStd ← rsqrt varEps
    let normalized ← mulBroadcast diff invStd (Shape.broadcastable_nc_channel batch channels)
    let normalized : StaticTensor [batch, channels] dt device := StaticTensor.assumeShape normalized

    let weightB ← reshapeUnsafe params.weight [1, channels]
    let biasB ← reshapeUnsafe params.bias [1, channels]
    let scaled ← mulBroadcast normalized weightB (Shape.broadcastable_nc_channel batch channels)
    let scaled : StaticTensor [batch, channels] dt device := StaticTensor.assumeShape scaled
    let result ← addBroadcast scaled biasB (Shape.broadcastable_nc_channel batch channels)
    pure (StaticTensor.assumeShape result)
  else
    let meanB ← reshapeUnsafe params.runningMean [1, channels]
    let diff ← subBroadcast x meanB (Shape.broadcastable_nc_channel batch channels)
    let diff : StaticTensor [batch, channels] dt device := StaticTensor.assumeShape diff

    let epsT ← Tensor.full (device := device) [1, channels] dt params.eps
    let varB ← reshapeUnsafe params.runningVar [1, channels]
    let varEps ← addBroadcast varB epsT (Shape.broadcastable_refl [1, channels])
    let varEps : StaticTensor [1, channels] dt device := StaticTensor.assumeShape varEps
    let invStd ← rsqrt varEps
    let normalized ← mulBroadcast diff invStd (Shape.broadcastable_nc_channel batch channels)
    let normalized : StaticTensor [batch, channels] dt device := StaticTensor.assumeShape normalized

    let weightB ← reshapeUnsafe params.weight [1, channels]
    let biasB ← reshapeUnsafe params.bias [1, channels]
    let scaled ← mulBroadcast normalized weightB (Shape.broadcastable_nc_channel batch channels)
    let scaled : StaticTensor [batch, channels] dt device := StaticTensor.assumeShape scaled
    let result ← addBroadcast scaled biasB (Shape.broadcastable_nc_channel batch channels)
    pure (StaticTensor.assumeShape result)

/-- Set training mode -/
def train {device : Backend.DeviceType} (params : BatchNormParams numFeatures dt device)
    : BatchNormParams numFeatures dt device :=
  { params with training := true }

/-- Set eval mode -/
def eval {device : Backend.DeviceType} (params : BatchNormParams numFeatures dt device)
    : BatchNormParams numFeatures dt device :=
  { params with training := false }

/-- Get trainable parameters (weight and bias). -/
def parameters {device : Backend.DeviceType} (params : BatchNormParams numFeatures dt device) : List UOp :=
  [params.weight.uop, params.bias.uop]

/-- Number of trainable parameters. -/
def numParams {device : Backend.DeviceType} (_ : BatchNormParams numFeatures dt device) : Nat :=
  numFeatures * 2

end BatchNormParams

/-! ## Convenience constructors -/

/-- Create RMSNorm for a given feature dimension. -/
def rmsNorm (device : Backend.DeviceType := .CPU) (dim : Nat) (dt : DType := .float32) (eps : Float32 := 1e-6)
    : TensorM (RMSNormParams dim dt device) :=
  RMSNormParams.create (device := device) dim dt eps

/-- Create LayerNorm for matrix feature width `dim`. -/
def layerNorm (device : Backend.DeviceType := .CPU) (dim : Nat) (dt : DType := .float32) (eps : Float32 := 1e-5)
    : TensorM (LayerNormParams dim dt device) :=
  LayerNormParams.create (device := device) dim dt eps

/-- Create BatchNorm1d for `numFeatures` channels. -/
def batchNorm1d (numFeatures : Nat) (dt : DType := .float32) (device : Backend.DeviceType := .CPU) (eps : Float32 := 1e-5)
    : TensorM (BatchNormParams numFeatures dt device) :=
  BatchNormParams.create (device := device) numFeatures dt eps

/-- Create BatchNorm2d for `numFeatures` channels. -/
def batchNorm2d (numFeatures : Nat) (dt : DType := .float32) (device : Backend.DeviceType := .CPU) (eps : Float32 := 1e-5)
    : TensorM (BatchNormParams numFeatures dt device) :=
  BatchNormParams.create (device := device) numFeatures dt eps

end TinyGrad4.NN
