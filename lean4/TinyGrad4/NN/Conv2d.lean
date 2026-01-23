import Float64
import TinyGrad4.Tensor.Tensor
import TinyGrad4.Tensor.Math
import TinyGrad4.Tensor.Movement

/-!
# Conv2d Layer

Applies a 2D convolution over an input signal composed of several input planes.

Mirrors tinygrad's `nn.Conv2d` with Kaiming uniform initialization.

## Shape semantics
- Input:  [batch, in_channels, height, width]
- Weight: [out_channels, in_channels, kernel_height, kernel_width]
- Bias:   [out_channels]
- Output: [batch, out_channels, out_height, out_width]

where:
  out_height = (height + 2*padding - dilation*(kernel_height-1) - 1) / stride + 1
  out_width  = (width + 2*padding - dilation*(kernel_width-1) - 1) / stride + 1
-/

namespace TinyGrad4.NN

open TinyGrad4
open StaticTensor

/-- Conv2d layer parameters -/
structure Conv2dParams (inChannels outChannels kernelH kernelW : Nat) (dt : DType) (device : Backend.DeviceType) where
  /-- Weight tensor [outChannels, inChannels, kernelH, kernelW] -/
  weight : StaticTensor [outChannels, inChannels, kernelH, kernelW] dt device
  /-- Optional bias [outChannels] -/
  bias : Option (Vector outChannels dt device)
  /-- Padding (same on all sides) -/
  padding : Nat
  /-- Stride -/
  stride : Nat
  /-- Dilation -/
  dilation : Nat

namespace Conv2dParams

/-- Create Conv2d layer with Kaiming uniform initialization.
    bound = 1 / sqrt(in_channels * kernel_h * kernel_w), matching tinygrad's nn.Conv2d -/
def create (device : Backend.DeviceType := .CPU) (inChannels outChannels : Nat) (kernelSize : Nat := 3)
    (dt : DType := .float32) (useBias : Bool := true)
    (padding : Nat := 0) (stride : Nat := 1) (dilation : Nat := 1)
    (seed : Nat := 42) : TensorM (Conv2dParams inChannels outChannels kernelSize kernelSize dt device) := do
  -- Fan-in = in_channels * kernel_h * kernel_w
  let fanIn := inChannels * kernelSize * kernelSize
  let bound := (1.0 / Float64.sqrt (Float64.ofNat fanIn)).toFloat32

  -- Weight: uniform(-bound, bound) shaped [outChannels, inChannels, kernelSize, kernelSize]
  let weight ← uniformInit (device := device) [outChannels, inChannels, kernelSize, kernelSize] dt (-bound) bound seed

  -- Bias: uniform(-bound, bound) shaped [outChannels]
  let bias ← if useBias then
    let b ← uniformInit (device := device) [outChannels] dt (-bound) bound (seed + 1)
    pure (some b)
  else
    pure none

  pure { weight, bias, padding, stride, dilation }

/-- Create Conv2d layer with asymmetric kernel size -/
def createAsym (device : Backend.DeviceType := .CPU) (inChannels outChannels kernelH kernelW : Nat)
    (dt : DType := .float32) (useBias : Bool := true)
    (padding : Nat := 0) (stride : Nat := 1) (dilation : Nat := 1)
    (seed : Nat := 42) : TensorM (Conv2dParams inChannels outChannels kernelH kernelW dt device) := do
  let fanIn := inChannels * kernelH * kernelW
  let bound := (1.0 / Float64.sqrt (Float64.ofNat fanIn)).toFloat32

  let weight ← uniformInit (device := device) [outChannels, inChannels, kernelH, kernelW] dt (-bound) bound seed

  let bias ← if useBias then
    let b ← uniformInit (device := device) [outChannels] dt (-bound) bound (seed + 1)
    pure (some b)
  else
    pure none

  pure { weight, bias, padding, stride, dilation }

/-- Forward pass: apply convolution
    Input:  [batch, inChannels, height, width]
    Output: [batch, outChannels, outHeight, outWidth] -/
def forward {batch height width : Nat} {device : Backend.DeviceType}
    (params : Conv2dParams inChannels outChannels kernelH kernelW dt device)
    (x : StaticTensor [batch, inChannels, height, width] dt device)
    : TensorM (StaticTensor (Shape.conv2dOut [batch, inChannels, height, width]
                                             [outChannels, inChannels, kernelH, kernelW]
                                             params.padding params.stride params.dilation) dt device) := do
  conv2d x params.weight params.bias params.padding params.stride params.dilation

/-- Get all trainable parameters -/
def parameters {device : Backend.DeviceType} (params : Conv2dParams inChannels outChannels kernelH kernelW dt device)
    : List UOp :=
  match params.bias with
  | none => [params.weight.uop]
  | some b => [params.weight.uop, b.uop]

/-- Number of parameters -/
def numParams {device : Backend.DeviceType} (params : Conv2dParams inChannels outChannels kernelH kernelW dt device) : Nat :=
  let weightParams := outChannels * inChannels * kernelH * kernelW
  match params.bias with
  | none => weightParams
  | some _ => weightParams + outChannels

end Conv2dParams

/-- Convenience: Create common square-kernel Conv2d -/
abbrev Conv2d (inChannels outChannels kernelSize : Nat) (dt : DType := .float32)
    (device : Backend.DeviceType := .CPU) :=
  Conv2dParams inChannels outChannels kernelSize kernelSize dt device

/-- MaxPool2d layer (stateless, just parameters) -/
structure MaxPool2dParams where
  kernelSize : Nat
  stride : Nat := 0  -- 0 means same as kernelSize
  padding : Nat := 0

namespace MaxPool2dParams

def create (kernelSize : Nat) (stride : Nat := 0) (padding : Nat := 0) : MaxPool2dParams :=
  { kernelSize, stride := if stride == 0 then kernelSize else stride, padding }

def forward {batch channels height width : Nat} {dt : DType} {device : Backend.DeviceType}
    (params : MaxPool2dParams)
    (x : StaticTensor [batch, channels, height, width] dt device)
    : TensorM (StaticTensor (Shape.pool2dShape [batch, channels, height, width]
                                               params.kernelSize params.padding params.stride) dt device) := do
  maxPool2d x params.kernelSize params.stride params.padding

end MaxPool2dParams

/-- AvgPool2d layer (stateless, just parameters) -/
structure AvgPool2dParams where
  kernelSize : Nat
  stride : Nat := 0  -- 0 means same as kernelSize
  padding : Nat := 0

namespace AvgPool2dParams

def create (kernelSize : Nat) (stride : Nat := 0) (padding : Nat := 0) : AvgPool2dParams :=
  { kernelSize, stride := if stride == 0 then kernelSize else stride, padding }

def forward {batch channels height width : Nat} {dt : DType} {device : Backend.DeviceType}
    (params : AvgPool2dParams)
    (x : StaticTensor [batch, channels, height, width] dt device)
    : TensorM (StaticTensor (Shape.pool2dShape [batch, channels, height, width]
                                               params.kernelSize params.padding params.stride) dt device) := do
  avgPool2d x params.kernelSize params.stride params.padding

end AvgPool2dParams

end TinyGrad4.NN
