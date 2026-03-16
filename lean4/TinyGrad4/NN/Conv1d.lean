import Float64
import TinyGrad4.Tensor.Tensor
import TinyGrad4.Tensor.Math
import TinyGrad4.Tensor.Movement

/-!
# Conv1d Layer

Applies a 1D convolution over an input signal composed of several input planes.

Mirrors tinygrad's `nn.Conv1d` with Kaiming uniform initialization.

## Shape semantics
- Input:  [batch, in_channels, width]
- Weight: [out_channels, in_channels, kernel_width]
- Bias:   [out_channels]
- Output: [batch, out_channels, out_width]

where:
  out_width = (width + 2*padding - dilation*(kernel_width-1) - 1) / stride + 1
-/

namespace TinyGrad4.NN

open TinyGrad4
open StaticTensor

/-- Conv1d layer parameters -/
structure Conv1dParams (inChannels outChannels kernelW : Nat) (dt : DType) (device : Backend.DeviceType) where
  /-- Weight tensor [outChannels, inChannels, kernelW] -/
  weight : StaticTensor [outChannels, inChannels, kernelW] dt device
  /-- Optional bias [outChannels] -/
  bias : Option (Vector outChannels dt device)
  /-- Padding (same on both sides) -/
  padding : Nat
  /-- Stride -/
  stride : Nat
  /-- Dilation -/
  dilation : Nat

namespace Conv1dParams

/-- Create Conv1d layer with Kaiming uniform initialization.
    bound = 1 / sqrt(in_channels * kernel_w), matching tinygrad's nn.Conv1d -/
def create (device : Backend.DeviceType := .CPU) (inChannels outChannels : Nat) (kernelSize : Nat := 3)
    (dt : DType := .float32) (useBias : Bool := true)
    (padding : Nat := 0) (stride : Nat := 1) (dilation : Nat := 1)
    (seed : Nat := 42) : TensorM (Conv1dParams inChannels outChannels kernelSize dt device) := do
  let fanIn := inChannels * kernelSize
  let bound := (1.0 / Float64.sqrt (Float64.ofNat fanIn)).toFloat32

  let weight ← uniformInit (device := device) [outChannels, inChannels, kernelSize] dt (-bound) bound seed

  let bias ← if useBias then
    let b ← uniformInit (device := device) [outChannels] dt (-bound) bound (seed + 1)
    pure (some b)
  else
    pure none

  pure { weight, bias, padding, stride, dilation }

/-- Forward pass: apply convolution
    Input:  [batch, inChannels, width]
    Output: [batch, outChannels, outWidth] -/
def forward {batch width : Nat} {device : Backend.DeviceType}
    (params : Conv1dParams inChannels outChannels kernelW dt device)
    (x : StaticTensor [batch, inChannels, width] dt device)
    : TensorM (StaticTensor (Shape.conv1dOut [batch, inChannels, width]
                                             [outChannels, inChannels, kernelW]
                                             params.padding params.stride params.dilation) dt device) := do
  conv1d x params.weight params.bias params.padding params.stride params.dilation

/-- Get all trainable parameters -/
def parameters {device : Backend.DeviceType} (params : Conv1dParams inChannels outChannels kernelW dt device)
    : List UOp :=
  match params.bias with
  | none => [params.weight.uop]
  | some b => [params.weight.uop, b.uop]

/-- Number of parameters -/
def numParams {device : Backend.DeviceType} (params : Conv1dParams inChannels outChannels kernelW dt device) : Nat :=
  let weightParams := outChannels * inChannels * kernelW
  match params.bias with
  | none => weightParams
  | some _ => weightParams + outChannels

end Conv1dParams

/-- Convenience: Create common Conv1d -/
abbrev Conv1d (inChannels outChannels kernelSize : Nat) (dt : DType := .float32)
    (device : Backend.DeviceType := .CPU) :=
  Conv1dParams inChannels outChannels kernelSize dt device

end TinyGrad4.NN
