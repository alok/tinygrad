import Float64
import TinyGrad4.Tensor.Tensor
import TinyGrad4.Tensor.Math

/-!
# Dropout Layer

Applies dropout regularization during training.

During training: randomly zeroes some elements with probability `p`,
then scales remaining elements by `1/(1-p)` to maintain expected values.

During evaluation: returns input unchanged (identity function).

Mirrors tinygrad's `Tensor.dropout` method.

## References
- Paper: https://jmlr.org/papers/v15/srivastava14a.html
-/

namespace TinyGrad4.NN

open TinyGrad4
open StaticTensor

private def broadcastProof {s1 s2 : List Nat} : Shape.broadcastable s1 s2 = true := by
  exact sorry_proof

/-- Dropout parameters -/
structure DropoutParams where
  /-- Dropout probability (fraction of elements to zero out) -/
  p : Float32
  /-- Whether in training mode (dropout active) or eval mode (pass-through) -/
  training : Bool
  deriving Repr

namespace DropoutParams

/-- Create Dropout layer with given probability.
    p=0.5 means 50% of elements are zeroed during training. -/
def create (p : Float32 := 0.5) : DropoutParams :=
  { p, training := true }

/-- Forward pass for dropout.

    Training mode:
    - Generate random mask where mask[i] = 1 if rand[i] > p, else 0
    - Output = input * mask / (1 - p)

    Eval mode:
    - Output = input (identity)

    The `seed` parameter controls the random mask generation.
    Different seeds produce different dropout patterns. -/
def forward {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (params : DropoutParams) (x : StaticTensor s d device) (seed : Nat)
    : TensorM (StaticTensor s d device) := do
  -- Eval mode or p=0: no dropout, return input as-is
  if !params.training || params.p == 0.0 then
    return x

  -- p=1: drop everything
  if params.p >= 1.0 then
    return ← Tensor.zerosLike x

  -- Generate random values in [0, 1)
  let randT ← Tensor.rand (device := device) s d seed

  -- Create threshold tensor
  let pT ← Tensor.full (device := device) s d params.p

  -- mask = (rand > p), i.e., keep element if rand > p
  -- This drops ~p fraction of elements
  let mask ← cmpgtBroadcast randT pT broadcastProof

  -- Convert bool mask to float: where(mask, 1.0, 0.0)
  let one ← Tensor.full (device := device) s d 1.0
  let zero ← Tensor.full (device := device) s d 0.0
  let maskF ← select mask one zero broadcastProof broadcastProof
  let maskF : StaticTensor s d device := StaticTensor.assumeShape maskF

  -- Scale factor: 1 / (1 - p) to maintain expected values
  let scale := 1.0 / (1.0 - params.p)
  let scaleT ← Tensor.full (device := device) s d scale

  -- output = x * mask * scale
  let masked ← mul x maskF
  mul masked scaleT

/-- Set training mode (dropout active) -/
def train (params : DropoutParams) : DropoutParams :=
  { params with training := true }

/-- Set eval mode (dropout disabled, pass-through) -/
def eval (params : DropoutParams) : DropoutParams :=
  { params with training := false }

/-- Dropout has no trainable parameters -/
def parameters (_ : DropoutParams) : List UOp := []

/-- Number of parameters (always 0 for dropout) -/
def numParams (_ : DropoutParams) : Nat := 0

end DropoutParams

/-- Convenience: Create Dropout layer -/
def dropout (p : Float32 := 0.5) : DropoutParams :=
  DropoutParams.create p

/-- Apply dropout directly to a tensor (functional API).
    Uses global training flag from params. -/
def dropoutForward {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor s d device) (p : Float32 := 0.5) (training : Bool := true) (seed : Nat := 0)
    : TensorM (StaticTensor s d device) := do
  let params := { p, training : DropoutParams }
  DropoutParams.forward params x seed

end TinyGrad4.NN
