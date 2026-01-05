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

/-- Coerce tensor to target shape -/
private def coerceShape {s1 s2 : List Nat} {d : DType}
    (t : StaticTensor s1 d) : StaticTensor s2 d :=
  { uop := t.uop, h_shape := sorry_proof, requiresGrad := t.requiresGrad }

/-- Forward pass for dropout.

    Training mode:
    - Generate random mask where {lit}`mask[i] = 1` if {lit}`rand[i] > p`, else 0
    - Output = {lit}`input * mask / (1 - p)`

    Eval mode:
    - Output = input (identity)

    The `seed` parameter controls the random mask generation.
    Different seeds produce different dropout patterns. -/
def forward {s : List Nat} {d : DType}
    (params : DropoutParams) (x : StaticTensor s d) (seed : Nat)
    : TensorM (StaticTensor s d) := do
  -- Eval mode or p=0: no dropout, return input as-is
  if !params.training || params.p == 0.0 then
    return x

  -- p=1: drop everything
  if params.p >= 1.0 then
    return ← Tensor.zerosLike x

  -- Generate random values in [0, 1)
  let randT ← Tensor.rand s d seed

  -- Create threshold tensor
  let pT ← Tensor.full s d params.p

  -- mask = (rand > p), i.e., keep element if rand > p
  -- This drops ~p fraction of elements
  let mask ← cmpgtB randT pT

  -- Convert bool mask to float: where(mask, 1.0, 0.0)
  let one ← Tensor.full s d 1.0
  let zero ← Tensor.full s d 0.0
  let maskF ← where_ mask one zero
  let maskF : StaticTensor s d := coerceShape maskF

  -- Scale factor: 1 / (1 - p) to maintain expected values
  let scale := 1.0 / (1.0 - params.p)
  let scaleT ← Tensor.full s d scale

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
def dropoutForward {s : List Nat} {d : DType}
    (x : StaticTensor s d) (p : Float32 := 0.5) (training : Bool := true) (seed : Nat := 0)
    : TensorM (StaticTensor s d) := do
  let params := { p, training : DropoutParams }
  DropoutParams.forward params x seed

end TinyGrad4.NN
