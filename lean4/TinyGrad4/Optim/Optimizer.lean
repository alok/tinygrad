import TinyGrad4.Tensor.Tensor
import TinyGrad4.Backend.Interpreter
import TinyGrad4.Gradient.Autodiff
import TinyGrad4.Optim.Adam

namespace TinyGrad4.Optim

open TinyGrad4
open TinyGrad4.Interpreter

/-!
# Optimizer Interface

Base class for all optimizers, mirroring tinygrad's `Optimizer`.

## Usage

```lean
-- Create optimizer
let opt := Adam.create (lr := 0.01)

-- Training loop
for _ in [:epochs] do
  let (loss, params) ← forward model input
  let (updates, opt') ← adamStep loss params opt
  -- Apply updates...
```
-/

/-- Optimizer interface using typeclass -/
class Optimizer (O : Type) where
  /-- Zero all gradients (reset for next iteration) -/
  zeroGrad : O → O

  /-- Perform a single optimization step.
      Returns: (updated parameter values as RawBuffers, updated optimizer state) -/
  step : O → (loss : Scalar d) → (params : List (StaticTensor s d)) → (env : Env)
       → TensorM (List RawBuffer × O)

/-- Get learning rate from optimizer config -/
class HasLearningRate (O : Type) where
  getLr : O → Float
  setLr : O → Float → O

/-- Optimizer with momentum support -/
class HasMomentum (O : Type) where
  getMomentum : O → Float
  setMomentum : O → Float → O

/-- Parameter group for different learning rates per layer -/
structure ParamGroup where
  /-- Parameter UIDs in this group -/
  params : List Nat
  /-- Learning rate multiplier for this group -/
  lrMult : Float := 1.0
  /-- Weight decay for this group -/
  weightDecay : Float := 0.0
  deriving Repr

/-- Helper: collect all parameters from a list of layers -/
def collectParams (layers : List (List UOp)) : List UOp :=
  layers.flatten

/-- Helper: apply updates to parameters (returns new tensors).
    Creates new StaticTensors from RawBuffer values. -/
def applyUpdates {s : List Nat} {d : DType}
    (params : List (StaticTensor s d))
    (updates : List RawBuffer)
    : TensorM (List (StaticTensor s d)) := do
  let mut result : List (StaticTensor s d) := []
  for (p, u) in params.zip updates do
    -- Create new tensor from updated values
    let newUop ← UOp.vconstRaw u s
    let reshaped ← UOp.reshape newUop s
    result := result ++ [{ uop := reshaped, h_shape := sorry_proof, requiresGrad := p.requiresGrad }]
  pure result

-- Instance: Adam implements Optimizer
instance : Optimizer Adam where
  zeroGrad opt := opt  -- Adam doesn't store gradients, nothing to zero
  step opt loss params env := adamStep loss params opt env

-- Instance: Adam has learning rate
instance : HasLearningRate Adam where
  getLr opt := opt.config.lr
  setLr opt lr := { opt with config := { opt.config with lr := lr } }

end TinyGrad4.Optim
