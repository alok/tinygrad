import TinyGrad4.NN.Linear
import TinyGrad4.NN.Norm
import TinyGrad4.NN.Embedding

/-!
# Neural Network Module

Provides neural network building blocks mirroring tinygrad's `nn` module.

## Layers
- `Linear`: Fully connected layer
- `RMSNorm`: Root Mean Square Layer Normalization
- `LayerNorm`: Layer Normalization
- `Embedding`: Embedding lookup table

## Usage

```lean
import TinyGrad4.NN

open TinyGrad4.NN

-- Create a simple network
def model : TensorM _ := do
  let linear1 ← LinearParams.create 784 256
  let linear2 ← LinearParams.create 256 10
  let norm ← rmsNorm 256
  pure (linear1, norm, linear2)
```
-/

-- All exports happen via the individual module imports above
-- Users can `open TinyGrad4.NN` to access LinearParams, RMSNormParams, etc.
