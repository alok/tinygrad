import TinyGrad4.Tensor.Tensor
import TinyGrad4.Tensor.Math
import TinyGrad4.Tensor.Movement

/-!
# Embedding Layer

A lookup table that stores embeddings of a fixed vocabulary.

Mirrors tinygrad's `nn.Embedding`.
-/

namespace TinyGrad4.NN

open TinyGrad4
open StaticTensor

/-- Embedding layer parameters -/
structure EmbeddingParams (vocabSize embedSize : Nat) (dt : DType) where
  /-- Embedding weight matrix [vocabSize, embedSize] -/
  weight : Matrix vocabSize embedSize dt

namespace EmbeddingParams

/-- Create embedding with Glorot uniform initialization.
    Matches tinygrad's Tensor.glorot_uniform -/
def create (vocabSize embedSize : Nat) (dt : DType := .float32) (seed : Nat := 42)
    : TensorM (EmbeddingParams vocabSize embedSize dt) := do
  -- Glorot bound = sqrt(6 / (fan_in + fan_out))
  let fanIn := vocabSize
  let fanOut := embedSize
  let bound := (Float.sqrt (6.0 / Float.ofNat (fanIn + fanOut))).toFloat32

  -- Random uniform in [-bound, bound]
  let r ← Tensor.rand [vocabSize, embedSize] dt seed
  let rangeT ← Tensor.full [vocabSize, embedSize] dt (2.0 * bound)
  let lowT ← Tensor.full [vocabSize, embedSize] dt (-bound)
  let scaled ← mul r rangeT
  let weight ← add scaled lowT

  pure { weight }

/-- Forward pass: lookup indices in embedding table.

    This is a simplified implementation that creates a one-hot encoding
    and does matmul. For efficiency, a proper gather operation would be better.

    Input: indices [batch] with values in [0, vocabSize)
    Output: embeddings [batch, embedSize] -/
def forward {batch : Nat} (params : EmbeddingParams vocabSize embedSize dt)
    (indices : Vector batch .int32)
    : TensorM (Matrix batch embedSize dt) := do
  -- Create one-hot: [batch, vocabSize]
  -- arange [vocabSize] broadcasted to [batch, vocabSize]
  let arangeVocab ← Tensor.arange vocabSize .int32
  let arangeExpanded ← expand arangeVocab [batch, vocabSize]

  -- indices expanded to [batch, vocabSize]
  let indicesExpanded ← expand indices [batch, vocabSize]

  -- Compare: one_hot[i,j] = 1 if indices[i] == j else 0
  let oneHotBool ← cmpeq indicesExpanded arangeExpanded

  -- Cast to target dtype
  let oneHot ← StaticTensor.cast oneHotBool dt

  -- Matmul: [batch, vocabSize] @ [vocabSize, embedSize] = [batch, embedSize]
  matmul oneHot params.weight

/-- Get trainable parameters -/
def parameters (params : EmbeddingParams vocabSize embedSize dt) : List UOp :=
  [params.weight.uop]

end EmbeddingParams

/-- Convenience constructor -/
def embedding (vocabSize embedSize : Nat) (dt : DType := .float32) (seed : Nat := 42)
    : TensorM (EmbeddingParams vocabSize embedSize dt) :=
  EmbeddingParams.create vocabSize embedSize dt seed

end TinyGrad4.NN
