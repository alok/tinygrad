import TinyGrad4.Shape
import TinyGrad4.DType
import TinyGrad4.Data.ArrayN

namespace TinyGrad4

/-- Batch of inputs/labels with static shapes and dtypes. -/
structure Batch (batch : Nat) (xShape yShape : Shape) (xDtype yDtype : DType) where
  x : DataArrayN (batch :: xShape) xDtype
  y : DataArrayN (batch :: yShape) yDtype

/-- Loader interface for typed batches. -/
class DataLoader (Loader : Type) (batch : Nat) (xShape yShape : Shape) (xDtype yDtype : DType) where
  numBatches : Loader → Nat
  getBatch : Loader → Nat → IO (Batch batch xShape yShape xDtype yDtype)

end TinyGrad4
