import TinyGrad4.Tensor.Tensor
import TinyGrad4.UOp.Typed

namespace TinyGrad4.Optim

/-!
# Optim initializers

Lightweight parameter initialization helpers that build typed TUOps.
-/

namespace Init

/-- Fill a tensor with a scalar value. -/
def full (shape : Shape) (dtype : DType) (value : Float32) : TensorM (StaticTensor shape dtype) := do
  let const ← TUOp.const dtype value
  let expanded ← TUOp.expand const shape
  pure (StaticTensor.ofTUOp expanded)

/-- Zeros initializer. -/
def zeros (shape : Shape) (dtype : DType := .float32) : TensorM (StaticTensor shape dtype) :=
  full shape dtype 0.0

/-- Ones initializer. -/
def ones (shape : Shape) (dtype : DType := .float32) : TensorM (StaticTensor shape dtype) :=
  full shape dtype 1.0

/-- Uniform initializer in {lit}`[0, 1)`. -/
def uniform (shape : Shape) (dtype : DType := .float32) (seed : Nat := 0) :
    TensorM (StaticTensor shape dtype) :=
  Tensor.rand shape dtype seed

/-- Normal initializer with mean 0 and stddev 1. -/
def normal (shape : Shape) (dtype : DType := .float32) (seed : Nat := 0) :
    TensorM (StaticTensor shape dtype) :=
  Tensor.randn shape dtype seed

end Init

end TinyGrad4.Optim
