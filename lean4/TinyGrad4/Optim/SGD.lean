import TinyGrad4.Tensor.Tensor
import TinyGrad4.Tensor.Math
import TinyGrad4.Backend.Interpreter
import TinyGrad4.Gradient.Autodiff

namespace TinyGrad4.Optim

open TinyGrad4.Interpreter
open TinyGrad4.StaticTensor

/-!
# SGD Optimizer

Simple Stochastic Gradient Descent optimizer.
Updates parameters: param = param - lr * grad
-/

/-- Parameter with its gradient -/
structure Parameter (shape : List Nat) (dtype : DType) where
  tensor : StaticTensor shape dtype
  grad : Option (StaticTensor shape dtype) := none

/-- SGD optimizer state -/
structure SGD where
  learningRate : Float := 0.01
  momentum : Float := 0.0

namespace SGD

/-- Create SGD optimizer -/
def create (lr : Float := 0.01) (momentum : Float := 0.0) : SGD :=
  { learningRate := lr, momentum := momentum }

/-- Apply gradient update to a tensor's data
    Returns updated data array -/
def updateArray (data grad : FlatArray) (lr : Float) : FlatArray :=
  FloatArray.zipWith (fun d g => d - lr * g) data grad

/-- ByteArray-level update for float32 buffers (fast path).

    Uses native float32 math, so it can differ slightly from `updateArray` (which uses Float = f64). -/
def updateRawF32 (data grad : RawBuffer) (lr : Float) : RawBuffer :=
  if data.dtype != .float32 || grad.dtype != .float32 then
    data
  else
    let out := Backend.Native.sgdUpdateF32 data.data grad.data lr
    if out.isEmpty then
      data
    else
      { dtype := .float32, data := out }

/-- Single step update given parameter UOp and gradient UOp
    Evaluates both, computes new value, returns updated array -/
def stepOne (paramUop gradUop : UOp) (lr : Float) (env : Env) : FlatArray :=
  let paramData := eval paramUop env
  let gradData := eval gradUop env
  updateArray paramData gradData lr

/-- Single step update that stays in RawBuffers (ByteArray float32). -/
def stepOneRaw (paramUop gradUop : UOp) (lr : Float) (env : Env) : RawBuffer :=
  let paramData := evalRaw paramUop env
  let gradData := evalRaw gradUop env
  updateRawF32 paramData gradData lr

/-- Cached eval + update (IO). Useful when the same graph repeats across steps. -/
def stepOneCached (paramUop gradUop : UOp) (lr : Float) (env : Env) : IO FlatArray := do
  let vals ← evalManyCached [paramUop, gradUop] env
  let paramData := vals.getD paramUop.uid (zeros (listProd paramUop.shape))
  let gradData := vals.getD gradUop.uid (zeros (listProd gradUop.shape))
  pure (updateArray paramData gradData lr)

/-- Cached eval + raw update (IO). -/
def stepOneRawCached (paramUop gradUop : UOp) (lr : Float) (env : Env) : IO RawBuffer := do
  let vals ← evalManyRawCached [paramUop, gradUop] env
  let paramData := vals.getD paramUop.uid (RawBuffer.zeros paramUop.dtype (listProd paramUop.shape))
  let gradData := vals.getD gradUop.uid (RawBuffer.zeros gradUop.dtype (listProd gradUop.shape))
  pure (updateRawF32 paramData gradData lr)

end SGD

/-- Convenience: compute gradients and return updated parameter values
    This is a pure function that doesn't mutate anything -/
def sgdStep {s : List Nat} {d : DType}
    (loss : Scalar d)
    (params : List (StaticTensor s d))
    (lr : Float)
    (env : Env := ∅)
    : TensorM (List FlatArray) := do
  -- Compute gradients
  let paramUops := params.map (·.uop)
  let gradMap ← backward loss paramUops

  -- Update each parameter
  let updates := params.map fun p =>
    match gradMap[p.uop.uid]? with
    | some gradUop =>
      let paramData := eval p.uop env
      let gradData := eval gradUop env
      SGD.updateArray paramData gradData lr
    | none =>
      -- No gradient, keep original
      eval p.uop env

  pure updates

/-- Like `sgdStep`, but returns raw float32 buffers and uses the byte-level update kernel. -/
def sgdStepRaw {s : List Nat} {d : DType}
    (loss : Scalar d)
    (params : List (StaticTensor s d))
    (lr : Float)
    (env : Env := ∅)
    : TensorM (List RawBuffer) := do
  let paramUops := params.map (·.uop)
  let gradMap ← backward loss paramUops

  let updates := params.map fun p =>
    match gradMap[p.uop.uid]? with
    | some gradUop => SGD.stepOneRaw p.uop gradUop lr env
    | none => evalRaw p.uop env

  pure updates

end TinyGrad4.Optim
