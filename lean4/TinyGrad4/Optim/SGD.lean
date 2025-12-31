import TinyGrad4.Tensor.Tensor
import TinyGrad4.Tensor.Math
import TinyGrad4.Backend.Interpreter
import TinyGrad4.Gradient.Autodiff

namespace TinyGrad4.Optim

open TinyGrad4.Interpreter
open TinyGrad4.StaticTensor

/-!
# SGD Optimizer (UOp-based)

Simple Stochastic Gradient Descent optimizer.
Updates parameters: param = param - lr * grad

This implementation follows the tinygrad philosophy: express optimizer updates
as tensor operations (UOps), letting codegen handle all dtypes automatically.
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

/-- Build UOps for SGD update formula.
    Returns new_param as UOp: param - lr * grad -/
def buildUpdateUOp (param grad : UOp) (lr : Float) : TensorM UOp := do
  let dtype := param.dtype
  let shape := param.shape

  -- Create lr constant and expand to param shape
  let lrConst ← UOp.const dtype lr.toFloat32
  let lrB ← UOp.expand lrConst shape

  -- lr * grad
  let scaledGrad ← UOp.binaryOp .MUL lrB grad

  -- param - lr * grad
  UOp.binaryOp .SUB param scaledGrad

/-- Single step update using UOps.
    Returns: updated param RawBuffer -/
def stepOneUOp (paramUop gradUop : UOp) (lr : Float) (env : Env) : TensorM RawBuffer := do
  let paramNew ← buildUpdateUOp paramUop gradUop lr
  pure (eval paramNew env)

/-- ByteArray-level update for float32 buffers (fast native path).
    Uses native float32 math for maximum performance. -/
def updateRawF32 (data grad : RawBuffer) (lr : Float) : RawBuffer :=
  if data.dtype != .float32 || grad.dtype != .float32 then
    data
  else
    let out := Backend.Native.sgdUpdateF32 data.data grad.data lr
    if out.isEmpty then
      data
    else
      { dtype := .float32, data := out }

/-- Single step update that stays in RawBuffers (native fast path). -/
def stepOneRaw (paramUop gradUop : UOp) (lr : Float) (env : Env) : RawBuffer :=
  let paramData := eval paramUop env
  let gradData := eval gradUop env
  updateRawF32 paramData gradData lr

/-- Cached eval + raw update (IO). -/
def stepOneRawCached (paramUop gradUop : UOp) (lr : Float) (env : Env) : IO RawBuffer := do
  let vals ← evalManyCached [paramUop, gradUop] env
  let paramData := vals.getD paramUop.uid (RawBuffer.zeros paramUop.dtype (listProd paramUop.shape))
  let gradData := vals.getD gradUop.uid (RawBuffer.zeros gradUop.dtype (listProd gradUop.shape))
  pure (updateRawF32 paramData gradData lr)

end SGD

/-- Compute gradients and return updated parameter values using UOps.
    This is the tinygrad way: express updates as UOps, let codegen handle dtypes. -/
def sgdStep {s : List Nat} {d : DType}
    (loss : Scalar d)
    (params : List (StaticTensor s d))
    (lr : Float)
    (env : Env := ∅)
    : TensorM (List RawBuffer) := do
  -- Compute gradients
  let paramUops := params.map (·.uop)
  let gradMap ← backward loss paramUops

  -- Build update UOps for all params
  let mut updateUops : List UOp := []
  for p in params do
    match gradMap[p.uop.uid]? with
    | some gradUop =>
      let newParam ← SGD.buildUpdateUOp p.uop gradUop lr
      updateUops := updateUops ++ [newParam]
    | none =>
      updateUops := updateUops ++ [p.uop]

  -- Evaluate all updates together
  let results := evalMany updateUops env
  pure (updateUops.map fun u => results.getD u.uid (RawBuffer.zeros u.dtype (listProd u.shape)))

/-- Like `sgdStep`, but uses native float32 update kernel (faster for float32). -/
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
    | none => eval p.uop env

  pure updates

end TinyGrad4.Optim
