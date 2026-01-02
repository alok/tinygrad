import TinyGrad4.Tensor.Tensor
import TinyGrad4.Tensor.Math
import TinyGrad4.Backend.Interpreter
import TinyGrad4.Gradient.Autodiff
import TinyGrad4.UOp.Typed

namespace TinyGrad4.Optim

open TinyGrad4.Interpreter
open TinyGrad4.StaticTensor

/-!
# Adam and AdamW Optimizers (UOp-based)

Adam: Adaptive Moment Estimation optimizer.
AdamW: Adam with decoupled weight decay.

This implementation follows the tinygrad philosophy: express optimizer updates
as tensor operations (UOps), letting codegen handle all dtypes automatically.

Papers:
- Adam: https://arxiv.org/abs/1412.6980
- AdamW: https://arxiv.org/abs/1711.05101v3
-/

/-- Adam optimizer configuration -/
structure AdamConfig where
  /-- Learning rate -/
  lr : Float := 0.001
  /-- First moment decay rate -/
  beta1 : Float := 0.9
  /-- Second moment decay rate -/
  beta2 : Float := 0.999
  /-- Numerical stability epsilon -/
  eps : Float := 1e-8
  /-- Weight decay (0 for vanilla Adam, >0 for AdamW) -/
  weightDecay : Float := 0.0
  deriving Repr

/-- Adam moment state for a parameter (stored as RawBuffer for dtype support) -/
structure AdamMoments where
  /-- First moment estimate -/
  m : RawBuffer
  /-- Second moment estimate -/
  v : RawBuffer
  /-- Shape of the parameter -/
  shape : Shape
  deriving Repr

/-- Adam optimizer state -/
structure Adam where
  /-- Configuration -/
  config : AdamConfig
  /-- Step count (for bias correction) -/
  step : Nat := 0
  /-- Running product of beta1^t for bias correction -/
  beta1_t : Float := 1.0
  /-- Running product of beta2^t for bias correction -/
  beta2_t : Float := 1.0
  /-- Per-parameter moments (indexed by uid) -/
  moments : Std.HashMap UOpId AdamMoments := {}
  deriving Repr

namespace Adam

/-- Create Adam optimizer -/
def create (lr : Float := 0.001) (beta1 : Float := 0.9) (beta2 : Float := 0.999)
    (eps : Float := 1e-8) : Adam :=
  { config := { lr, beta1, beta2, eps, weightDecay := 0.0 } }

/-- Create AdamW optimizer -/
def createW (lr : Float := 0.001) (beta1 : Float := 0.9) (beta2 : Float := 0.999)
    (eps : Float := 1e-8) (weightDecay : Float := 0.01) : Adam :=
  { config := { lr, beta1, beta2, eps, weightDecay } }

/-- Initialize moments for a parameter -/
private def initMoments (dtype : DType) (shape : Shape) : AdamMoments :=
  let numel := listProd shape
  { m := RawBuffer.zeros dtype numel
    v := RawBuffer.zeros dtype numel
    shape := shape }

/-- Build UOps for Adam update formula.
    Returns (new_param, new_m, new_v) as UOps to be evaluated together.

    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    param = param - lr * m_hat / (sqrt(v_hat) + eps)
-/
def buildUpdateUOps (param grad mBuf vBuf : UOp) (cfg : AdamConfig)
    (beta1_t beta2_t : Float) : TensorM (UOp × UOp × UOp) := do
  let dtype := param.dtype
  let shape := param.shape

  -- Create scalar constants
  let beta1 ← UOp.const dtype cfg.beta1.toFloat32
  let beta2 ← UOp.const dtype cfg.beta2.toFloat32
  let one ← UOp.const dtype 1.0
  let lr ← UOp.const dtype cfg.lr.toFloat32
  let eps ← UOp.const dtype cfg.eps.toFloat32

  -- Bias correction denominators
  let biasCorr1 ← UOp.const dtype (1.0 - beta1_t).toFloat32
  let biasCorr2 ← UOp.const dtype (1.0 - beta2_t).toFloat32

  -- Expand scalars to param shape
  let beta1B ← UOp.expand beta1 shape
  let beta2B ← UOp.expand beta2 shape
  let oneB ← UOp.expand one shape
  let lrB ← UOp.expand lr shape
  let epsB ← UOp.expand eps shape
  let biasCorr1B ← UOp.expand biasCorr1 shape
  let biasCorr2B ← UOp.expand biasCorr2 shape

  -- Update moments: m_new = beta1 * m + (1 - beta1) * grad
  let oneMinusBeta1 ← UOp.binaryOp .SUB oneB beta1B
  let mScaled ← UOp.binaryOp .MUL beta1B mBuf
  let gScaled ← UOp.binaryOp .MUL oneMinusBeta1 grad
  let mNew ← UOp.binaryOp .ADD mScaled gScaled

  -- v_new = beta2 * v + (1 - beta2) * grad^2
  let oneMinusBeta2 ← UOp.binaryOp .SUB oneB beta2B
  let vScaled ← UOp.binaryOp .MUL beta2B vBuf
  let gradSq ← UOp.binaryOp .MUL grad grad
  let gSqScaled ← UOp.binaryOp .MUL oneMinusBeta2 gradSq
  let vNew ← UOp.binaryOp .ADD vScaled gSqScaled

  -- Bias correction: m_hat = m_new / (1 - beta1^t)
  let mHat ← UOp.binaryOp .IDIV mNew biasCorr1B
  let vHat ← UOp.binaryOp .IDIV vNew biasCorr2B

  -- Weight decay (AdamW style - applied to param before update)
  let paramDecayed ← if cfg.weightDecay > 0.0 then do
    let wd ← UOp.const dtype cfg.weightDecay.toFloat32
    let wdB ← UOp.expand wd shape
    let lrWd ← UOp.binaryOp .MUL lrB wdB
    let decay ← UOp.binaryOp .SUB oneB lrWd
    UOp.binaryOp .MUL param decay
  else
    pure param

  -- Update: param_new = param - lr * m_hat / (sqrt(v_hat) + eps)
  let sqrtV ← UOp.unaryOp .SQRT vHat
  let denom ← UOp.binaryOp .ADD sqrtV epsB
  let update ← UOp.binaryOp .IDIV mHat denom
  let scaledUpdate ← UOp.binaryOp .MUL lrB update
  let paramNew ← UOp.binaryOp .SUB paramDecayed scaledUpdate

  pure (paramNew, mNew, vNew)

/-- Single step update for one parameter.
    Returns: (updated param RawBuffer, updated optimizer) -/
def stepOne (opt : Adam) (paramUop gradUop : UOp) (env : Env)
    : TensorM (RawBuffer × Adam) := do
  let uid := paramUop.uid
  let dtype := paramUop.dtype
  let shape := paramUop.shape

  -- Update beta powers for bias correction
  let newBeta1_t := opt.beta1_t * opt.config.beta1
  let newBeta2_t := opt.beta2_t * opt.config.beta2
  let step := opt.step + 1

  -- Get or initialize moments
  let moments := match opt.moments.get? uid with
    | some m => m
    | none => initMoments dtype shape

  -- Build UOps for moments (convert RawBuffer to UOp)
  let mUop ← TUOp.vconstRaw moments.m shape
  let vUop ← TUOp.vconstRaw moments.v shape

  -- Build update UOps
  let (paramNew, mNew, vNew) ← buildUpdateUOps paramUop gradUop mUop.raw vUop.raw opt.config newBeta1_t newBeta2_t

  -- Evaluate all at once
  let results := evalMany [paramNew, mNew, vNew] env
  let paramResult := results.getD paramNew.uid (RawBuffer.zeros dtype (listProd shape))
  let mResult := results.getD mNew.uid (RawBuffer.zeros dtype (listProd shape))
  let vResult := results.getD vNew.uid (RawBuffer.zeros dtype (listProd shape))

  -- Update state
  let newMoments : AdamMoments := { m := mResult, v := vResult, shape := shape }
  let newOpt : Adam := { opt with
    step := step
    beta1_t := newBeta1_t
    beta2_t := newBeta2_t
    moments := opt.moments.insert uid newMoments
  }

  pure (paramResult, newOpt)

end Adam

/-- Compute gradients and apply Adam updates.
    Returns updated parameter values as RawBuffers. -/
def adamStep {s : List Nat} {d : DType}
    (loss : Scalar d)
    (params : List (StaticTensor s d))
    (opt : Adam)
    (env : Env := ∅)
    : TensorM (List RawBuffer × Adam) := do
  -- Compute gradients
  let paramUops := params.map (·.uop)
  let gradMap ← backward loss paramUops

  -- Update each parameter
  let mut updates : List RawBuffer := []
  let mut currentOpt := opt

  for p in params do
    match gradMap[p.uop.uid]? with
    | some gradUop =>
      let (newVal, newOpt) ← Adam.stepOne currentOpt p.uop gradUop env
      updates := updates ++ [newVal]
      currentOpt := newOpt
    | none =>
      -- No gradient, keep original
      updates := updates ++ [eval p.uop env]

  pure (updates, currentOpt)

/-- Convenience: Adam optimizer -/
def adam (lr : Float := 0.001) : Adam := Adam.create lr

/-- Convenience: AdamW optimizer -/
def adamW (lr : Float := 0.001) (weightDecay : Float := 0.01) : Adam :=
  Adam.createW lr (weightDecay := weightDecay)

end TinyGrad4.Optim
