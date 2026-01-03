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

/-- Build TUOps for Adam update formula.
    Returns (new_param, new_m, new_v) as TUOps to be evaluated together.

    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    param = param - lr * m_hat / (sqrt(v_hat) + eps)
-/
def buildUpdateTUOps {opP opG opM opV : Ops} {s : Shape} {r : Nat} {d : DType}
    (param : TUOp opP s r d) (grad : TUOp opG s r d) (mBuf : TUOp opM s r d) (vBuf : TUOp opV s r d)
    (cfg : AdamConfig) (beta1_t beta2_t : Float) :
    TUOpM (TUOp .SUB s r d × TUOp .ADD s r d × TUOp .ADD s r d) := do
  let asSRD {op : Ops} {s' : Shape} {r' : Nat} {d' : DType} (u : TUOp op s' r' d') : TUOp op s r d :=
    TUOp.mkUnsafe (TUOp.castDType (TUOp.castShape u s) d).raw
  let bin {opx opy : Ops} {rx ry : Nat} (op : Ops) (x : TUOp opx s rx d) (y : TUOp opy s ry d) :
      TUOpM (TUOp op s r d) := do
    let x' : TUOp opx s r d := TUOp.mkUnsafe x.raw
    let y' : TUOp opy s r d := TUOp.mkUnsafe y.raw
    let res ← TUOp.binaryOpB (out := Shape.broadcastOut s s) op x' y'
    let res := TUOp.castShape res s
    let res := TUOp.castDType res d
    pure (TUOp.mkUnsafe res.raw)

  -- Create scalar constants
  let beta1 ← TUOp.const d cfg.beta1.toFloat32
  let beta2 ← TUOp.const d cfg.beta2.toFloat32
  let one ← TUOp.const d 1.0
  let lr ← TUOp.const d cfg.lr.toFloat32
  let eps ← TUOp.const d cfg.eps.toFloat32

  -- Bias correction denominators
  let biasCorr1 ← TUOp.const d (1.0 - beta1_t).toFloat32
  let biasCorr2 ← TUOp.const d (1.0 - beta2_t).toFloat32

  -- Expand scalars to param shape
  let beta1B ← TUOp.expand beta1 s
  let beta2B ← TUOp.expand beta2 s
  let oneB ← TUOp.expand one s
  let lrB ← TUOp.expand lr s
  let epsB ← TUOp.expand eps s
  let biasCorr1B ← TUOp.expand biasCorr1 s
  let biasCorr2B ← TUOp.expand biasCorr2 s

  -- Update moments: m_new = beta1 * m + (1 - beta1) * grad
  let oneMinusBeta1 ← bin .SUB oneB beta1B
  let mScaled ← bin .MUL beta1B mBuf
  let gScaled ← bin .MUL oneMinusBeta1 grad
  let mNew := asSRD (← bin .ADD mScaled gScaled)

  -- v_new = beta2 * v + (1 - beta2) * grad^2
  let oneMinusBeta2 ← bin .SUB oneB beta2B
  let vScaled ← bin .MUL beta2B vBuf
  let gradSq ← bin .MUL grad grad
  let gSqScaled ← bin .MUL oneMinusBeta2 gradSq
  let vNew := asSRD (← bin .ADD vScaled gSqScaled)

  -- Bias correction: m_hat = m_new / (1 - beta1^t)
  let mHat := asSRD (← bin .IDIV mNew biasCorr1B)
  let vHat := asSRD (← bin .IDIV vNew biasCorr2B)

  -- Weight decay (AdamW style - applied to param before update)
  let paramDecayedRaw ← if cfg.weightDecay > 0.0 then do
    let wd ← TUOp.const d cfg.weightDecay.toFloat32
    let wdB ← TUOp.expand wd s
    let lrWd ← bin .MUL lrB wdB
    let decay ← bin .SUB oneB lrWd
    let decayed ← bin .MUL param decay
    pure decayed.raw
  else
    pure param.raw
  let paramDecayed : TUOp paramDecayedRaw.op s r d := TUOp.mkUnsafe paramDecayedRaw

  -- Update: param_new = param - lr * m_hat / (sqrt(v_hat) + eps)
  let sqrtV ← TUOp.unaryOp .SQRT vHat
  let denom ← bin .ADD sqrtV epsB
  let update ← bin .IDIV mHat denom
  let scaledUpdate ← bin .MUL lrB update
  let paramNew := asSRD (← bin .SUB paramDecayed scaledUpdate)

  pure (paramNew, mNew, vNew)

/-- Build UOps for Adam update formula.
    Returns (new_param, new_m, new_v) as UOps to be evaluated together. -/
def buildUpdateUOps (param grad mBuf vBuf : UOp) (cfg : AdamConfig)
    (beta1_t beta2_t : Float) : TensorM (UOp × UOp × UOp) := do
  let shape := param.shape
  let dtype := param.dtype
  let paramT : TUOp param.op shape (TUOp.rankOf shape) dtype := TUOp.mkUnsafe param
  let gradT0 := TUOp.ofRaw grad
  let gradT := TUOp.castDType (TUOp.castShape gradT0 shape) dtype
  let gradT : TUOp grad.op shape (TUOp.rankOf shape) dtype := TUOp.mkUnsafe gradT.raw
  let mT0 := TUOp.ofRaw mBuf
  let mT := TUOp.castDType (TUOp.castShape mT0 shape) dtype
  let mT : TUOp mBuf.op shape (TUOp.rankOf shape) dtype := TUOp.mkUnsafe mT.raw
  let vT0 := TUOp.ofRaw vBuf
  let vT := TUOp.castDType (TUOp.castShape vT0 shape) dtype
  let vT : TUOp vBuf.op shape (TUOp.rankOf shape) dtype := TUOp.mkUnsafe vT.raw
  let (paramNew, mNew, vNew) ← buildUpdateTUOps paramT gradT mT vT cfg beta1_t beta2_t
  pure (paramNew.raw, mNew.raw, vNew.raw)

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
  let mUop0 ← TUOp.vconstRaw moments.m shape
  let vUop0 ← TUOp.vconstRaw moments.v shape
  let mUop := TUOp.castDType (TUOp.castShape mUop0 shape) dtype
  let vUop := TUOp.castDType (TUOp.castShape vUop0 shape) dtype
  let mUop : TUOp mUop.raw.op shape (TUOp.rankOf shape) dtype := TUOp.mkUnsafe mUop.raw
  let vUop : TUOp vUop.raw.op shape (TUOp.rankOf shape) dtype := TUOp.mkUnsafe vUop.raw

  -- Build update TUOps
  let paramT : TUOp paramUop.op shape (TUOp.rankOf shape) dtype := TUOp.mkUnsafe paramUop
  let gradT0 := TUOp.ofRaw gradUop
  let gradT := TUOp.castDType (TUOp.castShape gradT0 shape) dtype
  let gradT : TUOp gradUop.op shape (TUOp.rankOf shape) dtype := TUOp.mkUnsafe gradT.raw
  let (paramNew, mNew, vNew) ← buildUpdateTUOps paramT gradT mUop vUop opt.config newBeta1_t newBeta2_t

  -- Evaluate all at once
  let paramNewU := paramNew.raw
  let mNewU := mNew.raw
  let vNewU := vNew.raw
  let results := evalMany [paramNewU, mNewU, vNewU] env
  let paramResult := results.getD paramNewU.uid (RawBuffer.zeros dtype (listProd shape))
  let mResult := results.getD mNewU.uid (RawBuffer.zeros dtype (listProd shape))
  let vResult := results.getD vNewU.uid (RawBuffer.zeros dtype (listProd shape))

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
