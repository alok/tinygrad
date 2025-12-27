import TinyGrad4.Ops
import TinyGrad4.UOp.UOp

namespace TinyGrad4

namespace Gradient

inductive TapePolicy where
  | saveInput
  | saveOutput
  | saveMask
  | recompute
  deriving DecidableEq, Repr, Inhabited

/-- Backward metadata describing the tape we need (or choose) for an op. -/
structure AdjointSpec where
  tape : TapePolicy := .recompute
  deriving Repr

end Gradient

namespace Ops

def adjointSpec (op : Ops) : Gradient.AdjointSpec :=
  match op with
  | .MAX | .WHERE => { tape := .saveMask }
  | _ => { tape := .recompute }

end Ops

namespace Gradient

private def isZeroConst (u : UOp) : Bool :=
  u.op == .CONST && u.shape == [] && u.arg.getFloat.getD 1.0 == 0.0

/-- Adjoint spec for a concrete UOp, with pattern-sensitive refinements. -/
def adjointSpec (u : UOp) : AdjointSpec :=
  if u.op == .MAX && u.src.length == 2 then
    let x := u.src[0]!
    let y := u.src[1]!
    if isZeroConst x || isZeroConst y then
      { tape := .saveOutput }
    else
      Ops.adjointSpec u.op
  else
    Ops.adjointSpec u.op

def isReluLike (u : UOp) : Bool :=
  u.op == .MAX && (adjointSpec u).tape == .saveOutput

end Gradient

end TinyGrad4
