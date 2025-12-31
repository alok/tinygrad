import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import Std.Data.HashMap

namespace TinyGrad4.Backend.FusedSGD

open Std

/-!
# Fused SGD Pattern Matching

Matches the SGD update pattern:
  param_new = param - lr * grad

In UOp terms:
  SUB
  ├── param (BUFFER or LOAD)
  └── MUL
      ├── lr (CONST scalar float32)
      └── grad (BUFFER or LOAD)

The lr can be on either side of the MUL.
-/

structure Plan where
  w : UOpId
  grad : UOpId
  lr : Float
  deriving Repr

namespace Plan

def mapIds (p : Plan) (f : UOpId → UOpId) : Plan :=
  { p with w := f p.w, grad := f p.grad }

end Plan

/-- Try to find a buffer at the leaves of a UOp -/
private partial def findInputBuffer (u : UOp) : Option UOpId :=
  if u.op == .BUFFER || u.op == .LOAD then
    some u.uid
  else
    match u.src with
    | s :: _ => findInputBuffer s
    | [] => none

/-- Check if u is a scalar float32 constant and return its value -/
private def getScalarF32 (u : UOp) : Option Float :=
  if u.op == .CONST && u.shape == [] && u.dtype == .float32 then
    match u.arg with
    | .constF32Bits bits => some (Float32.ofBits bits |>.toFloat)
    | _ => none
  else none

/--
Pattern match for SGD update:
  param - lr * grad
where lr is a scalar constant.
-/
def compile (u : UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : Option Plan := Id.run do
  -- Only handle float32
  if u.dtype != .float32 then return none

  -- Ignore keep/refCnt for this simple pattern
  let _ := (keep, refCnt)

  -- Root must be SUB
  if u.op != .SUB then return none

  -- Match: param - mulNode
  let [param, mulNode] := u.src | return none

  -- mulNode must be MUL
  if mulNode.op != .MUL then return none
  let [a, b] := mulNode.src | return none

  -- One of a/b is scalar constant (lr), other is grad
  let lrGrad? : Option (Float × UOp) :=
    match getScalarF32 a with
    | some lr => some (lr, b)
    | none => match getScalarF32 b with
      | some lr => some (lr, a)
      | none => none
  let some (lrVal, grad) := lrGrad? | return none

  -- param and grad should be same shape
  if param.shape != grad.shape then return none

  -- Find buffer IDs
  let some wId := findInputBuffer param | return none
  let some gradId := findInputBuffer grad | return none

  return some { w := wId, grad := gradId, lr := lrVal }

/-- Return array of plan variants (currently just 0 or 1) -/
def compileVariants (u : UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : Array Plan :=
  match compile u keep refCnt with
  | some p => #[p]
  | none => #[]

end TinyGrad4.Backend.FusedSGD
