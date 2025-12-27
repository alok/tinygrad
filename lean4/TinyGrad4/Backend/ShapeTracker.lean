import TinyGrad4.Shape
import TinyGrad4.UOp.UOp
import TinyGrad4.Backend.View

namespace TinyGrad4.Backend

/-- Shape tracker: a stack of views. -/
structure ShapeTracker where
  views : Array View
  deriving Repr

structure StackInfo where
  shapes : Array (Array Nat)
  strides : Array (Array Int64)
  offsets : Array Int64
  maskStarts : Array (Array Nat)
  maskEnds : Array (Array Nat)
  deriving Repr

namespace ShapeTracker

/-- Top view (fallback to contiguous scalar). -/
def top (st : ShapeTracker) : View :=
  match st.views.back? with
  | some v => v
  | none => View.contiguous []

/-- Replace the top view (or start a new stack). -/
def replaceTop (st : ShapeTracker) (v : View) : ShapeTracker :=
  if st.views.isEmpty then
    { views := #[v] }
  else
    { views := st.views.set! (st.views.size - 1) v }

/-- Build a tracker from views. -/
def ofViews (views : Array View) : ShapeTracker :=
  { views }

/-- Contiguous tracker for a given shape. -/
def contiguous (shape : Shape) : ShapeTracker :=
  ofViews #[View.contiguous shape]

/-- Apply a movement op to a tracker (stacking if needed). -/
def applyMovement (st : ShapeTracker) (u : UOp) : Option ShapeTracker := do
  let v := top st
  match View.applyMovement v u with
  | some v' => some (replaceTop st v')
  | none =>
    let base := View.contiguous v.kernelShape.toList
    let v' ← View.applyMovement base u
    return { views := st.views.push v' }

/-- Attempt to recover a tracker from a UOp. -/
partial def ofUOp? (u : UOp) : Option (UOpId × ShapeTracker) := do
  if u.op.isMovement then
    match u.src with
    | [src] =>
      let (base, st) ← ofUOp? src
      let st' ← applyMovement st u
      return (base, st')
    | _ => none
  else
    return (u.uid, contiguous u.shape)

/-- Expand to a target shape. -/
def expand (st : ShapeTracker) (target : Shape) : Option ShapeTracker := do
  let v ← View.expand (top st) target
  return replaceTop st v

/-- Kernel shape for lowering (from the top view). -/
def kernelShape (st : ShapeTracker) : Array Nat :=
  (top st).kernelShape

/-- Stack metadata for the top view. -/
def stackInfo (st : ShapeTracker) : StackInfo :=
  let views := if st.views.isEmpty then #[top st] else st.views
  { shapes := views.map (fun v => v.kernelShape)
    strides := views.map (fun v => v.strides)
    offsets := views.map (fun v => v.offset)
    maskStarts := views.map (fun v => v.maskStart)
    maskEnds := views.map (fun v => v.maskEnd) }

/-- Whether stacking metadata is required. -/
def needsStack (_st : ShapeTracker) : Bool :=
  _st.views.size > 1

end ShapeTracker

end TinyGrad4.Backend
