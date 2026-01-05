import TinyGrad4.Ops
import TinyGrad4.Shape
import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import TinyGrad4.Backend.Pattern
import Std.Data.HashMap

namespace TinyGrad4.Backend.FusedSoftmax

open Std
open TinyGrad4.Backend.Pattern

/-!
# Fused Softmax Pattern Matching

Matches the stable softmax pattern:
  {lit}`softmax(x, axis) = exp(x - max(x, axis)) / sum(exp(x - max(x, axis)), axis)`

In UOp terms with tinygrad's base-2 ops:
  {lit}`FDIV`
  {lit}`├── EXP2(LOG2E * (x - REDUCE_AXIS(MAX, x, axis)))`
  {lit}`└── REDUCE_AXIS(ADD, EXP2(...), axis)`

Also matches log-softmax variant.

## Refactored to use Pattern module

This module now uses composable pattern matchers from `TinyGrad4.Backend.Pattern`
instead of duplicating helper functions. The Pattern module provides:
- Primitive matchers: `asDiv?`, `asExp2?`, `asReduceAdd?`, etc.
- Composite patterns: `Pattern.softmax?`
- Utilities: `collectCover`, `findInputBuffer`
-/

structure Plan where
  /-- Root node ID (the FDIV or final SUB for log-softmax) -/
  root : UOpId
  /-- All nodes covered by this fusion -/
  cover : UOpIdSet
  /-- Input buffer ID -/
  input : UOpId
  /-- Product of batch dimensions (all dims except softmax axis) -/
  outer : Nat
  /-- Size of softmax dimension -/
  inner : Nat
  /-- Input shape for proper indexing -/
  inputShape : Array Nat
  /-- Axis along which softmax is computed -/
  axis : Nat
  /-- Scale factor bits (1.0 for softmax) for Native kernels -/
  scaleBits : UInt32 := (1.0 : Float).toFloat32.toBits
  /-- ln(2) bits for log operations -/
  ln2Bits : UInt32 := (Float.log 2.0).toFloat32.toBits
  /-- Whether this is log-softmax (true) or regular softmax (false) -/
  log : Bool
  deriving Repr

namespace Plan

def mapIds (p : Plan) (f : UOpId → UOpId) : Plan :=
  { p with
    root := f p.root
    cover := UOpIdSet.map p.cover f
    input := f p.input }

end Plan

/-- Compute {lit}`outer*inner` from shape and axis. -/
private def computeOuterInner (shape : Array Nat) (axis : Nat) : Nat × Nat :=
  let outer := (shape.toList.take axis).foldl (· * ·) 1
  let inner := shape.getD axis 1
  let trailing := (shape.toList.drop (axis + 1)).foldl (· * ·) 1
  (outer * trailing, inner)

/--
Pattern match for stable softmax using composable Pattern module.

Uses `Pattern.softmax?` for structural matching, then builds the Plan
with computed dimensions for the interpreter.
-/
def compile (u : UOp) (_keep : UOpIdSet) (_refCnt : HashMap UOpId Nat) : Option Plan := do
  -- Only handle float32
  guard (u.dtype == .float32)

  -- Use Pattern.softmax? for structural matching
  let info ← softmax? u

  -- Find the actual input buffer
  let inputId ← UOp.findInputBuffer info.input

  -- Validate and extract axis
  guard (!info.axes.isEmpty)
  let axis := info.axis
  let inputShape := info.input.shape.toArray
  guard (axis < inputShape.size)

  -- Compute dimensions
  let (outer, inner) := computeOuterInner inputShape axis

  -- Build cover set using Pattern utility
  let cover := collectCover u

  pure {
    root := u.uid
    cover
    input := inputId
    outer
    inner
    inputShape
    axis
    log := info.isLog
  }

/-- Return array of plan variants (currently just 0 or 1) -/
def compileVariants (u : UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : Array Plan :=
  match compile u keep refCnt with
  | some p => #[p]
  | none => #[]

end TinyGrad4.Backend.FusedSoftmax
