import TinyGrad4.Basic
import TinyGrad4.Shape

namespace TinyGrad4
namespace Spec

/-!
# Indexing semantics (NumPy-like, reference)

We model basic indexing (ints/slices/ellipsis/newaxis) with shape inference.
Advanced indexing and assignment rules are TODOs, using this as the canonical spec layer.
-/

/-- Slice spec (start/stop/step), following NumPy semantics. -/
structure Slice where
  start? : Option Int := none
  stop? : Option Int := none
  step? : Option Int := none
  deriving Repr, BEq, Inhabited

namespace Slice

/-- The full slice `:`. -/
def all : Slice := {}

/-- Step with default 1. -/
def step (s : Slice) : Int := s.step?.getD 1

end Slice

/-- Normalize a possibly-negative index into `[0, dim)` or return `none` if out of bounds. -/
def normalizeIndex (idx : Int) (dim : Nat) : Option Nat :=
  let dimI := Int.ofNat dim
  let i := if idx < 0 then idx + dimI else idx
  if i < 0 then
    none
  else if i >= dimI then
    none
  else
    some i.toNat

private def clampInt (x lo hi : Int) : Int :=
  max lo (min hi x)

/-- Normalize a slice to concrete `(start, stop, step)` with bounds applied. -/
def normalizeSlice (s : Slice) (dim : Nat) : Option (Int × Int × Int) :=
  let step := s.step?.getD 1
  if step == 0 then
    none
  else
    let dimI := Int.ofNat dim
    if step > 0 then
      let start := s.start?.getD 0
      let stop := s.stop?.getD dimI
      let start' := if start < 0 then start + dimI else start
      let stop' := if stop < 0 then stop + dimI else stop
      let start'' := clampInt start' 0 dimI
      let stop'' := clampInt stop' 0 dimI
      some (start'', stop'', step)
    else
      let start := s.start?.getD (dimI - 1)
      let stop := s.stop?.getD (-1)
      let start' := if start < 0 then start + dimI else start
      let stop' := if stop < 0 then stop + dimI else stop
      let start'' := clampInt start' (-1) (dimI - 1)
      let stop'' := clampInt stop' (-1) (dimI - 1)
      some (start'', stop'', step)

/-- Compute the number of elements selected by a slice (returns `none` if step=0). -/
def sliceSize? (s : Slice) (dim : Nat) : Option Nat :=
  match normalizeSlice s dim with
  | none => none
  | some (start, stop, step) =>
    if step > 0 then
      if start >= stop then
        some 0
      else
        let n := (stop - 1 - start) / step + 1
        some n.toNat
    else
      if start <= stop then
        some 0
      else
        let step' := -step
        let n := (start - 1 - stop) / step' + 1
        some n.toNat

/-- Basic indexing items (ints/slices/ellipsis/newaxis). -/
inductive BasicIndexItem where
  | int (idx : Int)
  | slice (s : Slice)
  | ellipsis
  | newaxis
  deriving Repr, BEq, Inhabited

namespace BasicIndexItem

def consumesDim : BasicIndexItem → Bool
  | .int _ => true
  | .slice _ => true
  | _ => false

end BasicIndexItem

private def countEllipsis (items : List BasicIndexItem) : Nat :=
  items.foldl (fun acc item =>
    match item with
    | .ellipsis => acc + 1
    | _ => acc
  ) 0

private def countConsumes (items : List BasicIndexItem) : Nat :=
  items.foldl (fun acc item =>
    if BasicIndexItem.consumesDim item then acc + 1 else acc
  ) 0

/--
Expand a single ellipsis to full slices, and append trailing slices when indices are short.
Returns `none` if indices over-consume the shape or contain multiple ellipses.
-/
def expandEllipsis (shape : Shape) (items : List BasicIndexItem) : Option (List BasicIndexItem) :=
  let ellCount := countEllipsis items
  if ellCount > 1 then
    none
  else
    let rank := shape.length
    let consumeCount := countConsumes items
    if consumeCount > rank then
      none
    else
      let missing := rank - consumeCount
      let fill := List.replicate missing (.slice Slice.all)
      if ellCount == 0 then
        some (items ++ fill)
      else
        let rec loop (items : List BasicIndexItem) : List BasicIndexItem :=
          match items with
          | [] => []
          | .ellipsis :: rest => fill ++ rest
          | item :: rest => item :: loop rest
        some (loop items)

/-- Infer the output shape for basic indexing. -/
def inferBasicIndexShape (shape : Shape) (items : List BasicIndexItem) : Option Shape :=
  match expandEllipsis shape items with
  | none => none
  | some items =>
    let rec loop (dims : List Nat) (items : List BasicIndexItem) (out : List Nat) : Option Shape :=
      match items with
      | [] =>
        if dims.isEmpty then some out.reverse else none
      | item :: rest =>
        match item with
        | .newaxis => loop dims rest (1 :: out)
        | .int idx =>
          match dims with
          | [] => none
          | dim :: dims' =>
            match normalizeIndex idx dim with
            | none => none
            | some _ => loop dims' rest out
        | .slice sl =>
          match dims with
          | [] => none
          | dim :: dims' =>
            match sliceSize? sl dim with
            | none => none
            | some size => loop dims' rest (size :: out)
        | .ellipsis => none
    loop shape items []

/-!
TODO:
- Advanced indexing: broadcast index tensors and gather (see TensorLib.Index.Advanced).
- Mixed basic/advanced indexing shape rules.
- Assignment: broadcast RHS to indexed LHS shape (TensorLib.Index.assign).
-/

end Spec
end TinyGrad4
