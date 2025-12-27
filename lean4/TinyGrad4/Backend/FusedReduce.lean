import TinyGrad4.Ops
import TinyGrad4.Shape
import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import TinyGrad4.Backend.FusedEwise
import Std.Data.HashMap

namespace TinyGrad4.Backend.FusedReduce

open Std

structure Plan where
  root : UOpId
  cover : UOpIdSet
  ewise : FusedEwise.Plan
  reduceOp : Ops
  fullShape : Array Nat
  axes : Array Nat
  deriving Repr

namespace Plan

def mapIds (p : Plan) (f : UOpId → UOpId) : Plan :=
  { p with
    root := f p.root
    cover := UOpIdSet.map p.cover f
    ewise := p.ewise.mapIds f }

end Plan

private def insertNatAsc (x : Nat) : List Nat → List Nat
  | [] => [x]
  | y :: ys => if x <= y then x :: y :: ys else y :: insertNatAsc x ys

private def sortNatAsc (xs : List Nat) : List Nat :=
  xs.foldl (fun acc x => insertNatAsc x acc) []

private def dedupNat (xs : List Nat) : List Nat := Id.run do
  let mut out : List Nat := []
  for x in xs do
    if !(out.contains x) then
      out := out ++ [x]
  return out

def compile (u : UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : Option Plan := Id.run do
  if u.op != .REDUCE_AXIS then
    return none
  if u.dtype != .float32 then
    return none
  let src :=
    match u.src with
    | [s] => some s
    | _ => none
  let some src := src | return none
  if src.dtype != .float32 then
    return none
  let (reduceOp, axes) :=
    match u.arg with
    | .reduceWithAxes op ax => (some op, ax)
    | _ => (none, [])
  let some reduceOp := reduceOp | return none
  if reduceOp != .ADD && reduceOp != .MAX then
    return none
  if axes.isEmpty then
    return none
  let axes' := sortNatAsc (dedupNat axes)
  if axes'.isEmpty then
    return none
  let keepdim := u.shape.length == src.shape.length
  let expected := Shape.reduce src.shape axes' keepdim
  if expected != u.shape then
    return none
  let fullShape := src.shape.toArray
  let okAxes := axes'.all (fun ax => ax < fullShape.size)
  if !okAxes then
    return none
  let some ewise := FusedEwise.compileForReduce src keep refCnt | return none
  let cover := UOpIdSet.add ewise.cover u.uid
  return some
    { root := u.uid
      cover
      ewise
      reduceOp
      fullShape
      axes := axes'.toArray }

def compileVariants (u : UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : Array Plan :=
  match compile u keep refCnt with
  | some p => #[p]
  | none => #[]

end TinyGrad4.Backend.FusedReduce
