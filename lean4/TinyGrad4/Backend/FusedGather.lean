import TinyGrad4.Ops
import TinyGrad4.Shape
import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import TinyGrad4.Backend.Pattern
import TinyGrad4.Backend.ShapeTracker
import TinyGrad4.Backend.View
import Std.Data.HashMap

namespace TinyGrad4.Backend.FusedGather

open Std
open TinyGrad4.Backend.Pattern

structure Plan where
  root : UOpId
  cover : UOpIdSet
  xBase : UOpId
  xView : View
  idxBase : UOpId
  idxView : View
  idxItemsize : Nat
  idxSigned : Bool
  maskShape : Array Nat
  reduceAxis : Nat
  deriving Repr

namespace Plan

def mapIds (p : Plan) (f : UOpId → UOpId) : Plan :=
  { p with
    root := f p.root
    cover := UOpIdSet.map p.cover f
    xBase := f p.xBase
    idxBase := f p.idxBase }

end Plan

private def viewFor (u : UOp) (target : Shape) : Option (UOpId × View) := do
  let (base, st0) ← Backend.ShapeTracker.ofUOp? u
  let st1 ← Backend.ShapeTracker.expand st0 target
  if Backend.ShapeTracker.needsStack st1 then
    none
  else
    let v := Backend.ShapeTracker.top st1
    if !Backend.View.isFullMask v then
      none
    else
      let mut ok := true
      for i in [:v.strides.size] do
        if v.strides[i]! < 0 then
          ok := false
      if !ok then
        none
      else
        some (base, v)

private partial def baseUOp (u : UOp) : UOp :=
  if u.op.isMovement then
    match u.src with
    | [s] => baseUOp s
    | _ => u
  else
    u

private def isClassConst (u : UOp) : Bool :=
  if u.op == .VCONST then
    u.dtype.isInt
  else if u.op == .CAST then
    match u.src with
    | [s] =>
        let s0 := baseUOp s
        s0.op == .VCONST && u.dtype.isInt
    | _ => false
  else
    false

def compile (u : UOp) (_keep : UOpIdSet) (_refCnt : HashMap UOpId Nat) : Option Plan := do
  guard (u.op == .REDUCE_AXIS)
  let (reduceOp, axes) ← getReduceInfo u.arg
  guard (reduceOp == .ADD)
  let axis ←
    match axes with
    | [a] => some a
    | _ => none
  let src ←
    match u.src with
    | [s] => some s
    | _ => none
  let (cond, x, y) ← UOp.asWhere? src
  guard (UOp.isZeroConst y)
  let (a, b) ← UOp.asCmpeq? cond
  let maskShape := src.shape
  guard (axis < maskShape.length)
  let expectedOut := Shape.reduce maskShape [axis] false
  guard (u.shape == expectedOut)

  let (aBase, aView) ← viewFor a maskShape
  let (bBase, bView) ← viewFor b maskShape

  let aStride := aView.strides.getD axis 0
  let bStride := bView.strides.getD axis 0

  let (idxBase, idxView, idxItemsize, idxSigned, classBase) ←
    if aStride == 0 && bStride != 0 then
      some (aBase, aView, a.dtype.itemsize, a.dtype.isSigned, baseUOp b)
    else if bStride == 0 && aStride != 0 then
      some (bBase, bView, b.dtype.itemsize, b.dtype.isSigned, baseUOp a)
    else
      none

  guard (idxView.kernelShape == maskShape.toArray)
  guard (idxView.maskStart.size == maskShape.length)
  guard (idxView.maskEnd.size == maskShape.length)

  -- Require class operand to be a constant int vector (arange-like).
  guard (isClassConst classBase)

  -- Require index dtype to be integer.
  guard (a.dtype.isInt && b.dtype.isInt)

  let (xBase, xView) ← viewFor x maskShape

  let cover := collectCover u
  pure
    { root := u.uid
      cover
      xBase
      xView
      idxBase
      idxView
      idxItemsize
      idxSigned
      maskShape := maskShape.toArray
      reduceAxis := axis }

def compileVariants (u : UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : Array Plan :=
  match compile u keep refCnt with
  | some p => #[p]
  | none => #[]

end TinyGrad4.Backend.FusedGather
