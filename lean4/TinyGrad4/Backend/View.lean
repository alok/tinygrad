import TinyGrad4.Shape
import TinyGrad4.UOp.UOp

namespace TinyGrad4.Backend

/-- View description for backend lowering. -/
structure View where
  kernelShape : Array Nat
  strides : Array Int64
  offset : Int64
  maskStart : Array Nat
  maskEnd : Array Nat
  deriving Repr

namespace View

private def i64OfNat (n : Nat) : Int64 :=
  Int64.ofInt (Int.ofNat n)

private def stridesOf (shape : Shape) : Array Int64 :=
  let strides := Shape.unitStrides shape
  (strides.map (fun i => Int64.ofInt i)).toArray

/-- Contiguous row-major view. -/
def contiguous (shape : Shape) : View :=
  let k := shape.toArray
  { kernelShape := k
    strides := stridesOf shape
    offset := 0
    maskStart := Array.replicate k.size 0
    maskEnd := k }

/-- Rank of a view. -/
def rank (v : View) : Nat :=
  v.kernelShape.size

/-- Mask is full when it covers the whole kernel shape. -/
def isFullMask (v : View) : Bool := Id.run do
  if v.maskStart.size != v.kernelShape.size || v.maskEnd.size != v.kernelShape.size then
    return false
  let mut ok := true
  for i in [:v.kernelShape.size] do
    if v.maskStart[i]! != 0 || v.maskEnd[i]! != v.kernelShape[i]! then
      ok := false
  return ok

private def isContiguousStrict (v : View) : Bool := Id.run do
  if v.strides.size != v.kernelShape.size then
    return false
  let expected := stridesOf v.kernelShape.toList
  let mut ok := true
  for i in [:v.kernelShape.size] do
    if v.strides[i]! != expected[i]! then
      ok := false
  return ok

/-- Contiguous or broadcast view (stride=0 allowed, no negative strides). -/
def isContiguousOrBroadcast (v : View) : Bool := Id.run do
  if v.strides.size != v.kernelShape.size then
    return false
  let expected := stridesOf v.kernelShape.toList
  let mut ok := true
  for i in [:v.kernelShape.size] do
    let s := v.strides[i]!
    if s < 0 then
      ok := false
    else if s == expected[i]! then
      ok := ok && true
    else if s == 0 then
      ok := ok && true
    else
      ok := false
  return ok

/-- Expand to target shape (broadcast-friendly).
    Handles rank mismatches by prepending dimensions with stride 0. -/
def expand (v : View) (target : Shape) : Option View := Id.run do
  let tgt := target.toArray
  let inRank := v.kernelShape.size
  let outRank := tgt.size
  if v.maskStart.size != inRank || v.maskEnd.size != inRank then
    return none
  -- Handle rank mismatch: prepend 1s to input shape
  let prependCount := if outRank > inRank then outRank - inRank else 0
  let paddedShape := (Array.replicate prependCount 1) ++ v.kernelShape
  let paddedStrides := (Array.replicate prependCount 0) ++ v.strides
  let paddedMaskStart := (Array.replicate prependCount 0) ++ v.maskStart
  let paddedMaskEnd := (Array.replicate prependCount 1) ++ v.maskEnd
  if paddedShape.size != outRank then
    return none
  let mut strides := Array.emptyWithCapacity outRank
  let mut maskStart := Array.emptyWithCapacity outRank
  let mut maskEnd := Array.emptyWithCapacity outRank
  for i in [:outRank] do
    let inDim := paddedShape[i]!
    let outDim := tgt[i]!
    if inDim == outDim then
      strides := strides.push paddedStrides[i]!
      maskStart := maskStart.push paddedMaskStart[i]!
      maskEnd := maskEnd.push paddedMaskEnd[i]!
    else if inDim == 1 then
      if paddedMaskStart[i]! != 0 || paddedMaskEnd[i]! != 1 then
        return none
      strides := strides.push 0
      maskStart := maskStart.push 0
      maskEnd := maskEnd.push outDim
    else
      return none
  return some { v with kernelShape := tgt, strides, maskStart, maskEnd }

private def reshape (v : View) (target : Shape) : Option View :=
  if listProd target != listProd v.kernelShape.toList then
    none
  else if !isFullMask v || !isContiguousStrict v then
    none
  else
    let k := target.toArray
    some
      { kernelShape := k
        strides := stridesOf target
        offset := v.offset
        maskStart := Array.replicate k.size 0
        maskEnd := k }

private def permute (v : View) (perm : List Nat) : Option View := Id.run do
  if !Shape.permuteValid v.kernelShape.toList perm then
    return none
  let permArr := perm.toArray
  let mut k := Array.emptyWithCapacity permArr.size
  let mut strides := Array.emptyWithCapacity permArr.size
  let mut maskStart := Array.emptyWithCapacity permArr.size
  let mut maskEnd := Array.emptyWithCapacity permArr.size
  for i in [:permArr.size] do
    let idx := permArr[i]!
    if idx >= v.kernelShape.size then
      return none
    k := k.push v.kernelShape[idx]!
    strides := strides.push v.strides[idx]!
    maskStart := maskStart.push v.maskStart[idx]!
    maskEnd := maskEnd.push v.maskEnd[idx]!
  return some { v with kernelShape := k, strides, maskStart, maskEnd }

private def pad (v : View) (padding : List (Nat × Nat)) : Option View := Id.run do
  if padding.length != v.kernelShape.size then
    return none
  let padArr := padding.toArray
  let mut k := Array.emptyWithCapacity v.kernelShape.size
  let mut maskStart := Array.emptyWithCapacity v.kernelShape.size
  let mut maskEnd := Array.emptyWithCapacity v.kernelShape.size
  let mut offset := v.offset
  for i in [:v.kernelShape.size] do
    let (l, r) := padArr[i]!
    let dim := v.kernelShape[i]!
    k := k.push (dim + l + r)
    maskStart := maskStart.push (v.maskStart[i]! + l)
    maskEnd := maskEnd.push (v.maskEnd[i]! + l)
    offset := offset - i64OfNat l * v.strides[i]!
  return some { v with kernelShape := k, maskStart, maskEnd, offset }

private def shrink (v : View) (bounds : List (Nat × Nat)) : Option View := Id.run do
  if bounds.length != v.kernelShape.size then
    return none
  let bArr := bounds.toArray
  let mut k := Array.emptyWithCapacity v.kernelShape.size
  let mut maskStart := Array.emptyWithCapacity v.kernelShape.size
  let mut maskEnd := Array.emptyWithCapacity v.kernelShape.size
  let mut offset := v.offset
  for i in [:v.kernelShape.size] do
    let (start, stop) := bArr[i]!
    let dim := v.kernelShape[i]!
    if stop < start || stop > dim then
      return none
    let outDim := stop - start
    k := k.push outDim
    offset := offset + i64OfNat start * v.strides[i]!
    let ms := v.maskStart[i]!
    let me := v.maskEnd[i]!
    let interStart := Nat.max ms start
    let interEnd := Nat.min me stop
    if interEnd <= interStart then
      maskStart := maskStart.push 0
      maskEnd := maskEnd.push 0
    else
      maskStart := maskStart.push (interStart - start)
      maskEnd := maskEnd.push (interEnd - start)
  return some { v with kernelShape := k, maskStart, maskEnd, offset }

private def flip (v : View) (axes : List Nat) : Option View := Id.run do
  if !isFullMask v then
    return none
  let rank := v.kernelShape.size
  let mut strides := v.strides
  let mut offset := v.offset
  for ax in axes do
    if ax >= rank then
      return none
    let dim := v.kernelShape[ax]!
    let stride := strides[ax]!
    strides := strides.set! ax (-stride)
    if dim > 0 then
      offset := offset + i64OfNat (dim - 1) * stride
  return some { v with strides, offset }

/-- Apply a single movement op to a view. -/
def applyMovement (v : View) (u : UOp) : Option View := do
  if !u.op.isMovement then
    none
  else
    match u.op with
    | .RESHAPE => reshape v u.shape
    | .EXPAND => expand v u.shape
    | .PERMUTE =>
      match u.arg.getPermutation with
      | some perm => permute v perm
      | none => none
    | .PAD =>
      match u.arg with
      | .padding padding => pad v padding
      | _ => none
    | .SHRINK =>
      match u.arg with
      | .bounds bounds => shrink v bounds
      | _ => none
    | .FLIP =>
      match u.arg with
      | .axes axes => flip v axes
      | _ => none
    | _ => none

/-- Attempt to recover a view directly from a UOp. -/
partial def ofUOp? (u : UOp) : Option (UOpId × View) := do
  if u.op.isMovement then
    match u.src with
    | [src] =>
      let (base, v) ← ofUOp? src
      let v' ← applyMovement v u
      return (base, v')
    | _ => none
  else
    return (u.uid, contiguous u.shape)

end View

end TinyGrad4.Backend
