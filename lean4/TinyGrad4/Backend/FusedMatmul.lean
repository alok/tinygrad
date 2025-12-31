import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import TinyGrad4.Shape
import TinyGrad4.Backend.View
import TinyGrad4.Backend.ShapeTracker
import TinyGrad4.Gradient.Adjoint
import Std.Data.HashMap

namespace TinyGrad4.Backend.FusedMatmul

open Std

private def hasPadReshapeChain (u : UOp) : Bool := Id.run do
  let mut cur := u
  let mut seenReshape := false
  while cur.op.isMovement && cur.src.length == 1 do
    if cur.op == .RESHAPE then
      seenReshape := true
    if seenReshape && cur.op == .PAD then
      return true
    cur := cur.src[0]!
  return false

/-!
# Fused Matmul Patterns (CPU portable)

Less-minimal but faster evaluation-time optimization:
fuse common post-matmul patterns into a single kernel.

Currently implemented:
- `ADD(CONTRACT(a, b), bias)` where `bias` broadcasts to the `[m, n]` output.
- `ADD(ADD(CONTRACT(a, b), bias0), bias1)` where both broadcasts to `[m, n]` (common: bias + residual).
- `MAX(CONTRACT(a, b), 0)` fused by treating `0` as a bias.
- Works for 2D and batched matmul (`(..., m, k) @ (..., k, n)`), float32 only.
- Requires the `CONTRACT` result to be single-use (RC == 1) and not a root to avoid duplicated matmuls.
-/

structure Plan where
  root : UOpId
  cover : UOpIdSet
  a : UOpId
  b : UOpId
  aBase : UOpId
  bBase : UOpId
  aNumel : Nat
  bNumel : Nat
  aKernelNumel : Nat
  bKernelNumel : Nat
  aStrides : Array Int64
  aOffset : Int64
  aMaskStarts : Array Nat
  aMaskEnds : Array Nat
  bStrides : Array Int64
  bOffset : Int64
  bMaskStarts : Array Nat
  bMaskEnds : Array Nat
  aStackShapes : Array (Array Nat)
  aStackStrides : Array (Array Int64)
  aStackOffsets : Array Int64
  aStackMaskStarts : Array (Array Nat)
  aStackMaskEnds : Array (Array Nat)
  bStackShapes : Array (Array Nat)
  bStackStrides : Array (Array Int64)
  bStackOffsets : Array Int64
  bStackMaskStarts : Array (Array Nat)
  bStackMaskEnds : Array (Array Nat)
  needsStack : Bool
  aFast : Bool
  bFast : Bool
  bias : UOpId
  biasNumel : Nat
  biasKernelNumel : Nat
  biasShape : Array Nat
  biasShape2d : Array Nat
  biasStrides : Array Int64
  biasOffset : Int64
  biasMaskStarts : Array Nat
  biasMaskEnds : Array Nat
  biasFast : Bool
  bias2 : Option UOpId
  bias2Numel : Nat
  bias2KernelNumel : Nat
  bias2Shape : Array Nat
  bias2Shape2d : Array Nat
  bias2Strides : Array Int64
  bias2Offset : Int64
  bias2MaskStarts : Array Nat
  bias2MaskEnds : Array Nat
  bias2Fast : Bool
  aStarts : Array Nat
  bStarts : Array Nat
  biasStarts : Array Nat
  bias2Starts : Array Nat
  m : Nat
  k : Nat
  n : Nat
  scaleBits : Option UInt32
  relu : Bool
  deriving Repr

def Plan.mapIds (p : Plan) (f : UOpId → UOpId) : Plan :=
  { p with
    root := f p.root
    cover := UOpIdSet.map p.cover f
    a := f p.a
    b := f p.b
    aBase := f p.aBase
    bBase := f p.bBase
    bias := f p.bias
    bias2 := p.bias2.map f }

private def isZeroConst (u : UOp) : Bool :=
  u.op == .CONST && u.dtype == .float32 && u.shape == [] && u.arg.getFloat.getD 1.0 == 0.0

private def isConstScalarF32 (u : UOp) : Bool :=
  u.op == .CONST && u.dtype == .float32 && u.shape == []

private def constBitsAsF32 (u : UOp) : UInt32 :=
  match u.arg with
  | .constF32Bits bits => bits
  | _ =>
    match u.arg.getFloat with
    | some v => v.toFloat32.toBits
    | none => 0

private structure MatmulTerm where
  term : UOp
  contract : UOp
  scaleBits : Option UInt32

private def matchMatmulTerm (t : UOp) : Option MatmulTerm := do
  if t.op == .CONTRACT then
    pure { term := t, contract := t, scaleBits := none }
  else if t.op == .MUL && t.dtype == .float32 && t.src.length == 2 then
    let a := t.src[0]!
    let b := t.src[1]!
    if a.op == .CONTRACT && isConstScalarF32 b then
      pure { term := t, contract := a, scaleBits := some (constBitsAsF32 b) }
    else if b.op == .CONTRACT && isConstScalarF32 a then
      pure { term := t, contract := b, scaleBits := some (constBitsAsF32 a) }
    else
      failure
  else
    failure

private def unflattenIndex (flatIdx : Nat) (shape : Shape) : List Nat :=
  shape.foldr (fun dim (idx, acc) =>
    let (q, r) := (idx / dim, idx % dim)
    (q, r :: acc)
  ) (flatIdx, []) |>.2

private def stridesOf (shape : Shape) : List Nat :=
  shape.foldr (fun dim (strides, prod) => (prod :: strides, prod * dim)) ([], 1) |>.1

private def offsetOf (indices strides : List Nat) : Nat :=
  (indices.zip strides).foldl (fun acc (idx, stride) => acc + idx * stride) 0

private def i64ToNat (x : Int64) : Nat :=
  if x < 0 then 0 else Int.toNat (Int64.toInt x)

private def batchStartsBytesFromView (v : Backend.View) (batchOut : Shape) (elemBytes : Nat) : Array Nat := Id.run do
  let rank := v.rank
  let batchRank := if rank >= 2 then rank - 2 else 0
  let batchShape := (v.kernelShape.toList).take batchRank
  let lenBatch := batchOut.length
  let batchOutNumel := listProd batchOut

  let batchShape' := List.replicate (lenBatch - batchShape.length) 1 ++ batchShape
  let batchStrides0 := v.strides.toList.take batchShape.length
  let batchStrides0Nat := batchStrides0.map i64ToNat
  let batchStrides := List.replicate (lenBatch - batchStrides0Nat.length) 0 ++ batchStrides0Nat
  let off0 := v.offset
  let off0Nat := i64ToNat off0

  let mut out : Array Nat := Array.emptyWithCapacity batchOutNumel
  for bi in [:batchOutNumel] do
    let batchIdx := unflattenIndex bi batchOut
    let idx := listZipWith (fun i d => if d == 1 then 0 else i) batchIdx batchShape'
    out := out.push ((off0Nat + offsetOf idx batchStrides) * elemBytes)
  return out

private def last2Shape (s : Shape) : Shape :=
  let r := s.length
  if r <= 2 then s else s.drop (r - 2)

private def setUnion (a b : UOpIdSet) : UOpIdSet :=
  b.fold (init := a) fun acc uid => UOpIdSet.add acc uid

private partial def collectAddTerms (u : UOp) (rootId : UOpId) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : Option (List UOp) := do
  if u.op == .ADD && u.dtype == .float32 && u.src.length == 2 &&
      (u.uid == rootId || (!UOpIdSet.member keep u.uid && refCnt.getD u.uid 0 == 1)) then
    let t0 ← collectAddTerms u.src[0]! rootId keep refCnt
    let t1 ← collectAddTerms u.src[1]! rootId keep refCnt
    pure (t0 ++ t1)
  else
    pure [u]

private partial def collectAddCover (u : UOp) (rootId : UOpId) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : Option UOpIdSet := do
  if u.op == .ADD && u.dtype == .float32 && u.src.length == 2 &&
      (u.uid == rootId || (!UOpIdSet.member keep u.uid && refCnt.getD u.uid 0 == 1)) then
    let c0 ← collectAddCover u.src[0]! rootId keep refCnt
    let c1 ← collectAddCover u.src[1]! rootId keep refCnt
    pure (UOpIdSet.add (setUnion c0 c1) u.uid)
  else
    pure UOpIdSet.mkEmpty

private def compileWith (u : UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) (virtualizeA virtualizeB : Bool) : Option Plan := do
  if u.dtype != .float32 || u.src.length != 2 then
    failure

  let x := u.src[0]!
  let y := u.src[1]!
  let (relu, innerNode, zeroConst) :=
    if u.op == .MAX && isZeroConst x && (y.op == .ADD || y.op == .CONTRACT || y.op == .MUL) then
      (true, y, some x)
    else if u.op == .MAX && isZeroConst y && (x.op == .ADD || x.op == .CONTRACT || x.op == .MUL) then
      (true, x, some y)
    else
      (false, u, none)

  if relu && !Gradient.isReluLike u then
    failure

  if !relu && u.op != .ADD then
    failure

  if relu then
    if UOpIdSet.member keep innerNode.uid then
      failure
    if refCnt.getD innerNode.uid 0 != 1 then
      failure

  let terms ←
    if innerNode.op == .ADD then
      collectAddTerms innerNode innerNode.uid keep refCnt
    else
      some [innerNode]

  let matmulTerms := terms.filterMap matchMatmulTerm
  let matmulTerm ←
    match matmulTerms with
    | t :: [] => some t
    | _ => none
  let contract := matmulTerm.contract
  if matmulTerm.term.uid != contract.uid then
    if UOpIdSet.member keep matmulTerm.term.uid then
      failure
    if refCnt.getD matmulTerm.term.uid 0 != 1 then
      failure

  if matmulTerm.term.shape != u.shape || contract.shape != u.shape then
    failure

  let biases0 := terms.filter (fun t => t.uid != matmulTerm.term.uid)
  let biases :=
    if biases0.isEmpty && relu then
      match zeroConst with
      | some z => [z]
      | none => []
    else
      biases0

  let (bias, bias2) ←
    match biases with
    | b0 :: [] => some (b0, none)
    | b0 :: b1 :: [] => some (b0, some b1)
    | _ => none

  if biases.isEmpty then
    failure

  if contract.dtype != .float32 || bias.dtype != .float32 then
    failure
  if let some b2 := bias2 then
    if b2.dtype != .float32 then
      failure
  if UOpIdSet.member keep contract.uid then
    failure
  if refCnt.getD contract.uid 0 != 1 then
    failure

  if contract.src.length != 2 then
    failure
  let a := contract.src[0]!
  let b := contract.src[1]!
  if a.dtype != .float32 || b.dtype != .float32 then
    failure
  let rA := a.shape.length
  let rB := b.shape.length
  if rA < 2 || rB < 2 then
    failure

  let m := listGetD a.shape (rA - 2) 0
  let k := listGetD a.shape (rA - 1) 0
  let k2 := listGetD b.shape (rB - 2) 0
  let n := listGetD b.shape (rB - 1) 0
  if k != k2 then
    failure
  if contract.shape != innerNode.shape || contract.shape != u.shape then
    failure

  let okBroadcast (b : UOp) : Option Unit := do
    match Shape.broadcast b.shape contract.shape with
    | none => failure
    | some outSh =>
      if outSh != contract.shape then
        failure
    pure ()

  okBroadcast bias
  if let some b2 := bias2 then
    okBroadcast b2

  let outRank := contract.shape.length
  let batchOut := if outRank > 2 then contract.shape.take (outRank - 2) else []

  let mut cover := UOpIdSet.add UOpIdSet.mkEmpty u.uid
  cover := UOpIdSet.add cover contract.uid
  if matmulTerm.term.uid != contract.uid then
    cover := UOpIdSet.add cover matmulTerm.term.uid
  if relu then
    cover := UOpIdSet.add cover innerNode.uid
  if innerNode.op == .ADD then
    let addCover ← collectAddCover innerNode innerNode.uid keep refCnt
    cover := setUnion cover addCover

  let outShape := contract.shape
  let aTarget : Shape := batchOut ++ [m, k]
  let bTarget : Shape := batchOut ++ [k, n]
  let outTarget : Shape := outShape

  let viewStackFor (u : UOp) (target : Shape) (virtualize : Bool) : UOpId × Backend.ShapeTracker := Id.run do
    let canTry := virtualize && u.op.isMovement && !UOpIdSet.member keep u.uid && refCnt.getD u.uid 0 == 1
    if !canTry then
      let v0 := Backend.View.contiguous u.shape
      let v := (v0.expand target).getD (Backend.View.contiguous target)
      return (u.uid, Backend.ShapeTracker.ofViews #[v])
    if hasPadReshapeChain u then
      match Backend.ShapeTracker.ofUOp? u with
      | some (base, st0) =>
        let st := (st0.expand target).getD (Backend.ShapeTracker.contiguous target)
        return (base, st)
      | none =>
        let v0 := Backend.View.contiguous u.shape
        let v := (v0.expand target).getD (Backend.View.contiguous target)
        return (u.uid, Backend.ShapeTracker.ofViews #[v])
    match Backend.View.ofUOp? u with
    | some (base, v0) =>
      let v := (v0.expand target).getD (Backend.View.contiguous target)
      return (base, Backend.ShapeTracker.ofViews #[v])
    | none =>
      match Backend.ShapeTracker.ofUOp? u with
      | some (base, st0) =>
        let st := (st0.expand target).getD (Backend.ShapeTracker.contiguous target)
        return (base, st)
      | none =>
        let v0 := Backend.View.contiguous u.shape
        let v := (v0.expand target).getD (Backend.View.contiguous target)
        return (u.uid, Backend.ShapeTracker.ofViews #[v])

  let (aBase, aSt) := viewStackFor a aTarget virtualizeA
  let aV := aSt.top
  let aFast := aV.isContiguousOrBroadcast && aV.isFullMask && aV.offset == (0 : Int64)
  let aKernelNumel := listProd (Backend.ShapeTracker.kernelShape aSt).toList
  let aStack := Backend.ShapeTracker.stackInfo aSt

  let (bBase, bSt) := viewStackFor b bTarget virtualizeB
  let bV := bSt.top
  let bFast := bV.isContiguousOrBroadcast && bV.isFullMask && bV.offset == (0 : Int64)
  let bKernelNumel := listProd (Backend.ShapeTracker.kernelShape bSt).toList
  let bStack := Backend.ShapeTracker.stackInfo bSt

  let biasV0 :=
    match Backend.View.ofUOp? bias with
    | some (_, v) => v
    | none => Backend.View.contiguous bias.shape
  let biasV :=
    match biasV0.expand outTarget with
    | some v => v
    | none => Backend.View.contiguous outTarget
  let biasFast := biasV.isContiguousOrBroadcast && biasV.isFullMask && biasV.offset == (0 : Int64)
  let biasKernelNumel := listProd biasV.kernelShape.toList

  let (bias2VOpt, bias2Strides, bias2Offset, bias2MaskStarts, bias2MaskEnds, bias2Fast) :=
    match bias2 with
    | some b2 =>
      let v0 :=
        match Backend.View.ofUOp? b2 with
        | some (_, v) => v
        | none => Backend.View.contiguous b2.shape
      let v :=
        match v0.expand outTarget with
        | some v => v
        | none => Backend.View.contiguous outTarget
      let fast := v.isContiguousOrBroadcast && v.isFullMask && v.offset == (0 : Int64)
      (some v, v.strides, v.offset, v.maskStart, v.maskEnd, fast)
    | none =>
      (none, #[], 0, #[], #[], true)
  let bias2KernelNumel :=
    match bias2VOpt with
    | some v => listProd v.kernelShape.toList
    | none => 0

  let aStarts := if batchOut.isEmpty then #[] else batchStartsBytesFromView aV batchOut 4
  let bStarts := if batchOut.isEmpty then #[] else batchStartsBytesFromView bV batchOut 4
  let biasStarts := if batchOut.isEmpty then #[] else batchStartsBytesFromView biasV batchOut 4
  let bias2Starts :=
    match bias2VOpt with
    | some v => if batchOut.isEmpty then #[] else batchStartsBytesFromView v batchOut 4
    | none => #[]

  let markMovementChain (cover : UOpIdSet) (u : UOp) : UOpIdSet := Id.run do
    let mut cover := cover
    let mut cur := u
    while cur.op.isMovement && cur.src.length == 1 do
      if !UOpIdSet.member keep cur.uid && refCnt.getD cur.uid 0 == 1 then
        cover := UOpIdSet.add cover cur.uid
      cur := cur.src[0]!
    return cover

  if a.op.isMovement && !UOpIdSet.member keep a.uid && refCnt.getD a.uid 0 == 1 && aBase != a.uid then
    cover := markMovementChain cover a
  if b.op.isMovement && !UOpIdSet.member keep b.uid && refCnt.getD b.uid 0 == 1 && bBase != b.uid then
    cover := markMovementChain cover b

  let needsStack := Backend.ShapeTracker.needsStack aSt || Backend.ShapeTracker.needsStack bSt

  pure
    { root := u.uid
      cover
      a := a.uid
      b := b.uid
      aBase
      bBase
      bias := bias.uid
      aNumel := listProd a.shape
      bNumel := listProd b.shape
      aKernelNumel
      bKernelNumel
      aStrides := aV.strides
      aOffset := aV.offset
      aMaskStarts := aV.maskStart
      aMaskEnds := aV.maskEnd
      bStrides := bV.strides
      bOffset := bV.offset
      bMaskStarts := bV.maskStart
      bMaskEnds := bV.maskEnd
      aStackShapes := aStack.shapes
      aStackStrides := aStack.strides
      aStackOffsets := aStack.offsets
      aStackMaskStarts := aStack.maskStarts
      aStackMaskEnds := aStack.maskEnds
      bStackShapes := bStack.shapes
      bStackStrides := bStack.strides
      bStackOffsets := bStack.offsets
      bStackMaskStarts := bStack.maskStarts
      bStackMaskEnds := bStack.maskEnds
      needsStack
      aFast
      bFast
      biasNumel := listProd bias.shape
      biasKernelNumel
      biasShape := bias.shape.toArray
      biasShape2d := (last2Shape bias.shape).toArray
      biasStrides := biasV.strides
      biasOffset := biasV.offset
      biasMaskStarts := biasV.maskStart
      biasMaskEnds := biasV.maskEnd
      biasFast
      bias2 := bias2.map (fun b => b.uid)
      bias2Numel := bias2.map (fun b => listProd b.shape) |>.getD 0
      bias2KernelNumel
      bias2Shape := bias2.map (fun b => b.shape.toArray) |>.getD #[]
      bias2Shape2d := bias2.map (fun b => (last2Shape b.shape).toArray) |>.getD #[]
      bias2Strides
      bias2Offset
      bias2MaskStarts
      bias2MaskEnds
      bias2Fast
      aStarts
      bStarts
      biasStarts
      bias2Starts
      m, k, n, scaleBits := matmulTerm.scaleBits, relu }

def compile (u : UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : Option Plan :=
  compileWith u keep refCnt true true

def compileVariants (u : UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : Array Plan := Id.run do
  let mut out : Array Plan := #[]
  match compileWith u keep refCnt true true with
  | none => return out
  | some p0 =>
    out := out.push p0
    let aVirt := p0.aBase != p0.a
    let bVirt := p0.bBase != p0.b
    if aVirt then
      match compileWith u keep refCnt false true with
      | some p => out := out.push p
      | none => pure ()
    if bVirt then
      match compileWith u keep refCnt true false with
      | some p => out := out.push p
      | none => pure ()
    if aVirt || bVirt then
      match compileWith u keep refCnt false false with
      | some p => out := out.push p
      | none => pure ()
    return out

end TinyGrad4.Backend.FusedMatmul
