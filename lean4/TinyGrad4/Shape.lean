import TinyGrad4.Basic

namespace TinyGrad4

/-!
# Shape System for TinyGrad4

Static shapes encoded as `List Nat` for compile-time verification.
-/

/-- Shape is a list of dimension sizes (all known at compile time) -/
abbrev Shape := List Nat

namespace Shape

/-- Total number of elements in a shape -/
def numel (s : Shape) : Nat := listProd s

/-- Number of dimensions (rank) -/
def rank (s : Shape) : Nat := s.length

/-- Get dimension at index (0 if out of bounds) -/
def dim (s : Shape) (i : Nat) : Nat := listGetD s i 0

/-- Check if two shapes are broadcastable -/
def broadcastable (s1 s2 : Shape) : Bool :=
  let len := max s1.length s2.length
  let s1' := List.replicate (len - s1.length) 1 ++ s1
  let s2' := List.replicate (len - s2.length) 1 ++ s2
  listAll (fun (d1, d2) => d1 == d2 || d1 == 1 || d2 == 1) (s1'.zip s2')

/-- Compute broadcast output shape -/
def broadcast (s1 s2 : Shape) : Option Shape :=
  if broadcastable s1 s2 then
    let len := max s1.length s2.length
    let s1' := List.replicate (len - s1.length) 1 ++ s1
    let s2' := List.replicate (len - s2.length) 1 ++ s2
    some ((s1'.zip s2').map fun (d1, d2) => max d1 d2)
  else
    none

/-- Broadcast output shape, computed unconditionally.

This is useful for dependent typing: if `broadcast s1 s2 = some out`,
then `broadcastOut s1 s2` is definitionally the same `out`.
When shapes are not broadcastable, callers should still check and reject the operation.
-/
def broadcastOut (s1 s2 : Shape) : Shape :=
  let len := max s1.length s2.length
  let s1' := List.replicate (len - s1.length) 1 ++ s1
  let s2' := List.replicate (len - s2.length) 1 ++ s2
  (s1'.zip s2').map fun (d1, d2) => max d1 d2

/-- Typeclass witness for valid broadcast shapes. -/
class BroadcastShape (s1 s2 : Shape) (out : Shape) : Prop where
  h_out : out = broadcastOut s1 s2
  h_valid : broadcastable s1 s2 = true

instance broadcastShapeInst (s1 s2 : Shape) :
    BroadcastShape s1 s2 (broadcastOut s1 s2) where
  h_out := rfl
  h_valid := sorry_proof

/-- Check if concat is valid (same rank, same dims except axis). -/
def concatValid (s1 s2 : Shape) (axis : Nat) : Bool :=
  s1.length == s2.length &&
  axis < s1.length &&
  listAll (fun i => if i == axis then true else dim s1 i == dim s2 i) (listRange s1.length)

/-- Check if concat is valid for a list of shapes (same rank, same dims except axis). -/
def concatListValid (shapes : List Shape) (axis : Nat) : Bool :=
  match shapes with
  | [] => false
  | s :: ss =>
    axis < s.length &&
    listAll (fun s' => concatValid s s' axis) ss

/-- Concat output shape, computed unconditionally.

If inputs are invalid, callers should check and reject the operation.
-/
def concatOut (s1 s2 : Shape) (axis : Nat) : Shape :=
  (listRange s1.length).map fun i =>
    if i == axis then dim s1 i + dim s2 i else dim s1 i

/-- Concat output shape for a list, computed unconditionally.

If inputs are invalid, callers should check and reject the operation.
-/
def concatOutList (shapes : List Shape) (axis : Nat) : Shape :=
  match shapes with
  | [] => []
  | s :: ss =>
    let axisDim := (s :: ss).foldl (fun acc sh => acc + dim sh axis) 0
    (listRange s.length).map fun i =>
      if i == axis then axisDim else dim s i

/-- Typeclass witness for valid concat shapes. -/
class ConcatShape (s1 s2 : Shape) (axis : Nat) (out : Shape) : Prop where
  h_out : out = concatOut s1 s2 axis
  h_valid : concatValid s1 s2 axis = true

instance concatShapeInst (s1 s2 : Shape) (axis : Nat) :
    ConcatShape s1 s2 axis (concatOut s1 s2 axis) where
  h_out := rfl
  h_valid := sorry_proof

/-- Typeclass witness for valid concat shapes (list). -/
class ConcatListShape (shapes : List Shape) (axis : Nat) (out : Shape) : Prop where
  h_out : out = concatOutList shapes axis
  h_valid : concatListValid shapes axis = true

instance concatListShapeInst (shapes : List Shape) (axis : Nat) :
    ConcatListShape shapes axis (concatOutList shapes axis) where
  h_out := rfl
  h_valid := sorry_proof

/-- Insert a dimension of size `dim` at `axis` (axis clamped to [0, rank]). -/
def insertDim (s : Shape) (axis : Nat) (dim : Nat) : Shape :=
  let pre := s.take axis
  let post := s.drop axis
  pre ++ [dim] ++ post

/-- Stack output shape (new axis length = number of tensors). -/
def stackOut (shapes : List Shape) (axis : Nat) : Shape :=
  match shapes with
  | [] => []
  | s :: _ => insertDim s axis shapes.length

/-- Row-major unit strides for a shape. -/
def unitStrides (s : Shape) : List Int :=
  s.foldr (fun dim (strides, prod) =>
    let stride : Int := Int.ofNat prod
    (stride :: strides, prod * dim)
  ) ([], 1) |>.1

/-- Compute broadcasted strides from `(fromShape, fromStrides)` to `toShape`.
Returns `none` when broadcasting is invalid. -/
def broadcastStrides (fromShape : Shape) (fromStrides : List Int) (toShape : Shape) : Option (List Int) :=
  if fromShape.length != fromStrides.length then
    none
  else
    let rec loop (src : List (Nat × Int)) (dst : List Nat) : Option (List Int) :=
      match src, dst with
      | [], [] => some []
      | (xDim, xStride) :: xs, yDim :: ys =>
        if xDim == yDim then
          match loop xs ys with
          | some rest => some (xStride :: rest)
          | none => none
        else if xDim == 1 then
          match loop xs ys with
          | some rest => some (0 :: rest)
          | none => none
        else
          none
      | [], _ :: ys =>
        match loop [] ys with
        | some rest => some (0 :: rest)
        | none => none
      | _ :: _, [] => none
    let fromPairs := (fromShape.zip fromStrides).reverse
    let out := loop fromPairs toShape.reverse
    out.map List.reverse

/-- Check if reshape is valid (product preserved) -/
def reshapeValid (from_ to : Shape) : Bool :=
  from_.numel == to.numel

/-- Check if expand is valid (can only expand dims of size 1) -/
def expandValid (from_ to : Shape) : Bool :=
  from_.length == to.length &&
  listAll (fun (d1, d2) => d1 == d2 || d1 == 1) (from_.zip to)

/-- Check if permutation is valid -/
def permuteValid (s : Shape) (perm : List Nat) : Bool :=
  perm.length == s.length &&
  listAll (fun i => perm.contains i) (listRange s.length)

/-- Apply permutation to shape -/
def permute (s : Shape) (perm : List Nat) : Shape :=
  perm.map (fun i => listGetD s i 0)

/-- Apply padding to shape -/
def pad (s : Shape) (padding : List (Nat × Nat)) : Shape :=
  listZipWith (fun d (l, r) => d + l + r) s padding

/-- Apply shrink (slice) to shape -/
def shrink (_ : Shape) (bounds : List (Nat × Nat)) : Shape :=
  bounds.map fun (start, stop) => stop - start

/-- Check if shrink bounds are valid -/
def shrinkValid (s : Shape) (bounds : List (Nat × Nat)) : Bool :=
  s.length == bounds.length &&
  listAll (fun ((d, (start, stop)) : Nat × Nat × Nat) =>
    start <= stop && stop <= d) (s.zip bounds)

/-- Apply reduction (set reduced dims to 1) -/
def reduce (s : Shape) (axes : List Nat) (keepdim : Bool := true) : Shape :=
  if keepdim then
    (listEnum s).map fun (i, d) => if axes.contains i then 1 else d
  else
    (listEnum s).filterMap fun (i, d) => if axes.contains i then none else some d

/-- Check if reduce axes are within bounds. -/
def reduceValid (s : Shape) (axes : List Nat) : Bool :=
  listAll (fun a => decide (a < s.length)) axes

/-- Typeclass witness for valid reduce shapes. -/
class ReduceShape (s : Shape) (axes : List Nat) (keepdim : Bool) (out : Shape) : Prop where
  h_out : out = reduce s axes keepdim
  h_valid : reduceValid s axes = true

instance reduceShapeInst (s : Shape) (axes : List Nat) (keepdim : Bool) :
    ReduceShape s axes keepdim (reduce s axes keepdim) where
  h_out := rfl
  h_valid := sorry_proof

/-- Matmul output shape for tensors with rank ≥ 2.
    Semantics follow tinygrad/PyTorch:
    (..., m, k) @ (..., k, n) -> (..., m, n) with broadcast on leading dims. -/
def matmulShape (s1 s2 : Shape) : Option Shape :=
  let r1 := s1.length
  let r2 := s2.length
  if r1 < 2 || r2 < 2 then
    none
  else
    let m := listGetD s1 (r1 - 2) 0
    let k1 := listGetD s1 (r1 - 1) 0
    let k2 := listGetD s2 (r2 - 2) 0
    let n := listGetD s2 (r2 - 1) 0
    if k1 != k2 then
      none
    else
      let batch1 := s1.take (r1 - 2)
      let batch2 := s2.take (r2 - 2)
      match broadcast batch1 batch2 with
      | some batchOut => some (batchOut ++ [m, n])
      | none => none

/-- Matmul output shape computed unconditionally (use with `MatmulShape`). -/
def matmulOut (s1 s2 : Shape) : Shape :=
  let r1 := s1.length
  let r2 := s2.length
  let m := listGetD s1 (r1 - 2) 0
  let n := listGetD s2 (r2 - 1) 0
  let batch1 := s1.take (r1 - 2)
  let batch2 := s2.take (r2 - 2)
  broadcastOut batch1 batch2 ++ [m, n]

/-- Check if matmul is valid -/
def matmulValid (s1 s2 : Shape) : Bool :=
  (matmulShape s1 s2).isSome

/-- Typeclass witness for valid matmul shapes. -/
class MatmulShape (s1 s2 : Shape) (out : Shape) : Prop where
  h_out : out = matmulOut s1 s2
  h_valid : matmulValid s1 s2 = true

instance matmulShapeInst (s1 s2 : Shape) :
    MatmulShape s1 s2 (matmulOut s1 s2) where
  h_out := rfl
  h_valid := sorry_proof

-- ============================================================================
-- Repeat shape computation
-- ============================================================================

/-- Compute output shape after repeat along each dimension.
    repeats[i] specifies how many times to repeat dimension i.
    Pads repeats with 1s on the left if shorter than shape. -/
def repeatOut (s : Shape) (repeats : List Nat) : Shape :=
  let len := max s.length repeats.length
  let s' := List.replicate (len - s.length) 1 ++ s
  let r' := List.replicate (len - repeats.length) 1 ++ repeats
  listZipWith (· * ·) s' r'

-- ============================================================================
-- Conv2d shape computation
-- ============================================================================

/-- Compute output spatial dimension for convolution.
    outputDim = (inputDim + 2*padding - dilation*(kernelSize-1) - 1) / stride + 1 -/
def convOutDim (inputDim padding dilation kernelSize stride : Nat) : Nat :=
  let dilatedKernel := dilation * (kernelSize - 1) + 1
  let paddedInput := inputDim + 2 * padding
  if paddedInput < dilatedKernel then 0
  else (paddedInput - dilatedKernel) / stride + 1

/-- Compute conv1d output shape.
    Input: [batch, cin, w]
    Weight: [cout, cin, kW]
    Output: [batch, cout, wOut] -/
def conv1dShape (inputShape weightShape : Shape) (padding stride dilation : Nat) : Option Shape :=
  match inputShape, weightShape with
  | [batch, cin, w], [cout, cin', kW] =>
    if cin != cin' then none
    else
      let wOut := convOutDim w padding dilation kW stride
      some [batch, cout, wOut]
  | _, _ => none

/-- Conv1d output shape, computed unconditionally.
    Returns empty list if inputs are invalid. -/
def conv1dOut (inputShape weightShape : Shape) (padding stride dilation : Nat) : Shape :=
  match conv1dShape inputShape weightShape padding stride dilation with
  | some s => s
  | none => []

/-- Check if conv1d parameters are valid -/
def conv1dValid (inputShape weightShape : Shape) (padding stride dilation : Nat) : Bool :=
  (conv1dShape inputShape weightShape padding stride dilation).isSome

/-- Compute conv2d output shape.
    Input: [batch, cin, h, w]
    Weight: [cout, cin, kH, kW]
    Output: [batch, cout, hOut, wOut] -/
def conv2dShape (inputShape weightShape : Shape) (padding stride dilation : Nat) : Option Shape :=
  match inputShape, weightShape with
  | [batch, cin, h, w], [cout, cin', kH, kW] =>
    if cin != cin' then none
    else
      let hOut := convOutDim h padding dilation kH stride
      let wOut := convOutDim w padding dilation kW stride
      some [batch, cout, hOut, wOut]
  | _, _ => none

/-- Conv2d output shape, computed unconditionally.
    Returns empty list if inputs are invalid. -/
def conv2dOut (inputShape weightShape : Shape) (padding stride dilation : Nat) : Shape :=
  match conv2dShape inputShape weightShape padding stride dilation with
  | some s => s
  | none => []

/-- Check if conv2d parameters are valid -/
def conv2dValid (inputShape weightShape : Shape) (padding stride dilation : Nat) : Bool :=
  (conv2dShape inputShape weightShape padding stride dilation).isSome

/-- Compute pool2d output shape (for maxPool2d/avgPool2d).
    Input: [batch, channels, h, w]
    Output: [batch, channels, hOut, wOut]
    Channel dimension is preserved. -/
def pool2dShape (inputShape : Shape) (kernelSize padding stride : Nat) : Shape :=
  match inputShape with
  | [batch, channels, h, w] =>
    let hOut := convOutDim h padding 1 kernelSize stride
    let wOut := convOutDim w padding 1 kernelSize stride
    [batch, channels, hOut, wOut]
  | _ => []

-- ============================================================================
-- Pool (im2col) shape computation
-- ============================================================================

/-- Compute output shape after pool/im2col operation.
    Input spatial dims are last `kernelSize.length` dimensions.
    Output adds kernel dimensions at the end.

    Input:  [..., h, w]
    Kernel: [kH, kW]
    Output: [..., hOut, wOut, kH, kW] -/
def poolOut (s : Shape) (kernelSize stride dilation : List Nat) : Shape :=
  let k := kernelSize.length
  let pre := s.take (s.length - k)
  let spatial := s.drop (s.length - k)
  -- convOutDim(inputDim, padding, dilation, kernelSize, stride)
  let outSpatial := listZipWith5 convOutDim spatial
    (List.replicate k 0)  -- padding = 0 for pool
    dilation kernelSize stride
  pre ++ outSpatial ++ kernelSize

end Shape

/-- Proof that reshape preserves element count -/
structure ReshapeProof (from_ to : Shape) : Prop where
  numel_eq : from_.numel = to.numel

/-- Decidable instance for ReshapeProof -/
instance (from_ to : Shape) : Decidable (ReshapeProof from_ to) :=
  if h : from_.numel = to.numel
  then isTrue ⟨h⟩
  else isFalse fun ⟨h'⟩ => h h'

end TinyGrad4
