import TinyGrad4.Tensor.Tensor

namespace TinyGrad4

namespace StaticTensor

def reshape {s : List Nat} {d : DType} (t : StaticTensor s d)
    (newShape : List Nat)
    : TensorM (StaticTensor newShape d) := do
  let reshaped ← TUOp.reshape t.tuop newShape
  pure (StaticTensor.ofTUOp reshaped t.requiresGrad)

def flatten {s : List Nat} {d : DType} (t : StaticTensor s d)
    : TensorM (StaticTensor [listProd s] d) := do
  -- Use actual UOp shape in case type parameter doesn't match (e.g. after sorry_proof cast)
  let actualShape := t.uop.shape
  let reshaped ← UOp.reshape t.uop [listProd actualShape]
  pure { uop := reshaped, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def expand {s : List Nat} {d : DType} (t : StaticTensor s d)
    (newShape : List Nat)
    : TensorM (StaticTensor newShape d) := do
  let expanded ← TUOp.expand t.tuop newShape
  pure (StaticTensor.ofTUOp expanded t.requiresGrad)

def unsqueeze {s : List Nat} {d : DType} (t : StaticTensor s d)
    (axis : Nat)
    : TensorM (StaticTensor (Shape.insertDim s axis 1) d) := do
  -- Use actual UOp shape in case type parameter doesn't match (e.g. after sorry_proof cast)
  let actualShape := t.uop.shape
  let newShape := Shape.insertDim actualShape axis 1
  let reshaped ← UOp.reshape t.uop newShape
  pure { uop := reshaped, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def permute {s : List Nat} {d : DType} (t : StaticTensor s d)
    (perm : List Nat)
    : TensorM (StaticTensor (Shape.permute s perm) d) := do
  let permuted ← TUOp.permute t.tuop perm
  pure (StaticTensor.ofTUOp permuted t.requiresGrad)

def T {m n : Nat} {d : DType} (t : Matrix m n d) : TensorM (Matrix n m d) :=
  permute t [1, 0]

def pad {s : List Nat} {d : DType} (t : StaticTensor s d)
    (padding : List (Nat × Nat))
    : TensorM (StaticTensor (Shape.pad s padding) d) := do
  let padded ← TUOp.pad t.tuop padding
  pure (StaticTensor.ofTUOp padded t.requiresGrad)

def shrink {s : List Nat} {d : DType} (t : StaticTensor s d)
    (bounds : List (Nat × Nat))
    : TensorM (StaticTensor (Shape.shrink s bounds) d) := do
  let shrunk ← TUOp.shrink t.tuop bounds
  pure (StaticTensor.ofTUOp shrunk t.requiresGrad)

def flip {s : List Nat} {d : DType} (t : StaticTensor s d)
    (axes : List Nat)
    : TensorM (StaticTensor s d) := do
  let flipped ← TUOp.flip t.tuop axes
  pure (StaticTensor.ofTUOp flipped t.requiresGrad)

def stack {d : DType} {shapes : List Shape} (ts : TensorList d shapes) (axis : Nat)
    : TensorM (StaticTensor (Shape.stackOut shapes axis) d) := do
  let rec go {shapes : List Shape} (ts : TensorList d shapes) : TensorM (List UOp) := do
    match ts with
    | .nil => pure []
    | .cons t rest =>
      let t' ← unsqueeze t axis
      let rest' ← go rest
      pure (t'.uop :: rest')
  match ts with
  | .nil => panic! "stack: empty list"
  | _ =>
    let uops ← go ts
    let out ← UOp.cat uops axis
    let reqGrad := TensorList.anyRequiresGrad ts
    pure { uop := out, requiresGrad := reqGrad, h_shape := sorry_proof }

/-- Repeat tensor along each dimension.
    repeats[i] specifies how many times to repeat dimension i.

    Implementation: reshape to interleave 1s, expand, reshape to merge.
    For example, shape [2, 3] with repeats [4, 5]:
      [2, 3] → [1, 2, 1, 3] → [4, 2, 5, 3] → [8, 15]
-/
def tile {s : List Nat} {d : DType} (t : StaticTensor s d)
    (repeats : List Nat)
    : TensorM (StaticTensor (Shape.repeatOut s repeats) d) := do
  -- Use actual UOp shape in case type parameter doesn't match (e.g. after sorry_proof cast)
  let actualShape := t.uop.shape
  -- Align repeats with shape (pad with 1s on left)
  let len := max actualShape.length repeats.length
  let s' := List.replicate (len - actualShape.length) 1 ++ actualShape
  let r' := List.replicate (len - repeats.length) 1 ++ repeats

  -- Build interleaved shapes
  -- unsqueezedShape: [1, s0, 1, s1, ...]  (where repeats[i] != 1, insert 1 before s[i])
  -- expandedShape:   [r0, s0, r1, s1, ...] (expand the 1s to repeats)
  -- finalShape:      [r0*s0, r1*s1, ...]   (merge adjacent dims)
  let unsqueezedShape := (s'.zip r').foldl (fun acc (si, ri) =>
    if ri == 1 then acc ++ [si]
    else acc ++ [1, si]) []

  let expandedShape := (s'.zip r').foldl (fun acc (si, ri) =>
    if ri == 1 then acc ++ [si]
    else acc ++ [ri, si]) []

  let finalShape := listZipWith (· * ·) s' r'

  -- Apply transforms
  let t1 ← reshape t unsqueezedShape
  let t2 ← expand t1 expandedShape
  let t3 ← reshape t2 finalShape
  pure { uop := t3.uop, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Shrink tensor to target shape (each dim becomes min of current and target).
    This is a convenience wrapper over shrink with computed bounds. -/
def shrinkTo {s : List Nat} {d : DType} (t : StaticTensor s d)
    (targetShape : List Nat)
    : TensorM (StaticTensor targetShape d) := do
  -- Use actual UOp shape in case type parameter doesn't match (e.g. after sorry_proof cast)
  let actualShape := t.uop.shape
  -- Compute bounds: (0, min(s[i], target[i]))
  let bounds := listZipWith (fun si ti => (0, min si ti)) actualShape targetShape
  let shrunk ← shrink t bounds
  pure { uop := shrunk.uop, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Ceiling division -/
private def ceildiv (a b : Nat) : Nat :=
  if b == 0 then 0 else (a + b - 1) / b

/-- Safe max that works with Nat -/
private def smax (a b : Nat) : Nat := max a b

/-- Pool/im2col operation: unfolds patches from spatial dimensions.

    Input:  [..., h, w]
    Kernel: [kH, kW]
    Output: [..., hOut, wOut, kH, kW]

    This extracts all (kH × kW) patches from the spatial dimensions,
    with the given stride and dilation.

    Implementation follows Python tinygrad's _pool:
    1. Repeat to ensure we have enough elements
    2. Shrink and reshape to create patch dimension
    3. Permute to move kernel dims to the end
-/
def pool {s : List Nat} {d : DType} (t : StaticTensor s d)
    (kernelSize : List Nat)
    (stride : List Nat := [1, 1])
    (dilation : List Nat := [1, 1])
    : TensorM (StaticTensor (Shape.poolOut s kernelSize stride dilation) d) := do
  let k := kernelSize.length
  -- Use actual UOp shape in case type parameter doesn't match (e.g. after sorry_proof cast)
  let actualShape := t.uop.shape
  let noop := actualShape.take (actualShape.length - k)  -- prefix dimensions (batch, channels)
  let i_ := actualShape.drop (actualShape.length - k)    -- spatial dimensions

  -- Output spatial dimensions
  let o_ := listZipWith4 (fun i di ki si =>
    ceildiv (i - di * (ki - 1)) si) i_ dilation kernelSize stride

  -- Scaling factor to ensure shrink is possible
  let f_ := listZipWith4 (fun oi si ii di =>
    smax 1 (ceildiv (oi * si - di) ii)) o_ stride i_ dilation

  -- Repeats to ensure we don't need padding
  let repeats := List.replicate noop.length 1 ++
    listZipWith4 (fun ki ii di fi =>
      ceildiv (ki * (ii * fi + di)) ii) kernelSize i_ dilation f_

  -- Step 1: Tile (repeat)
  let x ← tile t repeats

  -- Step 2: Shrink to exact size needed
  let shrinkTarget := noop ++
    listZipWith4 (fun ki ii di fi => ki * (ii * fi + di)) kernelSize i_ dilation f_
  let x ← shrinkTo x shrinkTarget

  -- Step 3: Reshape to separate kernel dims
  -- [..., k0*(i0*f0+d0)] → [..., k0, (i0*f0+d0)]
  let reshapeShape := noop ++
    (listZipWith4 (fun ki ii di fi => [ki, ii * fi + di]) kernelSize i_ dilation f_).flatten
  let x ← reshape x reshapeShape

  -- Step 4: Handle stride - shrink and reshape
  let strideShape1 := noop ++
    (listZipWith3 (fun ki oi si => [ki, oi * si]) kernelSize o_ stride).flatten
  let x ← shrinkTo x strideShape1

  let strideShape2 := noop ++
    (listZipWith3 (fun ki oi si => [ki, oi, si]) kernelSize o_ stride).flatten
  let x ← reshape x strideShape2

  -- Step 5: Shrink stride dim to 1 and flatten
  let finalShrinkShape := noop ++
    (listZipWith (fun ki oi => [ki, oi, 1]) kernelSize o_).flatten
  let x ← shrinkTo x finalShrinkShape

  let flatShape := noop ++
    (listZipWith (fun ki oi => [ki, oi]) kernelSize o_).flatten
  let x ← reshape x flatShape

  -- Step 6: Permute to move spatial output dims before kernel dims
  -- Current: [..., k0, o0, k1, o1, ...]
  -- Target:  [..., o0, o1, ..., k0, k1, ...]
  let noopPerm := listRange noop.length
  let outPerm := (listRange k).map (fun i => noop.length + i * 2 + 1)  -- o0, o1, ...
  let kernelPerm := (listRange k).map (fun i => noop.length + i * 2)  -- k0, k1, ...
  let perm := noopPerm ++ outPerm ++ kernelPerm

  let result ← permute x perm
  pure { uop := result.uop, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

end StaticTensor

end TinyGrad4
