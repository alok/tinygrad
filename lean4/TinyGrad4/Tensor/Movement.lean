import Float64
import TinyGrad4.Tensor.Tensor

namespace TinyGrad4

namespace StaticTensor

abbrev SomeTensor (d : DType) (device : Backend.DeviceType) := Sigma fun s => StaticTensor s d device

private def build {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (u : UOp) (requiresGrad : Bool := false) : StaticTensor s d device :=
  StaticTensor.ofUOp u (requiresGrad := requiresGrad)

private def resolveAxis (rank axis : Nat) : Nat :=
  if axis < rank then axis
  else panic! s!"axis {axis} out of range for rank {rank}"

private def replaceDim (shape : Shape) (axis newDim : Nat) : Shape :=
  (listEnum shape).map fun p =>
    if p.1 == axis then newDim else p.2

private def sliceBounds (shape : Shape) (axis start stop : Nat) : List (Nat × Nat) :=
  (listEnum shape).map fun p =>
    if p.1 == axis then (start, stop) else (0, p.2)

private def chunkSizesFromSplit (dim splitSize : Nat) : List Nat :=
  Id.run do
    let mut remaining := dim
    let mut out : List Nat := []
    while remaining > 0 do
      let k := min splitSize remaining
      out := out ++ [k]
      remaining := remaining - k
    pure out

private def normalizeRollShift (n : Nat) (shift : Int) : Nat :=
  if n == 0 then
    0
  else if shift >= 0 then
    Int.toNat shift % n
  else
    let back := Int.toNat (-shift) % n
    if back == 0 then 0 else n - back

def reshapeUnsafe {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device)
    (newShape : List Nat)
    : TensorM (StaticTensor newShape d device) := do
  let reshaped ← UOp.reshape t.uop newShape
  pure (StaticTensor.ofUOp reshaped (requiresGrad := t.requiresGrad))

def reshape {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device)
    (newShape : List Nat) (h : Shape.reshapeValid s newShape = true)
    : TensorM (StaticTensor newShape d device) := do
  let h' : Shape.reshapeValid t.uop.shape newShape = true := by
    simpa [t.h_shape] using h
  let reshaped ← UOp.reshapeValid t.uop newShape h'
  pure (build reshaped (requiresGrad := t.requiresGrad))

def flatten {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device)
    : TensorM (StaticTensor [listProd s] d device) := do
  let h : Shape.reshapeValid s [listProd s] = true := by
    simp [Shape.reshapeValid, Shape.numel, listProd]
  let h' : Shape.reshapeValid t.uop.shape [listProd s] = true := by
    simpa [t.h_shape] using h
  let reshaped ← UOp.reshapeValid t.uop [listProd s] h'
  pure (build reshaped (requiresGrad := t.requiresGrad))

def expandUnsafe {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device)
    (newShape : List Nat)
    : TensorM (StaticTensor newShape d device) := do
  let expanded ← UOp.expand t.uop newShape
  pure (StaticTensor.ofUOp expanded (requiresGrad := t.requiresGrad))

def expand {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device)
    (newShape : List Nat) (h : Shape.expandValid s newShape = true)
    : TensorM (StaticTensor newShape d device) := do
  let h' : Shape.expandValid t.uop.shape newShape = true := by
    simpa [t.h_shape] using h
  let expanded ← UOp.expandValid t.uop newShape h'
  pure (build expanded (requiresGrad := t.requiresGrad))

def unsqueezeUnsafe {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device)
    (axis : Nat)
    : TensorM (StaticTensor (Shape.insertDim s axis 1) d device) := do
  let newShape := Shape.insertDim s axis 1
  let reshaped ← UOp.reshape t.uop newShape
  pure (StaticTensor.ofUOp reshaped (requiresGrad := t.requiresGrad))

def unsqueeze {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device)
    (axis : Nat)
    : TensorM (StaticTensor (Shape.insertDim s axis 1) d device) := do
  unsqueezeUnsafe t axis

def permuteUnsafe {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device)
    (perm : List Nat)
    : TensorM (StaticTensor (Shape.permute s perm) d device) := do
  let permuted ← UOp.permute t.uop perm
  pure (StaticTensor.ofUOp permuted (requiresGrad := t.requiresGrad))

def permute {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device)
    (perm : List Nat) (h : Shape.permuteValid s perm = true)
    : TensorM (StaticTensor (Shape.permute s perm) d device) := do
  let h' : Shape.permuteValid t.uop.shape perm = true := by
    simpa [t.h_shape] using h
  let permuted ← UOp.permuteValid t.uop perm h'
  pure (build permuted (requiresGrad := t.requiresGrad))

def T {m n : Nat} {d : DType} {device : Backend.DeviceType} (t : Matrix m n d device) : TensorM (Matrix n m d device) :=
  permute t [1, 0] (by simp [Shape.permuteValid, listAll, listRange])

def padUnsafe {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device)
    (padding : List (Nat × Nat))
    : TensorM (StaticTensor (Shape.pad s padding) d device) := do
  let padded ← UOp.pad t.uop padding
  pure (StaticTensor.ofUOp padded (requiresGrad := t.requiresGrad))

def pad {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device)
    (padding : List (Nat × Nat)) (h : Shape.padValid s padding = true)
    : TensorM (StaticTensor (Shape.pad s padding) d device) := do
  let h' : Shape.padValid t.uop.shape padding = true := by
    simpa [t.h_shape] using h
  let padded ← UOp.padValid t.uop padding h'
  pure (build padded (requiresGrad := t.requiresGrad))

def shrinkUnsafe {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device)
    (bounds : List (Nat × Nat))
    : TensorM (StaticTensor (Shape.shrink s bounds) d device) := do
  let shrunk ← UOp.shrink t.uop bounds
  pure (StaticTensor.ofUOp shrunk (requiresGrad := t.requiresGrad))

def shrink {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device)
    (bounds : List (Nat × Nat)) (h : Shape.shrinkValid s bounds = true)
    : TensorM (StaticTensor (Shape.shrink s bounds) d device) := do
  let h' : Shape.shrinkValid t.uop.shape bounds = true := by
    simpa [t.h_shape] using h
  let shrunk ← UOp.shrinkValid t.uop bounds h'
  pure (build shrunk (requiresGrad := t.requiresGrad))

def flipUnsafe {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device)
    (axes : List Nat)
    : TensorM (StaticTensor s d device) := do
  let flipped ← UOp.flip t.uop axes
  pure (StaticTensor.ofUOp flipped (requiresGrad := t.requiresGrad))

def flip {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device)
    (axes : List Nat) (h : Shape.flipValid s axes = true)
    : TensorM (StaticTensor s d device) := do
  let h' : Shape.flipValid t.uop.shape axes = true := by
    simpa [t.h_shape] using h
  let flipped ← UOp.flipValid t.uop axes h'
  pure (build flipped (requiresGrad := t.requiresGrad))

def stackUnsafe {d : DType} {device : Backend.DeviceType} {shapes : List Shape} (ts : TensorList d device shapes) (axis : Nat)
    : TensorM (StaticTensor (Shape.stackOut shapes axis) d device) := do
  let rec go {shapes : List Shape} (ts : TensorList d device shapes) : TensorM (List UOp) := do
    match ts with
    | .nil => pure []
    | .cons t rest =>
      let t' ← unsqueezeUnsafe t axis
      let rest' ← go rest
      pure (t'.uop :: rest')
  match ts with
  | .nil => panic! "stack: empty list"
  | _ =>
    let uops ← go ts
    let out ← UOp.cat uops axis
    let reqGrad := TensorList.anyRequiresGrad ts
    pure (StaticTensor.ofUOp out (requiresGrad := reqGrad))

def stack {d : DType} {device : Backend.DeviceType} {shapes : List Shape} (ts : TensorList d device shapes) (axis : Nat)
    : TensorM (StaticTensor (Shape.stackOut shapes axis) d device) := do
  stackUnsafe ts axis

/-- Split by an explicit list of chunk sizes along one axis. -/
def splitSizes {s : Shape} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) (sizes : List Nat) (axis : Nat := 0)
    : TensorM (List (SomeTensor d device)) := do
  let axis' := resolveAxis s.length axis
  let dim := listGetD s axis' 0
  if listSum sizes != dim then
    panic! s!"splitSizes: sizes sum {listSum sizes} must equal axis dim {dim}"
  let mut out : List (SomeTensor d device) := []
  let mut start := 0
  for sz in sizes do
    let stop := start + sz
    let bounds := sliceBounds s axis' start stop
    let chunkShape := replaceDim s axis' sz
    let u ← UOp.shrink t.uop bounds
    let chunk : StaticTensor chunkShape d device := StaticTensor.ofUOp u (requiresGrad := t.requiresGrad)
    out := out ++ [⟨chunkShape, chunk⟩]
    start := stop
  pure out

/-- Split into chunks of at most `splitSize` along one axis. -/
def split {s : Shape} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) (splitSize : Nat) (axis : Nat := 0)
    : TensorM (List (SomeTensor d device)) := do
  if splitSize == 0 then
    panic! "split: splitSize must be > 0"
  let axis' := resolveAxis s.length axis
  let dim := listGetD s axis' 0
  let sizes := chunkSizesFromSplit dim splitSize
  splitSizes t sizes axis'

/-- Split into `chunks` approximately equal pieces along one axis. -/
def chunk {s : Shape} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) (chunks : Nat) (axis : Nat := 0)
    : TensorM (List (SomeTensor d device)) := do
  if chunks == 0 then
    panic! "chunk: chunks must be > 0"
  let axis' := resolveAxis s.length axis
  let dim := listGetD s axis' 0
  if dim == 0 then
    pure []
  else
    let splitSize := (dim + chunks - 1) / chunks
    split t splitSize axis'

/-- Roll elements along an axis. Positive shifts move values toward higher indices. -/
def roll {s : Shape} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) (shift : Int) (axis : Nat := 0)
    : TensorM (StaticTensor s d device) := do
  let axis' := resolveAxis s.length axis
  let n := listGetD s axis' 0
  if n == 0 then
    pure t
  else
    let k := normalizeRollShift n shift
    if k == 0 then
      pure t
    else
      let leftBounds := sliceBounds s axis' (n - k) n
      let rightBounds := sliceBounds s axis' 0 (n - k)
      let left ← UOp.shrink t.uop leftBounds
      let right ← UOp.shrink t.uop rightBounds
      let out ← UOp.cat [left, right] axis'
      pure (StaticTensor.ofUOp out (requiresGrad := t.requiresGrad))

/-- Pad trailing elements of each axis up to `targetShape`. -/
def padTo {s : Shape} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) (targetShape : Shape)
    : TensorM (StaticTensor targetShape d device) := do
  if targetShape.length != s.length then
    panic! s!"padTo: rank mismatch {s.length} vs {targetShape.length}"
  let mut padding : List (Nat × Nat) := []
  for (srcDim, dstDim) in s.zip targetShape do
    if dstDim < srcDim then
      panic! s!"padTo: target dim {dstDim} must be >= source dim {srcDim}"
    padding := padding ++ [(0, dstDim - srcDim)]
  let padded ← UOp.pad t.uop padding
  pure (StaticTensor.ofUOp padded (requiresGrad := t.requiresGrad))

/-- Repeat tensor along each dimension.
    repeats[i] specifies how many times to repeat dimension i.

    Implementation: reshape to interleave 1s, expand, reshape to merge.
    For example, shape [2, 3] with repeats [4, 5]:
      [2, 3] → [1, 2, 1, 3] → [4, 2, 5, 3] → [8, 15]
-/
def tile {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device)
    (repeats : List Nat)
    : TensorM (StaticTensor (Shape.repeatOut s repeats) d device) := do
  -- Align repeats with shape (pad with 1s on left)
  let len := max s.length repeats.length
  let s' := List.replicate (len - s.length) 1 ++ s
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
  let t1 ← reshapeUnsafe t unsqueezedShape
  let t2 ← expandUnsafe t1 expandedShape
  let t3 ← reshapeUnsafe t2 finalShape
  pure (StaticTensor.ofUOp t3.uop (requiresGrad := t.requiresGrad))

/-- Shrink tensor to target shape (each dim becomes min of current and target).
    This is a convenience wrapper over shrink with computed bounds. -/
def shrinkTo {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device)
    (targetShape : List Nat)
    : TensorM (StaticTensor targetShape d device) := do
  -- Compute bounds: (0, min(s[i], target[i]))
  let bounds := listZipWith (fun si ti => (0, min si ti)) s targetShape
  let shrunk ← shrinkUnsafe t bounds
  pure (StaticTensor.ofUOp shrunk.uop (requiresGrad := t.requiresGrad))

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
def pool {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device)
    (kernelSize : List Nat)
    (stride : List Nat := [1, 1])
    (dilation : List Nat := [1, 1])
    : TensorM (StaticTensor (Shape.poolOut s kernelSize stride dilation) d device) := do
  let k := kernelSize.length
  let noop := s.take (s.length - k)  -- prefix dimensions (batch, channels)
  let i_ := s.drop (s.length - k)    -- spatial dimensions

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
  let x ← reshapeUnsafe x reshapeShape

  -- Step 4: Handle stride - shrink and reshape
  let strideShape1 := noop ++
    (listZipWith3 (fun ki oi si => [ki, oi * si]) kernelSize o_ stride).flatten
  let x ← shrinkTo x strideShape1

  let strideShape2 := noop ++
    (listZipWith3 (fun ki oi si => [ki, oi, si]) kernelSize o_ stride).flatten
  let x ← reshapeUnsafe x strideShape2

  -- Step 5: Shrink stride dim to 1 and flatten
  let finalShrinkShape := noop ++
    (listZipWith (fun ki oi => [ki, oi, 1]) kernelSize o_).flatten
  let x ← shrinkTo x finalShrinkShape

  let flatShape := noop ++
    (listZipWith (fun ki oi => [ki, oi]) kernelSize o_).flatten
  let x ← reshapeUnsafe x flatShape

  -- Step 6: Permute to move spatial output dims before kernel dims
  -- Current: [..., k0, o0, k1, o1, ...]
  -- Target:  [..., o0, o1, ..., k0, k1, ...]
  let noopPerm := listRange noop.length
  let outPerm := (listRange k).map (fun i => noop.length + i * 2 + 1)  -- o0, o1, ...
  let kernelPerm := (listRange k).map (fun i => noop.length + i * 2)  -- k0, k1, ...
  let perm := noopPerm ++ outPerm ++ kernelPerm

  let result ← permuteUnsafe x perm
  pure (StaticTensor.ofUOp result.uop (requiresGrad := t.requiresGrad))

/-- Unfold along one dimension using pool/im2col primitives.
    Static lane currently supports unfolding the last axis. -/
def unfold {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) (dim : Nat) (size : Nat) (step : Nat)
    : TensorM (StaticTensor (Shape.poolOut s [size] [step] [1]) d device) := do
  if size == 0 then
    panic! "unfold: size must be > 0"
  if step == 0 then
    panic! "unfold: step must be > 0"
  let dim' := resolveAxis s.length dim
  if dim' != s.length - 1 then
    panic! s!"unfold: static lane supports only last-axis unfolding (got dim={dim'})"
  let dimSize := listGetD s dim' 0
  if size > dimSize then
    panic! s!"unfold: size {size} exceeds axis dim {dimSize}"
  pool t [size] [step] [1]

end StaticTensor

end TinyGrad4
