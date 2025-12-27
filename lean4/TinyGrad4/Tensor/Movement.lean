import TinyGrad4.Tensor.Tensor

namespace TinyGrad4

namespace StaticTensor

def reshape {s : List Nat} {d : DType} (t : StaticTensor s d)
    (newShape : List Nat)
    : TensorM (StaticTensor newShape d) := do
  let reshaped ← UOp.reshape t.uop newShape
  pure { uop := reshaped, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def flatten {s : List Nat} {d : DType} (t : StaticTensor s d)
    : TensorM (StaticTensor [listProd s] d) := do
  reshape t [listProd s]

def expand {s : List Nat} {d : DType} (t : StaticTensor s d)
    (newShape : List Nat)
    : TensorM (StaticTensor newShape d) := do
  let expanded ← UOp.expand t.uop newShape
  pure { uop := expanded, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def unsqueeze {s : List Nat} {d : DType} (t : StaticTensor s d)
    (axis : Nat)
    : TensorM (StaticTensor (Shape.insertDim s axis 1) d) := do
  let newShape := Shape.insertDim s axis 1
  let reshaped ← UOp.reshape t.uop newShape
  pure { uop := reshaped, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def permute {s : List Nat} {d : DType} (t : StaticTensor s d)
    (perm : List Nat)
    : TensorM (StaticTensor (Shape.permute s perm) d) := do
  let permuted ← UOp.permute t.uop perm
  pure { uop := permuted, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def T {m n : Nat} {d : DType} (t : Matrix m n d) : TensorM (Matrix n m d) :=
  permute t [1, 0]

def pad {s : List Nat} {d : DType} (t : StaticTensor s d)
    (padding : List (Nat × Nat))
    : TensorM (StaticTensor (Shape.pad s padding) d) := do
  let padded ← UOp.pad t.uop padding
  pure { uop := padded, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def shrink {s : List Nat} {d : DType} (t : StaticTensor s d)
    (bounds : List (Nat × Nat))
    : TensorM (StaticTensor (Shape.shrink s bounds) d) := do
  let shrunk ← UOp.shrink t.uop bounds
  pure { uop := shrunk, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def flip {s : List Nat} {d : DType} (t : StaticTensor s d)
    (axes : List Nat)
    : TensorM (StaticTensor s d) := do
  let flipped ← UOp.flip t.uop axes
  pure { uop := flipped, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

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

end StaticTensor

end TinyGrad4
