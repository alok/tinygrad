import Float64
import TinyGrad4.Kernel.Spec

/-!
# Kernel Laws (spec-level tensors and proved rewrite laws)

Spec tensors are total functions from intrinsically-typed indices
(`Shape.Index s`: per-dimension `Fin` bounds) to values. Rewrite laws are
stated extensionally and *proved* — `grind` discharges the equational and
index-arithmetic reasoning; the intrinsic index types make shape bookkeeping
part of the type.

Floating-point honesty: laws that are false for IEEE floats (associativity,
distributivity, …) are stated as *hypotheses* of the rewrites that need them —
not as axioms about `Float32`. A rewrite justified by `ewise_assoc h` is
exactly as trustworthy as its `h`.
-/

namespace TinyGrad4.Kernel

/-- Spec-level tensor: a total function from typed indices to values. -/
abbrev Tensor (s : Shape) (α : Type) : Type := Shape.Index s → α

/-- Extensional equality of spec tensors. -/
def TensorEq {s : Shape} {α : Type} (x y : Tensor s α) : Prop := ∀ idx, x idx = y idx

theorem TensorEq.refl {s : Shape} {α : Type} (x : Tensor s α) : TensorEq x x := fun _ => rfl

/-! ## Spec-level operations -/

def const (s : Shape) (v : α) : Tensor s α := fun _ => v

def ewiseUnary (f : α → α) (x : Tensor s α) : Tensor s α := fun idx => f (x idx)

def ewiseBinary (f : α → α → α) (x y : Tensor s α) : Tensor s α := fun idx => f (x idx) (y idx)

/-- Transport a tensor along a shape equality (the "type-level equivalence"
    escape hatch; `reindex rfl` is definitionally the identity). -/
def reindex {s s' : Shape} (h : s = s') (x : Tensor s α) : Tensor s' α :=
  fun idx => x (h ▸ idx)

/-- Broadcast a tensor into a shape of equal or higher rank. Indices that do
    not map back read `default`. Intended for `small.length ≤ big.length` with
    broadcast-compatible dims; for rank-reducing arguments `broadcastIndex`
    zero-pads the index rather than failing, so no `default` is observed. -/
def broadcastTensor (small big : Shape) (x : Tensor small α) [Inhabited α] : Tensor big α :=
  fun idx =>
    match broadcastIndex small big idx with
    | some i => x i
    | none => default

/-! ## Index lemmas (intrinsic bounds do the work) -/

theorem _root_.TinyGrad4.Shape.Index.toList_length {s : Shape} (idx : Shape.Index s) :
    idx.toList.length = s.length := by
  induction idx with
  | nil => rfl
  | cons i rest ih => simp [Shape.Index.toList, ih]

/-- `ofList?` inverts `toList`: a typed index round-trips. -/
theorem _root_.TinyGrad4.Shape.Index.ofList?_toList {s : Shape} (idx : Shape.Index s) :
    Shape.Index.ofList? s idx.toList = some idx := by
  induction idx with
  | nil => rfl
  | cons i rest ih => simp [Shape.Index.toList, Shape.Index.ofList?, i.isLt, ih]

/-- Broadcast index arithmetic is trivial on matching dims: a size-1 dim can
    only hold index 0 (`Fin 1`), which `grind` sees from `i.isLt`. -/
private theorem zip_map_broadcast_self {s : Shape} (idx : Shape.Index s) :
    ((s.zip idx.toList).map fun p => if p.1 = 1 then 0 else p.2) = idx.toList := by
  induction idx with
  | nil => rfl
  | cons i rest ih =>
    simp only [Shape.Index.toList, List.zip_cons_cons, List.map_cons, ih]
    have := i.isLt
    grind

/-- Broadcasting a shape into itself maps every index to itself. -/
theorem broadcastIndex_self {s : Shape} (idx : Shape.Index s) :
    broadcastIndex s s idx = some idx := by
  unfold broadcastIndex
  simp [Shape.Index.toList_length, zip_map_broadcast_self, Shape.Index.ofList?_toList]

/-! ## Rewrite laws -/

namespace Laws

variable {α : Type} {s : Shape}

/-- Right identity: `x + 0 = x`, given the scalar law as a hypothesis. -/
theorem ewise_add_zero_right (ops : ScalarOps α)
    (h : ∀ a : α, ops.add a ops.zero = a) (x : Tensor s α) :
    TensorEq (ewiseBinary ops.add x (const s ops.zero)) x := by
  intro idx; grind [ewiseBinary, const]

/-- Commutativity lifts pointwise. -/
theorem ewise_comm (f : α → α → α) (h : ∀ a b, f a b = f b a) (x y : Tensor s α) :
    TensorEq (ewiseBinary f x y) (ewiseBinary f y x) := by
  intro idx; grind [ewiseBinary]

/-- Associativity lifts pointwise. -/
theorem ewise_assoc (f : α → α → α) (h : ∀ a b c, f (f a b) c = f a (f b c))
    (x y z : Tensor s α) :
    TensorEq (ewiseBinary f (ewiseBinary f x y) z) (ewiseBinary f x (ewiseBinary f y z)) := by
  intro idx; grind [ewiseBinary]

/-- Fusing two unary maps is a single map of the composition. -/
theorem ewise_fuse (f g : α → α) (x : Tensor s α) :
    TensorEq (ewiseUnary f (ewiseUnary g x)) (ewiseUnary (f ∘ g) x) :=
  fun _ => rfl

/-- Broadcasting a tensor into its own shape is the identity. -/
theorem broadcast_id [Inhabited α] (x : Tensor s α) :
    TensorEq (broadcastTensor s s x) x := by
  intro idx
  simp [broadcastTensor, broadcastIndex_self]

/-- Transport along `rfl` is the identity (definitionally). -/
theorem reindex_rfl (x : Tensor s α) : TensorEq (reindex rfl x) x :=
  fun _ => rfl

end Laws

end TinyGrad4.Kernel
