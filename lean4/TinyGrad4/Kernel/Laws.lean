import TinyGrad4.Kernel.Spec

namespace TinyGrad4

/-!
# Kernel Laws (Rewrite Lemmas / Assumptions)

This file is intentionally "sorry-friendly":
- We state the algebraic properties we want to use for aggressive rewriting.
- Many of these are not true for real IEEE floats (NaNs, rounding), so they should be treated as
  *optimization assumptions* rather than theorems about Lean's native `Float32`.

The usual workflow is:
1) Use these laws to justify graph rewrites (fusion, CSE, strength reduction, etc).
2) Later, either:
   - restrict rewrites to cases where they are valid for your FP model, or
   - swap these axioms for a more realistic model.
-/

namespace Kernel.Laws

open Kernel

/-! ## Convenience -/

def const {α : Type} (s : Shape) (v : α) : Tensor s α :=
  fun _ => v

/-! ## Scalar algebra (assumptions) -/

variable {α : Type}
variable (ops : ScalarOps α)

axiom add_assoc : ∀ a b c : α, ops.add (ops.add a b) c = ops.add a (ops.add b c)
axiom add_comm : ∀ a b : α, ops.add a b = ops.add b a
axiom add_zero_left : ∀ a : α, ops.add ops.zero a = a
axiom add_zero_right : ∀ a : α, ops.add a ops.zero = a

axiom mul_assoc : ∀ a b c : α, ops.mul (ops.mul a b) c = ops.mul a (ops.mul b c)
axiom mul_comm : ∀ a b : α, ops.mul a b = ops.mul b a
axiom mul_one_left : ∃ one : α, (∀ a : α, ops.mul one a = a)
axiom mul_one_right : ∃ one : α, (∀ a : α, ops.mul a one = a)

axiom left_distrib : ∀ a b c : α, ops.mul a (ops.add b c) = ops.add (ops.mul a b) (ops.mul a c)
axiom right_distrib : ∀ a b c : α, ops.mul (ops.add a b) c = ops.add (ops.mul a c) (ops.mul b c)

/-! ## Tensor-level rewrite lemmas (usually proven by extensionality) -/

theorem ewise_add_zero_right {s : Shape} (x : Tensor s α) :
    TensorEq (ewiseBinary (fun a b => ops.add a b) x (const s ops.zero)) x :=
  sorry_proof

theorem broadcast_id {s : Shape} (x : Tensor s α) :
    TensorEq (broadcastTensor s s x) x :=
  sorry_proof

theorem reduce_keepdim_noop_of_shape {s : Shape} (axes : List Nat) (x : Tensor s α) (h : Shape.reduce s axes true = s) :
    TensorEq (reduceKeepdim ops .sum (s := s) axes x) (by simpa [h] using x) :=
  sorry_proof

end Kernel.Laws

end TinyGrad4
