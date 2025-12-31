/-!
# TimeM (trusted time-cost annotations)

`TimeM α` is a tiny monad that pairs a return value with a *trusted* (not verified) time cost.

This is meant for compiler heuristics and cost models:
- prove correctness about `.ret`
- reason about performance using `.time`

The cost model is intentionally informal and may evolve; the point is to make the dependency explicit in the code.
-/

namespace TinyGrad4.Backend

/-- A monad that tracks a trusted time cost (in arbitrary units). -/
structure TimeM (α : Type) where
  ret : α
  time : Nat
  deriving Repr

namespace TimeM

@[simp] def pure {α} (a : α) : TimeM α :=
  ⟨a, 0⟩

@[simp] def bind {α β} (m : TimeM α) (f : α → TimeM β) : TimeM β :=
  let r := f m.ret
  ⟨r.ret, m.time + r.time⟩

instance : Monad TimeM where
  pure := pure
  bind := bind

/-- Charge `c` time units and return `a`. -/
@[simp] def tick {α} (a : α) (c : Nat := 1) : TimeM α :=
  ⟨a, c⟩

/-- Charge `c` time units. -/
@[simp] def tickUnit (c : Nat := 1) : TimeM Unit :=
  tick () c

end TimeM

end TinyGrad4.Backend

