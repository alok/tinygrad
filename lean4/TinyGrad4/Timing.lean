import Std

/-!
# Timing Core

Lightweight, cycle-free timing primitives for use across the codebase.
Higher-level profiling lives in `TinyGrad4.Data.Timing`.
-/

namespace TinyGrad4

/-! ## MonadTimeNS -/

/-- Monotonic time source in nanoseconds. -/
class MonadTimeNS (m : Type → Type) where
  monoNs : m Nat

instance : MonadTimeNS IO where
  monoNs := IO.monoNanosNow

end TinyGrad4
