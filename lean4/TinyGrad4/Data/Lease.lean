/-!
# Lease - Scoped resource lifetime

Lightweight wrapper that pairs a value with a release action.
Used for pooled buffers (GPU/TPU) so callers can deterministically return
resources back to the pool or free them.
-/

namespace TinyGrad4.Data

/-- Value paired with a release action. -/
structure Lease (T : Type) where
  value : T
  release : IO Unit

namespace Lease

/-- Release the lease. -/
def free (l : Lease T) : IO Unit :=
  l.release

/-- Run an action and always release afterward. -/
def withLease (l : Lease T) (f : T → IO α) : IO α := do
  try
    f l.value
  finally
    l.release

/-- Map the value, keeping the same release. -/
def map (l : Lease T) (f : T → U) : Lease U :=
  { value := f l.value, release := l.release }

end Lease

end TinyGrad4.Data
