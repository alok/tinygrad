import Float64
import TinyGrad4.Spec.Broadcast
import TinyGrad4.Spec.Indexing
import TinyGrad4.Spec.Semantics
import TinyGrad4.Spec.Typed
import TinyGrad4.Spec.UOpSemantics

/-!
# Spec layer

Reference semantics for broadcasting, strides, indexing, executable tensor semantics, and proof-carrying typed
signatures, plus lower-level checked UOp signatures. Pure Lean, not runtime.
-/
