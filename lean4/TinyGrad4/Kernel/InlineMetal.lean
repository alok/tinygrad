import Float64
import TinyGrad4.Kernel.Codegen
import TinyGrad4.Backend.MetalEwise

/-!
# InlineKernel → Metal dispatch

Bridges compile-time-generated `InlineKernel` source to the Metal runtime.
Kept separate from `Kernel/Codegen.lean` so the codegen layer stays
backend-independent.
-/

namespace TinyGrad4.Kernel

open TinyGrad4.Backend

/-- Run an inline kernel on the GPU: one output element per input element,
    `n` input buffers, `numel` elements each. The shader was generated at
    elaboration time; this just compiles (cached) and launches it. -/
def InlineKernel.runMetal {n : Nat} (k : InlineKernel n)
    (inputs : Array RawBuffer) (numel : Nat) : IO RawBuffer := do
  if inputs.size != n then
    throw (IO.userError s!"InlineKernel.runMetal: expected {n} inputs, got {inputs.size}")
  MetalEwise.runEwiseKernel k.name k.metal inputs numel

end TinyGrad4.Kernel
