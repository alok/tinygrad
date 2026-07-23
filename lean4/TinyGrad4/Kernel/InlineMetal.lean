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
    elaboration time; this just compiles (cached) and launches it.
    When `numel % 4 == 0` the float4-vectorized variant is dispatched
    (numel/4 threads) for bandwidth; results are identical per lane. -/
def InlineKernel.runMetal {n : Nat} (k : InlineKernel n)
    (inputs : Array RawBuffer) (numel : Nat) : IO RawBuffer := do
  if inputs.size != n then
    throw (IO.userError s!"InlineKernel.runMetal: expected {n} inputs, got {inputs.size}")
  -- the generated kernel has no gid guard (dispatch is exact), so every input
  -- must actually hold numel elements or the GPU reads out of bounds
  let mut i := 0
  for inp in inputs do
    let need := numel * inp.dtype.itemsize
    if inp.data.size < need then
      throw (IO.userError
        s!"InlineKernel.runMetal: input {i} has {inp.data.size} bytes, needs {need}")
    i := i + 1
  if numel % 4 == 0 && numel > 0 then
    MetalEwise.runEwiseKernel (k.name ++ "_v4") k.metalVec inputs numel (threads := some (numel / 4))
  else
    MetalEwise.runEwiseKernel k.name k.metal inputs numel

end TinyGrad4.Kernel
