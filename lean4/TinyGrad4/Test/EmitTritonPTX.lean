import TinyGrad4.Backend.TritonEmit

/-!
# Emit Triton PTX (Test Wrapper)

Re-exports backend Triton PTX emit utilities for test/CLI use.
-/

namespace TinyGrad4.Test.EmitTritonPTX

abbrev EmitConfig := TinyGrad4.Backend.TritonEmit.EmitConfig

/-- Emit Triton PTX for an explicit config. -/
@[inline] def emit : EmitConfig → IO UInt32 :=
  TinyGrad4.Backend.TritonEmit.emit

/-- Emit Triton PTX using env config. -/
@[inline] def emitMain : IO UInt32 :=
  TinyGrad4.Backend.TritonEmit.emitFromEnv

/-- Auto-generate PTX if TG4_TRITON_AUTOGEN is set and the PTX is missing. -/
@[inline] def autogenIfNeeded : IO Unit :=
  TinyGrad4.Backend.TritonEmit.autogenIfNeeded

end TinyGrad4.Test.EmitTritonPTX
