import TinyGrad4.Backend.LeanPtxEmit

/-!
# Emit Lean PTX (Test Wrapper)

Re-exports Lean PTX emit utilities for test/CLI use.
-/

namespace TinyGrad4.Test.EmitLeanPTX

abbrev EmitConfig := TinyGrad4.Backend.LeanPtxEmit.EmitConfig

/-- Emit Lean PTX for an explicit config. -/
@[inline] def emit : EmitConfig → IO UInt32 :=
  TinyGrad4.Backend.LeanPtxEmit.emit

/-- Emit Lean PTX using env config. -/
@[inline] def emitMain : IO UInt32 :=
  TinyGrad4.Backend.LeanPtxEmit.emitFromEnv

end TinyGrad4.Test.EmitLeanPTX
