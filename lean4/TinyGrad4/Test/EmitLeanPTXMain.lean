import TinyGrad4.Test.EmitLeanPTX

/-- CLI entrypoint for emit_lean_ptx. -/
def main : IO UInt32 :=
  TinyGrad4.Test.EmitLeanPTX.emitMain
