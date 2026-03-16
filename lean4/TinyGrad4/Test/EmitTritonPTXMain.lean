import TinyGrad4.Test.EmitTritonPTX

/-- CLI entrypoint for emit_triton_ptx. -/
def main : IO UInt32 :=
  TinyGrad4.Test.EmitTritonPTX.emitMain
