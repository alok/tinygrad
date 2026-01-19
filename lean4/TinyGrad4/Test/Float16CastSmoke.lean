import Float64
import TinyGrad4.Backend.Native

-- Disable RawBuffer linter: test uses FloatArray for roundtrip checks
set_option linter.useRawBuffer false

private def approxEq (a b : Float64) (eps : Float64 := 1.0e-2) : Bool :=
  Float64.abs (a - b) <= eps

def main : IO UInt32 := do
  let input := FloatArray.mk #[1.0, -2.0, 0.5, 123.0]
  let bytes := TinyGrad4.Backend.Native.packF32FromF64 input
  let f16 := TinyGrad4.Backend.Native.f32ToF16 bytes
  let roundtrip := TinyGrad4.Backend.Native.f16ToF32 f16
  let out := TinyGrad4.Backend.Native.unpackF64FromF32 roundtrip

  let mut ok := true
  for i in [:input.size] do
    let orig := input.get! i
    let got := out.get! i
    if !approxEq orig got then
      ok := false
      IO.println s!"Mismatch at {i}: {orig} vs {got}"

  if ok then
    IO.println "Float16 cast smoke: ok"
  else
    IO.println "Float16 cast smoke: FAIL"

  return if ok then 0 else 1
