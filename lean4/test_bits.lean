import TinyGrad4.Tensor.Math
import TinyGrad4.Backend.Native
import TinyGrad4.Backend.Buffer

#eval IO.println s!"Float32(1.0).toBits = {(1.0 : Float).toFloat32.toBits}"
#eval IO.println s!"Float32 1.0f bits (expected 0x3F800000 = 1065353216) = {(1.0 : Float32).toBits}"
#eval IO.println s!"log2ef32 bits = {StaticTensor.log2ef32.toBits}"

-- Test the packF32 and actual softmax
def testSoftmax : IO Unit := do
  let x : Array Float := #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  let xb := TinyGrad4.Backend.Native.packF32FromF64 ⟨x⟩
  IO.println s!"Input bytes size: {xb.size}"
  IO.println s!"Input bytes (first 12): {xb.toList.take 12}"

  let scaleBits := (1.0 : Float32).toBits
  IO.println s!"Scale bits: {scaleBits}"

  let outBytes := TinyGrad4.Backend.Native.softmaxLastF32 xb 2 3 scaleBits
  IO.println s!"Output bytes size: {outBytes.size}"

  let outRaw : TinyGrad4.RawBuffer := { dtype := .float32, data := outBytes }
  let out := outRaw.toFloatArray.data
  IO.println s!"Output values: {out}"

#eval testSoftmax
