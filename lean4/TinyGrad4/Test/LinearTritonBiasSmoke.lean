import Float64
import TinyGrad4.Backend.CudaTritonMatmul
import TinyGrad4.Backend.Cuda
import TinyGrad4.Backend.Native
import TinyGrad4.Backend.Interpreter

open TinyGrad4

private def pushF32LE (out : ByteArray) (v : Float32) : ByteArray :=
  let bits := v.toBits
  let b0 := (bits &&& 0xFF).toUInt8
  let b1 := ((bits >>> 8) &&& 0xFF).toUInt8
  let b2 := ((bits >>> 16) &&& 0xFF).toUInt8
  let b3 := ((bits >>> 24) &&& 0xFF).toUInt8
  out.push b0 |>.push b1 |>.push b2 |>.push b3

private def fillF32 (n : Nat) (v : Float32) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity (n * 4)
  for _ in [:n] do
    out := pushF32LE out v
  return out

private def roundF32ToF16 (v : Float32) : Float64 := Id.run do
  let f16 := TinyGrad4.Backend.Native.f32ToF16 (fillF32 1 v)
  let f32 := TinyGrad4.Backend.Native.f16ToF32 f16
  let vals := TinyGrad4.RawBuffer.unpackF32Bytes f32
  vals.getD 0 0.0

private def approxEq (a b : Float64) (eps : Float64 := 1.0e-1) : Bool :=
  Float64.abs (a - b) <= eps

private def sampleIndices (n : Nat) : Array Nat :=
  if n == 0 then #[]
  else if n == 1 then #[0]
  else
    let mid := n / 2
    #[0, 1, 2, mid, n - 1]

/-- Smoke test: run Triton matmul + bias kernel and compare a few samples to expected. -/
def main : IO UInt32 := do
  let available ← TinyGrad4.Backend.Cuda.isAvailable
  if !available then
    IO.println "Linear Triton bias smoke: skipped (CUDA not available)"
    return 0

  let m := 256
  let n := 256
  let k := 256
  let cfg? ←
    match TinyGrad4.Backend.CudaTritonMatmul.choosePreset .float32 m n k with
    | none => pure none
    | some preset => TinyGrad4.Backend.CudaTritonMatmul.ensureConfigBias preset m n k

  match cfg? with
  | none =>
    IO.println "Linear Triton bias smoke: FAIL (no Triton bias config available)"
    return 1
  | some cfg =>
    let aBytes := fillF32 (m * k) 1.0
    let bBytes := fillF32 (k * n) 1.0
    let biasBytes := fillF32 n 0.5

    let aBuf : RawBuffer := { dtype := .float32, data := aBytes }
    let bBuf : RawBuffer := { dtype := .float32, data := bBytes }
    let biasBuf : RawBuffer := { dtype := .float32, data := biasBytes }

    let tritonOut ← TinyGrad4.Backend.CudaTritonMatmul.matmulF32ViaF16Bias cfg aBuf bBuf biasBuf m k n
    let expected := roundF32ToF16 ((Float64.ofNat k + 0.5).toFloat32)
    let vals := TinyGrad4.RawBuffer.unpackF32Bytes tritonOut.data
    let samples := sampleIndices (m * n)
    let mut ok := true
    for idx in samples do
      let got := vals.getD idx 0.0
      if !approxEq got expected then
        ok := false
        IO.println s!"Mismatch at {idx}: triton={got} expected={expected}"

    if ok then
      IO.println "Linear Triton bias smoke: ok"
      return 0
    else
      IO.println "Linear Triton bias smoke: FAIL"
      return 1
