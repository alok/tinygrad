import Float64
import TinyGrad4.Backend.Cuda
import TinyGrad4.Backend.CudaTritonMatmul
import TinyGrad4.Backend.Native

-- Disable RawBuffer linter: this test works directly with ByteArray buffers
set_option linter.useRawBuffer false

private def readU32LE (b : ByteArray) (offset : Nat) : UInt32 :=
  let b0 := b.get! offset
  let b1 := b.get! (offset + 1)
  let b2 := b.get! (offset + 2)
  let b3 := b.get! (offset + 3)
  (UInt32.ofNat b0.toNat) |||
    ((UInt32.ofNat b1.toNat) <<< 8) |||
    ((UInt32.ofNat b2.toNat) <<< 16) |||
    ((UInt32.ofNat b3.toNat) <<< 24)

private def readF32At (b : ByteArray) (idx : Nat) : Float64 :=
  let base := idx * 4
  let bits := readU32LE b base
  (Float32.ofBits bits).toFloat

private def approxEq (a b : Float64) (eps : Float64 := 1.0e-1) : Bool :=
  Float64.abs (a - b) <= eps

private def sampleIndices (n : Nat) : Array Nat :=
  if n == 0 then #[]
  else if n == 1 then #[0]
  else
    let mid := n / 2
    #[0, 1, 2, mid, n - 1]

private def makeData (n : Nat) (seed : Nat) : FloatArray := Id.run do
  let mut out := Array.emptyWithCapacity n
  let mut s := UInt64.ofNat seed
  for _ in [:n] do
    s := s * 6364136223846793005 + 1
    let v := (Float64.ofNat (s.toNat % 1000)) / 100.0
    out := out.push v
  return FloatArray.mk out

/-- Smoke test: run Triton matmul if configured in env, compare a few samples to CPU. -/
def main : IO UInt32 := do
  let cfg? ← TinyGrad4.Backend.CudaTritonMatmul.getConfigFromEnv
  match cfg? with
  | none =>
    IO.println "Triton matmul smoke: skipped (TG4_TRITON_PTX not set)"
    return 0
  | some cfg =>
    let available ← TinyGrad4.Backend.Cuda.isAvailable
    if !available then
      IO.println "Triton matmul smoke: skipped (CUDA not available)"
      return 0

    let m := cfg.expectedM
    let n := cfg.expectedN
    let k := cfg.expectedK
    let aNumel := m * k
    let bNumel := k * n

    let aVals := makeData aNumel 123
    let bVals := makeData bNumel 456
    let aBytes := TinyGrad4.Backend.Native.packF32FromF64 aVals
    let bBytes := TinyGrad4.Backend.Native.packF32FromF64 bVals

    let aBuf : TinyGrad4.RawBuffer := { dtype := .float32, data := aBytes }
    let bBuf : TinyGrad4.RawBuffer := { dtype := .float32, data := bBytes }

    let tritonOut ← TinyGrad4.Backend.CudaTritonMatmul.matmulF32ViaF16 cfg aBuf bBuf m k n
    let cpuOut := TinyGrad4.Backend.Native.matmulF32 aBytes bBytes m k n

    let samples := sampleIndices (m * n)
    let mut ok := true
    for idx in samples do
      let got := readF32At tritonOut.data idx
      let ref := readF32At cpuOut idx
      if !approxEq got ref then
        ok := false
        IO.println s!"Mismatch at {idx}: triton={got} cpu={ref}"

    if ok then
      IO.println "Triton matmul smoke: ok"
      return 0
    else
      IO.println "Triton matmul smoke: FAIL"
      return 1
