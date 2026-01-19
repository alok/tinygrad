import Float64
import TinyGrad4.Backend.Cuda
import TinyGrad4.Backend.CudaTritonMatmul
import TinyGrad4.Backend.Native

private def pushF32LE (out : ByteArray) (v : Float32) : ByteArray :=
  let bits := v.toBits
  let b0 := (bits &&& 0xFF).toUInt8
  let b1 := ((bits >>> 8) &&& 0xFF).toUInt8
  let b2 := ((bits >>> 16) &&& 0xFF).toUInt8
  let b3 := ((bits >>> 24) &&& 0xFF).toUInt8
  out.push b0 |>.push b1 |>.push b2 |>.push b3

private def packF32 (vals : Array Float32) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity (vals.size * 4)
  for v in vals do
    out := pushF32LE out v
  return out

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

private def makeData (n : Nat) (seed : Nat) : Array Float32 := Id.run do
  let mut out := Array.emptyWithCapacity n
  let mut s := UInt64.ofNat seed
  for _ in [:n] do
    s := s * 6364136223846793005 + 1
    let v := ((Float64.ofNat (s.toNat % 1000)) / 100.0).toFloat32
    out := out.push v
  return out

/-- Smoke test: run Triton matmul if configured in env, compare a few samples to CPU. -/
def main : IO UInt32 := do
  let available ← TinyGrad4.Backend.Cuda.isAvailable
  if !available then
    IO.println "Triton matmul smoke: skipped (CUDA not available)"
    return 0

  let cfgEnv? ← TinyGrad4.Backend.CudaTritonMatmul.getConfigFromEnv
  let (m, n, k, cfg?) ←
    match cfgEnv? with
    | some cfg =>
      pure (cfg.expectedM, cfg.expectedN, cfg.expectedK, some cfg)
    | none => do
      let m := 256
      let n := 256
      let k := 256
      let preset? := TinyGrad4.Backend.CudaTritonMatmul.choosePreset .float32 m n k
      let cfg? ←
        match preset? with
        | none => pure none
        | some preset => TinyGrad4.Backend.CudaTritonMatmul.ensureConfig preset m n k
      pure (m, n, k, cfg?)

  match cfg? with
  | none =>
    IO.println "Triton matmul smoke: FAIL (no Triton config available)"
    return 1
  | some cfg =>
    let aNumel := m * k
    let bNumel := k * n

    let aVals := makeData aNumel 123
    let bVals := makeData bNumel 456
    let aBytes := packF32 aVals
    let bBytes := packF32 bVals

    let aBuf : TinyGrad4.RawBuffer := { dtype := .float32, data := aBytes }
    let bBuf : TinyGrad4.RawBuffer := { dtype := .float32, data := bBytes }

    let tritonOut ← TinyGrad4.Backend.CudaTritonMatmul.matmulF32ViaF16 cfg aBuf bBuf m k n
    let cpuOut := TinyGrad4.Backend.Native.matmulF32 aBytes bBytes m k n
    let cpuOutF16 := TinyGrad4.Backend.Native.f32ToF16 cpuOut
    let cpuOutRounded := TinyGrad4.Backend.Native.f16ToF32 cpuOutF16

    let samples := sampleIndices (m * n)
    let mut ok := true
    for idx in samples do
      let got := readF32At tritonOut.data idx
      let ref := readF32At cpuOutRounded idx
      if !approxEq got ref then
        ok := false
        IO.println s!"Mismatch at {idx}: triton={got} cpu={ref}"

    if ok then
      IO.println "Triton matmul smoke: ok"
      return 0
    else
      IO.println "Triton matmul smoke: FAIL"
      return 1
