import Float64
import TinyGrad4.Tensor.Tensor
import TinyGrad4.Tensor.Math
import TinyGrad4.NN.Linear
import TinyGrad4.Backend.CudaTritonMatmul
import TinyGrad4.Backend.Cuda
import TinyGrad4.Backend.Native
import TinyGrad4.Backend.Interpreter

open TinyGrad4

private def approxEq (a b : Float64) (eps : Float64 := 1.0e-2) : Bool :=
  Float64.abs (a - b) <= eps

private def pushF32LE (out : ByteArray) (v : Float32) : ByteArray :=
  let bits := v.toBits
  let b0 := (bits &&& 0xFF).toUInt8
  let b1 := ((bits >>> 8) &&& 0xFF).toUInt8
  let b2 := ((bits >>> 16) &&& 0xFF).toUInt8
  let b3 := ((bits >>> 24) &&& 0xFF).toUInt8
  out.push b0 |>.push b1 |>.push b2 |>.push b3

private def packF32 (v : Float32) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity 4
  out := pushF32LE out v
  return out

private def roundF32ToF16 (v : Float32) : Float64 := Id.run do
  let f16 := TinyGrad4.Backend.Native.f32ToF16 (packF32 v)
  let f32 := TinyGrad4.Backend.Native.f16ToF32 f16
  let vals := RawBuffer.unpackF32Bytes f32
  vals.getD 0 0.0

private def checkExpected (vals : Array Float64) (expected : Float64) : IO Bool := do
  if vals.isEmpty then
    IO.println "Linear Triton scale+relu smoke: FAIL (no output values)"
    return false
  let idxs :=
    if vals.size <= 2 then
      (List.range vals.size)
    else
      [0, vals.size / 2, vals.size - 1]
  let mut ok := true
  for idx in idxs do
    let got := vals[idx]!
    if !approxEq got expected then
      ok := false
      IO.println s!"Linear Triton scale+relu smoke: FAIL (idx {idx} {got} != {expected})"
  return ok

private def buildLinearScaleRelu (batch inFeatures outFeatures : Nat)
    (scale : Float32) (biasVal : Float32)
    : TensorM (Matrix batch outFeatures .float32) := do
  let x ← Tensor.full [batch, inFeatures] .float32 1.0
  let w ← Tensor.full [inFeatures, outFeatures] .float32 1.0
  let b ← Tensor.full [outFeatures] .float32 biasVal
  let y ← TinyGrad4.StaticTensor.matmul x w
  let yScaled ← TinyGrad4.StaticTensor.scale y scale
  let yBias ← TinyGrad4.StaticTensor.addB yScaled b
  let yRelu ← TinyGrad4.StaticTensor.relu yBias
  pure { uop := yRelu.uop, requiresGrad := yRelu.requiresGrad, h_shape := sorry_proof }

/-- Smoke test: build a linear layer with scale+relu and eval via IO path. -/
def main : IO UInt32 := do
  let available ← TinyGrad4.Backend.Cuda.isAvailable
  if !available then
    IO.println "Linear Triton scale+relu smoke: skipped (CUDA not available)"
    return 0

  let cfgEnv? ← TinyGrad4.Backend.CudaTritonMatmul.getConfigFromEnv
  let (m, k, n, cfg?) ←
    match cfgEnv? with
    | some cfg =>
      pure (cfg.expectedM, cfg.expectedK, cfg.expectedN, some cfg)
    | none => do
      let m := 256
      let k := 256
      let n := 256
      let preset? := TinyGrad4.Backend.CudaTritonMatmul.choosePreset .float32 m n k
      let cfg? ←
        match preset? with
        | none => pure none
        | some preset => TinyGrad4.Backend.CudaTritonMatmul.ensureConfig preset m n k
      pure (m, k, n, cfg?)

  match cfg? with
  | none =>
    IO.println "Linear Triton scale+relu smoke: FAIL (no Triton config available)"
    return 1
  | some _ =>
    let scale : Float32 := 0.5
    let biasVal : Float32 := -200.0
    let t := runTensorM (buildLinearScaleRelu m k n scale biasVal)
    let out ← TinyGrad4.Interpreter.evalTensorIO t
    let expectedBytes := m * n * 4
    if out.data.size != expectedBytes then
      IO.println s!"Linear Triton scale+relu smoke: FAIL (size {out.data.size} != {expectedBytes})"
      return 1

    let base := roundF32ToF16 ((Float64.ofNat k).toFloat32)
    let base32 := base.toFloat32
    let expected32 := max (base32 * scale + biasVal) 0.0
    let expected := expected32.toFloat
    let vals := RawBuffer.unpackF32Bytes out.data
    let ok ← checkExpected vals expected
    if !ok then
      return 1

    IO.println "Linear Triton scale+relu smoke: ok"
    return 0
