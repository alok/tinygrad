import Float64
import TinyGrad4.Tensor.Tensor
import TinyGrad4.Tensor.Math
import TinyGrad4.NN.Linear
import TinyGrad4.Backend.CudaTritonMatmul
import TinyGrad4.Backend.Cuda
import TinyGrad4.Backend.Interpreter
import TinyGrad4.Test.EmitTritonPTX

open TinyGrad4

private def approxEq (a b : Float64) (eps : Float64 := 1.0e-2) : Bool :=
  Float64.abs (a - b) <= eps

private def checkExpected (vals : Array Float64) (expected : Float64) : IO Bool := do
  if vals.isEmpty then
    IO.println "Linear Triton smoke: FAIL (no output values)"
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
      IO.println s!"Linear Triton smoke: FAIL (idx {idx} {got} != {expected})"
  return ok

private def defaultEmitConfig : TinyGrad4.Test.EmitTritonPTX.EmitConfig :=
  { ptxPath := "tmp/triton_matmul.ptx",
    blockM := 64,
    blockN := 128,
    blockK := 64,
    numWarps := 4,
    numStages := 2,
    m := 256,
    n := 256,
    k := 256 }

private def buildLinear (batch inFeatures outFeatures : Nat)
    : TensorM (Matrix batch outFeatures .float32) := do
  let x ← Tensor.full [batch, inFeatures] .float32 1.0
  let w ← Tensor.full [inFeatures, outFeatures] .float32 1.0
  let b ← Tensor.full [outFeatures] .float32 0.5
  let y ← TinyGrad4.NN.linear' x w (some b)
  pure y

/-- Smoke test: build a linear layer with shapes from TG4_TRITON_* and eval via IO path. -/
def main : IO UInt32 := do
  let available ← TinyGrad4.Backend.Cuda.isAvailable
  if !available then
    IO.println "Linear Triton smoke: skipped (CUDA not available)"
    return 0

  TinyGrad4.Test.EmitTritonPTX.autogenIfNeeded
  let cfg? ← TinyGrad4.Backend.CudaTritonMatmul.getConfigFromEnv
  let cfg? ←
    match cfg? with
    | some cfg => pure (some cfg)
    | none => do
      let emitCfg := defaultEmitConfig
      let ptxPath := System.FilePath.mk emitCfg.ptxPath
      let existed ← ptxPath.pathExists
      let ok ←
        if existed then
          pure true
        else do
          let rc ← TinyGrad4.Test.EmitTritonPTX.emit emitCfg
          pure (rc == 0)
      let ptxExists ← ptxPath.pathExists
      if !ok || !ptxExists then
        pure none
      else
        let cfg : TinyGrad4.Backend.CudaTritonMatmul.TritonMatmulConfig := {
          kernelName := "matmul_kernel",
          ptxPath := ptxPath,
          blockM := emitCfg.blockM,
          blockN := emitCfg.blockN,
          blockK := emitCfg.blockK,
          numWarps := emitCfg.numWarps,
          sharedBytes := 0,
          expectedM := emitCfg.m,
          expectedN := emitCfg.n,
          expectedK := emitCfg.k
        }
        TinyGrad4.Backend.CudaTritonMatmul.setConfig (some cfg)
        pure (some cfg)

  match cfg? with
  | none =>
    IO.println "Linear Triton smoke: FAIL (no Triton config available)"
    return 1
  | some cfg =>
    let batch := cfg.expectedM
    let inFeatures := cfg.expectedK
    let outFeatures := cfg.expectedN

    let t := runTensorM (buildLinear batch inFeatures outFeatures)
    let out ← TinyGrad4.Interpreter.evalTensorIO t
    let expectedBytes := batch * outFeatures * 4
    if out.data.size != expectedBytes then
      IO.println s!"Linear Triton smoke: FAIL (size {out.data.size} != {expectedBytes})"
      return 1
    let expected := Float64.ofNat inFeatures + 0.5
    let vals := RawBuffer.unpackF32Bytes out.data
    let ok ← checkExpected vals expected
    if !ok then
      return 1

    IO.println "Linear Triton smoke: ok"
    return 0
