import Float64
import TinyGrad4.Tensor.Tensor
import TinyGrad4.Tensor.Math
import TinyGrad4.NN.Linear
import TinyGrad4.Backend.CudaTritonMatmul
import TinyGrad4.Backend.Cuda
import TinyGrad4.Backend.Interpreter

open TinyGrad4

private def buildLinear (batch inFeatures outFeatures : Nat)
    : TensorM (Matrix batch outFeatures .float32) := do
  let x ← Tensor.full [batch, inFeatures] .float32 1.0
  let w ← Tensor.full [inFeatures, outFeatures] .float32 1.0
  let b ← Tensor.full [outFeatures] .float32 0.5
  let y ← TinyGrad4.NN.linear' x w (some b)
  pure y

/-- Smoke test: build a linear layer with shapes from TG4_TRITON_* and eval via IO path. -/
def main : IO UInt32 := do
  let cfg? ← TinyGrad4.Backend.CudaTritonMatmul.getConfigFromEnv
  match cfg? with
  | none =>
    IO.println "Linear Triton smoke: skipped (TG4_TRITON_PTX not set)"
    return 0
  | some cfg =>
    let available ← TinyGrad4.Backend.Cuda.isAvailable
    if !available then
      IO.println "Linear Triton smoke: skipped (CUDA not available)"
      return 0

    let batch := cfg.expectedM
    let inFeatures := cfg.expectedK
    let outFeatures := cfg.expectedN

    let t := runTensorM (buildLinear batch inFeatures outFeatures)
    let out ← TinyGrad4.Interpreter.evalTensorIO t
    let expectedBytes := batch * outFeatures * 4
    if out.data.size != expectedBytes then
      IO.println s!"Linear Triton smoke: FAIL (size {out.data.size} != {expectedBytes})"
      return 1

    IO.println "Linear Triton smoke: ok"
    return 0
