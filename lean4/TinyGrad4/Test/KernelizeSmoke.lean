import TinyGrad4

/-!
# Kernelize Smoke Test (Compile Once, Run Many)

This test ensures the fusion planner can be "lowered" into explicit `.KERNEL` boundaries:
- we compile an attention-like graph into a kernelized IR (`Interpreter.compileManyCached`)
- we execute it via `Interpreter.evalCompiledRaw` (no planning at runtime)
- we check the output against an unfused baseline
- we assert the compiled graph contains at least one `.KERNEL` node
-/

namespace TinyGrad4.Test.KernelizeSmoke

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

open TinyGrad4
open Interpreter
open Backend

private def startsBytes (batch : Nat) (matNumel : Nat) : Array Nat := Id.run do
  let mut out : Array Nat := Array.emptyWithCapacity batch
  for i in [:batch] do
    out := out.push (i * matNumel * 4)
  return out

private def assertAllClose (arr : FloatArray) (expected : FloatArray) (tol : Float) (label : String) : IO Unit := do
  if arr.size != expected.size then
    throw (IO.userError s!"{label}: size {arr.size} != {expected.size}")
  for i in [:arr.size] do
    let v := arr[i]!
    let e := expected[i]!
    let diff := Float.abs (v - e)
    if diff > tol then
      throw (IO.userError s!"{label}: idx {i} value {v} expected {e} diff {diff} > {tol}")

private def testKernelizeAttentionB2T3D2 : IO Unit := do
  let b := 2
  let t := 3
  let d := 2
  let invSqrtD : Float32 := 0.70710677

  let qData : FloatArray := ⟨#[
    0.1, 0.2,  0.3, 0.4,  0.5, 0.6,
    0.7, 0.8,  0.9, 1.0,  1.1, 1.2
  ]⟩
  let kData : FloatArray := ⟨#[
    1.0, 0.9,  0.8, 0.7,  0.6, 0.5,
    0.4, 0.3,  0.2, 0.1,  0.0, -0.1
  ]⟩
  let vData : FloatArray := ⟨#[
    0.1, 0.0,  0.0, 0.1,  0.2, 0.2,
    -0.1, 0.3,  0.4, -0.2,  0.0, 0.2
  ]⟩
  let maskData : FloatArray := ⟨#[
    0.0, -0.1, -0.2,
    0.0, -0.1, -0.2,
    0.0, -0.1, -0.2
  ]⟩

  let qb := Native.packF32FromF64 qData
  let kb := Native.packF32FromF64 kData
  let vb := Native.packF32FromF64 vData
  let maskb := Native.packF32FromF64 maskData

  let (qU, kU, vU, maskU, scoresMaskedU, outU) := runTensorM do
    let q ← Tensor.buffer [b, t, d] .float32
    let k ← Tensor.buffer [b, t, d] .float32
    let v ← Tensor.buffer [b, t, d] .float32
    let mask ← Tensor.buffer [t, t] .float32

    let kT ← StaticTensor.permute k [0, 2, 1]
    let scores ← UOp.contract2D q.uop kT.uop
    let scale ← UOp.const .float32 invSqrtD
    let scoresScaled ← UOp.mul scores scale
    let scoresMasked ← UOp.add scoresScaled mask.uop
    let scoresMaskedT : StaticTensor [b, t, t] .float32 :=
      { uop := scoresMasked, requiresGrad := false, h_shape := sorry_proof }
    let probs ← StaticTensor.softmax scoresMaskedT
    let out ← UOp.contract2D probs.uop v.uop
    pure (q.uop, k.uop, v.uop, mask.uop, scoresMasked, out)

  let env : Env :=
    let env0 : Env := ∅
    let env1 := Interpreter.setBuffer env0 qU { dtype := .float32, data := qb }
    let env2 := Interpreter.setBuffer env1 kU { dtype := .float32, data := kb }
    let env3 := Interpreter.setBuffer env2 vU { dtype := .float32, data := vb }
    Interpreter.setBuffer env3 maskU { dtype := .float32, data := maskb }

  let compiled ← Interpreter.compileManyCached [outU]
  let hasKernel := compiled.nodes.any (fun u => u.op == .KERNEL)
  if !hasKernel then
    throw (IO.userError "kernelize: expected at least one .KERNEL node in compiled graph")

  match compiled.implMap[scoresMaskedU.uid]? with
  | some (.fusedMatmul plan) =>
    match plan.scaleBits with
    | some bits =>
      if bits != invSqrtD.toBits then
        throw (IO.userError s!"kernelize: fusedMatmul scaleBits mismatch, got {bits}, expected {invSqrtD.toBits}")
    | none =>
      throw (IO.userError "kernelize: expected fusedMatmul scaleBits")
  | _ =>
    throw (IO.userError "kernelize: expected scoresMasked to be fusedMatmul")

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outArr := (outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))).decode

  let kTBytes := Native.permuteF32 kb #[b, t, d] #[0, 2, 1]

  let qStarts := startsBytes b (t * d)
  let kStarts := startsBytes b (d * t)
  let scoresBytes := Native.matmulBatchedF32 qb kTBytes qStarts kStarts t d t
  let attScale := Native.fullF32Bits 1 invSqrtD.toBits
  let scoresScaledBytes := Native.mulBcastF32 scoresBytes attScale #[b, t, t] #[] #[b, t, t]
  let scoresMaskedBytes := Native.addBcastF32 scoresScaledBytes maskb #[b, t, t] #[t, t] #[b, t, t]

  let softmaxScaleBits := StaticTensor.log2ef32.toBits
  let probsBytes := Native.softmaxLastF32 scoresMaskedBytes (b * t) t softmaxScaleBits

  let pStarts := startsBytes b (t * t)
  let vStarts := startsBytes b (t * d)
  let outBytes := Native.matmulBatchedF32 probsBytes vb pStarts vStarts t t d
  let expected := RawBuffer.decode { dtype := .float32, data := outBytes }

  assertAllClose outArr expected 0.001 "kernelize attention"

def runAll : IO Unit := do
  IO.println "=== KernelizeSmoke Tests ==="
  testKernelizeAttentionB2T3D2
  IO.println "✓ kernelize attention"
  IO.println "=== KernelizeSmoke OK ==="

end TinyGrad4.Test.KernelizeSmoke

