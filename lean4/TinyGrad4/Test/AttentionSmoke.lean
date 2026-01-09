import TinyGrad4

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

/-!
# Attention Smoke Test (Fusion + Correctness)

This test builds a tiny attention-like pipeline:

- `scores = q @ kᵀ + mask` (batched matmul + broadcasted bias)
- `scoresScaled = (q @ kᵀ) * (1/sqrt(d)) + mask` (attention scaling fused into the matmul kernel)
- `p = softmax(scores)` (last-axis)
- `out = p @ v` (batched matmul)

We check:
- the fused matmul+bias and fused softmax planners match the expected subgraphs
- the end-to-end output matches a baseline computed via unfused kernels
-/

namespace TinyGrad4.Test.AttentionSmoke

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

private def testAttentionB2T3D2 : IO Unit := do
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

  let (qU, kU, vU, maskU, scoresMaskedU, probsU, outU) := runTensorM do
    let q ← Tensor.buffer [b, t, d] .float32
    let k ← Tensor.buffer [b, t, d] .float32
    let v ← Tensor.buffer [b, t, d] .float32
    let mask ← Tensor.buffer [t, t] .float32

    let kT ← StaticTensor.permute k [0, 2, 1]
    let scores ← UOp.contract2D q.uop kT.uop
    let scale ← UOp.const .float32 invSqrtD
    let scoresScaled ← UOp.mul scores scale
    let scoresMasked ← UOp.add scoresScaled mask.uop
    let scoresMaskedT : Float32^[b, t, t] :=
      { uop := scoresMasked, requiresGrad := false, h_shape := sorry_proof }
    let probs ← StaticTensor.softmax scoresMaskedT
    let out ← UOp.contract2D probs.uop v.uop
    pure (q.uop, k.uop, v.uop, mask.uop, scoresMasked, probs.uop, out)

  let env : Env :=
    let env0 : Env := ∅
    let env1 := Interpreter.setBuffer env0 qU { dtype := .float32, data := qb }
    let env2 := Interpreter.setBuffer env1 kU { dtype := .float32, data := kb }
    let env3 := Interpreter.setBuffer env2 vU { dtype := .float32, data := vb }
    Interpreter.setBuffer env3 maskU { dtype := .float32, data := maskb }

  -- Ensure planners match on the real graph.
  let nodes := UOp.toposort outU
  let keepIds := UOpIdSet.add UOpIdSet.mkEmpty outU.uid

  let refCnt0 : Std.HashMap UOpId Nat := Id.run do
    let mut refCnt0 : Std.HashMap UOpId Nat := ∅
    for u in nodes do
      for s in u.src do
        refCnt0 := refCnt0.insert s.uid (refCnt0.getD s.uid 0 + 1)
    return refCnt0

  match FusedMatmul.compile scoresMaskedU keepIds refCnt0 with
  | none => throw (IO.userError "attention: expected fused matmul+bias plan")
  | some plan =>
    if plan.aStarts.isEmpty then
      throw (IO.userError "attention: expected batched fused matmul plan (aStarts non-empty)")
    match plan.scaleBits with
    | some bits =>
      if bits != invSqrtD.toBits then
        throw (IO.userError s!"attention: fused matmul scaleBits mismatch, got {bits} expected {invSqrtD.toBits}")
    | none =>
      throw (IO.userError "attention: expected fused matmul to capture attention scale")

  match FusedSoftmax.compile probsU keepIds refCnt0 with
  | none => throw (IO.userError "attention: expected fused softmax plan")
  | some plan =>
    if plan.outer != b * t || plan.inner != t then
      throw (IO.userError s!"attention: softmax plan outer/inner mismatch: outer={plan.outer} inner={plan.inner}")

  -- Interpreter output (should choose fusions).
  let outArr := (← Interpreter.evalRawCached outU env).decode

  -- Baseline output (explicitly unfused kernels): permute, matmul, add(bcast), softmax, matmul.
  let kTBytes := Native.permuteF32 kb #[b, t, d] #[0, 2, 1]

  let qStarts := startsBytes b (t * d)
  let kStarts := startsBytes b (d * t)
  let scoresBytes := Native.matmulBatchedF32 qb kTBytes qStarts kStarts t d t
  let attScale := Native.fullF32Bits 1 invSqrtD.toBits
  let scoresScaledBytes := Native.mulBcastF32 scoresBytes attScale #[b, t, t] #[] #[b, t, t]
  let scoresMaskedBytes := Native.addBcastF32 scoresScaledBytes maskb #[b, t, t] #[t, t] #[b, t, t]

  let scaleBits := StaticTensor.log2ef32.toBits
  let probsBytes := Native.softmaxLastF32 scoresMaskedBytes (b * t) t scaleBits

  let pStarts := startsBytes b (t * t)
  let vStarts := startsBytes b (t * d)
  let outBytes := Native.matmulBatchedF32 probsBytes vb pStarts vStarts t t d
  let expected := RawBuffer.decode { dtype := .float32, data := outBytes }

  assertAllClose outArr expected 0.001 "attention"

def runAll : IO Unit := do
  IO.println "=== AttentionSmoke Tests ==="
  testAttentionB2T3D2
  IO.println "✓ attention smoke"
  IO.println "=== AttentionSmoke OK ==="

end TinyGrad4.Test.AttentionSmoke

