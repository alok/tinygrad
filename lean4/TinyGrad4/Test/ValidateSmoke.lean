import TinyGrad4

/-!
# UOp Validate Smoke Test

Ensures our IR invariants checker (`UOp.validateMany`) succeeds on:
- a normal high-level graph
- the `.KERNEL`-lowered graph produced by `Interpreter.compileManyCached`
-/

namespace TinyGrad4.Test.ValidateSmoke

open TinyGrad4
open Interpreter

private def assertValid (roots : List UOp) (label : String) : IO Unit := do
  let errs := UOp.validateMany roots
  if errs.size != 0 then
    let shown := errs.toList.take 10 |>.map (fun e => e.render)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"{label}: {errs.size} validation errors\n{msg}")

private def testValidateAttentionLike : IO Unit := do
  let b := 2
  let t := 3
  let d := 2

  let (_, _, _, _, outU) := runTensorM do
    let q ← Tensor.buffer [b, t, d] .float32
    let k ← Tensor.buffer [b, t, d] .float32
    let v ← Tensor.buffer [b, t, d] .float32
    let mask ← Tensor.buffer [t, t] .float32

    let kT ← StaticTensor.permute k [0, 2, 1]
    let scores ← UOp.contract2D q.uop kT.uop
    let scoresMasked ← UOp.add scores mask.uop
    let scoresMaskedT : StaticTensor [b, t, t] .float32 :=
      { uop := scoresMasked, requiresGrad := false, h_shape := sorry_proof }
    let probs ← StaticTensor.softmax scoresMaskedT
    let out ← UOp.contract2D probs.uop v.uop
    pure (q.uop, k.uop, v.uop, mask.uop, out)

  assertValid [outU] "validate high-level graph"
  let compiled ← Interpreter.compileManyCached [outU]
  assertValid compiled.roots "validate kernelized graph"

def runAll : IO Unit := do
  IO.println "=== ValidateSmoke Tests ==="
  testValidateAttentionLike
  IO.println "✓ validate high-level + kernelized"
  IO.println "=== ValidateSmoke OK ==="

end TinyGrad4.Test.ValidateSmoke

