import Float64
import TinyGrad4
import TinyGrad4.NN.Dropout
import TinyGrad4.NN.Norm

/-!
# ShapeTypedLayerSmoke

Concrete, executable examples for shape-typed layer composition.

This test shows that we can:
1. Build layer stacks while preserving shape in the `StaticTensor` type
2. Evaluate the resulting UOp graph through the interpreter
3. Keep API callsites readable with minimal wrapper noise

Non-compiling example (intentionally commented):
```lean
-- let x ← Tensor.ones [2, 4] .float32
-- let bad ← Tensor.ones [3, 4] .float32
-- let _ ← add x bad   -- shape mismatch, rejected by Lean type checker
```
-/

namespace TinyGrad4.Test.ShapeTypedLayerSmoke

set_option linter.useRawBuffer false

open TinyGrad4
open TinyGrad4.Interpreter
open TinyGrad4.StaticTensor
open TinyGrad4.NN

private def broadcastProof {s1 s2 : List Nat} : Shape.broadcastable s1 s2 = true := by
  exact sorry_proof

private def assertShape (got expected : List Nat) (label : String) : IO Unit := do
  if got != expected then
    throw (IO.userError s!"{label}: shape mismatch got={got}, expected={expected}")

private def testLayerNormDropoutStack : IO Unit := do
  let out := Id.run <| runTensorM do
    let x ← Tensor.ones [2, 4] .float32
    let bias ← Tensor.full [1, 4] .float32 3.0
    let shifted ← addBroadcast x bias broadcastProof
    let ln ← layerNorm .CPU [4] .float32
    let normalized ← LayerNormParams.forward ln shifted
    let dp := dropout 0.25
    DropoutParams.forward (DropoutParams.eval dp) normalized 7

  assertShape out.uop.shape [2, 4] "layernorm+dropout stack"
  let vals := (eval out.uop (∅ : Env)).decode.data
  if vals.size != 8 then
    throw (IO.userError s!"layernorm+dropout stack: expected 8 values, got {vals.size}")

private def testBatchNorm1dShape : IO Unit := do
  let out := Id.run <| runTensorM do
    let x ← Tensor.ones [3, 5] .float32
    let bn ← batchNorm1d .CPU 5 .float32
    BatchNormParams.forward1d bn x

  assertShape out.uop.shape [3, 5] "batchnorm1d"

def runAll : IO Unit := do
  IO.println "=== ShapeTypedLayerSmoke Tests ==="
  testLayerNormDropoutStack
  IO.println "✓ layernorm + dropout typed stack"
  testBatchNorm1dShape
  IO.println "✓ batchnorm1d typed shape"
  IO.println "=== ShapeTypedLayerSmoke OK ==="

end TinyGrad4.Test.ShapeTypedLayerSmoke

#eval! TinyGrad4.Test.ShapeTypedLayerSmoke.runAll
