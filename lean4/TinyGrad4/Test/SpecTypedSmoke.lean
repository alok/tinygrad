import Float64
import TinyGrad4.Spec

/-!
# SpecTypedSmoke

Smoke checks for the proof-carrying `TinyGrad4.Spec.Typed` layer.
-/

namespace TinyGrad4.Test.SpecTypedSmoke

open TinyGrad4
open TinyGrad4.Spec

private def assertEq {α} [DecidableEq α] (got expected : α) (msg : String) : IO Unit := do
  if got != expected then
    throw (IO.userError msg)

private def testTypedConstructors : IO Unit := do
  assertEq (Typed.erase (Typed.full [2, 3] .float32))
    ({ shape := [2, 3], dtype := .float32 } : TensorDesc)
    "typed full"
  assertEq (Typed.erase (Typed.eye 2 4 .float16))
    ({ shape := [2, 4], dtype := .float16 } : TensorDesc)
    "typed eye"
  assertEq (Typed.erase (Typed.arange 7 .int32))
    ({ shape := [7], dtype := .int32 } : TensorDesc)
    "typed arange"

private def testTypedElementwise : IO Unit := do
  let lhs := Typed.full [2, 3] .float16
  let rhs := Typed.full [1, 3] .float32
  let out := Typed.binary .ADD lhs rhs (by simp [Shape.broadcastable, listAll])
  assertEq (Typed.erase out)
    ({ shape := [2, 3], dtype := .float32 } : TensorDesc)
    "typed binary"

private def testTypedIndexing : IO Unit := do
  let src := Typed.full [2, 3] .float32
  let idx := Typed.full [2, 2] .int32
  let gathered := Typed.gather src idx 1 (by simp [DType.isInt, DType.isSigned]) (by
    simp [gatherShapeOk, listAll, listRange, listGetD])
  assertEq (Typed.erase gathered)
    ({ shape := [2, 2], dtype := .float32 } : TensorDesc)
    "typed gather"

  let diagonal := Typed.diagonal (Typed.full [3, 3] .float32)
  assertEq (Typed.erase diagonal)
    ({ shape := [3], dtype := .float32 } : TensorDesc)
    "typed diagonal"

  let packed := Typed.maskedSelectPacked (Typed.full [3, 3] .float32) (Typed.fullBool [3, 3])
  assertEq (Typed.erase packed.1)
    ({ shape := [9], dtype := .float32 } : TensorDesc)
    "typed maskedSelectPacked payload"
  assertEq (Typed.erase packed.2)
    ({ shape := [], dtype := .int32 } : TensorDesc)
    "typed maskedSelectPacked count"

private def testTypedScatterAndNn : IO Unit := do
  let self := Typed.full [1, 1, 16] .float32
  let idx := Typed.full [1, 1, 4] .int32
  let src := Typed.full [1, 1, 4] .float32
  let scattered := Typed.scatter self idx src 2 (by simp [DType.isInt, DType.isSigned]) (by
    simp [scatterShapeOk, listAll, listRange, listGetD])
  assertEq (Typed.erase scattered)
    ({ shape := [1, 1, 16], dtype := .float32 } : TensorDesc)
    "typed scatter"

  let linear := Typed.linear (Typed.full [8, 32] .float32) (Typed.full [32, 64] .float32)
  assertEq (Typed.erase linear)
    ({ shape := [8, 64], dtype := .float32 } : TensorDesc)
    "typed linear"

  let conv := Typed.conv2d (Typed.full [1, 3, 32, 32] .float32) (Typed.full [16, 3, 3, 3] .float32) 1 2 1
  assertEq (Typed.erase conv)
    ({ shape := [1, 16, 16, 16], dtype := .float32 } : TensorDesc)
    "typed conv2d"

  let bn := Typed.batchnormNCHW
    (Typed.full [2, 32, 8, 8] .float32)
    (Typed.full [32] .float32)
    (Typed.full [32] .float32)
  assertEq (Typed.erase bn)
    ({ shape := [2, 32, 8, 8], dtype := .float32 } : TensorDesc)
    "typed batchnorm"

def runAll : IO Unit := do
  IO.println "=== SpecTypedSmoke Tests ==="
  testTypedConstructors
  IO.println "✓ typed constructors"
  testTypedElementwise
  IO.println "✓ typed elementwise"
  testTypedIndexing
  IO.println "✓ typed indexing"
  testTypedScatterAndNn
  IO.println "✓ typed scatter + nn"
  IO.println "=== SpecTypedSmoke OK ==="

end TinyGrad4.Test.SpecTypedSmoke

#eval! TinyGrad4.Test.SpecTypedSmoke.runAll
