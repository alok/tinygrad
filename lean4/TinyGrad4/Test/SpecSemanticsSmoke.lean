import Float64
import TinyGrad4.Spec

/-!
# SpecSemanticsSmoke

Quick checks for the executable tensor spec layer: movement, indexing, reduction, elementwise, concat, and matmul.
-/

namespace TinyGrad4.Test.SpecSemanticsSmoke

open TinyGrad4
open TinyGrad4.Spec

private def assertEq {α} [DecidableEq α] (got expected : α) (msg : String) : IO Unit := do
  if got != expected then
    throw (IO.userError msg)

private def testMovementAndIndexing : IO Unit := do
  let x : TensorDesc := { shape := [2, 3], dtype := .float32 }
  assertEq (MovementOp.apply? x (.permute [1, 0]))
    (some { shape := [3, 2], dtype := .float32 })
    "permute"
  assertEq (MovementOp.apply? { shape := [1, 3], dtype := .float32 } (.expand [4, 3]))
    (some { shape := [4, 3], dtype := .float32 })
    "expand"
  assertEq (MovementOp.apply? x (.expand [4, 3])) none "invalid expand"

  let items : List BasicIndexItem := [.ellipsis, .int 1, .slice Slice.all]
  assertEq (basicIndex? { shape := [10, 20, 30], dtype := .float32 } items)
    (some { shape := [10, 30], dtype := .float32 })
    "basic index"

private def testReductions : IO Unit := do
  let x : TensorDesc := { shape := [2, 3, 4], dtype := .float32 }
  let spec : ReduceSpec := { op := .ADD, axes := [1], keepdim := false }
  assertEq (ReduceSpec.apply? x spec)
    (some { shape := [2, 4], dtype := .float32 })
    "sum reduce"

  let full : ReduceSpec := { op := .MAX }
  assertEq (ReduceSpec.apply? x full)
    (some { shape := [1, 1, 1], dtype := .float32 })
    "reduce all dims"

private def testElementwise : IO Unit := do
  let lhs : TensorDesc := { shape := [2, 3], dtype := .float16 }
  let rhs : TensorDesc := { shape := [1, 3], dtype := .float32 }
  assertEq (binaryResult? .ADD lhs rhs)
    (some { shape := [2, 3], dtype := .float32 })
    "binary add broadcast"
  assertEq (binaryResult? .CMPLT lhs rhs)
    (some { shape := [2, 3], dtype := .bool })
    "binary comparison dtype"

  let cond : TensorDesc := { shape := [2, 1], dtype := .bool }
  let a : TensorDesc := { shape := [1, 3], dtype := .float32 }
  let b : TensorDesc := { shape := [2, 3], dtype := .float32 }
  assertEq (ternaryResult? .WHERE cond a b)
    (some { shape := [2, 3], dtype := .float32 })
    "where broadcast"
  assertEq (ternaryResult? .WHERE cond a { shape := [2, 3], dtype := .float16 })
    none
    "where dtype mismatch"

private def testCatAndMatmul : IO Unit := do
  let catInputs := [
    ({ shape := [2, 3], dtype := .float32 } : TensorDesc),
    ({ shape := [2, 5], dtype := .float32 } : TensorDesc)
  ]
  assertEq (catResult? catInputs 1)
    (some { shape := [2, 8], dtype := .float32 })
    "cat"

  let lhs : TensorDesc := { shape := [4, 8], dtype := .float16 }
  let rhs : TensorDesc := { shape := [8, 16], dtype := .float32 }
  assertEq (contract2DResult? lhs rhs)
    (some { shape := [4, 16], dtype := .float32 })
    "matmul"

def runAll : IO Unit := do
  IO.println "=== SpecSemanticsSmoke Tests ==="
  testMovementAndIndexing
  IO.println "✓ movement + indexing"
  testReductions
  IO.println "✓ reductions"
  testElementwise
  IO.println "✓ elementwise"
  testCatAndMatmul
  IO.println "✓ cat + matmul"
  IO.println "=== SpecSemanticsSmoke OK ==="

end TinyGrad4.Test.SpecSemanticsSmoke

#eval! TinyGrad4.Test.SpecSemanticsSmoke.runAll
