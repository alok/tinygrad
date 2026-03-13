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
  assertEq (constResult .float32) ({ shape := [], dtype := .float32 } : TensorDesc) "const"
  assertEq (vconstResult 4 .int32) ({ shape := [4], dtype := .int32 } : TensorDesc) "vconst"
  assertEq (fullResult [2, 3] .float32) x "full"
  assertEq (fullBoolResult [2, 3]) ({ shape := [2, 3], dtype := .bool } : TensorDesc) "fullBool"
  assertEq (eyeResult 2 4 .float16) ({ shape := [2, 4], dtype := .float16 } : TensorDesc) "eye"
  assertEq (arangeResult 7 .int32) ({ shape := [7], dtype := .int32 } : TensorDesc) "arange"
  assertEq (likeResult x) x "like"
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

private def testHighLevelIndexing : IO Unit := do
  let src : TensorDesc := { shape := [2, 3], dtype := .float32 }
  let idx : TensorDesc := { shape := [2, 2], dtype := .int32 }
  assertEq (gatherResult? src idx 1)
    (some { shape := [2, 2], dtype := .float32 })
    "gather"
  assertEq (takeResult? src { shape := [5], dtype := .int32 })
    (some { shape := [5], dtype := .float32 })
    "take"
  assertEq (diagResult? { shape := [3], dtype := .float32 })
    (some { shape := [3, 3], dtype := .float32 })
    "diag"
  assertEq (diagonalResult? { shape := [3, 3], dtype := .float32 })
    (some { shape := [3], dtype := .float32 })
    "diagonal"
  assertEq (triuResult? { shape := [3, 4], dtype := .float32 })
    (some { shape := [3, 4], dtype := .float32 })
    "triu"
  assertEq (unfoldResult? { shape := [8], dtype := .float32 } 0 3 2)
    (some { shape := [3, 3], dtype := .float32 })
    "unfold"
  assertEq (unfoldResult? { shape := [2, 8], dtype := .float32 } 0 3 2)
    none
    "unfold non-last axis"

private def testScatterAndBridges : IO Unit := do
  let self : TensorDesc := { shape := [1, 1, 16], dtype := .float32 }
  let idx : TensorDesc := { shape := [1, 1, 4], dtype := .int32 }
  let src : TensorDesc := { shape := [1, 1, 4], dtype := .float32 }
  assertEq (copyResult self) self "copy"
  assertEq (identityLikeUnaryResult? .DETACH self) (some self) "detach"
  assertEq (identityLikeUnaryResult? .CONTIGUOUS self) (some self) "contiguous"
  assertEq (identityLikeUnaryResult? .CONTIGUOUS_BACKWARD self) (some self) "contiguous backward"
  assertEq (scatterResult? self idx src 2)
    (some self)
    "scatter"
  assertEq (scatterReduceResult? self idx src 2 .sum false)
    (some self)
    "scatterReduce"
  assertEq (scatterScalarResult? self idx 2)
    (some self)
    "scatter scalar"
  assertEq (scatterReduceScalarResult? self idx 2 .prod true)
    (some self)
    "scatter scalar reduce"
  assertEq (scatterResult? self { shape := [1, 4], dtype := .int32 } { shape := [1, 4], dtype := .float32 } 1)
    none
    "scatter rank mismatch"

  assertEq (maskedSelectPackedResult? { shape := [3, 3], dtype := .float32 } { shape := [3, 3], dtype := .bool })
    (some ({ shape := [9], dtype := .float32 }, { shape := [], dtype := .int32 }))
    "maskedSelectPacked"
  assertEq (maskedSelectPackedResult? { shape := [3, 3], dtype := .float32 } { shape := [3, 2], dtype := .bool })
    none
    "maskedSelectPacked mismatch"

private def testNnShapes : IO Unit := do
  let x : TensorDesc := { shape := [8, 32], dtype := .float32 }
  let w : TensorDesc := { shape := [32, 64], dtype := .float32 }
  let b : TensorDesc := { shape := [64], dtype := .float32 }
  assertEq (linearResult? x w)
    (some { shape := [8, 64], dtype := .float32 })
    "linear"
  assertEq (linearBiasResult? x w b)
    (some { shape := [8, 64], dtype := .float32 })
    "linear bias"

  let convIn1 : TensorDesc := { shape := [1, 3, 32], dtype := .float32 }
  let convW1 : TensorDesc := { shape := [16, 3, 3], dtype := .float32 }
  assertEq (conv1dResult? convIn1 convW1 1 2 1)
    (some { shape := [1, 16, 16], dtype := .float32 })
    "conv1d"

  let convIn2 : TensorDesc := { shape := [1, 3, 32, 32], dtype := .float32 }
  let convW2 : TensorDesc := { shape := [16, 3, 3, 3], dtype := .float32 }
  assertEq (conv2dResult? convIn2 convW2 1 2 1)
    (some { shape := [1, 16, 16, 16], dtype := .float32 })
    "conv2d"

  assertEq (pool2dResult? convIn2 2 0 2)
    (some { shape := [1, 3, 16, 16], dtype := .float32 })
    "pool2d"

  let meanNC : TensorDesc := { shape := [32], dtype := .float32 }
  let invstdNC : TensorDesc := { shape := [32], dtype := .float32 }
  assertEq (batchnormNCResult? x meanNC invstdNC)
    (some x)
    "batchnorm NC"

  let x4 : TensorDesc := { shape := [2, 32, 8, 8], dtype := .float32 }
  let meanNCHW : TensorDesc := { shape := [32], dtype := .float32 }
  let invstdNCHW : TensorDesc := { shape := [32], dtype := .float32 }
  assertEq (batchnormNCHWResult? x4 meanNCHW invstdNCHW)
    (some x4)
    "batchnorm NCHW"

  let pooled : TensorDesc := { shape := [2, 32, 4, 4], dtype := .float32 }
  let idx : TensorDesc := { shape := [2, 32, 4, 4], dtype := .int32 }
  assertEq (maxUnpool2dOutResult? pooled idx 8 8)
    (some { shape := [2, 32, 8, 8], dtype := .float32 })
    "maxUnpool2dOut"

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
  testHighLevelIndexing
  IO.println "✓ high-level indexing"
  testScatterAndBridges
  IO.println "✓ scatter + bridges"
  testNnShapes
  IO.println "✓ nn shapes"
  testReductions
  IO.println "✓ reductions"
  testElementwise
  IO.println "✓ elementwise"
  testCatAndMatmul
  IO.println "✓ cat + matmul"
  IO.println "=== SpecSemanticsSmoke OK ==="

end TinyGrad4.Test.SpecSemanticsSmoke

#eval! TinyGrad4.Test.SpecSemanticsSmoke.runAll
