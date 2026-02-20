import Float64
import TinyGrad4
import TinyGrad4.Spec
import TinyGrad4.Test.Case
import TinyGrad4.Test.Profiles
import TinyGrad4.Test.Assertions

namespace TinyGrad4.Test.Port.IndexingProps

open TinyGrad4
open TinyGrad4.Interpreter
open TinyGrad4.StaticTensor
open TinyGrad4.Spec
open TinyGrad4.Test
open TinyGrad4.Test.Assertions

def testEllipsisAndSliceShape : IO Unit := do
  let shape : Shape := [10, 20, 30]
  let items : List BasicIndexItem := [.ellipsis, .int 2, .slice Slice.all]
  match inferBasicIndexShape shape items with
  | some out =>
    if out != [10, 30] then
      throw <| IO.userError s!"ellipsis/int/slice shape mismatch: got {out}, expected [10, 30]"
  | none =>
    throw <| IO.userError "ellipsis/int/slice should infer a valid shape"

def testNewaxisAndIntShape : IO Unit := do
  let items : List BasicIndexItem := [.newaxis, .slice Slice.all, .int 1]
  match inferBasicIndexShape [2, 3] items with
  | some out =>
    if out != [1, 2] then
      throw <| IO.userError s!"newaxis/slice/int shape mismatch: got {out}, expected [1, 2]"
  | none =>
    throw <| IO.userError "newaxis/slice/int should infer a valid shape"

def testNormalizeIndexExamples : IO Unit := do
  if normalizeIndex 2 5 != some 2 then
    throw <| IO.userError "normalizeIndex positive case failed"
  if normalizeIndex (-1) 5 != some 4 then
    throw <| IO.userError "normalizeIndex negative case failed"
  if normalizeIndex 7 5 != none then
    throw <| IO.userError "normalizeIndex out-of-range case failed"

def testNormalizeIndexBoundsPlausible (cfg : RunConfig) : IO Unit :=
  assertPlausible cfg "normalizeIndex bounds"
    (∀ (dim : Nat) (idx : Int) (j : Nat),
      normalizeIndex idx dim = some j → j < dim)

def testSingleIntIndexShapePlausible (cfg : RunConfig) : IO Unit :=
  assertPlausible cfg "single-int indexing rank-1 shape"
    (∀ (dim : Nat) (idx : Int),
      let out := inferBasicIndexShape [dim] [.int idx]
      out = none ∨ out = some [])

def testItemRuntimeError : IO Unit := do
  let status := runTensorM do
    let x ← Tensor.arange 3 .float32
    StaticTensor.itemChecked x
  match status with
  | .ok _ => throw <| IO.userError "itemChecked should report an error on non-scalar tensors"
  | .error _ => pure ()

def testMaskedSelectPackedCountBounds : IO Unit := do
  let (count0F, count1F, countMixF, packedMix) := runTensorM do
    let x0 ← Tensor.arange 9 .float32
    let x ← reshapeUnsafe x0 [3, 3]
    let mask0 ← Tensor.fullBool [3, 3] false
    let mask1 ← Tensor.fullBool [3, 3] true
    let five ← Tensor.full [3, 3] .float32 5.0
    let maskMix ← StaticTensor.cmplt x five
    let (_, count0) ← StaticTensor.maskedSelectPacked x mask0
    let (_, count1) ← StaticTensor.maskedSelectPacked x mask1
    let (packedMix, countMix) ← StaticTensor.maskedSelectPacked x maskMix
    let count0F ← StaticTensor.cast count0 .float32
    let count1F ← StaticTensor.cast count1 .float32
    let countMixF ← StaticTensor.cast countMix .float32
    pure (count0F, count1F, countMixF, packedMix)
  assertRawAllClose (evalTensor count0F) #[0.0] 0.001 "maskedSelectPacked all-false count"
  assertRawAllClose (evalTensor count1F) #[9.0] 0.001 "maskedSelectPacked all-true count"
  assertRawAllClose (evalTensor countMixF) #[5.0] 0.001 "maskedSelectPacked mixed count"
  assertRawAllClose (evalTensor packedMix) #[0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0] 0.001
    "maskedSelectPacked mixed prefix payload"

def cases : List TestCase :=
  [
    {
      name := "indexing.shape.ellipsis_slice"
      group := "indexing"
      minProfile := .fast
      pythonRefs := ["test/unit/test_indexing.py::test_ellipsis_index"]
      suite := fun _ => ioTest "ellipsis + int + slice shape inference" testEllipsisAndSliceShape
    },
    {
      name := "indexing.shape.newaxis_int"
      group := "indexing"
      minProfile := .fast
      pythonRefs := ["test/unit/test_indexing.py::test_none_index", "test/unit/test_indexing.py::test_single_int_index"]
      suite := fun _ => ioTest "newaxis + slice + int shape inference" testNewaxisAndIntShape
    },
    {
      name := "indexing.normalize.examples"
      group := "indexing"
      minProfile := .fast
      pythonRefs := ["test/unit/test_indexing.py::test_out_of_bound_index"]
      suite := fun _ => ioTest "normalizeIndex representative examples" testNormalizeIndexExamples
    },
    {
      name := "indexing.prop.normalize_bounds"
      group := "indexing"
      minProfile := .medium
      pythonRefs := ["test/unit/test_indexing.py::test_out_of_bound_index"]
      suite := fun cfg =>
        ioTest "Plausible: normalizeIndex bounds" (testNormalizeIndexBoundsPlausible cfg)
    },
    {
      name := "indexing.prop.single_int_shape"
      group := "indexing"
      minProfile := .fast
      pythonRefs := ["test/unit/test_indexing.py::test_single_int_index"]
      suite := fun cfg =>
        ioTest "Plausible: single-int indexing shape" (testSingleIntIndexShapePlausible cfg)
    },
    {
      name := "indexing.runtime.item_non_scalar_error"
      group := "indexing"
      minProfile := .fast
      pythonRefs := ["test/test_tensor.py::test_item"]
      suite := fun _ =>
        ioTest "item rejects non-scalar tensors" testItemRuntimeError
    },
    {
      name := "indexing.runtime.masked_select_packed_counts"
      group := "indexing"
      minProfile := .medium
      pythonRefs := ["test/test_ops.py::test_masked_select"]
      suite := fun _ =>
        ioTest "maskedSelectPacked count bounds and prefix payload" testMaskedSelectPackedCountBounds
    }
  ]

end TinyGrad4.Test.Port.IndexingProps
