import Float64
import TinyGrad4.Spec
import TinyGrad4.Test.Case
import TinyGrad4.Test.Profiles
import TinyGrad4.Test.Assertions

namespace TinyGrad4.Test.Port.IndexingProps

open TinyGrad4
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
    }
  ]

end TinyGrad4.Test.Port.IndexingProps
