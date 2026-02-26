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

def testScatterAxisDimMismatch : IO Unit := do
  let (scatterOut, scatterReduceSumOut, scatterReduceMeanOut, scatterReduceAmaxOut, scatterReduceAminOut, scatterReduceProdOut,
      scatterReduceSumIncludeSelfOut, scatterReduceProdIncludeSelfOut, scatterReduceMeanIncludeSelfOut,
      scatterReduceAmaxIncludeSelfOut, scatterReduceAminIncludeSelfOut, scatterAddScalarOut, scatterMulScalarOut,
      scatterOutNegDim, scatterReduceSumNegDimOut, scatterAddScalarNegDimOut) := runTensorM do
    let base ← Tensor.zeros [1, 1, 16] .float32
    let i0 ← Tensor.full [1] .int32 5.0
    let i1 ← Tensor.full [1] .int32 7.0
    let i2 ← Tensor.full [1] .int32 13.0
    let i3 ← Tensor.full [1] .int32 15.0
    let idx01 ← StaticTensor.cat i0 i1 0 (by native_decide)
    let idx23 ← StaticTensor.cat i2 i3 0 (by native_decide)
    let idxFlat ← StaticTensor.cat idx01 idx23 0 (by native_decide)
    let idx ← reshapeUnsafe idxFlat [1, 1, 4]

    let s0 ← Tensor.full [1] .float32 6.0
    let s1 ← Tensor.full [1] .float32 8.0
    let s2 ← Tensor.full [1] .float32 14.0
    let s3 ← Tensor.full [1] .float32 16.0
    let src01 ← StaticTensor.cat s0 s1 0 (by native_decide)
    let src23 ← StaticTensor.cat s2 s3 0 (by native_decide)
    let srcFlat ← StaticTensor.cat src01 src23 0 (by native_decide)
    let src ← reshapeUnsafe srcFlat [1, 1, 4]
    let scatterOut ← scatter base 2 idx src
    let scatterOutNegDim ← scatterDim base (-1) idx src

    let ridx0 ← Tensor.full [1] .int32 5.0
    let ridx1 ← Tensor.full [1] .int32 5.0
    let ridx2 ← Tensor.full [1] .int32 5.0
    let ridx3 ← Tensor.full [1] .int32 2.0
    let ridx01 ← StaticTensor.cat ridx0 ridx1 0 (by native_decide)
    let ridx23 ← StaticTensor.cat ridx2 ridx3 0 (by native_decide)
    let ridxFlat ← StaticTensor.cat ridx01 ridx23 0 (by native_decide)
    let ridx ← reshapeUnsafe ridxFlat [1, 1, 4]

    let rs0 ← Tensor.full [1] .float32 1.0
    let rs1 ← Tensor.full [1] .float32 2.0
    let rs2 ← Tensor.full [1] .float32 3.0
    let rs3 ← Tensor.full [1] .float32 4.0
    let rsrc01 ← StaticTensor.cat rs0 rs1 0 (by native_decide)
    let rsrc23 ← StaticTensor.cat rs2 rs3 0 (by native_decide)
    let rsrcFlat ← StaticTensor.cat rsrc01 rsrc23 0 (by native_decide)
    let rsrc ← reshapeUnsafe rsrcFlat [1, 1, 4]
    let scatterReduceSumOut ← scatterReduce base 2 ridx rsrc .sum false
    let scatterReduceMeanOut ← scatterReduce base 2 ridx rsrc .mean false
    let scatterReduceAmaxOut ← scatterReduce base 2 ridx rsrc .amax false
    let scatterReduceAminOut ← scatterReduce base 2 ridx rsrc .amin false
    let ps0 ← Tensor.full [1] .float32 2.0
    let ps1 ← Tensor.full [1] .float32 3.0
    let ps2 ← Tensor.full [1] .float32 4.0
    let ps3 ← Tensor.full [1] .float32 5.0
    let psrc01 ← StaticTensor.cat ps0 ps1 0 (by native_decide)
    let psrc23 ← StaticTensor.cat ps2 ps3 0 (by native_decide)
    let psrcFlat ← StaticTensor.cat psrc01 psrc23 0 (by native_decide)
    let psrc ← reshapeUnsafe psrcFlat [1, 1, 4]
    let scatterReduceProdOut ← scatterReduce base 2 ridx psrc .prod false
    let onesBase ← Tensor.ones [1, 1, 16] .float32
    let scatterReduceSumIncludeSelfOut ← scatterReduce onesBase 2 ridx rsrc .sum true
    let scatterReduceProdIncludeSelfOut ← scatterReduce onesBase 2 ridx psrc .prod true
    let scatterReduceMeanIncludeSelfOut ← scatterReduce onesBase 2 ridx rsrc .mean true
    let scatterReduceAmaxIncludeSelfOut ← scatterReduce onesBase 2 ridx rsrc .amax true
    let highBase ← Tensor.full [1, 1, 16] .float32 10.0
    let scatterReduceAminIncludeSelfOut ← scatterReduce highBase 2 ridx rsrc .amin true
    let twosBase ← Tensor.full [1, 1, 16] .float32 2.0
    let scatterAddScalarOut ← scatterAddScalar twosBase 2 ridx 1.5
    let scatterMulScalarOut ← scatterMultiplyScalar twosBase 2 ridx 1.5
    let scatterReduceSumNegDimOut ← scatterReduceDim base (-1) ridx rsrc .sum false
    let scatterAddScalarNegDimOut ← scatterAddScalarDim twosBase (-1) ridx 1.5
    pure (scatterOut, scatterReduceSumOut, scatterReduceMeanOut, scatterReduceAmaxOut, scatterReduceAminOut, scatterReduceProdOut,
      scatterReduceSumIncludeSelfOut, scatterReduceProdIncludeSelfOut, scatterReduceMeanIncludeSelfOut,
      scatterReduceAmaxIncludeSelfOut, scatterReduceAminIncludeSelfOut, scatterAddScalarOut, scatterMulScalarOut,
      scatterOutNegDim, scatterReduceSumNegDimOut, scatterAddScalarNegDimOut)

  assertRawAllClose (evalTensor scatterOut)
    #[0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 0.0, 16.0] 0.001
    "scatter dim-mismatch lane"
  assertRawAllClose (evalTensor scatterReduceSumOut)
    #[0.0, 0.0, 4.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.001
    "scatterReduce sum dim-mismatch lane"
  assertRawAllClose (evalTensor scatterReduceMeanOut)
    #[0.0, 0.0, 4.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.001
    "scatterReduce mean dim-mismatch lane"
  assertRawAllClose (evalTensor scatterReduceAmaxOut)
    #[0.0, 0.0, 4.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.001
    "scatterReduce amax dim-mismatch lane"
  assertRawAllClose (evalTensor scatterReduceAminOut)
    #[0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.001
    "scatterReduce amin dim-mismatch lane"
  assertRawAllClose (evalTensor scatterReduceProdOut)
    #[0.0, 0.0, 5.0, 0.0, 0.0, 24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.001
    "scatterReduce prod dim-mismatch lane"
  assertRawAllClose (evalTensor scatterReduceSumIncludeSelfOut)
    #[1.0, 1.0, 5.0, 1.0, 1.0, 7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 0.001
    "scatterReduce sum include_self dim-mismatch lane"
  assertRawAllClose (evalTensor scatterReduceProdIncludeSelfOut)
    #[1.0, 1.0, 5.0, 1.0, 1.0, 24.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 0.001
    "scatterReduce prod include_self dim-mismatch lane"
  assertRawAllClose (evalTensor scatterReduceMeanIncludeSelfOut)
    #[1.0, 1.0, 2.5, 1.0, 1.0, 1.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 0.001
    "scatterReduce mean include_self dim-mismatch lane"
  assertRawAllClose (evalTensor scatterReduceAmaxIncludeSelfOut)
    #[1.0, 1.0, 4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 0.001
    "scatterReduce amax include_self dim-mismatch lane"
  assertRawAllClose (evalTensor scatterReduceAminIncludeSelfOut)
    #[10.0, 10.0, 4.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0] 0.001
    "scatterReduce amin include_self dim-mismatch lane"
  assertRawAllClose (evalTensor scatterAddScalarOut)
    #[2.0, 2.0, 3.5, 2.0, 2.0, 6.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0] 0.001
    "scatter add scalar dim-mismatch lane"
  assertRawAllClose (evalTensor scatterMulScalarOut)
    #[2.0, 2.0, 3.0, 2.0, 2.0, 6.75, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0] 0.001
    "scatter multiply scalar dim-mismatch lane"
  assertRawAllClose (evalTensor scatterOutNegDim)
    #[0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 0.0, 16.0] 0.001
    "scatter dim=-1 lane"
  assertRawAllClose (evalTensor scatterReduceSumNegDimOut)
    #[0.0, 0.0, 4.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.001
    "scatterReduce sum dim=-1 lane"
  assertRawAllClose (evalTensor scatterAddScalarNegDimOut)
    #[2.0, 2.0, 3.5, 2.0, 2.0, 6.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0] 0.001
    "scatter add scalar dim=-1 lane"

def testScatterDim1Mismatch : IO Unit := do
  let (scatterOut, scatterReduceSumOut, scatterAddScalarOut) := runTensorM do
    let base ← Tensor.zeros [1, 4, 4] .float32
    let i0 ← Tensor.full [1] .int32 0.0
    let i1 ← Tensor.full [1] .int32 1.0
    let i2 ← Tensor.full [1] .int32 2.0
    let i3 ← Tensor.full [1] .int32 3.0
    let r00 ← StaticTensor.cat i0 i1 0 (by native_decide)
    let r01 ← StaticTensor.cat i2 i3 0 (by native_decide)
    let row0 ← StaticTensor.cat r00 r01 0 (by native_decide)
    let r10 ← StaticTensor.cat i3 i2 0 (by native_decide)
    let r11 ← StaticTensor.cat i1 i0 0 (by native_decide)
    let row1 ← StaticTensor.cat r10 r11 0 (by native_decide)
    let idx01 ← StaticTensor.cat row0 row1 0 (by native_decide)
    let idx ← reshapeUnsafe idx01 [1, 2, 4]
    let src0 ← Tensor.arange 8 .float32
    let src1 ← addScalar src0 1.0
    let src ← reshapeUnsafe src1 [1, 2, 4]
    let scatterOut ← scatter base 1 idx src

    let j0 ← Tensor.full [1] .int32 1.0
    let j1 ← Tensor.full [1] .int32 1.0
    let j2 ← Tensor.full [1] .int32 2.0
    let j3 ← Tensor.full [1] .int32 2.0
    let k0 ← Tensor.full [1] .int32 1.0
    let k1 ← Tensor.full [1] .int32 3.0
    let k2 ← Tensor.full [1] .int32 3.0
    let k3 ← Tensor.full [1] .int32 0.0
    let s00 ← StaticTensor.cat j0 j1 0 (by native_decide)
    let s01 ← StaticTensor.cat j2 j3 0 (by native_decide)
    let ridxRow0 ← StaticTensor.cat s00 s01 0 (by native_decide)
    let s10 ← StaticTensor.cat k0 k1 0 (by native_decide)
    let s11 ← StaticTensor.cat k2 k3 0 (by native_decide)
    let ridxRow1 ← StaticTensor.cat s10 s11 0 (by native_decide)
    let ridx01 ← StaticTensor.cat ridxRow0 ridxRow1 0 (by native_decide)
    let ridx ← reshapeUnsafe ridx01 [1, 2, 4]
    let rsrc ← reshapeUnsafe src1 [1, 2, 4]
    let scatterReduceSumOut ← scatterReduce base 1 ridx rsrc .sum false
    let twosBase ← Tensor.full [1, 4, 4] .float32 2.0
    let scatterAddScalarOut ← scatterAddScalar twosBase 1 ridx 1.5
    pure (scatterOut, scatterReduceSumOut, scatterAddScalarOut)

  assertRawAllClose (evalTensor scatterOut)
    #[1.0, 0.0, 0.0, 8.0, 0.0, 2.0, 7.0, 0.0, 0.0, 6.0, 3.0, 0.0, 5.0, 0.0, 0.0, 4.0] 0.001
    "scatter dim=1 mismatch lane"
  assertRawAllClose (evalTensor scatterReduceSumOut)
    #[0.0, 0.0, 0.0, 8.0, 6.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 6.0, 7.0, 0.0] 0.001
    "scatterReduce sum dim=1 mismatch lane"
  assertRawAllClose (evalTensor scatterAddScalarOut)
    #[2.0, 2.0, 2.0, 3.5, 5.0, 3.5, 2.0, 2.0, 2.0, 2.0, 3.5, 3.5, 2.0, 3.5, 3.5, 2.0] 0.001
    "scatter add scalar dim=1 mismatch lane"

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
    },
    {
      name := "indexing.runtime.scatter_dim_mismatch"
      group := "indexing"
      minProfile := .medium
      pythonRefs := [
        "test/test_ops.py::test_scatter",
        "test/test_ops.py::test_scatter_add",
        "test/test_ops.py::test_scatter_mul",
        "test/test_ops.py::test_scatter_reduce",
        "test/test_ops.py::test_max_unpool2d"
      ]
      suite := fun _ =>
        ioTest "scatter/scatterReduce dim-mismatch parity lane" testScatterAxisDimMismatch
    },
    {
      name := "indexing.runtime.scatter_dim1_mismatch"
      group := "indexing"
      minProfile := .medium
      pythonRefs := ["test/test_ops.py::test_scatter", "test/test_ops.py::test_scatter_reduce", "test/test_ops.py::test_scatter_add"]
      suite := fun _ =>
        ioTest "scatter/scatterReduce/scatterAdd dim=1 mismatch parity lane" testScatterDim1Mismatch
    }
  ]

end TinyGrad4.Test.Port.IndexingProps
