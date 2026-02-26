import Float64
import TinyGrad4
import TinyGrad4.Test.Case
import TinyGrad4.Test.Assertions

namespace TinyGrad4.Test.Port.FixtureOracle

open TinyGrad4
open TinyGrad4.Interpreter
open TinyGrad4.StaticTensor
open TinyGrad4.Test
open TinyGrad4.Test.Assertions

private structure FixtureCase where
  id : String
  shape : Shape
  expected : Array Float64
  pythonRef : String

private def parseExpected (j : Lean.Json) : Except String (Array Float64) := do
  let arr ← (← j.getObjVal? "expected").getArr?
  let mut out : Array Float64 := #[]
  for v in arr do
    let n ← v.getNum?
    out := out.push n.toFloat
  pure out

private def parseShape (j : Lean.Json) : Except String Shape := do
  let arr ← (← j.getObjVal? "shape").getArr?
  let mut out : Shape := []
  for v in arr do
    let n ← v.getNat?
    out := out ++ [n]
  pure out

private def parseCase (j : Lean.Json) : Except String FixtureCase := do
  let id ← (← j.getObjVal? "id").getStr?
  let shape ← parseShape j
  let expected ← parseExpected j
  let pythonRef ← (← j.getObjVal? "python_ref").getStr?
  pure { id, shape, expected, pythonRef }

private def loadCases (path : System.FilePath) : IO (Array FixtureCase) := do
  let raw ← IO.FS.readFile path
  let parsed ←
    match Lean.Json.parse raw with
    | .ok j => pure j
    | .error err => throw <| IO.userError s!"failed to parse fixture JSON: {err}"
  let casesArr ←
    match (parsed.getObjVal? "cases") with
    | .ok v =>
      match v.getArr? with
      | .ok arr => pure arr
      | .error err => throw <| IO.userError s!"cases must be an array: {err}"
    | .error err => throw <| IO.userError s!"missing cases field: {err}"
  let mut out : Array FixtureCase := #[]
  for item in casesArr do
    match parseCase item with
    | .ok c => out := out.push c
    | .error err => throw <| IO.userError s!"invalid fixture entry: {err}"
  pure out

private def runCase (id : String) : IO (Shape × RawBuffer) := do
  match id with
  | "eye_3_5" =>
    let t := runTensorM do
      Tensor.eye 3 5 .float32
    pure (t.uop.shape, evalTensor t)
  | "randperm_10_seed0" =>
    let t := runTensorM do
      Tensor.randperm 10 .float32 0
    pure (t.uop.shape, evalTensor t)
  | "meshgrid_ij_x" =>
    let t := runTensorM do
      let x ← Tensor.arange 3 .float32
      let y ← Tensor.linspace 3.0 6.0 4 .float32
      let (gx, _) ← Tensor.meshgridIJ x y
      pure gx
    pure (t.uop.shape, evalTensor t)
  | "meshgrid_ij_y" =>
    let t := runTensorM do
      let x ← Tensor.arange 3 .float32
      let y ← Tensor.linspace 3.0 6.0 4 .float32
      let (_, gy) ← Tensor.meshgridIJ x y
      pure gy
    pure (t.uop.shape, evalTensor t)
  | "meshgrid_xy_x" =>
    let t := runTensorM do
      let x ← Tensor.arange 3 .float32
      let y ← Tensor.linspace 3.0 6.0 4 .float32
      let (gx, _) ← Tensor.meshgridXY x y
      pure gx
    pure (t.uop.shape, evalTensor t)
  | "meshgrid_xy_y" =>
    let t := runTensorM do
      let x ← Tensor.arange 3 .float32
      let y ← Tensor.linspace 3.0 6.0 4 .float32
      let (_, gy) ← Tensor.meshgridXY x y
      pure gy
    pure (t.uop.shape, evalTensor t)
  | "cumsum_axis1_2x3" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let ones6 ← Tensor.ones [6] .float32
      let shifted ← add base ones6
      let mat ← reshape shifted [2, 3] (by native_decide)
      StaticTensor.cumsumAxis mat 1
    pure (t.uop.shape, evalTensor t)
  | "cumprod_axis1_2x3" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let ones6 ← Tensor.ones [6] .float32
      let shifted ← add base ones6
      let mat ← reshape shifted [2, 3] (by native_decide)
      StaticTensor.cumprodAxis mat 1
    pure (t.uop.shape, evalTensor t)
  | "logsumexp_1d_6" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let ones6 ← Tensor.ones [6] .float32
      let shifted ← add base ones6
      StaticTensor.logsumexp shifted
    pure (t.uop.shape, evalTensor t)
  | "logcumsumexp_1d_6" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let ones6 ← Tensor.ones [6] .float32
      let shifted ← add base ones6
      StaticTensor.logcumsumexp shifted
    pure (t.uop.shape, evalTensor t)
  | "linspace_5_10_3" =>
    let t := runTensorM do
      Tensor.linspace 5.0 10.0 3 .float32
    pure (t.uop.shape, evalTensor t)
  | "relu_1d_6_shift25" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let shifted ← addScalar base (-2.5)
      StaticTensor.relu shifted
    pure (t.uop.shape, evalTensor t)
  | "tanh_1d_6_shift25" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let shifted ← addScalar base (-2.5)
      StaticTensor.tanh shifted
    pure (t.uop.shape, evalTensor t)
  | "silu_1d_6_shift25" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let shifted ← addScalar base (-2.5)
      StaticTensor.silu shifted
    pure (t.uop.shape, evalTensor t)
  | "gelu_1d_6_shift25" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let shifted ← addScalar base (-2.5)
      StaticTensor.gelu shifted
    pure (t.uop.shape, evalTensor t)
  | "max_1d_6_shift25" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let shifted ← addScalar base (-2.5)
      StaticTensor.max shifted
    pure (t.uop.shape, evalTensor t)
  | "min_1d_6_shift25" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let shifted ← addScalar base (-2.5)
      StaticTensor.min shifted
    pure (t.uop.shape, evalTensor t)
  | "softmax_last_2x3" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let mat ← reshape base [2, 3] (by native_decide)
      StaticTensor.softmax mat
    pure (t.uop.shape, evalTensor t)
  | "logsoftmax_last_2x3" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let mat ← reshape base [2, 3] (by native_decide)
      StaticTensor.logSoftmax mat
    pure (t.uop.shape, evalTensor t)
  | "softmax_axis0_2x3" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let mat ← reshape base [2, 3] (by native_decide)
      StaticTensor.softmaxAxisF mat ⟨0, by decide⟩
    pure (t.uop.shape, evalTensor t)
  | "round_1d_6_shift25" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let shifted ← addScalar base (-2.5)
      StaticTensor.round shifted
    pure (t.uop.shape, evalTensor t)
  | "sign_1d_6_shift25" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let shifted ← addScalar base (-2.5)
      StaticTensor.sign shifted
    pure (t.uop.shape, evalTensor t)
  | "lerp_scalar_1d_3" =>
    let t := runTensorM do
      let start ← Tensor.linspace 1.0 3.0 3 .float32
      let stop ← Tensor.linspace 4.0 6.0 3 .float32
      StaticTensor.lerpScalar start stop 0.5
    pure (t.uop.shape, evalTensor t)
  | "asin_vals7" =>
    let t := runTensorM do
      let base ← Tensor.arange 7 .float32
      let scaled ← scale base 0.3
      let x ← addScalar scaled (-0.9)
      StaticTensor.asin x
    pure (t.uop.shape, evalTensor t)
  | "acos_vals7" =>
    let t := runTensorM do
      let base ← Tensor.arange 7 .float32
      let scaled ← scale base 0.3
      let x ← addScalar scaled (-0.9)
      StaticTensor.acos x
    pure (t.uop.shape, evalTensor t)
  | "atan_1d_6_shift25" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let shifted ← addScalar base (-2.5)
      StaticTensor.atan shifted
    pure (t.uop.shape, evalTensor t)
  | "sinh_vals7" =>
    let t := runTensorM do
      let base ← Tensor.arange 7 .float32
      let scaled ← scale base 0.5
      let x ← addScalar scaled (-1.5)
      StaticTensor.sinh x
    pure (t.uop.shape, evalTensor t)
  | "cosh_vals7" =>
    let t := runTensorM do
      let base ← Tensor.arange 7 .float32
      let scaled ← scale base 0.5
      let x ← addScalar scaled (-1.5)
      StaticTensor.cosh x
    pure (t.uop.shape, evalTensor t)
  | "erf_vals7" =>
    let t := runTensorM do
      let base ← Tensor.arange 7 .float32
      let scaled ← scale base 0.5
      let x ← addScalar scaled (-1.5)
      StaticTensor.erf x
    pure (t.uop.shape, evalTensor t)
  | "softsign_1d_6_shift25" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let shifted ← addScalar base (-2.5)
      StaticTensor.softsign shifted
    pure (t.uop.shape, evalTensor t)
  | "mish_1d_6_shift25" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let shifted ← addScalar base (-2.5)
      StaticTensor.mish shifted
    pure (t.uop.shape, evalTensor t)
  | "celu_1d_6_shift25" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let shifted ← addScalar base (-2.5)
      StaticTensor.celu shifted
    pure (t.uop.shape, evalTensor t)
  | "selu_1d_6_shift25" =>
    let t := runTensorM do
      let base ← Tensor.arange 6 .float32
      let shifted ← addScalar base (-2.5)
      StaticTensor.selu shifted
    pure (t.uop.shape, evalTensor t)
  | "copysign_1d_4" =>
    let t := runTensorM do
      let m0 ← Tensor.full [1] .float32 1.0
      let m1 ← Tensor.full [1] .float32 (-2.0)
      let mz ← Tensor.zeros [1] .float32
      let m01 ← StaticTensor.cat m0 m1 0 (by native_decide)
      let m23 ← StaticTensor.cat mz mz 0 (by native_decide)
      let mag ← StaticTensor.cat m01 m23 0 (by native_decide)
      let s0 ← Tensor.full [1] .float32 (-1.0)
      let s1 ← Tensor.full [1] .float32 1.0
      let sz := mz
      let sNegZero ← Tensor.full [1] .float32 (-0.0)
      let s01 ← StaticTensor.cat s0 s1 0 (by native_decide)
      let s23 ← StaticTensor.cat sNegZero sz 0 (by native_decide)
      let signSrc ← StaticTensor.cat s01 s23 0 (by native_decide)
      StaticTensor.copysign mag signSrc
    pure (t.uop.shape, evalTensor t)
  | "logaddexp_1d_3" =>
    let t := runTensorM do
      let la0 ← Tensor.full [1] .float32 100.0
      let la1 ← Tensor.full [1] .float32 (-100.0)
      let la2 ← Tensor.full [1] .float32 1.0
      let la01 ← StaticTensor.cat la0 la1 0 (by native_decide)
      let la ← StaticTensor.cat la01 la2 0 (by native_decide)
      let lb0 ← Tensor.full [1] .float32 99.0
      let lb1 ← Tensor.full [1] .float32 100.0
      let lb2 ← Tensor.full [1] .float32 (-2.0)
      let lb01 ← StaticTensor.cat lb0 lb1 0 (by native_decide)
      let lb ← StaticTensor.cat lb01 lb2 0 (by native_decide)
      StaticTensor.logaddexp la lb
    pure (t.uop.shape, evalTensor t)
  | "masked_fill_scalar_1d_6" =>
    let t := runTensorM do
      let base0 ← Tensor.arange 6 .float32
      let base ← addScalar base0 1.0
      let one ← Tensor.full [6] .float32 1.0
      let three ← Tensor.full [6] .float32 3.0
      let six ← Tensor.full [6] .float32 6.0
      let m1 ← StaticTensor.cmpeq base one
      let m3 ← StaticTensor.cmpeq base three
      let m6 ← StaticTensor.cmpeq base six
      let m13 ← StaticTensor.bitor m1 m3
      let mask ← StaticTensor.bitor m13 m6
      StaticTensor.maskedFillScalar base mask (-12.0)
    pure (t.uop.shape, evalTensor t)
  | "take_flat_2x3_idx5" =>
    let t := runTensorM do
      let base0 ← Tensor.arange 6 .float32
      let base ← reshapeUnsafe base0 [2, 3]
      let idxA ← Tensor.arange 3 .int32
      let idxB ← Tensor.arange 2 .int32
      let idx ← StaticTensor.cat idxA idxB 0 (by native_decide)
      StaticTensor.take base idx
    pure (t.uop.shape, evalTensor t)
  | "item_scalar_42" =>
    let t := runTensorM do
      let x ← Tensor.full [] .float32 42.0
      StaticTensor.item x
    pure (t.uop.shape, evalTensor t)
  | "triu_diag1_3x4" =>
    let t := runTensorM do
      let base0 ← Tensor.arange 12 .float32
      let base1 ← addScalar base0 1.0
      let mat ← reshapeUnsafe base1 [3, 4]
      StaticTensor.triu mat 1
    pure (t.uop.shape, evalTensor t)
  | "tril_diag_neg1_3x4" =>
    let t := runTensorM do
      let base0 ← Tensor.arange 12 .float32
      let base1 ← addScalar base0 1.0
      let mat ← reshapeUnsafe base1 [3, 4]
      StaticTensor.tril mat (-1)
    pure (t.uop.shape, evalTensor t)
  | "diag_vec3" =>
    let t := runTensorM do
      let v ← Tensor.linspace 1.0 3.0 3 .float32
      StaticTensor.diag v
    pure (t.uop.shape, evalTensor t)
  | "diagonal_mat3" =>
    let t := runTensorM do
      let base ← Tensor.arange 9 .float32
      let mat ← reshapeUnsafe base [3, 3]
      StaticTensor.diagonal mat
    pure (t.uop.shape, evalTensor t)
  | "unfold_8_2_2" =>
    let t := runTensorM do
      let v ← Tensor.arange 8 .float32
      StaticTensor.unfold v 0 2 2
    pure (t.uop.shape, evalTensor t)
  | "masked_select_packed_payload_3x3" =>
    let t := runTensorM do
      let base ← Tensor.arange 9 .float32
      let x ← reshapeUnsafe base [3, 3]
      let z0 ← Tensor.full [3, 3] .float32 0.0
      let z2 ← Tensor.full [3, 3] .float32 2.0
      let z4 ← Tensor.full [3, 3] .float32 4.0
      let z8 ← Tensor.full [3, 3] .float32 8.0
      let m0 ← StaticTensor.cmpeq x z0
      let m2 ← StaticTensor.cmpeq x z2
      let m4 ← StaticTensor.cmpeq x z4
      let m8 ← StaticTensor.cmpeq x z8
      let m02 ← StaticTensor.bitor m0 m2
      let m48 ← StaticTensor.bitor m4 m8
      let mask ← StaticTensor.bitor m02 m48
      let (packed, _) ← StaticTensor.maskedSelectPacked x mask
      pure packed
    pure (t.uop.shape, evalTensor t)
  | "masked_select_packed_count_3x3" =>
    let t := runTensorM do
      let base ← Tensor.arange 9 .float32
      let x ← reshapeUnsafe base [3, 3]
      let z0 ← Tensor.full [3, 3] .float32 0.0
      let z2 ← Tensor.full [3, 3] .float32 2.0
      let z4 ← Tensor.full [3, 3] .float32 4.0
      let z8 ← Tensor.full [3, 3] .float32 8.0
      let m0 ← StaticTensor.cmpeq x z0
      let m2 ← StaticTensor.cmpeq x z2
      let m4 ← StaticTensor.cmpeq x z4
      let m8 ← StaticTensor.cmpeq x z8
      let m02 ← StaticTensor.bitor m0 m2
      let m48 ← StaticTensor.bitor m4 m8
      let mask ← StaticTensor.bitor m02 m48
      let (_, count) ← StaticTensor.maskedSelectPacked x mask
      StaticTensor.cast count .float32
    pure (t.uop.shape, evalTensor t)
  | "scatter_dim_mismatch_1x1x16" =>
    let t := runTensorM do
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
      scatter base 2 idx src
    pure (t.uop.shape, evalTensor t)
  | "scatter_reduce_sum_dim_mismatch_1x1x16" =>
    let t := runTensorM do
      let base ← Tensor.zeros [1, 1, 16] .float32

      let i0 ← Tensor.full [1] .int32 5.0
      let i1 ← Tensor.full [1] .int32 5.0
      let i2 ← Tensor.full [1] .int32 5.0
      let i3 ← Tensor.full [1] .int32 2.0
      let idx01 ← StaticTensor.cat i0 i1 0 (by native_decide)
      let idx23 ← StaticTensor.cat i2 i3 0 (by native_decide)
      let idxFlat ← StaticTensor.cat idx01 idx23 0 (by native_decide)
      let idx ← reshapeUnsafe idxFlat [1, 1, 4]

      let s0 ← Tensor.full [1] .float32 1.0
      let s1 ← Tensor.full [1] .float32 2.0
      let s2 ← Tensor.full [1] .float32 3.0
      let s3 ← Tensor.full [1] .float32 4.0
      let src01 ← StaticTensor.cat s0 s1 0 (by native_decide)
      let src23 ← StaticTensor.cat s2 s3 0 (by native_decide)
      let srcFlat ← StaticTensor.cat src01 src23 0 (by native_decide)
      let src ← reshapeUnsafe srcFlat [1, 1, 4]
      scatterReduce base 2 idx src .sum false
    pure (t.uop.shape, evalTensor t)
  | "scatter_reduce_mean_dim_mismatch_1x1x16" =>
    let t := runTensorM do
      let base ← Tensor.zeros [1, 1, 16] .float32

      let i0 ← Tensor.full [1] .int32 5.0
      let i1 ← Tensor.full [1] .int32 5.0
      let i2 ← Tensor.full [1] .int32 5.0
      let i3 ← Tensor.full [1] .int32 2.0
      let idx01 ← StaticTensor.cat i0 i1 0 (by native_decide)
      let idx23 ← StaticTensor.cat i2 i3 0 (by native_decide)
      let idxFlat ← StaticTensor.cat idx01 idx23 0 (by native_decide)
      let idx ← reshapeUnsafe idxFlat [1, 1, 4]

      let s0 ← Tensor.full [1] .float32 1.0
      let s1 ← Tensor.full [1] .float32 2.0
      let s2 ← Tensor.full [1] .float32 3.0
      let s3 ← Tensor.full [1] .float32 4.0
      let src01 ← StaticTensor.cat s0 s1 0 (by native_decide)
      let src23 ← StaticTensor.cat s2 s3 0 (by native_decide)
      let srcFlat ← StaticTensor.cat src01 src23 0 (by native_decide)
      let src ← reshapeUnsafe srcFlat [1, 1, 4]
      scatterReduce base 2 idx src .mean false
    pure (t.uop.shape, evalTensor t)
  | "scatter_reduce_amax_dim_mismatch_1x1x16" =>
    let t := runTensorM do
      let base ← Tensor.zeros [1, 1, 16] .float32

      let i0 ← Tensor.full [1] .int32 5.0
      let i1 ← Tensor.full [1] .int32 5.0
      let i2 ← Tensor.full [1] .int32 5.0
      let i3 ← Tensor.full [1] .int32 2.0
      let idx01 ← StaticTensor.cat i0 i1 0 (by native_decide)
      let idx23 ← StaticTensor.cat i2 i3 0 (by native_decide)
      let idxFlat ← StaticTensor.cat idx01 idx23 0 (by native_decide)
      let idx ← reshapeUnsafe idxFlat [1, 1, 4]

      let s0 ← Tensor.full [1] .float32 1.0
      let s1 ← Tensor.full [1] .float32 2.0
      let s2 ← Tensor.full [1] .float32 3.0
      let s3 ← Tensor.full [1] .float32 4.0
      let src01 ← StaticTensor.cat s0 s1 0 (by native_decide)
      let src23 ← StaticTensor.cat s2 s3 0 (by native_decide)
      let srcFlat ← StaticTensor.cat src01 src23 0 (by native_decide)
      let src ← reshapeUnsafe srcFlat [1, 1, 4]
      scatterReduce base 2 idx src .amax false
    pure (t.uop.shape, evalTensor t)
  | "scatter_reduce_amin_dim_mismatch_1x1x16" =>
    let t := runTensorM do
      let base ← Tensor.zeros [1, 1, 16] .float32

      let i0 ← Tensor.full [1] .int32 5.0
      let i1 ← Tensor.full [1] .int32 5.0
      let i2 ← Tensor.full [1] .int32 5.0
      let i3 ← Tensor.full [1] .int32 2.0
      let idx01 ← StaticTensor.cat i0 i1 0 (by native_decide)
      let idx23 ← StaticTensor.cat i2 i3 0 (by native_decide)
      let idxFlat ← StaticTensor.cat idx01 idx23 0 (by native_decide)
      let idx ← reshapeUnsafe idxFlat [1, 1, 4]

      let s0 ← Tensor.full [1] .float32 1.0
      let s1 ← Tensor.full [1] .float32 2.0
      let s2 ← Tensor.full [1] .float32 3.0
      let s3 ← Tensor.full [1] .float32 4.0
      let src01 ← StaticTensor.cat s0 s1 0 (by native_decide)
      let src23 ← StaticTensor.cat s2 s3 0 (by native_decide)
      let srcFlat ← StaticTensor.cat src01 src23 0 (by native_decide)
      let src ← reshapeUnsafe srcFlat [1, 1, 4]
      scatterReduce base 2 idx src .amin false
    pure (t.uop.shape, evalTensor t)
  | "scatter_reduce_prod_dim_mismatch_1x1x16" =>
    let t := runTensorM do
      let base ← Tensor.zeros [1, 1, 16] .float32

      let i0 ← Tensor.full [1] .int32 5.0
      let i1 ← Tensor.full [1] .int32 5.0
      let i2 ← Tensor.full [1] .int32 5.0
      let i3 ← Tensor.full [1] .int32 2.0
      let idx01 ← StaticTensor.cat i0 i1 0 (by native_decide)
      let idx23 ← StaticTensor.cat i2 i3 0 (by native_decide)
      let idxFlat ← StaticTensor.cat idx01 idx23 0 (by native_decide)
      let idx ← reshapeUnsafe idxFlat [1, 1, 4]

      let s0 ← Tensor.full [1] .float32 2.0
      let s1 ← Tensor.full [1] .float32 3.0
      let s2 ← Tensor.full [1] .float32 4.0
      let s3 ← Tensor.full [1] .float32 5.0
      let src01 ← StaticTensor.cat s0 s1 0 (by native_decide)
      let src23 ← StaticTensor.cat s2 s3 0 (by native_decide)
      let srcFlat ← StaticTensor.cat src01 src23 0 (by native_decide)
      let src ← reshapeUnsafe srcFlat [1, 1, 4]
      scatterReduce base 2 idx src .prod false
    pure (t.uop.shape, evalTensor t)
  | "conv_transpose2d_core_1x1x2x2" =>
    let t := runTensorM do
      let x0 ← Tensor.arange 4 .float32
      let x1 ← addScalar x0 1.0
      let x ← reshapeUnsafe x1 [1, 1, 2, 2]
      let w ← Tensor.ones [1, 1, 2, 2] .float32
      convTranspose2d x w none 0 2 1 1
    pure (t.uop.shape, evalTensor t)
  | "max_unpool2d_default_1x1x2x2" =>
    let t := runTensorM do
      let p0 ← Tensor.full [1] .float32 6.0
      let p1 ← Tensor.full [1] .float32 8.0
      let p2 ← Tensor.full [1] .float32 14.0
      let p3 ← Tensor.full [1] .float32 16.0
      let p01 ← StaticTensor.cat p0 p1 0 (by native_decide)
      let p23 ← StaticTensor.cat p2 p3 0 (by native_decide)
      let pFlat ← StaticTensor.cat p01 p23 0 (by native_decide)
      let pooled ← reshapeUnsafe pFlat [1, 1, 2, 2]

      let i0 ← Tensor.full [1] .int32 5.0
      let i1 ← Tensor.full [1] .int32 7.0
      let i2 ← Tensor.full [1] .int32 13.0
      let i3 ← Tensor.full [1] .int32 15.0
      let i01 ← StaticTensor.cat i0 i1 0 (by native_decide)
      let i23 ← StaticTensor.cat i2 i3 0 (by native_decide)
      let iFlat ← StaticTensor.cat i01 i23 0 (by native_decide)
      let idx ← reshapeUnsafe iFlat [1, 1, 2, 2]
      maxUnpool2d pooled idx 2 2
    pure (t.uop.shape, evalTensor t)
  | "max_unpool2d_out_1x1x2x2" =>
    let t := runTensorM do
      let p0 ← Tensor.full [1] .float32 6.0
      let p1 ← Tensor.full [1] .float32 8.0
      let p2 ← Tensor.full [1] .float32 14.0
      let p3 ← Tensor.full [1] .float32 16.0
      let p01 ← StaticTensor.cat p0 p1 0 (by native_decide)
      let p23 ← StaticTensor.cat p2 p3 0 (by native_decide)
      let pFlat ← StaticTensor.cat p01 p23 0 (by native_decide)
      let pooled ← reshapeUnsafe pFlat [1, 1, 2, 2]

      let i0 ← Tensor.full [1] .int32 5.0
      let i1 ← Tensor.full [1] .int32 7.0
      let i2 ← Tensor.full [1] .int32 13.0
      let i3 ← Tensor.full [1] .int32 15.0
      let i01 ← StaticTensor.cat i0 i1 0 (by native_decide)
      let i23 ← StaticTensor.cat i2 i3 0 (by native_decide)
      let iFlat ← StaticTensor.cat i01 i23 0 (by native_decide)
      let idx ← reshapeUnsafe iFlat [1, 1, 2, 2]
      maxUnpool2dOut (outH := 4) (outW := 4) pooled idx
    pure (t.uop.shape, evalTensor t)
  | "batchnorm_nc_2x3" =>
    let t := runTensorM do
      let x0 ← Tensor.arange 6 .float32
      let x ← reshapeUnsafe x0 [2, 3]
      let mean ← Tensor.linspace 1.5 3.5 3 .float32
      let invstd ← Tensor.full [3] .float32 0.6666667
      StaticTensor.batchnorm x none none mean invstd
    pure (t.uop.shape, evalTensor t)
  | "batchnorm_nchw_1x2x2x2_affine" =>
    let t := runTensorM do
      let x0 ← Tensor.arange 8 .float32
      let x ← reshapeUnsafe x0 [1, 2, 2, 2]
      let m0 ← Tensor.full [1] .float32 1.5
      let m1 ← Tensor.full [1] .float32 5.5
      let mean ← StaticTensor.cat m0 m1 0 (by native_decide)
      let i0 ← Tensor.full [1] .float32 1.0
      let i1 ← Tensor.full [1] .float32 0.5
      let invstd ← StaticTensor.cat i0 i1 0 (by native_decide)
      let w0 ← Tensor.full [1] .float32 2.0
      let w1 ← Tensor.full [1] .float32 (-1.0)
      let weight ← StaticTensor.cat w0 w1 0 (by native_decide)
      let b0 ← Tensor.full [1] .float32 0.5
      let b1 ← Tensor.full [1] .float32 1.0
      let bias ← StaticTensor.cat b0 b1 0 (by native_decide)
      StaticTensor.batchnormNCHW x (some weight) (some bias) mean invstd
    pure (t.uop.shape, evalTensor t)
  | "dropout_p1_zero_1d8" =>
    let t := runTensorM do
      let x ← Tensor.ones [8] .float32
      StaticTensor.dropout x 1.0 true 0
    pure (t.uop.shape, evalTensor t)
  | _ =>
    throw <| IO.userError s!"unknown fixture case id: {id}"

def testCoreOpsFixtureParity : IO Unit := do
  let cases ← loadCases "lean4/testdata/parity/core_ops.json"
  if cases.isEmpty then
    throw <| IO.userError "fixture file has no cases"
  for c in cases do
    let (shape, raw) ← runCase c.id
    assertShape shape c.shape s!"fixture shape parity: {c.id}"
    assertRawAllClose raw c.expected 0.001 s!"fixture value parity: {c.id} ({c.pythonRef})"

def cases : List TestCase :=
  [
    {
      name := "fixture.core_ops.python_oracle"
      group := "curated"
      minProfile := .slow
      pythonRefs := ["lean4/testdata/parity/core_ops.json"]
      suite := fun _ => ioTest "python oracle fixtures: core ops parity" testCoreOpsFixtureParity
    }
  ]

end TinyGrad4.Test.Port.FixtureOracle
