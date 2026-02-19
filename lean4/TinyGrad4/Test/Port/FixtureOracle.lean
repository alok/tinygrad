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
