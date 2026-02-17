import Float64
import TinyGrad4
import TinyGrad4.Test.Case
import TinyGrad4.Test.Profiles
import TinyGrad4.Test.Assertions

namespace TinyGrad4.Test.Port.ExtendedParity

open TinyGrad4
open TinyGrad4.Interpreter
open TinyGrad4.StaticTensor
open TinyGrad4.Test
open TinyGrad4.Test.Assertions

private def assertPermutationRange (raw : RawBuffer) (n : Nat) (label : String) : IO Unit := do
  if raw.numF32 != n then
    throw <| IO.userError s!"{label}: size {raw.numF32} != {n}"
  for i in [:n] do
    let target := Float64.ofNat i
    let mut found := false
    for j in [:n] do
      if Float64.abs (raw.getF32 j - target) < 0.1 then
        found := true
    if !found then
      throw <| IO.userError s!"{label}: missing value {i}"
  for i in [:n] do
    for j in [i + 1:n] do
      if Float64.abs (raw.getF32 i - raw.getF32 j) < 0.1 then
        throw <| IO.userError s!"{label}: duplicate values at {i} and {j}"

def testEyeValues : IO Unit := do
  let eye := runTensorM do
    Tensor.eye 3 5 .float32
  assertShape eye.uop.shape [3, 5] "Tensor.eye shape"
  assertRawAllClose (evalTensor eye)
    #[1.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0] 0.0001 "Tensor.eye values"

def testRandpermDeterminism : IO Unit := do
  let (a, b, c) := runTensorM do
    let a ← Tensor.randperm 10 .float32 0
    let b ← Tensor.randperm 10 .float32 0
    let c ← Tensor.randperm 10 .float32 1
    pure (a, b, c)
  let ra := evalTensor a
  let rb := evalTensor b
  let rc := evalTensor c
  if ra.data != rb.data then
    throw <| IO.userError "Tensor.randperm should be deterministic for identical seeds"
  if ra.data == rc.data then
    throw <| IO.userError "Tensor.randperm should differ for different seeds"
  assertPermutationRange ra 10 "Tensor.randperm permutation range"

def testMeshgridIJ : IO Unit := do
  let (gx, gy) := runTensorM do
    let x ← Tensor.arange 3 .float32
    let y ← Tensor.linspace 3.0 6.0 4 .float32
    Tensor.meshgridIJ x y
  assertShape gx.uop.shape [3, 4] "meshgridIJ x-shape"
  assertShape gy.uop.shape [3, 4] "meshgridIJ y-shape"
  assertRawAllClose (evalTensor gx)
    #[0.0, 0.0, 0.0, 0.0,
      1.0, 1.0, 1.0, 1.0,
      2.0, 2.0, 2.0, 2.0] 0.0001 "meshgridIJ x-values"
  assertRawAllClose (evalTensor gy)
    #[3.0, 4.0, 5.0, 6.0,
      3.0, 4.0, 5.0, 6.0,
      3.0, 4.0, 5.0, 6.0] 0.0001 "meshgridIJ y-values"

def testMeshgridXY : IO Unit := do
  let (gx, gy) := runTensorM do
    let x ← Tensor.arange 3 .float32
    let y ← Tensor.linspace 3.0 6.0 4 .float32
    Tensor.meshgridXY x y
  assertShape gx.uop.shape [4, 3] "meshgridXY x-shape"
  assertShape gy.uop.shape [4, 3] "meshgridXY y-shape"
  assertRawAllClose (evalTensor gx)
    #[0.0, 1.0, 2.0,
      0.0, 1.0, 2.0,
      0.0, 1.0, 2.0,
      0.0, 1.0, 2.0] 0.0001 "meshgridXY x-values"
  assertRawAllClose (evalTensor gy)
    #[3.0, 3.0, 3.0,
      4.0, 4.0, 4.0,
      5.0, 5.0, 5.0,
      6.0, 6.0, 6.0] 0.0001 "meshgridXY y-values"

def testSplitChunkRollPadTo : IO Unit := do
  let (splits, chunks, rolledPos, rolledNeg, padded) := runTensorM do
    let base ← Tensor.arange 10 .float32
    let splits ← StaticTensor.split base 3
    let chunks ← StaticTensor.chunk base 3
    let v5 ← Tensor.arange 5 .float32
    let rolledPos ← StaticTensor.roll v5 2
    let rolledNeg ← StaticTensor.roll v5 (-1)
    let matBase ← Tensor.arange 4 .float32
    let mat ← reshapeUnsafe matBase [2, 2]
    let padded ← StaticTensor.padTo mat [2, 4]
    pure (splits, chunks, rolledPos, rolledNeg, padded)

  match splits with
  | [s0, s1, s2, s3] =>
    assertShape s0.1 [3] "split[0] shape"
    assertShape s1.1 [3] "split[1] shape"
    assertShape s2.1 [3] "split[2] shape"
    assertShape s3.1 [1] "split[3] shape"
    assertRawAllClose (evalTensor s0.2) #[0.0, 1.0, 2.0] 0.0001 "split[0] values"
    assertRawAllClose (evalTensor s3.2) #[9.0] 0.0001 "split[3] values"
  | _ =>
    throw <| IO.userError s!"split expected 4 chunks, got {splits.length}"

  match chunks with
  | [c0, c1, c2] =>
    assertShape c0.1 [4] "chunk[0] shape"
    assertShape c1.1 [4] "chunk[1] shape"
    assertShape c2.1 [2] "chunk[2] shape"
  | _ =>
    throw <| IO.userError s!"chunk expected 3 chunks, got {chunks.length}"

  assertRawAllClose (evalTensor rolledPos) #[3.0, 4.0, 0.0, 1.0, 2.0] 0.0001 "roll +2 values"
  assertRawAllClose (evalTensor rolledNeg) #[1.0, 2.0, 3.0, 4.0, 0.0] 0.0001 "roll -1 values"
  assertShape padded.uop.shape [2, 4] "padTo shape"
  assertRawAllClose (evalTensor padded) #[0.0, 1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0] 0.0001 "padTo values"

def testReductionParity : IO Unit := do
  let (prodAll, varAll, stdAll, vmean, smean, csum, cprod, cmax, lse, lcse) := runTensorM do
    let base ← Tensor.arange 6 .float32
    let ones6 ← Tensor.ones [6] .float32
    let shifted ← add base ones6
    let v ← reshapeUnsafe shifted [2, 3]
    let prodAll ← StaticTensor.prod shifted
    let varAll ← StaticTensor.var shifted
    let stdAll ← StaticTensor.std shifted
    let vmean ← StaticTensor.varMean shifted
    let smean ← StaticTensor.stdMean shifted
    let csum ← StaticTensor.cumsumAxis v 1
    let cprod ← StaticTensor.cumprodAxis v 1
    let cmax ← StaticTensor.cummaxAxis v 1
    let lse ← StaticTensor.logsumexp shifted
    let lcse ← StaticTensor.logcumsumexp shifted
    pure (prodAll, varAll, stdAll, vmean, smean, csum, cprod, cmax, lse, lcse)

  assertRawAllClose (evalTensor prodAll) #[720.0] 0.001 "prod all values"
  assertRawAllClose (evalTensor varAll) #[2.9166667] 0.001 "var all values"
  assertRawAllClose (evalTensor stdAll) #[1.7078252] 0.001 "std all values"
  assertRawAllClose (evalTensor vmean.1) #[2.9166667] 0.001 "varMean variance"
  assertRawAllClose (evalTensor vmean.2) #[3.5] 0.001 "varMean mean"
  assertRawAllClose (evalTensor smean.1) #[1.7078252] 0.001 "stdMean std"
  assertRawAllClose (evalTensor smean.2) #[3.5] 0.001 "stdMean mean"
  assertRawAllClose (evalTensor csum) #[1.0, 3.0, 6.0, 4.0, 9.0, 15.0] 0.001 "cumsum axis values"
  assertRawAllClose (evalTensor cprod) #[1.0, 2.0, 6.0, 4.0, 20.0, 120.0] 0.001 "cumprod axis values"
  assertRawAllClose (evalTensor cmax) #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0] 0.001 "cummax axis values"
  assertRawAllClose (evalTensor lse) #[6.4561934] 0.001 "logsumexp values"
  assertRawAllClose (evalTensor lcse)
    #[1.0, 2.3132617, 3.407606, 4.44019, 5.4519143, 6.4561934] 0.001 "logcumsumexp values"

def testNNConvPoolSmoke : IO Unit := do
  let (convOut, maxOut, avgOut) := runTensorM do
    let x ← Tensor.ones [1, 1, 4, 4] .float32
    let w ← Tensor.ones [1, 1, 3, 3] .float32
    let convOut ← conv2d x w none 0 1 1
    let maxOut ← maxPool2d x 2 2
    let avgOut ← avgPool2d x 2 2
    pure (convOut, maxOut, avgOut)
  assertShape convOut.uop.shape [1, 1, 2, 2] "conv2d output shape"
  assertShape maxOut.uop.shape [1, 1, 2, 2] "maxPool2d output shape"
  assertShape avgOut.uop.shape [1, 1, 2, 2] "avgPool2d output shape"
  assertRawAllClose (evalTensor convOut) #[9.0, 9.0, 9.0, 9.0] 0.001 "conv2d output values"
  assertRawAllClose (evalTensor maxOut) #[1.0, 1.0, 1.0, 1.0] 0.001 "maxPool2d output values"
  assertRawAllClose (evalTensor avgOut) #[1.0, 1.0, 1.0, 1.0] 0.001 "avgPool2d output values"

def cases : List TestCase :=
  [
    {
      name := "ops.creation.eye"
      group := "ops"
      minProfile := .fast
      pythonRefs := ["test/test_ops.py::test_eye"]
      suite := fun _ => ioTest "Tensor.eye parity" testEyeValues
    },
    {
      name := "tensor.rand.randperm"
      group := "tensor"
      minProfile := .fast
      pythonRefs := ["test/test_tensor.py::test_randperm"]
      suite := fun _ => ioTest "Tensor.randperm determinism and permutation" testRandpermDeterminism
    },
    {
      name := "ops.creation.meshgrid_ij"
      group := "ops"
      minProfile := .fast
      pythonRefs := ["test/test_ops.py::test_meshgrid"]
      suite := fun _ => ioTest "meshgrid IJ indexing parity" testMeshgridIJ
    },
    {
      name := "ops.creation.meshgrid_xy"
      group := "ops"
      minProfile := .fast
      pythonRefs := ["test/test_ops.py::test_meshgrid"]
      suite := fun _ => ioTest "meshgrid XY indexing parity" testMeshgridXY
    },
    {
      name := "ops.move.split_chunk_roll_pad_to"
      group := "ops"
      minProfile := .medium
      pythonRefs := ["test/test_ops.py::test_split", "test/test_ops.py::test_chunk", "test/test_ops.py::test_roll"]
      suite := fun _ => ioTest "split/chunk/roll/padTo parity" testSplitChunkRollPadTo
    },
    {
      name := "ops.reduce.extended"
      group := "ops"
      minProfile := .medium
      pythonRefs := [
        "test/test_ops.py::test_prod",
        "test/test_ops.py::test_var",
        "test/test_ops.py::test_std",
        "test/test_ops.py::test_cumsum",
        "test/test_ops.py::test_cumprod",
        "test/test_ops.py::test_cummax",
        "test/test_ops.py::test_logsumexp",
        "test/test_ops.py::test_logcumsumexp"
      ]
      suite := fun _ => ioTest "extended reduction parity" testReductionParity
    },
    {
      name := "curated.nn.conv_pool_smoke"
      group := "curated"
      minProfile := .medium
      pythonRefs := ["test/unit/test_conv.py::test_conv2d", "test/test_ops.py::test_sum_fake"]
      suite := fun _ => ioTest "nn conv/pool deterministic smoke" testNNConvPoolSmoke
    }
  ]

end TinyGrad4.Test.Port.ExtendedParity
