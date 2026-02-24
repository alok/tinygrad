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

private def assertRawAllConst (raw : RawBuffer) (v tol : Float64) (label : String) : IO Unit := do
  for i in [:raw.numF32] do
    if Float64.abs (raw.getF32 i - v) > tol then
      throw <| IO.userError s!"{label}: index {i} expected {v}, got {raw.getF32 i}"

private def assertDropoutScaledBinary (raw : RawBuffer) (scale tol : Float64) (label : String) : IO Unit := do
  let mut sawZero := false
  let mut sawScaled := false
  for i in [:raw.numF32] do
    let x := raw.getF32 i
    if Float64.abs x <= tol then
      sawZero := true
    else if Float64.abs (x - scale) <= tol then
      sawScaled := true
    else
      throw <| IO.userError s!"{label}: index {i} expected 0 or {scale}, got {x}"
  if !sawZero then
    throw <| IO.userError s!"{label}: expected at least one dropped element"
  if !sawScaled then
    throw <| IO.userError s!"{label}: expected at least one kept element"

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

def testLinspaceParity : IO Unit := do
  let (f3, f1, f0, i3CastBack) := runTensorM do
    let f3 ← Tensor.linspace 5.0 10.0 3 .float32
    let f1 ← Tensor.linspace 5.0 10.0 1 .float32
    let f0 ← Tensor.linspace 5.0 10.0 0 .float32
    let i3 ← Tensor.linspace 5.0 10.0 3 .int32
    let i3CastBack ← StaticTensor.cast i3 .float32
    pure (f3, f1, f0, i3CastBack)
  assertShape f3.uop.shape [3] "linspace float32 shape (3 steps)"
  assertShape f1.uop.shape [1] "linspace float32 shape (1 step)"
  assertShape f0.uop.shape [0] "linspace float32 shape (0 steps)"
  assertShape i3CastBack.uop.shape [3] "linspace int32 shape (3 steps)"
  assertRawAllClose (evalTensor f3) #[5.0, 7.5, 10.0] 0.0001 "linspace float32 values (3 steps)"
  assertRawAllClose (evalTensor f1) #[5.0] 0.0001 "linspace float32 values (1 step)"
  assertRawAllClose (evalTensor f0) #[] 0.0001 "linspace float32 values (0 steps)"
  assertRawAllClose (evalTensor i3CastBack) #[5.0, 7.0, 10.0] 0.0001 "linspace int32 cast values"

def testCatStackParity : IO Unit := do
  let (catOut, stack0, stack1) := runTensorM do
    let leftBase ← Tensor.arange 4 .float32
    let left ← reshapeUnsafe leftBase [2, 2]
    let right ← Tensor.full [2, 2] .float32 9.0
    let catOut ← StaticTensor.cat left right 1 (by native_decide)
    let v0 ← Tensor.arange 2 .float32
    let v1 ← Tensor.full [2] .float32 4.0
    let v2 ← Tensor.full [2] .float32 (-1.0)
    let ts : TensorList .float32 .CPU [[2], [2], [2]] := .cons v0 (.cons v1 (.cons v2 .nil))
    let stack0 ← StaticTensor.stack ts 0
    let stack1 ← StaticTensor.stack ts 1
    pure (catOut, stack0, stack1)
  assertShape catOut.uop.shape [2, 4] "cat output shape"
  assertShape stack0.uop.shape [3, 2] "stack axis0 shape"
  assertShape stack1.uop.shape [2, 3] "stack axis1 shape"
  assertRawAllClose (evalTensor catOut) #[0.0, 1.0, 9.0, 9.0, 2.0, 3.0, 9.0, 9.0] 0.0001 "cat values"
  assertRawAllClose (evalTensor stack0) #[0.0, 1.0, 4.0, 4.0, -1.0, -1.0] 0.0001 "stack axis0 values"
  assertRawAllClose (evalTensor stack1) #[0.0, 4.0, -1.0, 1.0, 4.0, -1.0] 0.0001 "stack axis1 values"

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

def testMinMaxReductionParity : IO Unit := do
  let (mx, mn) := runTensorM do
    let base ← Tensor.arange 6 .float32
    let shifted ← addScalar base (-2.5)
    let mx ← StaticTensor.max shifted
    let mn ← StaticTensor.min shifted
    pure (mx, mn)
  assertShape mx.uop.shape [] "max reduction scalar shape"
  assertShape mn.uop.shape [] "min reduction scalar shape"
  assertRawAllClose (evalTensor mx) #[2.5] 0.0001 "max reduction value"
  assertRawAllClose (evalTensor mn) #[-2.5] 0.0001 "min reduction value"

def testSoftmaxLogSoftmaxParity : IO Unit := do
  let (softmaxLast, logSoftmaxLast, softmaxAxis0) := runTensorM do
    let base ← Tensor.arange 6 .float32
    let mat ← reshapeUnsafe base [2, 3]
    let softmaxLast ← StaticTensor.softmax mat
    let logSoftmaxLast ← StaticTensor.logSoftmax mat
    let softmaxAxis0 ← StaticTensor.softmaxAxis mat 0
    pure (softmaxLast, logSoftmaxLast, softmaxAxis0)
  assertShape softmaxLast.uop.shape [2, 3] "softmax last-axis shape"
  assertShape logSoftmaxLast.uop.shape [2, 3] "logSoftmax last-axis shape"
  assertShape softmaxAxis0.uop.shape [2, 3] "softmax axis0 shape"
  assertRawAllClose (evalTensor softmaxLast)
    #[0.0900306, 0.2447285, 0.6652409,
      0.0900306, 0.2447285, 0.6652409] 0.001 "softmax last-axis values"
  assertRawAllClose (evalTensor logSoftmaxLast)
    #[-2.407606, -1.407606, -0.407606,
      -2.407606, -1.407606, -0.407606] 0.001 "logSoftmax last-axis values"
  assertRawAllClose (evalTensor softmaxAxis0)
    #[0.0474259, 0.0474259, 0.0474259,
      0.9525741, 0.9525741, 0.9525741] 0.001 "softmax axis0 values"

def testActivationParity : IO Unit := do
  let (reluOut, tanhOut, siluOut, geluOut) := runTensorM do
    let base ← Tensor.arange 6 .float32
    let x ← addScalar base (-2.5)
    let reluOut ← StaticTensor.relu x
    let tanhOut ← StaticTensor.tanh x
    let siluOut ← StaticTensor.silu x
    let geluOut ← StaticTensor.gelu x
    pure (reluOut, tanhOut, siluOut, geluOut)
  let expectedRelu := #[0.0, 0.0, 0.0, 0.5, 1.5, 2.5]
  let expectedTanh := #[-0.9866143, -0.9051483, -0.4621172, 0.4621172, 0.9051483, 0.9866143]
  let expectedSilu := #[-0.1896454, -0.2736383, -0.1887703, 0.3112297, 1.2263616, 2.3103545]
  let expectedGelu := #[-0.0150843, -0.1004284, -0.1542860, 0.3457140, 1.3995717, 2.4849157]
  assertRawAllClose (evalTensor reluOut) expectedRelu 0.001 "relu values"
  assertRawAllClose (evalTensor tanhOut) expectedTanh 0.001 "tanh values"
  assertRawAllClose (evalTensor siluOut) expectedSilu 0.001 "silu values"
  assertRawAllClose (evalTensor geluOut) expectedGelu 0.001 "gelu values"

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

def testBatchnormParity : IO Unit := do
  let (bnNc, bnNchw) := runTensorM do
    let x0 ← Tensor.arange 6 .float32
    let xNc ← reshapeUnsafe x0 [2, 3]
    let meanNc ← Tensor.linspace 1.5 3.5 3 .float32
    let invstdNc ← Tensor.full [3] .float32 0.6666667
    let bnNc ← StaticTensor.batchnorm xNc none none meanNc invstdNc

    let x4Base ← Tensor.arange 8 .float32
    let x4 ← reshapeUnsafe x4Base [1, 2, 2, 2]
    let m0 ← Tensor.full [1] .float32 1.5
    let m1 ← Tensor.full [1] .float32 5.5
    let meanNchw ← StaticTensor.cat m0 m1 0 (by native_decide)
    let i0 ← Tensor.full [1] .float32 1.0
    let i1 ← Tensor.full [1] .float32 0.5
    let invstdNchw ← StaticTensor.cat i0 i1 0 (by native_decide)
    let w0 ← Tensor.full [1] .float32 2.0
    let w1 ← Tensor.full [1] .float32 (-1.0)
    let weightNchw ← StaticTensor.cat w0 w1 0 (by native_decide)
    let b0 ← Tensor.full [1] .float32 0.5
    let b1 ← Tensor.full [1] .float32 1.0
    let biasNchw ← StaticTensor.cat b0 b1 0 (by native_decide)
    let bnNchw ← StaticTensor.batchnormNCHW x4 (some weightNchw) (some biasNchw) meanNchw invstdNchw
    pure (bnNc, bnNchw)
  assertRawAllClose (evalTensor bnNc) #[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0] 0.001 "batchnorm NC parity"
  assertRawAllClose (evalTensor bnNchw)
    #[-2.5, -0.5, 1.5, 3.5, 1.75, 1.25, 0.75, 0.25] 0.001 "batchnorm NCHW affine parity"

def testDropoutParity : IO Unit := do
  let (trainA, trainB, trainC, evalOut, p0Out, p1Out) := runTensorM do
    let x ← Tensor.ones [32] .float32
    let trainA ← StaticTensor.dropout x 0.5 true 7
    let trainB ← StaticTensor.dropout x 0.5 true 7
    let trainC ← StaticTensor.dropout x 0.5 true 8
    let evalOut ← StaticTensor.dropout x 0.5 false 7
    let p0Out ← StaticTensor.dropout x 0.0 true 7
    let p1Out ← StaticTensor.dropout x 1.0 true 7
    pure (trainA, trainB, trainC, evalOut, p0Out, p1Out)
  let rawA := evalTensor trainA
  let rawB := evalTensor trainB
  let rawC := evalTensor trainC
  if rawA.data != rawB.data then
    throw <| IO.userError "dropout should be deterministic for identical seeds"
  if rawA.data == rawC.data then
    throw <| IO.userError "dropout should differ for distinct seeds"
  assertDropoutScaledBinary rawA 2.0 0.001 "dropout training mask+scale parity"
  assertRawAllConst (evalTensor evalOut) 1.0 0.001 "dropout eval identity parity"
  assertRawAllConst (evalTensor p0Out) 1.0 0.001 "dropout p=0 identity parity"
  assertRawAllConst (evalTensor p1Out) 0.0 0.001 "dropout p=1 zero parity"

def testRoundSignLerpParity : IO Unit := do
  let (rounded, signed, lerpS, lerpT) := runTensorM do
    let base ← Tensor.arange 6 .float32
    let shifted ← addScalar base (-2.5)
    let rounded ← StaticTensor.round shifted
    let signed ← StaticTensor.sign shifted
    let start ← Tensor.linspace 1.0 3.0 3 .float32
    let end_ ← Tensor.linspace 4.0 6.0 3 .float32
    let lerpS ← StaticTensor.lerpScalar start end_ 0.5
    let w ← Tensor.linspace 0.0 1.0 3 .float32
    let lerpT ← StaticTensor.lerp start end_ w
    pure (rounded, signed, lerpS, lerpT)
  assertRawAllClose (evalTensor rounded) #[-2.0, -2.0, 0.0, 0.0, 2.0, 2.0] 0.001 "round ties-to-even parity"
  assertRawAllClose (evalTensor signed) #[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0] 0.001 "sign parity"
  assertRawAllClose (evalTensor lerpS) #[2.5, 3.5, 4.5] 0.001 "lerp scalar parity"
  assertRawAllClose (evalTensor lerpT) #[1.0, 3.5, 6.0] 0.001 "lerp tensor parity"

def testArcTrigParity : IO Unit := do
  let (asinOut, acosOut, atanOut) := runTensorM do
    let base ← Tensor.arange 7 .float32
    let scaled ← scale base 0.3
    let x ← addScalar scaled (-0.9)
    let asinOut ← StaticTensor.asin x
    let acosOut ← StaticTensor.acos x
    let base6 ← Tensor.arange 6 .float32
    let shifted6 ← addScalar base6 (-2.5)
    let atanOut ← StaticTensor.atan shifted6
    pure (asinOut, acosOut, atanOut)
  assertRawAllClose (evalTensor asinOut)
    #[-1.1197694, -0.6435010, -0.3046926, 0.0, 0.3046927, 0.6435011, 1.1197694] 0.01 "asin parity"
  assertRawAllClose (evalTensor acosOut)
    #[2.6905658, 2.2142973, 1.8754889, 1.5707964, 1.2661036, 0.9272952, 0.4510269] 0.01 "acos parity"
  assertRawAllClose (evalTensor atanOut)
    #[-1.1902900, -0.9827937, -0.4636477, 0.4636477, 0.9827937, 1.1902900] 0.01 "atan parity"

def testHyperbolicErfParity : IO Unit := do
  let (sinhOut, coshOut, erfOut) := runTensorM do
    let base ← Tensor.arange 7 .float32
    let scaled ← scale base 0.5
    let x ← addScalar scaled (-1.5)
    let sinhOut ← StaticTensor.sinh x
    let coshOut ← StaticTensor.cosh x
    let erfOut ← StaticTensor.erf x
    pure (sinhOut, coshOut, erfOut)
  assertRawAllClose (evalTensor sinhOut)
    #[-2.1292794, -1.1752013, -0.5210953, 0.0, 0.5210953, 1.1752013, 2.1292794] 0.01 "sinh parity"
  assertRawAllClose (evalTensor coshOut)
    #[2.3524096, 1.5430807, 1.1276259, 1.0, 1.1276259, 1.5430807, 2.3524096] 0.01 "cosh parity"
  assertRawAllClose (evalTensor erfOut)
    #[-0.9661053, -0.8427007, -0.5205001, 0.0, 0.5205001, 0.8427007, 0.9661053] 0.01 "erf parity"

def testActivationExtensionsParity : IO Unit := do
  let (softsignOut, mishOut, celuOut, seluOut) := runTensorM do
    let base ← Tensor.arange 6 .float32
    let x ← addScalar base (-2.5)
    let softsignOut ← StaticTensor.softsign x
    let mishOut ← StaticTensor.mish x
    let celuOut ← StaticTensor.celu x
    let seluOut ← StaticTensor.selu x
    pure (softsignOut, mishOut, celuOut, seluOut)
  assertRawAllClose (evalTensor softsignOut)
    #[-0.7142857, -0.6000000, -0.3333333, 0.3333333, 0.6000000, 0.7142857] 0.01 "softsign parity"
  assertRawAllClose (evalTensor mishOut)
    #[-0.1968163, -0.2980998, -0.2207437, 0.3752452, 1.4033782, 2.4713922] 0.01 "mish parity"
  assertRawAllClose (evalTensor celuOut)
    #[-0.9179150, -0.7768698, -0.3934693, 0.5, 1.5, 2.5] 0.01 "celu parity"
  assertRawAllClose (evalTensor seluOut)
    #[-1.6137811, -1.3658104, -0.6917562, 0.5253500, 1.5760500, 2.6267500] 0.01 "selu parity"

def testCopysignLogaddexpParity : IO Unit := do
  let (copysignOut, recipSignMaskF, logaddexpOut) := runTensorM do
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

    let copysignOut ← StaticTensor.copysign mag signSrc
    let recipOut ← StaticTensor.recip copysignOut
    let zero4 ← Tensor.zeros [4] .float32
    let recipSignMask ← StaticTensor.cmplt recipOut zero4
    let recipSignMaskF ← StaticTensor.cast recipSignMask .float32

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
    let logaddexpOut ← StaticTensor.logaddexp la lb
    pure (copysignOut, recipSignMaskF, logaddexpOut)
  assertRawAllClose (evalTensor copysignOut) #[-1.0, 2.0, 0.0, 0.0] 0.001 "copysign magnitude parity"
  assertRawAllClose (evalTensor recipSignMaskF) #[1.0, 0.0, 0.0, 0.0] 0.001 "copysign signed-zero parity"
  assertRawAllClose (evalTensor logaddexpOut) #[100.31326, 100.0, 1.0485873] 0.01 "logaddexp parity"

def testDiagDiagonalParity : IO Unit := do
  let (diagOut, diagonalOut) := runTensorM do
    let v ← Tensor.linspace 1.0 3.0 3 .float32
    let diagOut ← StaticTensor.diag v
    let mBase ← Tensor.arange 9 .float32
    let m ← reshapeUnsafe mBase [3, 3]
    let diagonalOut ← StaticTensor.diagonal m
    pure (diagOut, diagonalOut)
  assertShape diagOut.uop.shape [3, 3] "diag shape"
  assertShape diagonalOut.uop.shape [3] "diagonal shape"
  assertRawAllClose (evalTensor diagOut) #[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0] 0.001 "diag values"
  assertRawAllClose (evalTensor diagonalOut) #[0.0, 4.0, 8.0] 0.001 "diagonal values"

def testMaskedFillParity : IO Unit := do
  let (filledScalar, filledTensor) := runTensorM do
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
    let filledScalar ← StaticTensor.maskedFillScalar base mask (-12.0)
    let negBase ← StaticTensor.neg base
    let filledTensor ← StaticTensor.maskedFill base mask negBase
    pure (filledScalar, filledTensor)
  assertRawAllClose (evalTensor filledScalar) #[-12.0, 2.0, -12.0, 4.0, 5.0, -12.0] 0.001 "maskedFill scalar parity"
  assertRawAllClose (evalTensor filledTensor) #[-1.0, 2.0, -3.0, 4.0, 5.0, -6.0] 0.001 "maskedFill tensor parity"

def testItemParity : IO Unit := do
  let scalarOut := runTensorM do
    let x ← Tensor.full [] .float32 42.0
    StaticTensor.item x
  assertRawAllClose (evalTensor scalarOut) #[42.0] 0.001 "item scalar value"

  let nonScalarStatus := runTensorM do
    let x ← Tensor.arange 3 .float32
    StaticTensor.itemChecked x
  match nonScalarStatus with
  | .ok _ => throw <| IO.userError "itemChecked should report an error on non-scalar input"
  | .error _ => pure ()

def testTakeUnfoldParity : IO Unit := do
  let (taken, unfold2, unfold3) := runTensorM do
    let base0 ← Tensor.arange 6 .float32
    let base ← reshapeUnsafe base0 [2, 3]
    let idxA ← Tensor.arange 3 .int32
    let idxB ← Tensor.arange 2 .int32
    let idx ← StaticTensor.cat idxA idxB 0 (by native_decide)
    let taken ← StaticTensor.take base idx
    let v8 ← Tensor.arange 8 .float32
    let unfold2 ← StaticTensor.unfold v8 0 2 2
    let unfold3 ← StaticTensor.unfold v8 0 3 2
    pure (taken, unfold2, unfold3)
  assertShape taken.uop.shape [5] "take output shape"
  assertRawAllClose (evalTensor taken) #[0.0, 1.0, 2.0, 0.0, 1.0] 0.001 "take flattened gather parity"
  assertShape unfold2.uop.shape [4, 2] "unfold size=2 step=2 shape"
  assertRawAllClose (evalTensor unfold2) #[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] 0.001 "unfold size=2 step=2 values"
  assertShape unfold3.uop.shape [3, 3] "unfold size=3 step=2 shape"
  assertRawAllClose (evalTensor unfold3) #[0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0] 0.001 "unfold size=3 step=2 values"

def testTriangularParity : IO Unit := do
  let (triu1, triuNeg1, tril1, trilNeg1) := runTensorM do
    let base0 ← Tensor.arange 12 .float32
    let base1 ← addScalar base0 1.0
    let mat ← reshapeUnsafe base1 [3, 4]
    let triu1 ← StaticTensor.triu mat 1
    let triuNeg1 ← StaticTensor.triu mat (-1)
    let tril1 ← StaticTensor.tril mat 1
    let trilNeg1 ← StaticTensor.tril mat (-1)
    pure (triu1, triuNeg1, tril1, trilNeg1)
  assertRawAllClose (evalTensor triu1)
    #[0.0, 2.0, 3.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 12.0] 0.001 "triu diagonal=1 values"
  assertRawAllClose (evalTensor triuNeg1)
    #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 10.0, 11.0, 12.0] 0.001 "triu diagonal=-1 values"
  assertRawAllClose (evalTensor tril1)
    #[1.0, 2.0, 0.0, 0.0, 5.0, 6.0, 7.0, 0.0, 9.0, 10.0, 11.0, 12.0] 0.001 "tril diagonal=1 values"
  assertRawAllClose (evalTensor trilNeg1)
    #[0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 9.0, 10.0, 0.0, 0.0] 0.001 "tril diagonal=-1 values"

def testMaskedSelectPackedBoundary : IO Unit := do
  let (packed0, count0F, packed1, count1F) := runTensorM do
    let x0 ← Tensor.arange 9 .float32
    let x ← reshapeUnsafe x0 [3, 3]
    let mask0 ← Tensor.fullBool [3, 3] false
    let mask1 ← Tensor.fullBool [3, 3] true
    let (packed0, count0) ← StaticTensor.maskedSelectPacked x mask0
    let (packed1, count1) ← StaticTensor.maskedSelectPacked x mask1
    let count0F ← StaticTensor.cast count0 .float32
    let count1F ← StaticTensor.cast count1 .float32
    pure (packed0, count0F, packed1, count1F)
  assertRawAllClose (evalTensor packed0) #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.001 "maskedSelectPacked all-false payload"
  assertRawAllClose (evalTensor count0F) #[0.0] 0.001 "maskedSelectPacked all-false count"
  assertRawAllClose (evalTensor packed1) #[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] 0.001 "maskedSelectPacked all-true payload"
  assertRawAllClose (evalTensor count1F) #[9.0] 0.001 "maskedSelectPacked all-true count"

def testMaskedSelectPackedPrefix : IO Unit := do
  let (packed, countF) := runTensorM do
    let x0 ← Tensor.arange 9 .float32
    let x ← reshapeUnsafe x0 [3, 3]
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
    let (packed, count) ← StaticTensor.maskedSelectPacked x mask
    let countF ← StaticTensor.cast count .float32
    pure (packed, countF)
  assertRawAllClose (evalTensor countF) #[4.0] 0.001 "maskedSelectPacked mixed count"
  assertRawAllClose (evalTensor packed) #[0.0, 2.0, 4.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0] 0.001 "maskedSelectPacked mixed prefix"

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
      name := "ops.creation.linspace"
      group := "ops"
      minProfile := .fast
      pythonRefs := ["test/test_ops.py::test_linspace"]
      suite := fun _ => ioTest "linspace parity for edge step counts and dtype" testLinspaceParity
    },
    {
      name := "ops.move.cat_stack"
      group := "ops"
      minProfile := .fast
      pythonRefs := ["test/test_ops.py::test_cat", "test/test_ops.py::test_stack", "test/test_tensor.py::test_cat"]
      suite := fun _ => ioTest "cat/stack movement parity" testCatStackParity
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
      name := "ops.reduce.min_max_full"
      group := "ops"
      minProfile := .fast
      pythonRefs := ["test/test_ops.py::test_min", "test/test_ops.py::test_max"]
      suite := fun _ => ioTest "full-tensor min/max reduction parity" testMinMaxReductionParity
    },
    {
      name := "ops.softmax.logsoftmax"
      group := "ops"
      minProfile := .fast
      pythonRefs := [
        "test/test_ops.py::test_softmax",
        "test/test_ops.py::test_log_softmax",
        "test/test_ops.py::test_softmax_other_axis"
      ]
      suite := fun _ => ioTest "softmax/logSoftmax parity including axis=0" testSoftmaxLogSoftmaxParity
    },
    {
      name := "ops.elemwise.activations"
      group := "ops"
      minProfile := .fast
      pythonRefs := [
        "test/test_ops.py::test_relu",
        "test/test_ops.py::test_tanh",
        "test/test_ops.py::test_silu",
        "test/test_ops.py::test_gelu"
      ]
      suite := fun _ => ioTest "relu/tanh/silu/gelu parity" testActivationParity
    },
    {
      name := "ops.elemwise.round_sign_lerp"
      group := "ops"
      minProfile := .fast
      pythonRefs := ["test/test_ops.py::test_round", "test/test_ops.py::test_sign", "test/test_ops.py::test_lerp"]
      suite := fun _ => ioTest "round/sign/lerp parity" testRoundSignLerpParity
    },
    {
      name := "ops.elemwise.arc_trig"
      group := "ops"
      minProfile := .fast
      pythonRefs := ["test/test_ops.py::test_asin", "test/test_ops.py::test_acos", "test/test_ops.py::test_atan"]
      suite := fun _ => ioTest "asin/acos/atan parity" testArcTrigParity
    },
    {
      name := "ops.elemwise.hyperbolic_erf"
      group := "ops"
      minProfile := .fast
      pythonRefs := ["test/test_ops.py::test_sinh", "test/test_ops.py::test_cosh", "test/test_ops.py::test_erf"]
      suite := fun _ => ioTest "sinh/cosh/erf parity" testHyperbolicErfParity
    },
    {
      name := "ops.elemwise.activation_extensions"
      group := "ops"
      minProfile := .fast
      pythonRefs := ["test/test_ops.py::test_softsign", "test/test_ops.py::test_mish", "test/test_ops.py::test_celu", "test/test_ops.py::test_selu"]
      suite := fun _ => ioTest "softsign/mish/celu/selu parity" testActivationExtensionsParity
    },
    {
      name := "ops.elemwise.copysign_logaddexp"
      group := "ops"
      minProfile := .fast
      pythonRefs := ["test/test_ops.py::test_copysign", "test/test_ops.py::test_logaddexp"]
      suite := fun _ => ioTest "copysign/logaddexp parity" testCopysignLogaddexpParity
    },
    {
      name := "ops.indexing.diag_diagonal"
      group := "ops"
      minProfile := .fast
      pythonRefs := ["test/test_ops.py::test_diag", "test/test_ops.py::test_diagonal"]
      suite := fun _ => ioTest "diag/diagonal parity" testDiagDiagonalParity
    },
    {
      name := "ops.indexing.masked_fill"
      group := "ops"
      minProfile := .fast
      pythonRefs := ["test/test_ops.py::test_masked_fill"]
      suite := fun _ => ioTest "masked_fill parity" testMaskedFillParity
    },
    {
      name := "ops.indexing.item"
      group := "ops"
      minProfile := .fast
      pythonRefs := ["test/test_tensor.py::test_item"]
      suite := fun _ => ioTest "item scalar/failure parity" testItemParity
    },
    {
      name := "ops.indexing.take_unfold"
      group := "ops"
      minProfile := .medium
      pythonRefs := ["test/test_ops.py::test_take", "test/test_ops.py::test_unfold"]
      suite := fun _ => ioTest "take/unfold parity" testTakeUnfoldParity
    },
    {
      name := "ops.indexing.triangular"
      group := "ops"
      minProfile := .medium
      pythonRefs := ["test/test_ops.py::test_triu", "test/test_ops.py::test_tril"]
      suite := fun _ => ioTest "triu/tril parity with offsets" testTriangularParity
    },
    {
      name := "ops.indexing.masked_select_packed_boundaries"
      group := "ops"
      minProfile := .slow
      pythonRefs := ["test/test_ops.py::test_masked_select"]
      suite := fun _ => ioTest "maskedSelectPacked all-false/all-true boundaries" testMaskedSelectPackedBoundary
    },
    {
      name := "ops.indexing.masked_select_packed_prefix"
      group := "ops"
      minProfile := .slow
      pythonRefs := ["test/test_ops.py::test_masked_select"]
      suite := fun _ => ioTest "maskedSelectPacked prefix-order parity" testMaskedSelectPackedPrefix
    },
    {
      name := "curated.nn.conv_pool_smoke"
      group := "curated"
      minProfile := .medium
      pythonRefs := ["test/unit/test_conv.py::test_conv2d", "test/test_ops.py::test_sum_fake"]
      suite := fun _ => ioTest "nn conv/pool deterministic smoke" testNNConvPoolSmoke
    },
    {
      name := "ops.nn.batchnorm_channel_axis"
      group := "ops"
      minProfile := .medium
      pythonRefs := ["test/test_nn.py::test_batchnorm_axis"]
      suite := fun _ => ioTest "batchnorm channel-axis parity (NC + NCHW affine)" testBatchnormParity
    },
    {
      name := "ops.nn.dropout_seeded_semantics"
      group := "ops"
      minProfile := .slow
      pythonRefs := ["test/test_tensor.py::test_dropout", "test/test_edgecases.py::test_dropout_rate_one"]
      suite := fun _ => ioTest "dropout seeded semantics parity (train/eval/p0/p1)" testDropoutParity
    }
  ]

end TinyGrad4.Test.Port.ExtendedParity
