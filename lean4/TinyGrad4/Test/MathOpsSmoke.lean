import TinyGrad4

/-!
# MathOpsSmoke

Smoke tests for new activations, norms, and losses.
-/

namespace TinyGrad4.Test.MathOpsSmoke

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

open TinyGrad4
open Interpreter
open StaticTensor

private def assertShape (got expected : Shape) (label : String) : IO Unit := do
  if got != expected then
    throw (IO.userError s!"{label}: shape {got} != {expected}")

private def assertDType (got expected : DType) (label : String) : IO Unit := do
  if got != expected then
    throw (IO.userError s!"{label}: dtype {repr got} != {repr expected}")

private def assertEqBytes (got expected : ByteArray) (label : String) : IO Unit := do
  if got != expected then
    throw (IO.userError s!"{label}: {got} != {expected}")

private def expectI16Bytes (shape : Shape) (vals : Array Int16) (label : String) : IO ByteArray := do
  match DataArrayN.ofArrayI16? shape vals with
  | some arr => pure arr.data
  | none => throw (IO.userError s!"{label}: ofArrayI16? returned none")

private def assertAllBetween (raw : RawBuffer) (lo hi : Float) (label : String) : IO Unit := do
  let vals := raw.toFloatArray
  for i in [:vals.size] do
    let v := vals[i]!
    if v < lo || v >= hi then
      throw (IO.userError s!"{label}: idx {i} {v} not in [{lo}, {hi})")

private def assertEqF32 (raw : RawBuffer) (expected : Array Float) (label : String) : IO Unit := do
  let got := raw.toFloatArray
  if got.size != expected.size then
    throw (IO.userError s!"{label}: size {got.size} != {expected.size}")
  for i in [:expected.size] do
    if got[i]! != expected[i]! then
      throw (IO.userError s!"{label}: idx {i} {got[i]!} != {expected[i]!}")

private def assertEqI32 (got : Array Int32) (expected : Array Int32) (label : String) : IO Unit := do
  if got.size != expected.size then
    throw (IO.userError s!"{label}: size {got.size} != {expected.size}")
  for i in [:expected.size] do
    if got[i]! != expected[i]! then
      throw (IO.userError s!"{label}: idx {i} {got[i]!} != {expected[i]!}")

def testCastToAndDType : IO Unit := do
  let (castF, castI, toI) := runTensorM do
    let a ← Tensor.fullInt [2] .int16 (-1)
    let b ← Tensor.full [2] .float32 3.0
    let castF ← cast a .float32
    let castI ← cast b .int16
    let toI ← to b .int16
    pure (castF, castI, toI)
  let fVals := evalTensor castF
  assertEqF32 fVals #[-1.0, -1.0] "cast int16 -> f32"
  let iRaw := evalTensor castI
  assertDType iRaw.dtype .int16 "cast f32 -> int16 dtype"
  let expectedI ← expectI16Bytes [2] #[Int16.ofInt 3, Int16.ofInt 3] "cast f32 -> int16 data"
  assertEqBytes iRaw.data expectedI "cast f32 -> int16 data"
  let toRaw := evalTensor toI
  assertDType toRaw.dtype .int16 "to dtype"

def testDTypeUtils : IO Unit := do
  let t := runTensorM do
    Tensor.full [2, 3] .int16 1.0
  if t.elementSize != 2 then
    throw (IO.userError s!"elementSize: {t.elementSize} != 2")
  if t.nbytes != 12 then
    throw (IO.userError s!"nbytes: {t.nbytes} != 12")

def testCompareEval : IO Unit := do
  let (eqT, neT, gtT) := runTensorM do
    let a ← Tensor.full [3] .float32 1.0
    let b ← Tensor.full [3] .float32 1.0
    let c ← Tensor.full [3] .float32 2.0
    let eqT ← cmpeq a b
    let neT ← cmpne a c
    let gtT ← cmpgt c a
    pure (eqT, neT, gtT)
  let eqRaw := evalTensor eqT
  let neRaw := evalTensor neT
  let gtRaw := evalTensor gtT
  assertDType eqRaw.dtype .bool "cmpeq dtype"
  assertDType neRaw.dtype .bool "cmpne dtype"
  assertDType gtRaw.dtype .bool "cmpgt dtype"
  let expected := ByteArray.mk #[1, 1, 1]
  assertEqBytes eqRaw.data expected "cmpeq data"
  assertEqBytes neRaw.data expected "cmpne data"
  assertEqBytes gtRaw.data expected "cmpgt data"

def testBoolOpsEval : IO Unit := do
  let (andT, orT, xorT) := runTensorM do
    let ones ← Tensor.full [3] .bool 1.0
    let zero ← Tensor.full [] .bool 0.0
    let andT ← bitandB ones zero
    let orT ← bitorB ones zero
    let xorT ← bitxorB ones zero
    pure (andT, orT, xorT)
  let andRaw := evalTensor andT
  let orRaw := evalTensor orT
  let xorRaw := evalTensor xorT
  assertDType andRaw.dtype .bool "bitand dtype"
  assertDType orRaw.dtype .bool "bitor dtype"
  assertDType xorRaw.dtype .bool "bitxor dtype"
  let zeros := ByteArray.mk #[0, 0, 0]
  let ones := ByteArray.mk #[1, 1, 1]
  assertEqBytes andRaw.data zeros "bitand data"
  assertEqBytes orRaw.data ones "bitor data"
  assertEqBytes xorRaw.data ones "bitxor data"

def testCatEval : IO Unit := do
  let catF := runTensorM do
    let a ← Tensor.full [2, 1] .float32 1.0
    let b ← Tensor.full [2, 2] .float32 2.0
    cat a b 1
  let catFList := runTensorM do
    let a ← Tensor.full [2, 1] .float32 1.0
    let b ← Tensor.full [2, 2] .float32 2.0
    let c ← Tensor.full [2, 1] .float32 3.0
    let ts : TensorList .float32 [[2, 1], [2, 2], [2, 1]] :=
      .cons a (.cons b (.cons c .nil))
    catList ts 1
  let catB := runTensorM do
    let c ← Tensor.full [2] .bool 1.0
    let d ← Tensor.full [2] .bool 0.0
    cat c d 0
  let catBList := runTensorM do
    let c ← Tensor.full [2] .bool 1.0
    let d ← Tensor.full [2] .bool 0.0
    let ts : TensorList .bool [[2], [2]] :=
      .cons c (.cons d .nil)
    catList ts 0
  let fVals := evalTensor catF
  assertEqF32 fVals #[1.0, 2.0, 2.0, 1.0, 2.0, 2.0] "cat float32 data"
  let fListVals := evalTensor catFList
  assertEqF32 fListVals #[1.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.0, 3.0] "catList float32 data"
  let bRaw := evalTensor catB
  assertDType bRaw.dtype .bool "cat bool dtype"
  assertEqBytes bRaw.data (ByteArray.mk #[1, 1, 0, 0]) "cat bool data"
  let bListRaw := evalTensor catBList
  assertDType bListRaw.dtype .bool "catList bool dtype"
  assertEqBytes bListRaw.data (ByteArray.mk #[1, 1, 0, 0]) "catList bool data"

def testCatEvalInt16 : IO Unit := do
  let catI := runTensorM do
    let a ← Tensor.fullInt [2] .int16 (-1)
    let b ← Tensor.fullInt [1] .int16 2
    cat a b 0
  let raw := evalTensor catI
  assertDType raw.dtype .int16 "cat int16 dtype"
  let expectedVals : Array Int16 := #[Int16.ofInt (-1), Int16.ofInt (-1), Int16.ofInt 2]
  let expected ← expectI16Bytes [3] expectedVals "cat int16 data"
  assertEqBytes raw.data expected "cat int16 data"

def testStackEval : IO Unit := do
  let stacked0 := runTensorM do
    let a ← Tensor.full [2] .float32 1.0
    let b ← Tensor.full [2] .float32 2.0
    let ts : TensorList .float32 [[2], [2]] := .cons a (.cons b .nil)
    stack ts 0
  let vals0 := evalTensor stacked0
  assertEqF32 vals0 #[1.0, 1.0, 2.0, 2.0] "stack axis0"
  let stacked1 := runTensorM do
    let a ← Tensor.full [2] .float32 1.0
    let b ← Tensor.full [2] .float32 2.0
    let ts : TensorList .float32 [[2], [2]] := .cons a (.cons b .nil)
    stack ts 1
  let vals1 := evalTensor stacked1
  assertEqF32 vals1 #[1.0, 2.0, 1.0, 2.0] "stack axis1"

def testWhereInt16 : IO Unit := do
  let t := runTensorM do
    let c1 ← Tensor.fullBool [1] true
    let c0 ← Tensor.fullBool [1] false
    let cond ← cat c1 c0 0
    let x ← Tensor.fullInt [2] .int16 5
    let y ← Tensor.fullInt [2] .int16 9
    where_ cond x y
  let raw := evalTensor t
  assertDType raw.dtype .int16 "where int16 dtype"
  let expected ← expectI16Bytes [2] #[Int16.ofInt 5, Int16.ofInt 9] "where int16 data"
  assertEqBytes raw.data expected "where int16 data"

def testMovementInt16 : IO Unit := do
  let (base, permT, shrinkT, padT) := runTensorM do
    let t ← Tensor.buffer [2, 2] .int16
    let permT ← permute t [1, 0]
    let shrinkT ← shrink t [(0, 2), (0, 1)]
    let padT ← pad t [(1, 1), (1, 1)]
    pure (t, permT, shrinkT, padT)
  let vals : Array Int16 := #[Int16.ofInt 1, Int16.ofInt 2, Int16.ofInt 3, Int16.ofInt 4]
  let baseBytes ← expectI16Bytes [2, 2] vals "movement base"
  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .int16, data := baseBytes }

  let permRaw := evalTensor permT env
  assertDType permRaw.dtype .int16 "permute int16 dtype"
  let expectedPerm : Array Int16 := #[Int16.ofInt 1, Int16.ofInt 3, Int16.ofInt 2, Int16.ofInt 4]
  let expectedPermBytes ← expectI16Bytes [2, 2] expectedPerm "permute int16 data"
  assertEqBytes permRaw.data expectedPermBytes "permute int16 data"

  let shrinkRaw := evalTensor shrinkT env
  assertDType shrinkRaw.dtype .int16 "shrink int16 dtype"
  let expectedShrink : Array Int16 := #[Int16.ofInt 1, Int16.ofInt 3]
  let expectedShrinkBytes ← expectI16Bytes [2, 1] expectedShrink "shrink int16 data"
  assertEqBytes shrinkRaw.data expectedShrinkBytes "shrink int16 data"

  let padRaw := evalTensor padT env
  assertDType padRaw.dtype .int16 "pad int16 dtype"
  let expectedPad : Array Int16 :=
    #[Int16.ofInt 0, Int16.ofInt 0, Int16.ofInt 0, Int16.ofInt 0,
      Int16.ofInt 0, Int16.ofInt 1, Int16.ofInt 2, Int16.ofInt 0,
      Int16.ofInt 0, Int16.ofInt 3, Int16.ofInt 4, Int16.ofInt 0,
      Int16.ofInt 0, Int16.ofInt 0, Int16.ofInt 0, Int16.ofInt 0]
  let expectedPadBytes ← expectI16Bytes [4, 4] expectedPad "pad int16 data"
  assertEqBytes padRaw.data expectedPadBytes "pad int16 data"

def testGatherScatter : IO Unit := do
  let (gathered, scattered) := runTensorM do
    let base ← Tensor.arange 6
    let base2 ← reshape base [2, 3]
    let idxF ← Tensor.arange 2
    let idx ← cast idxF .int32
    let gathered ← gatherLast base2 idx
    let vals ← Tensor.full [2] .float32 5.0
    let scattered : StaticTensor [2, 3] .float32 ← scatterLast vals idx
    pure (gathered, scattered)
  let gVals := evalTensor gathered
  assertEqF32 gVals #[0.0, 4.0] "gatherLast"
  let sVals := evalTensor scattered
  assertEqF32 sVals #[5.0, 0.0, 0.0, 0.0, 5.0, 0.0] "scatterLast"

def testArangeLinspaceRand : IO Unit := do
  let (a, l, r, rn, ri, zLike, oLike, fLike) := runTensorM do
    let a ← Tensor.arange 5
    let l ← Tensor.linspace 0.0 1.0 5
    let r ← Tensor.rand [4] .float32 123
    let rn ← Tensor.randn [4] .float32 7
    let ri ← Tensor.randint [5] (-2) 3 .int16 9
    let base ← Tensor.full [2] .float32 3.0
    let zLike ← Tensor.zerosLike base
    let oLike ← Tensor.onesLike base
    let fLike ← Tensor.fullLike base 2.0
    pure (a, l, r, rn, ri, zLike, oLike, fLike)
  let aVals := evalTensor a
  assertEqF32 aVals #[0.0, 1.0, 2.0, 3.0, 4.0] "arange"
  let lVals := evalTensor l
  assertEqF32 lVals #[0.0, 0.25, 0.5, 0.75, 1.0] "linspace"
  let rVals := evalTensor r
  assertAllBetween rVals 0.0 1.0 "rand"
  let rnVals := evalTensor rn
  assertAllBetween rnVals (-10.0) 10.0 "randn"
  let riRaw := evalTensor ri
  match DataArrayN.ofRawBuffer? [5] .int16 riRaw with
  | some arr =>
    let vals := DataArrayN.decodeI16 arr
    for i in [:vals.size] do
      let v := vals[i]!.toInt
      if v < -2 || v >= 3 then
        throw (IO.userError s!"randint: idx {i} {v} not in [-2, 3)")
  | none =>
    throw (IO.userError "randint: decode failed")
  let zVals := evalTensor zLike
  assertEqF32 zVals #[0.0, 0.0] "zerosLike"
  let oVals := evalTensor oLike
  assertEqF32 oVals #[1.0, 1.0] "onesLike"
  let fVals := evalTensor fLike
  assertEqF32 fVals #[2.0, 2.0] "fullLike"

def testMaxMinArg : IO Unit := do
  let (maxAll, minAll, maxAxisT, minAxisT, argmaxT, argminT) := runTensorM do
    let base ← Tensor.arange 6
    let t ← reshape base [2, 3]
    let maxAll ← max base
    let minAll ← min base
    let maxAxisT ← maxAxis t 1 false
    let minAxisT ← minAxis t 1 false
    let argmaxT ← argmax t
    let argminT ← argmin t
    pure (maxAll, minAll, maxAxisT, minAxisT, argmaxT, argminT)
  let maxVals := evalTensor maxAll
  assertEqF32 maxVals #[5.0] "max"
  let minVals := evalTensor minAll
  assertEqF32 minVals #[0.0] "min"
  let maxAxisVals := evalTensor maxAxisT
  assertEqF32 maxAxisVals #[2.0, 5.0] "maxAxis"
  let minAxisVals := evalTensor minAxisT
  assertEqF32 minAxisVals #[0.0, 3.0] "minAxis"
  let argmaxRaw := evalTensor argmaxT
  assertDType argmaxRaw.dtype .int32 "argmax dtype"
  match DataArrayN.ofRawBuffer? [2] .int32 argmaxRaw with
  | some arr =>
    let vals := DataArrayN.decodeI32 arr
    assertEqI32 vals #[Int32.ofInt 2, Int32.ofInt 2] "argmax"
  | none =>
    throw (IO.userError "argmax: decode failed")
  let argminRaw := evalTensor argminT
  assertDType argminRaw.dtype .int32 "argmin dtype"
  match DataArrayN.ofRawBuffer? [2] .int32 argminRaw with
  | some arr =>
    let vals := DataArrayN.decodeI32 arr
    assertEqI32 vals #[Int32.ofInt 0, Int32.ofInt 0] "argmin"
  | none =>
    throw (IO.userError "argmin: decode failed")
def testAbs : IO Unit := do
  let y := runTensorM do
    let x ← Tensor.buffer [3] .float32
    abs x
  assertShape y.uop.shape [3] "abs shape"
  assertDType y.uop.dtype .float32 "abs dtype"

def testSilu : IO Unit := do
  let y := runTensorM do
    let x ← Tensor.buffer [1] .float32
    silu x
  assertShape y.uop.shape [1] "silu shape"
  assertDType y.uop.dtype .float32 "silu dtype"

def testLeakyRelu : IO Unit := do
  let y := runTensorM do
    let x ← Tensor.buffer [2] .float32
    leakyRelu x 0.1
  assertShape y.uop.shape [2] "leakyRelu shape"
  assertDType y.uop.dtype .float32 "leakyRelu dtype"

def testElu : IO Unit := do
  let y := runTensorM do
    let x ← Tensor.buffer [2] .float32
    elu x 1.0
  assertShape y.uop.shape [2] "elu shape"
  assertDType y.uop.dtype .float32 "elu dtype"

def testLogSigmoid : IO Unit := do
  let y := runTensorM do
    let x ← Tensor.buffer [2] .float32
    logSigmoid x
  assertShape y.uop.shape [2] "logSigmoid shape"
  assertDType y.uop.dtype .float32 "logSigmoid dtype"

def testClamp : IO Unit := do
  let y := runTensorM do
    let x ← Tensor.buffer [3] .float32
    clamp x 0.0 1.0
  assertShape y.uop.shape [3] "clamp shape"
  assertDType y.uop.dtype .float32 "clamp dtype"

def testExtraMath : IO Unit := do
  let (rs, r6, hs, hw, ht, sq, tr, fl, ce) := runTensorM do
    let x ← Tensor.buffer [2] .float32
    let rs ← rsqrt x
    let r6 ← relu6 x
    let hs ← hardsigmoid x
    let hw ← hardswish x
    let ht ← hardtanh x (-1.0) 1.0
    let sq ← square x
    let tr ← trunc x
    let fl ← floor x
    let ce ← ceil x
    pure (rs, r6, hs, hw, ht, sq, tr, fl, ce)
  assertShape rs.uop.shape [2] "rsqrt shape"
  assertShape r6.uop.shape [2] "relu6 shape"
  assertShape hs.uop.shape [2] "hardsigmoid shape"
  assertShape hw.uop.shape [2] "hardswish shape"
  assertShape ht.uop.shape [2] "hardtanh shape"
  assertShape sq.uop.shape [2] "square shape"
  assertShape tr.uop.shape [2] "trunc shape"
  assertShape fl.uop.shape [2] "floor shape"
  assertShape ce.uop.shape [2] "ceil shape"
  assertDType rs.uop.dtype .float32 "rsqrt dtype"
  assertDType r6.uop.dtype .float32 "relu6 dtype"
  assertDType hs.uop.dtype .float32 "hardsigmoid dtype"
  assertDType hw.uop.dtype .float32 "hardswish dtype"
  assertDType ht.uop.dtype .float32 "hardtanh dtype"
  assertDType sq.uop.dtype .float32 "square dtype"
  assertDType tr.uop.dtype .float32 "trunc dtype"
  assertDType fl.uop.dtype .float32 "floor dtype"
  assertDType ce.uop.dtype .float32 "ceil dtype"

def testTrigOps : IO Unit := do
  -- Test shape and dtype
  let (sinT, cosT, tanT) := runTensorM do
    let x ← Tensor.buffer [2, 3] .float32
    let s ← sin x
    let c ← cos x
    let t ← tan x
    pure (s, c, t)
  assertShape sinT.uop.shape [2, 3] "sin shape"
  assertShape cosT.uop.shape [2, 3] "cos shape"
  assertShape tanT.uop.shape [2, 3] "tan shape"
  assertDType sinT.uop.dtype .float32 "sin dtype"
  assertDType cosT.uop.dtype .float32 "cos dtype"
  assertDType tanT.uop.dtype .float32 "tan dtype"

  -- Test actual values: sin(0) = 0, cos(0) = 1, sin(pi/2) ≈ 1
  let (sinVals, cosVals) := runTensorM do
    let x ← Tensor.full [2] .float32 0.0
    let s ← sin x
    let c ← cos x
    pure (s, c)
  let sVals := evalTensor sinVals
  let cVals := evalTensor cosVals
  -- sin(0) should be 0
  for i in [:sVals.size] do
    if Float.abs sVals[i]! > 0.0001 then
      throw (IO.userError s!"sin(0) = {sVals[i]!}, expected ~0")
  -- cos(0) should be 1
  for i in [:cVals.size] do
    if Float.abs (cVals[i]! - 1.0) > 0.0001 then
      throw (IO.userError s!"cos(0) = {cVals[i]!}, expected ~1")

def testPow : IO Unit := do
  -- Test shape and dtype
  let powT := runTensorM do
    let x ← Tensor.buffer [2, 3] .float32
    let y ← Tensor.buffer [2, 3] .float32
    pow x y
  assertShape powT.uop.shape [2, 3] "pow shape"
  assertDType powT.uop.dtype .float32 "pow dtype"

  -- Test actual values: 2^3 = 8, 3^2 = 9
  let powVals := runTensorM do
    let x ← Tensor.full [2] .float32 2.0
    let y ← Tensor.full [2] .float32 3.0
    pow x y
  let pVals := evalTensor powVals
  -- 2^3 should be 8
  for i in [:pVals.size] do
    if Float.abs (pVals[i]! - 8.0) > 0.0001 then
      throw (IO.userError s!"2^3 = {pVals[i]!}, expected 8")

def testRecip : IO Unit := do
  -- Test shape and dtype
  let recipT := runTensorM do
    let x ← Tensor.buffer [2, 3] .float32
    recip x
  assertShape recipT.uop.shape [2, 3] "recip shape"
  assertDType recipT.uop.dtype .float32 "recip dtype"

  -- Test actual values: 1/2 = 0.5
  let recipVals := runTensorM do
    let x ← Tensor.full [2] .float32 2.0
    recip x
  let rVals := evalTensor recipVals
  for i in [:rVals.size] do
    if Float.abs (rVals[i]! - 0.5) > 0.0001 then
      throw (IO.userError s!"1/2 = {rVals[i]!}, expected 0.5")

def testMeanVarAxis : IO Unit := do
  let (meanT, varT) := runTensorM do
    let x ← Tensor.buffer [2, 3] .float32
    let meanT ← meanAxis x 1 false
    let varT ← varAxis x 1 false
    pure (meanT, varT)
  assertShape meanT.uop.shape [2] "meanAxis shape"
  assertShape varT.uop.shape [2] "varAxis shape"
  assertDType meanT.uop.dtype .float32 "meanAxis dtype"
  assertDType varT.uop.dtype .float32 "varAxis dtype"

def testLogsumexpAxis : IO Unit := do
  let (lseKeep, lseDrop) := runTensorM do
    let x ← Tensor.buffer [2, 3, 4] .float32
    let lseKeep ← logsumexpAxis x 1 true
    let lseDrop ← logsumexpAxis x 1 false
    pure (lseKeep, lseDrop)
  assertShape lseKeep.uop.shape [2, 1, 4] "logsumexp keepdim shape"
  assertShape lseDrop.uop.shape [2, 4] "logsumexp dropdim shape"
  assertDType lseKeep.uop.dtype .float32 "logsumexp keepdim dtype"
  assertDType lseDrop.uop.dtype .float32 "logsumexp dropdim dtype"

def testSoftmaxAxis : IO Unit := do
  let y := runTensorM do
    let x ← Tensor.buffer [2, 3, 4] .float32
    softmaxAxis x 1
  assertShape y.uop.shape [2, 3, 4] "softmaxAxis shape"
  assertDType y.uop.dtype .float32 "softmaxAxis dtype"

def testLogSoftmaxAxis : IO Unit := do
  let y := runTensorM do
    let x ← Tensor.buffer [2, 3, 4] .float32
    logSoftmaxAxis x 2
  assertShape y.uop.shape [2, 3, 4] "logSoftmaxAxis shape"
  assertDType y.uop.dtype .float32 "logSoftmaxAxis dtype"

def testLayerNorm : IO Unit := do
  let y := runTensorM do
    let x ← Tensor.buffer [2, 3] .float32
    layerNorm x
  assertShape y.uop.shape [2, 3] "layerNorm shape"
  assertDType y.uop.dtype .float32 "layerNorm dtype"

def testRmsNorm : IO Unit := do
  let y := runTensorM do
    let x ← Tensor.buffer [2, 3] .float32
    rmsNorm x
  assertShape y.uop.shape [2, 3] "rmsNorm shape"
  assertDType y.uop.dtype .float32 "rmsNorm dtype"

def testBinaryCrossEntropy : IO Unit := do
  let loss := runTensorM do
    let pred ← Tensor.buffer [2] .float32
    let target ← Tensor.buffer [2] .float32
    binaryCrossEntropy pred target
  assertShape loss.uop.shape [] "binaryCrossEntropy shape"
  assertDType loss.uop.dtype .float32 "binaryCrossEntropy dtype"

def testBinaryCrossEntropyWithLogits : IO Unit := do
  let loss := runTensorM do
    let logits ← Tensor.buffer [2] .float32
    let target ← Tensor.buffer [2] .float32
    binaryCrossEntropyWithLogits logits target
  assertShape loss.uop.shape [] "binaryCrossEntropyWithLogits shape"
  assertDType loss.uop.dtype .float32 "binaryCrossEntropyWithLogits dtype"

def testMseLoss : IO Unit := do
  let loss := runTensorM do
    let pred ← Tensor.buffer [2, 3] .float32
    let target ← Tensor.buffer [2, 3] .float32
    mseLoss pred target
  assertShape loss.uop.shape [] "mseLoss shape"
  assertDType loss.uop.dtype .float32 "mseLoss dtype"

def testSmoothL1Loss : IO Unit := do
  let loss := runTensorM do
    let pred ← Tensor.buffer [2, 3] .float32
    let target ← Tensor.buffer [2, 3] .float32
    smoothL1Loss pred target 1.0
  assertShape loss.uop.shape [] "smoothL1Loss shape"
  assertDType loss.uop.dtype .float32 "smoothL1Loss dtype"

def runAll : IO Unit := do
  IO.println "=== MathOpsSmoke Tests ==="
  testCompareEval
  IO.println "✓ compare eval"
  testBoolOpsEval
  IO.println "✓ bool ops eval"
  testCastToAndDType
  IO.println "✓ cast/to/dtype utils"
  testDTypeUtils
  IO.println "✓ dtype utils"
  testCatEval
  IO.println "✓ cat eval"
  testCatEvalInt16
  IO.println "✓ cat int16 eval"
  testStackEval
  IO.println "✓ stack eval"
  testWhereInt16
  IO.println "✓ where int16 eval"
  testMovementInt16
  IO.println "✓ movement int16 eval"
  testGatherScatter
  IO.println "✓ gather/scatter eval"
  testArangeLinspaceRand
  IO.println "✓ arange/linspace/rand eval"
  testMaxMinArg
  IO.println "✓ max/min/argmax/argmin eval"
  testAbs
  IO.println "✓ abs"
  testSilu
  IO.println "✓ silu"
  testLeakyRelu
  IO.println "✓ leakyRelu"
  testElu
  IO.println "✓ elu"
  testLogSigmoid
  IO.println "✓ logSigmoid"
  testClamp
  IO.println "✓ clamp"
  testExtraMath
  IO.println "✓ extra math"
  testTrigOps
  IO.println "✓ trig ops (sin/cos/tan)"
  testPow
  IO.println "✓ pow"
  testRecip
  IO.println "✓ recip"
  testMeanVarAxis
  IO.println "✓ mean/var axis"
  testLogsumexpAxis
  IO.println "✓ logsumexp axis"
  testSoftmaxAxis
  IO.println "✓ softmax axis"
  testLogSoftmaxAxis
  IO.println "✓ logSoftmax axis"
  testLayerNorm
  IO.println "✓ layerNorm"
  testRmsNorm
  IO.println "✓ rmsNorm"
  testMseLoss
  IO.println "✓ mseLoss"
  testSmoothL1Loss
  IO.println "✓ smoothL1Loss"
  testBinaryCrossEntropy
  IO.println "✓ BCE"
  testBinaryCrossEntropyWithLogits
  IO.println "✓ BCEWithLogits"
  IO.println "=== MathOpsSmoke OK ==="

end TinyGrad4.Test.MathOpsSmoke
