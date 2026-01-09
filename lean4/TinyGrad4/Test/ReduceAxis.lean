import TinyGrad4

/-!
# ReduceAxis Interpreter Tests

Checks the `ByteArray`-backed interpreter fast paths for:
- sum/max reduction over the last axis
- softmax/log-softmax (which rely on those reductions)
-/

namespace TinyGrad4.Test.ReduceAxis

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

open TinyGrad4
open StaticTensor
open Interpreter

private def assertSize (arr : RawBuffer) (expected : Nat) (label : String) : IO Unit := do
  let vals := arr.toFloatArray
  if vals.size != expected then
    throw (IO.userError s!"{label}: size {vals.size} != {expected}")

private def assertAllClose (arr : RawBuffer) (expected : Float) (tol : Float) (label : String) : IO Unit := do
  let vals := arr.toFloatArray
  for i in [:vals.size] do
    let v := vals[i]!
    let diff := Float.abs (v - expected)
    if diff > tol then
      throw (IO.userError s!"{label}: idx {i} value {v} expected {expected} diff {diff} > {tol}")

def testSumAxisLast : IO Unit := do
  let res := runTensorM do
    let x ← Tensor.full [2, 3] .float32 2.0
    let y ← sumAxis x 1 true
    let env : Env := ∅
    pure (eval y.uop env)
  assertSize res 2 "sumAxisLast"
  assertAllClose res 6.0 0.01 "sumAxisLast"

def testSumAxisFirst : IO Unit := do
  let res := runTensorM do
    let x ← Tensor.full [2, 3] .float32 2.0
    let y ← sumAxis x 0 true
    let env : Env := ∅
    pure (eval y.uop env)
  assertSize res 3 "sumAxisFirst"
  assertAllClose res 4.0 0.01 "sumAxisFirst"

def testMaxAxisLast : IO Unit := do
  let res := runTensorM do
    let x ← Tensor.full [2, 3] .float32 2.0
    let y ← maxAxis x 1 true
    let env : Env := ∅
    pure (eval y.uop env)
  assertSize res 2 "maxAxisLast"
  assertAllClose res 2.0 0.01 "maxAxisLast"

def testMaxAxisFirst : IO Unit := do
  let res := runTensorM do
    let x ← Tensor.full [2, 3] .float32 2.0
    let y ← maxAxis x 0 true
    let env : Env := ∅
    pure (eval y.uop env)
  assertSize res 3 "maxAxisFirst"
  assertAllClose res 2.0 0.01 "maxAxisFirst"

def testLogSoftmaxUniform : IO Unit := do
  let res := runTensorM do
    let x ← Tensor.full [2, 3] .float32 1.0
    let y ← logSoftmax x
    let env : Env := ∅
    pure (eval y.uop env)
  assertSize res 6 "logSoftmaxUniform"
  -- Expected all -log(3) ~= -1.0986123
  assertAllClose res (-1.0986123) 0.02 "logSoftmaxUniform"

def testSumAxesKeepdim : IO Unit := do
  let res := runTensorM do
    let x ← Tensor.full [2, 3, 4] .float32 2.0
    let yUop ← UOp.sum x.uop [0, 2] true
    let env : Env := ∅
    pure (yUop.shape, eval yUop env)
  let (shape, arr) := res
  if shape != [1, 3, 1] then
    throw (IO.userError s!"sumAxesKeepdim: shape {shape} != [1,3,1]")
  assertSize arr 3 "sumAxesKeepdim"
  assertAllClose arr 16.0 0.01 "sumAxesKeepdim"

def testMaxAxesKeepdim : IO Unit := do
  let res := runTensorM do
    let x ← Tensor.full [2, 3, 4] .float32 2.0
    let yUop ← UOp.max_ x.uop [0, 2] true
    let env : Env := ∅
    pure (yUop.shape, eval yUop env)
  let (shape, arr) := res
  if shape != [1, 3, 1] then
    throw (IO.userError s!"maxAxesKeepdim: shape {shape} != [1,3,1]")
  assertSize arr 3 "maxAxesKeepdim"
  assertAllClose arr 2.0 0.01 "maxAxesKeepdim"

def runAll : IO Unit := do
  IO.println "=== ReduceAxis Tests ==="
  testSumAxisLast
  IO.println "✓ sumAxis last axis"
  testSumAxisFirst
  IO.println "✓ sumAxis first axis"
  testMaxAxisLast
  IO.println "✓ maxAxis last axis"
  testMaxAxisFirst
  IO.println "✓ maxAxis first axis"
  testLogSoftmaxUniform
  IO.println "✓ logSoftmax uniform"
  testSumAxesKeepdim
  IO.println "✓ sum axes [0,2] keepdim"
  testMaxAxesKeepdim
  IO.println "✓ max axes [0,2] keepdim"
  IO.println "=== ReduceAxis OK ==="

end TinyGrad4.Test.ReduceAxis

