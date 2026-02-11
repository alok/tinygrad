import Float64
import TinyGrad4

-- Disable RawBuffer linter for test files that need Array Float64 literals
set_option linter.useRawBuffer false

/-!
# ReduceAxis Fin API Smoke Test

This is the dependent-typing version of `ReduceAxis`:
we reduce with `Fin rank` axes so out-of-bounds axes are unrepresentable.
-/

namespace TinyGrad4.Test.ReduceAxisFinSmoke

open TinyGrad4
open StaticTensor
open Interpreter

private def assertSize (arr : RawBuffer) (expected : Nat) (label : String) : IO Unit := do
  let vals := arr.toFloatArray
  if vals.size != expected then
    throw (IO.userError s!"{label}: size {vals.size} != {expected}")

private def assertAllClose (arr : RawBuffer) (expected : Float64) (tol : Float64) (label : String) : IO Unit := do
  let vals := arr.toFloatArray
  for i in [:vals.size] do
    let v := vals[i]!
    let diff := Float64.abs (v - expected)
    if diff > tol then
      throw (IO.userError s!"{label}: idx {i} value {v} expected {expected} diff {diff} > {tol}")

def testSumAxisFin : IO Unit := do
  let res := runTensorM do
    let x ← Tensor.full [2, 3] .float32 2.0
    let axisLast : Fin 2 := ⟨1, by decide⟩
    let y ← sumAxisF x axisLast true
    pure (eval y.uop (∅ : Env))
  assertSize res 2 "sumAxisF"
  assertAllClose res 6.0 0.01 "sumAxisF"

def testMaxAxisFin : IO Unit := do
  let res := runTensorM do
    let x ← Tensor.full [2, 3] .float32 2.0
    let axisFirst : Fin 2 := ⟨0, by decide⟩
    let y ← maxAxisF x axisFirst true
    pure (eval y.uop (∅ : Env))
  assertSize res 3 "maxAxisF"
  assertAllClose res 2.0 0.01 "maxAxisF"

def testLogsumexpAxisFin : IO Unit := do
  let res := runTensorM do
    let x ← Tensor.full [2, 3] .float32 2.0
    let axisLast : Fin 2 := ⟨1, by decide⟩
    let y ← logsumexpAxisF x axisLast false
    pure (eval y.uop (∅ : Env))
  assertSize res 2 "logsumexpAxisF"
  assertAllClose res 3.0986123 0.02 "logsumexpAxisF"

def testSoftmaxAxisFin : IO Unit := do
  let res := runTensorM do
    let x ← Tensor.full [2, 3] .float32 2.0
    let axisLast : Fin 2 := ⟨1, by decide⟩
    let y ← softmaxAxisF x axisLast
    pure (eval y.uop (∅ : Env))
  assertSize res 6 "softmaxAxisF"
  assertAllClose res (1.0 / 3.0) 0.01 "softmaxAxisF"

def testLogSoftmaxAxisFin : IO Unit := do
  let res := runTensorM do
    let x ← Tensor.full [2, 3] .float32 2.0
    let axisLast : Fin 2 := ⟨1, by decide⟩
    let y ← logSoftmaxAxisF x axisLast
    pure (eval y.uop (∅ : Env))
  assertSize res 6 "logSoftmaxAxisF"
  assertAllClose res (-1.0986123) 0.02 "logSoftmaxAxisF"

def runAll : IO Unit := do
  IO.println "=== ReduceAxisFinSmoke Tests ==="
  testSumAxisFin
  IO.println "✓ sumAxisF"
  testMaxAxisFin
  IO.println "✓ maxAxisF"
  testLogsumexpAxisFin
  IO.println "✓ logsumexpAxisF"
  testSoftmaxAxisFin
  IO.println "✓ softmaxAxisF"
  testLogSoftmaxAxisFin
  IO.println "✓ logSoftmaxAxisF"
  IO.println "=== ReduceAxisFinSmoke OK ==="

end TinyGrad4.Test.ReduceAxisFinSmoke

#eval! TinyGrad4.Test.ReduceAxisFinSmoke.runAll
