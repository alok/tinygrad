import Float64
import TinyGrad4

/-!
# GatherScatterSmoke

Smoke tests for gather/scatter operations.
-/

namespace TinyGrad4.Test.GatherScatterSmoke

set_option linter.useRawBuffer false

open TinyGrad4
open Interpreter
open StaticTensor

private def fromArrayF32 (device : Backend.DeviceType := .CPU) (shape : Shape) (vals : Array Float32)
    : TensorM (StaticTensor shape .float32 device) := do
  let u ← UOp.vconstF32 vals
  let base : StaticTensor [vals.size] .float32 device  := StaticTensor.ofUOp u
  let reshaped ← UOp.reshape base.uop shape
  pure (StaticTensor.ofUOp reshaped (requiresGrad := false))

private def fromArrayI32 (device : Backend.DeviceType := .CPU) (shape : Shape) (vals : Array Int)
    : TensorM (StaticTensor shape .int32 device) := do
  let valsF := vals.map (fun v => (Float64.ofInt v).toFloat32)
  let baseF ← fromArrayF32 shape valsF
  cast baseF .int32

private def assertEqF32 (raw : RawBuffer) (expected : Array Float64) (label : String) : IO Unit := do
  let got := raw.toFloatArray
  if got.size != expected.size then
    throw (IO.userError s!"{label}: size {got.size} != {expected.size}")
  for i in [:expected.size] do
    if got[i]! != expected[i]! then
      throw (IO.userError s!"{label}: idx {i} {got[i]!} != {expected[i]!}")

def testGatherDim1 : IO Unit := do
  let (t, idx, out) := runTensorM do
    let t ← fromArrayF32 [2, 3] #[0, 1, 2, 3, 4, 5]
    let idx ← fromArrayI32 [2, 2] #[0, 2, 1, 0]
    let out ← gather t 1 idx
    pure (t, idx, out)
  let outRaw := evalTensor out
  assertEqF32 outRaw #[0, 2, 4, 3] "gather dim=1"
  let _ := t
  let _ := idx

def testGatherDim0 : IO Unit := do
  let out := runTensorM do
    let t ← fromArrayF32 [3, 2] #[0, 1, 2, 3, 4, 5]
    let idx ← fromArrayI32 [2, 2] #[2, 0, 1, 1]
    gather t 0 idx
  let outRaw := evalTensor out
  assertEqF32 outRaw #[4, 1, 2, 3] "gather dim=0"

def testScatterLastWins : IO Unit := do
  let out := runTensorM do
    let base ← Tensor.zeros [2, 3] .float32
    let idx ← fromArrayI32 [2, 3] #[0, 1, 0, 2, 1, 1]
    let src ← fromArrayF32 [2, 3] #[10, 20, 30, 40, 50, 60]
    scatter base 1 idx src
  let outRaw := evalTensor out
  assertEqF32 outRaw #[30, 20, 0, 0, 60, 40] "scatter last-wins"

def testScatterReduceSum : IO Unit := do
  let out := runTensorM do
    let base ← Tensor.zeros [2, 3] .float32
    let idx ← fromArrayI32 [2, 3] #[0, 1, 0, 2, 1, 1]
    let src ← fromArrayF32 [2, 3] #[10, 20, 30, 40, 50, 60]
    scatterReduce base 1 idx src .sum false
  let outRaw := evalTensor out
  assertEqF32 outRaw #[40, 20, 0, 0, 110, 40] "scatterReduce sum"

def runAll : IO Unit := do
  IO.println "=== GatherScatterSmoke ==="
  testGatherDim1
  testGatherDim0
  testScatterLastWins
  testScatterReduceSum
  IO.println "=== GatherScatterSmoke OK ==="

end TinyGrad4.Test.GatherScatterSmoke

#eval! TinyGrad4.Test.GatherScatterSmoke.runAll
