import TinyGrad4

/-!
# Fused Reduce Kernel Tests

Directly exercises the portable C fused reduce kernels:
- sum/max over last axis
- broadcasting inside the fused expression
- small elementwise programs (load/const/add)
-/

namespace TinyGrad4.Test.FusedReduce

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

open TinyGrad4
open Interpreter
open Backend

/-- Pack float64 array to float32 bytes -/
private def packF32 (data : Array Float) : ByteArray :=
  Native.packF32FromF64 ⟨data⟩

private def assertAllClose (arr : Array Float) (expected : Array Float) (tol : Float) (label : String) : IO Unit := do
  if arr.size != expected.size then
    throw (IO.userError s!"{label}: size {arr.size} != {expected.size}")
  for i in [:arr.size] do
    let v := arr[i]!
    let e := expected[i]!
    let diff := Float.abs (v - e)
    if diff > tol then
      throw (IO.userError s!"{label}: idx {i} value {v} expected {e} diff {diff} > {tol}")

private def testSumVec : IO Unit := do
  let x : Array Float := #[1.0, 2.0, 3.0, 4.0]
  let xb := packF32 x

  let inputs : Array ByteArray := #[xb]
  let shapes : Array (Array Nat) := #[#[4]]
  let dtypes : Array Nat := #[0]
  let fullShape : Array Nat := #[4]

  let prog : Array UInt64 := #[FusedEwise.instLoad 0]
  let outBytes := Native.fusedReduceSumLastF32 inputs shapes dtypes fullShape prog
  let out := (RawBuffer.decode { dtype := .float32, data := outBytes }).data
  assertAllClose out #[10.0] 0.0001 "fused reduce sum vec"

private def testSumVecPlusConst : IO Unit := do
  let x : Array Float := #[1.0, 2.0, 3.0, 4.0]
  let one : Array Float := #[1.0]  -- constant passed as input buffer (scalar)
  let xb := packF32 x
  let oneb := packF32 one

  let inputs : Array ByteArray := #[xb, oneb]
  let shapes : Array (Array Nat) := #[#[4], #[]]  -- scalar broadcasts
  let dtypes : Array Nat := #[0, 0]
  let fullShape : Array Nat := #[4]

  -- prog: load0; load1 (constant as buffer); add
  let prog : Array UInt64 :=
    #[FusedEwise.instLoad 0, FusedEwise.instLoad 1, FusedEwise.instBinary 0]
  let outBytes := Native.fusedReduceSumLastF32 inputs shapes dtypes fullShape prog
  let out := (RawBuffer.decode { dtype := .float32, data := outBytes }).data
  assertAllClose out #[14.0] 0.0001 "fused reduce sum (x+1)"

private def testMaxLastAxis2x4 : IO Unit := do
  let x : Array Float := #[1.0, -2.0, 3.0, 0.0,  5.0, 4.0, -1.0, 2.0]
  let xb := packF32 x

  let inputs : Array ByteArray := #[xb]
  let shapes : Array (Array Nat) := #[#[2, 4]]
  let dtypes : Array Nat := #[0]
  let fullShape : Array Nat := #[2, 4]

  let prog : Array UInt64 := #[FusedEwise.instLoad 0]
  let outBytes := Native.fusedReduceMaxLastF32 inputs shapes dtypes fullShape prog
  let out := (RawBuffer.decode { dtype := .float32, data := outBytes }).data
  assertAllClose out #[3.0, 5.0] 0.0001 "fused reduce max last axis"

private def testSumBcast2x4PlusScalar : IO Unit := do
  let x : Array Float := #[1.0, 2.0, 3.0, 4.0,  10.0, 20.0, 30.0, 40.0]
  let y : Array Float := #[1.0]
  let xb := packF32 x
  let yb := packF32 y

  let inputs : Array ByteArray := #[xb, yb]
  let shapes : Array (Array Nat) := #[#[2, 4], #[]]
  let dtypes : Array Nat := #[0, 0]
  let fullShape : Array Nat := #[2, 4]

  -- prog: load0; load1; add
  let prog : Array UInt64 := #[FusedEwise.instLoad 0, FusedEwise.instLoad 1, FusedEwise.instBinary 0]
  let outBytes := Native.fusedReduceSumLastF32 inputs shapes dtypes fullShape prog
  let out := (RawBuffer.decode { dtype := .float32, data := outBytes }).data
  assertAllClose out #[14.0, 104.0] 0.0001 "fused reduce sum (x + scalar)"

private def testSumAxis0_2x4 : IO Unit := do
  let x : Array Float := #[1.0, 2.0, 3.0, 4.0,  10.0, 20.0, 30.0, 40.0]
  let xb := packF32 x

  let inputs : Array ByteArray := #[xb]
  let shapes : Array (Array Nat) := #[#[2, 4]]
  let dtypes : Array Nat := #[0]
  let fullShape : Array Nat := #[2, 4]

  let prog : Array UInt64 := #[FusedEwise.instLoad 0]
  let outBytes := Native.fusedReduceSumAxisF32 inputs shapes dtypes fullShape 0 prog
  let out := (RawBuffer.decode { dtype := .float32, data := outBytes }).data
  assertAllClose out #[11.0, 22.0, 33.0, 44.0] 0.0001 "fused reduce sum axis0"

private def testMaxAxis0_2x4 : IO Unit := do
  let x : Array Float := #[1.0, 2.0, 3.0, 4.0,  10.0, 20.0, 30.0, 40.0]
  let xb := packF32 x

  let inputs : Array ByteArray := #[xb]
  let shapes : Array (Array Nat) := #[#[2, 4]]
  let dtypes : Array Nat := #[0]
  let fullShape : Array Nat := #[2, 4]

  let prog : Array UInt64 := #[FusedEwise.instLoad 0]
  let outBytes := Native.fusedReduceMaxAxisF32 inputs shapes dtypes fullShape 0 prog
  let out := (RawBuffer.decode { dtype := .float32, data := outBytes }).data
  assertAllClose out #[10.0, 20.0, 30.0, 40.0] 0.0001 "fused reduce max axis0"

private def testSumAll2x2 : IO Unit := do
  let x : Array Float := #[1.0, 2.0, 3.0, 4.0]
  let xb := packF32 x

  let inputs : Array ByteArray := #[xb]
  let shapes : Array (Array Nat) := #[#[2, 2]]
  let dtypes : Array Nat := #[0]
  let fullShape : Array Nat := #[2, 2]

  let prog : Array UInt64 := #[FusedEwise.instLoad 0]
  let outBytes := Native.fusedReduceSumAllF32 inputs shapes dtypes fullShape prog
  let out := (RawBuffer.decode { dtype := .float32, data := outBytes }).data
  assertAllClose out #[10.0] 0.0001 "fused reduce sum all"

private def testMaxAll2x2 : IO Unit := do
  let x : Array Float := #[1.0, 2.0, 3.0, 4.0]
  let xb := packF32 x

  let inputs : Array ByteArray := #[xb]
  let shapes : Array (Array Nat) := #[#[2, 2]]
  let dtypes : Array Nat := #[0]
  let fullShape : Array Nat := #[2, 2]

  let prog : Array UInt64 := #[FusedEwise.instLoad 0]
  let outBytes := Native.fusedReduceMaxAllF32 inputs shapes dtypes fullShape prog
  let out := (RawBuffer.decode { dtype := .float32, data := outBytes }).data
  assertAllClose out #[4.0] 0.0001 "fused reduce max all"

def runAll : IO Unit := do
  IO.println "=== FusedReduce Tests ==="
  testSumVec
  IO.println "✓ sum vec"
  testSumVecPlusConst
  IO.println "✓ sum (x+1)"
  testMaxLastAxis2x4
  IO.println "✓ max last axis"
  testSumBcast2x4PlusScalar
  IO.println "✓ sum with scalar broadcast"
  testSumAxis0_2x4
  IO.println "✓ sum axis0"
  testMaxAxis0_2x4
  IO.println "✓ max axis0"
  testSumAll2x2
  IO.println "✓ sum all"
  testMaxAll2x2
  IO.println "✓ max all"
  IO.println "=== FusedReduce OK ==="

end TinyGrad4.Test.FusedReduce

