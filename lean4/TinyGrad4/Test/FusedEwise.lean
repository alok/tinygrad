import TinyGrad4

/-!
# Fused Elementwise Kernel Tests

Directly exercises the portable C fused elementwise VM:
- broadcasting (scalar + vector)
- bool inputs + `where`
-/

namespace TinyGrad4.Test.FusedEwise

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

private def testAddScalarBcast : IO Unit := do
  let x : Array Float := #[1.0, 2.0, 3.0, 4.0]
  let y : Array Float := #[10.0]
  let xb := packF32 x
  let yb := packF32 y

  let inputs : Array ByteArray := #[xb, yb]
  let shapes : Array (Array Nat) := #[#[4], #[]]
  let dtypes : Array Nat := #[0, 0]
  let outShape : Array Nat := #[4]

  -- prog: load0; load1; add
  let prog : Array UInt64 :=
    #[FusedEwise.instLoad 0, FusedEwise.instLoad 1, FusedEwise.instBinary 0]

  let outBytes := Native.fusedEwiseF32 inputs shapes dtypes outShape prog
  let out := (RawBuffer.decode { dtype := .float32, data := outBytes }).data
  assertAllClose out #[11.0, 12.0, 13.0, 14.0] 0.0001 "fused add scalar bcast"

private def testWhereBool : IO Unit := do
  let cond : ByteArray := ByteArray.mk #[1, 0, 1, 0]
  let x : Array Float := #[1.0, 2.0, 3.0, 4.0]
  let y : Array Float := #[10.0, 20.0, 30.0, 40.0]
  let xb := packF32 x
  let yb := packF32 y

  let inputs : Array ByteArray := #[cond, xb, yb]
  let shapes : Array (Array Nat) := #[#[4], #[4], #[4]]
  let dtypes : Array Nat := #[1, 0, 0]
  let outShape : Array Nat := #[4]

  -- prog: load0(cond); load1; load2; where
  let prog : Array UInt64 :=
    #[FusedEwise.instLoad 0, FusedEwise.instLoad 1, FusedEwise.instLoad 2, FusedEwise.instWhere]

  let outBytes := Native.fusedEwiseF32 inputs shapes dtypes outShape prog
  let out := (RawBuffer.decode { dtype := .float32, data := outBytes }).data
  assertAllClose out #[1.0, 20.0, 3.0, 40.0] 0.0001 "fused where bool"

def runAll : IO Unit := do
  IO.println "=== FusedEwise Tests ==="
  testAddScalarBcast
  IO.println "✓ add scalar broadcast"
  testWhereBool
  IO.println "✓ where with bool"
  IO.println "=== FusedEwise OK ==="

end TinyGrad4.Test.FusedEwise


