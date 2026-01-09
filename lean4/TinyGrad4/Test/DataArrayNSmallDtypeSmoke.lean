import TinyGrad4
import TinyGrad4.Data.ArrayN

/-!
# DataArrayNSmallDtypeSmoke

Smoke tests for DataArrayN pack/unpack on small integer dtypes.
-/

namespace TinyGrad4.Test.DataArrayNSmallDtypeSmoke

open TinyGrad4

private def assertEqArray {α} [BEq α] [Repr α] [Inhabited α]
    (arr expected : Array α) (label : String) : IO Unit := do
  if arr.size != expected.size then
    throw (IO.userError s!"{label}: size mismatch {arr.size} != {expected.size}")
  for i in [:arr.size] do
    if arr[i]! != expected[i]! then
      throw (IO.userError s!"{label}: idx {i} got {repr arr[i]!} expected {repr expected[i]!}")

def testU8 : IO Unit := do
  let vals : Array UInt8 := #[0, 255, 1, 128]
  match DataArrayN.ofArrayU8? [4] vals with
  | some arr =>
    let decoded := DataArrayN.decodeU8 arr
    assertEqArray decoded vals "u8"
  | none =>
    throw (IO.userError "u8: ofArrayU8? returned none")

def testI8 : IO Unit := do
  let vals : Array Int8 := #[Int8.ofInt (-1), Int8.ofInt 0, Int8.ofInt 1, Int8.ofInt (-128)]
  match DataArrayN.ofArrayI8? [4] vals with
  | some arr =>
    let decoded := DataArrayN.decodeI8 arr
    assertEqArray decoded vals "i8"
  | none =>
    throw (IO.userError "i8: ofArrayI8? returned none")

def testU16 : IO Unit := do
  let vals : Array UInt16 := #[UInt16.ofNat 0, UInt16.ofNat 65535, UInt16.ofNat 1]
  match DataArrayN.ofArrayU16? [3] vals with
  | some arr =>
    let decoded := DataArrayN.decodeU16 arr
    assertEqArray decoded vals "u16"
  | none =>
    throw (IO.userError "u16: ofArrayU16? returned none")

def testI16 : IO Unit := do
  let vals : Array Int16 := #[Int16.ofInt (-1), Int16.ofInt 0, Int16.ofInt 1, Int16.ofInt (-32768)]
  match DataArrayN.ofArrayI16? [4] vals with
  | some arr =>
    let decoded := DataArrayN.decodeI16 arr
    assertEqArray decoded vals "i16"
  | none =>
    throw (IO.userError "i16: ofArrayI16? returned none")

def testReshape : IO Unit := do
  let vals : Array UInt8 := #[1, 2, 3, 4]
  match DataArrayN.ofArrayU8? [2, 2] vals with
  | some arr =>
    match DataArrayN.reshape? arr [4] with
    | some reshaped =>
      let decoded := DataArrayN.decodeU8 reshaped
      assertEqArray decoded vals "reshape u8"
    | none =>
      throw (IO.userError "reshape u8: reshape? returned none")
    match DataArrayN.reshape? arr [3] with
    | some _ =>
      throw (IO.userError "reshape u8: expected reshape? to fail on mismatched numel")
    | none =>
      pure ()
  | none =>
    throw (IO.userError "reshape u8: ofArrayU8? returned none")

def runAll : IO Unit := do
  IO.println "=== DataArrayNSmallDtypeSmoke Tests ==="
  testU8
  IO.println "✓ uint8 pack/unpack"
  testI8
  IO.println "✓ int8 pack/unpack"
  testU16
  IO.println "✓ uint16 pack/unpack"
  testI16
  IO.println "✓ int16 pack/unpack"
  testReshape
  IO.println "✓ reshape"
  IO.println "=== DataArrayNSmallDtypeSmoke OK ==="

end TinyGrad4.Test.DataArrayNSmallDtypeSmoke
