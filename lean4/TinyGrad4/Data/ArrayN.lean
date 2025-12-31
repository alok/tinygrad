import TinyGrad4.Basic
import TinyGrad4.Shape
import TinyGrad4.DType
import TinyGrad4.Backend.Buffer

namespace TinyGrad4

-- Disable RawBuffer linter: uses Array Float for internal packF32 helper
set_option linter.useRawBuffer false

/-- Packed data array with static shape and dtype at the type level. -/
structure DataArrayN (shape : Shape) (dtype : DType) where
  data : ByteArray

namespace DataArrayN

private def bytesFromUInt16 (v : UInt16) : Array UInt8 :=
  let b0 : UInt8 := UInt8.ofNat (UInt16.toNat (v &&& 0xFF))
  let b1 : UInt8 := UInt8.ofNat (UInt16.toNat ((v >>> 8) &&& 0xFF))
  #[b0, b1]

private def bytesFromUInt32 (v : UInt32) : Array UInt8 :=
  let b0 : UInt8 := UInt8.ofNat (UInt32.toNat (v &&& 0xFF))
  let b1 : UInt8 := UInt8.ofNat (UInt32.toNat ((v >>> 8) &&& 0xFF))
  let b2 : UInt8 := UInt8.ofNat (UInt32.toNat ((v >>> 16) &&& 0xFF))
  let b3 : UInt8 := UInt8.ofNat (UInt32.toNat ((v >>> 24) &&& 0xFF))
  #[b0, b1, b2, b3]

private def readU16LE (b : ByteArray) (offset : Nat) : UInt16 :=
  let b0 := b.get! offset
  let b1 := b.get! (offset + 1)
  (UInt16.ofNat b0.toNat) ||| ((UInt16.ofNat b1.toNat) <<< 8)

private def readU32LE (b : ByteArray) (offset : Nat) : UInt32 :=
  let b0 := b.get! offset
  let b1 := b.get! (offset + 1)
  let b2 := b.get! (offset + 2)
  let b3 := b.get! (offset + 3)
  (UInt32.ofNat b0.toNat) |||
    ((UInt32.ofNat b1.toNat) <<< 8) |||
    ((UInt32.ofNat b2.toNat) <<< 16) |||
    ((UInt32.ofNat b3.toNat) <<< 24)

private def pushBytes (out : ByteArray) (bytes : Array UInt8) : ByteArray := Id.run do
  let mut acc := out
  for b in bytes do
    acc := acc.push b
  return acc

private def packF32 (vals : Array Float) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity (vals.size * 4)
  for v in vals do
    let bits := (Float.toFloat32 v).toBits
    out := pushBytes out (bytesFromUInt32 bits)
  return out

private def int16ToUInt16 (v : Int16) : UInt16 :=
  UInt16.ofNat v.toBitVec.toNat

private def packI16 (vals : Array Int16) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity (vals.size * 2)
  for v in vals do
    out := pushBytes out (bytesFromUInt16 (int16ToUInt16 v))
  return out

/-- Construct float32 data from an Array Float (pads or truncates to shape). -/
def ofArrayF32 (shape : Shape) (vals : Array Float) : DataArrayN shape .float32 :=
  let expected := Shape.numel shape
  let vals' :=
    if vals.size == expected then
      vals
    else
      (fun n =>
        if n == expected then
          vals.extract 0 n
        else
          vals.extract 0 n ++ Array.replicate (expected - n) 0.0
      ) (Nat.min vals.size expected)
  { data := packF32 vals' }

/-- Construct int16 data when the length matches the shape. -/
def ofArrayI16? (shape : Shape) (vals : Array Int16) : Option (DataArrayN shape .int16) :=
  if vals.size != Shape.numel shape then
    none
  else
    some { data := packI16 vals }

/-- Wrap a RawBuffer when dtype and byte size match. -/
def ofRawBuffer? (shape : Shape) (dtype : DType) (buf : RawBuffer) : Option (DataArrayN shape dtype) :=
  if buf.dtype != dtype then
    none
  else if buf.data.size != Shape.numel shape * dtype.itemsize then
    none
  else
    some { data := buf.data }

/-- Decode int16 bytes into an Array Int16. -/
def decodeI16 {shape : Shape} (arr : DataArrayN shape .int16) : Array Int16 := Id.run do
  let n := Shape.numel shape
  let mut out := Array.emptyWithCapacity n
  for i in [:n] do
    let bits := readU16LE arr.data (i * 2)
    let v := Int16.ofBitVec (BitVec.ofNat 16 (UInt16.toNat bits))
    out := out.push v
  return out

/-- Decode int32 bytes into an Array Int32. -/
def decodeI32 {shape : Shape} (arr : DataArrayN shape .int32) : Array Int32 := Id.run do
  let n := Shape.numel shape
  let mut out := Array.emptyWithCapacity n
  for i in [:n] do
    let bits := readU32LE arr.data (i * 4)
    let v := Int32.ofBitVec (BitVec.ofNat 32 (UInt32.toNat bits))
    out := out.push v
  return out

end DataArrayN

end TinyGrad4
