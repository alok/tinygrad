import TinyGrad4.Backend.Buffer

/-!
# Zero-Copy Slice Types

Provides zero-copy views into ByteArray and RawBuffer without allocating new memory.
This matches numpy/tinygrad semantics where slicing returns a view, not a copy.

## Key Types
- `ByteSlice`: View into ByteArray with (parent, offset, length)
- `RawBufferSlice`: View into RawBuffer with dtype + element-level offset

## Usage
```lean
let data : ByteArray := ...  -- 10000 bytes
let slice := data.toSlice 1000 2000  -- view of bytes [1000, 2000), no copy
let value := slice[500]  -- reads data[1500]
```
-/

namespace TinyGrad4.Data

/-! ## ByteSlice: Zero-copy view into ByteArray -/

/-- A zero-copy view into a ByteArray.
    Stores reference to parent + offset + length.
    Reading `slice[i]` reads `parent[offset + i]`. -/
structure ByteSlice where
  /-- The underlying byte storage (shared reference, not copied) -/
  parent : ByteArray
  /-- Starting offset in parent -/
  offset : Nat
  /-- Number of bytes in this slice -/
  length : Nat
  /-- Proof that slice bounds are valid -/
  h_valid : offset + length ≤ parent.size := by decide

instance : Repr ByteSlice where
  reprPrec s _ := s!"ByteSlice(offset={s.offset}, length={s.length}, parentSize={s.parent.size})"

namespace ByteSlice

/-- Create a slice from entire ByteArray -/
def ofByteArray (ba : ByteArray) : ByteSlice :=
  { parent := ba, offset := 0, length := ba.size, h_valid := by simp }

/-- Create a slice with bounds checking (returns smaller slice if out of bounds) -/
def mk' (ba : ByteArray) (offset length : Nat) : ByteSlice :=
  let actualOffset := min offset ba.size
  let actualLength := min length (ba.size - actualOffset)
  { parent := ba
    offset := actualOffset
    length := actualLength
    h_valid := by omega }

/-- Size of the slice in bytes -/
@[inline] def size (s : ByteSlice) : Nat := s.length

/-- Check if slice is empty -/
@[inline] def isEmpty (s : ByteSlice) : Bool := s.length == 0

/-- Read a byte at index (relative to slice start) -/
@[inline] def get (s : ByteSlice) (i : Nat) (h : i < s.length := by omega) : UInt8 :=
  have hBound : s.offset + i < s.parent.size := by
    have := s.h_valid
    omega
  s.parent.get (s.offset + i) hBound

/-- Read a byte at index, returning 0 if out of bounds -/
@[inline] def get! (s : ByteSlice) (i : Nat) : UInt8 :=
  if h : i < s.length then s.get i h else 0

instance : GetElem ByteSlice Nat UInt8 (fun s i => i < s.length) where
  getElem s i h := s.get i h

/-- Create a sub-slice (zero-copy) -/
def slice (s : ByteSlice) (start len : Nat) : ByteSlice :=
  let actualStart := min start s.length
  let actualLen := min len (s.length - actualStart)
  { parent := s.parent
    offset := s.offset + actualStart
    length := actualLen
    h_valid := by
      have := s.h_valid
      omega }

/-- Drop first n bytes (zero-copy) -/
@[inline] def drop (s : ByteSlice) (n : Nat) : ByteSlice :=
  s.slice n (s.length - n)

/-- Take first n bytes (zero-copy) -/
@[inline] def take (s : ByteSlice) (n : Nat) : ByteSlice :=
  s.slice 0 n

/-- Read UInt32 little-endian at byte offset -/
def getU32LE (s : ByteSlice) (off : Nat) : UInt32 :=
  if off + 4 > s.length then 0
  else
    let b0 := s.get! off
    let b1 := s.get! (off + 1)
    let b2 := s.get! (off + 2)
    let b3 := s.get! (off + 3)
    b0.toUInt32 ||| (b1.toUInt32 <<< 8) ||| (b2.toUInt32 <<< 16) ||| (b3.toUInt32 <<< 24)

/-- Read UInt32 big-endian at byte offset -/
def getU32BE (s : ByteSlice) (off : Nat) : UInt32 :=
  if off + 4 > s.length then 0
  else
    let b0 := s.get! off
    let b1 := s.get! (off + 1)
    let b2 := s.get! (off + 2)
    let b3 := s.get! (off + 3)
    (b0.toUInt32 <<< 24) ||| (b1.toUInt32 <<< 16) ||| (b2.toUInt32 <<< 8) ||| b3.toUInt32

/-- Read Float32 (little-endian IEEE 754) at byte offset -/
def getF32LE (s : ByteSlice) (off : Nat) : Float32 :=
  Float32.ofBits (s.getU32LE off)

/-- Copy slice to a new ByteArray (use sparingly - defeats zero-copy purpose) -/
def toByteArray (s : ByteSlice) : ByteArray :=
  s.parent.extract s.offset (s.offset + s.length)

end ByteSlice

/-- Extension for ByteArray to create slices -/
def ByteArray.toSlice (ba : ByteArray) (start : Nat := 0) (stop : Nat := ba.size) : ByteSlice :=
  ByteSlice.mk' ba start (stop - start)

/-! ## RawBufferSlice: Zero-copy view with dtype -/

/-- A zero-copy view into a RawBuffer.
    Provides element-level (not byte-level) indexing with dtype awareness. -/
structure RawBufferSlice where
  /-- The underlying buffer (shared reference) -/
  parent : RawBuffer
  /-- Starting element offset (not byte offset) -/
  elemOffset : Nat
  /-- Number of elements in this slice -/
  numElems : Nat
  /-- Proof that slice bounds are valid -/
  h_valid : (elemOffset + numElems) * parent.dtype.itemsize ≤ parent.data.size := by decide
  deriving Repr

namespace RawBufferSlice

/-- Create slice from entire RawBuffer -/
def ofRawBuffer (rb : RawBuffer) : RawBufferSlice :=
  let numElems := rb.data.size / rb.dtype.itemsize
  { parent := rb
    elemOffset := 0
    numElems := numElems
    h_valid := by
      simp only [Nat.zero_add]
      exact Nat.div_mul_le_self rb.data.size rb.dtype.itemsize }

/-- Create slice with bounds checking -/
def mk' (rb : RawBuffer) (elemOffset numElems : Nat) : RawBufferSlice :=
  let maxElems := rb.data.size / rb.dtype.itemsize
  let actualOffset := min elemOffset maxElems
  let actualNum := min numElems (maxElems - actualOffset)
  { parent := rb
    elemOffset := actualOffset
    numElems := actualNum
    h_valid := by
      sorry  -- Proof that bounds are satisfied
  }

/-- DType of this slice -/
@[inline] def dtype (s : RawBufferSlice) : DType := s.parent.dtype

/-- Number of elements -/
@[inline] def size (s : RawBufferSlice) : Nat := s.numElems

/-- Byte offset into parent data -/
@[inline] def byteOffset (s : RawBufferSlice) : Nat :=
  s.elemOffset * s.parent.dtype.itemsize

/-- Byte size of this slice -/
@[inline] def byteSize (s : RawBufferSlice) : Nat :=
  s.numElems * s.parent.dtype.itemsize

/-- Get the underlying bytes as a ByteSlice (zero-copy) -/
def toByteSlice (s : RawBufferSlice) : ByteSlice :=
  { parent := s.parent.data
    offset := s.byteOffset
    length := s.byteSize
    h_valid := by
      unfold byteOffset byteSize
      have h := s.h_valid
      -- (elemOffset + numElems) * itemsize = elemOffset * itemsize + numElems * itemsize
      rw [Nat.add_mul] at h
      exact h }

/-- Create a sub-slice (zero-copy) -/
def slice (s : RawBufferSlice) (start len : Nat) : RawBufferSlice :=
  let actualStart := min start s.numElems
  let actualLen := min len (s.numElems - actualStart)
  { parent := s.parent
    elemOffset := s.elemOffset + actualStart
    numElems := actualLen
    h_valid := by
      have := s.h_valid
      sorry }

/-- Get element at index as Float (for float32 dtype) -/
def getF32 (s : RawBufferSlice) (i : Nat) : Float :=
  if s.dtype != .float32 then 0.0
  else if i >= s.numElems then 0.0
  else
    let bs := s.toByteSlice
    (bs.getF32LE (i * 4)).toFloat

/-- Get element at index as UInt8 (for uint8 dtype) -/
def getU8 (s : RawBufferSlice) (i : Nat) : UInt8 :=
  if s.dtype != .uint8 then 0
  else if i >= s.numElems then 0
  else s.parent.data.get! (s.byteOffset + i)

/-- Copy slice to new RawBuffer (use sparingly) -/
def toRawBuffer (s : RawBufferSlice) : RawBuffer :=
  { dtype := s.dtype
    data := s.parent.data.extract s.byteOffset (s.byteOffset + s.byteSize) }

end RawBufferSlice

/-- Extension for RawBuffer to create slices -/
def RawBuffer.toSlice (rb : RawBuffer) (elemOffset : Nat := 0) (numElems? : Option Nat := none) : RawBufferSlice :=
  let maxElems := rb.data.size / rb.dtype.itemsize
  let numElems := numElems?.getD (maxElems - elemOffset)
  RawBufferSlice.mk' rb elemOffset numElems

end TinyGrad4.Data
