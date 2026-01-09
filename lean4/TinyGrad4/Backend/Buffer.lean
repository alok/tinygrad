import TinyGrad4.DType

namespace TinyGrad4

/-!
# RawBuffer

`RawBuffer` is the runtime storage for interpreter evaluation:
- data lives in a packed `ByteArray` (unboxed bytes)
- `dtype` says how to interpret the bytes

This is the substrate for a future "everything static + superoptimized kernels" backend.
-/

structure RawBuffer where
  dtype : DType
  data : ByteArray

instance : Repr RawBuffer where
  reprPrec b _ :=
    s!"RawBuffer(dtype := {repr b.dtype}, byteSize := {b.data.size})"

namespace RawBuffer

def byteSize (b : RawBuffer) : Nat := b.data.size

/-- Convert F32 RawBuffer to FloatArray (Float64) for testing/inspection.
    Returns empty array for non-F32 dtypes. -/
def toFloatArray (b : RawBuffer) : FloatArray :=
  if b.dtype != .float32 then FloatArray.empty
  else
    let numElems := b.data.size / 4
    let readF32 (i : Nat) : Float :=
      let off := i * 4
      let b0 := b.data[off]!.toUInt32
      let b1 := b.data[off + 1]!.toUInt32
      let b2 := b.data[off + 2]!.toUInt32
      let b3 := b.data[off + 3]!.toUInt32
      -- Little-endian: low byte first
      let bits := b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24)
      (Float32.ofBits bits).toFloat
    FloatArray.mk ((Array.range numElems).map readF32)

/-- Alias for toFloatArray (backward compatibility with tests) -/
abbrev decode := toFloatArray

/-- Empty float32 buffer (for testing) -/
def emptyFloat32 : RawBuffer := { dtype := .float32, data := ByteArray.empty }

/-- Create float32 buffer with given size (zero-initialized) -/
def mkFloat32 (numElems : Nat) : RawBuffer :=
  { dtype := .float32, data := ByteArray.mk (Array.replicate (numElems * 4) 0) }

end RawBuffer

end TinyGrad4
