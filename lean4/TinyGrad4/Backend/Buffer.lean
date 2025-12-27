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

end RawBuffer

end TinyGrad4
