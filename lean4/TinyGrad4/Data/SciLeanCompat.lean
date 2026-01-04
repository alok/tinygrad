import SciLean.FFI.FloatArray
import TinyGrad4.Backend.Buffer
import TinyGrad4.Data.ArrayN

namespace TinyGrad4.Data.SciLeanCompat

open TinyGrad4

-- SciLean interop uses `FloatArray` to access FFI helpers.
set_option linter.useRawBuffer false

/-- Convert a SciLean `FloatArray` to a TinyGrad4 `RawBuffer` (float64). -/
def floatArrayToRawBuffer (arr : FloatArray) : RawBuffer :=
  { dtype := .float64, data := FloatArray.toByteArray arr }

/-- Convert a SciLean `FloatArray` to a raw ByteArray payload. -/
def floatArrayToByteArray (arr : FloatArray) : ByteArray :=
  FloatArray.toByteArray arr

/-- Convert a TinyGrad4 `RawBuffer` to a SciLean `FloatArray` when compatible. -/
def rawBufferToFloatArray? (buf : RawBuffer) : Option FloatArray :=
  if buf.dtype != .float64 then
    none
  else if h : buf.data.size % 8 = 0 then
    some (ByteArray.toFloatArray buf.data h)
  else
    none

/-- Attempt to reinterpret a SciLean `FloatArray` as a TinyGrad4 `DataArrayN`. -/
def floatArrayToDataArrayN (shape : Shape) (arr : FloatArray) : Option (DataArrayN shape .float64) :=
  DataArrayN.ofRawBuffer? shape .float64 (floatArrayToRawBuffer arr)

end TinyGrad4.Data.SciLeanCompat
