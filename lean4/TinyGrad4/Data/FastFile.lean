import Init.System.IO

namespace TinyGrad4.Data

namespace FastFile

/-- Raw POSIX file descriptor (FastFile). -/
abbrev RawFD := USize

/-- Open a file for read-only access (POSIX `open`). -/
@[extern "tg4_fast_open"] opaque openRO (path : @& String) : IO RawFD

/-- Close a raw file descriptor. -/
@[extern "tg4_fast_close"] opaque close (fd : RawFD) : IO Unit

/-- Rewind a raw file descriptor to the start. -/
@[extern "tg4_fast_rewind"] opaque rewind (fd : RawFD) : IO Unit

/-- Read into a preallocated ByteArray and return it.
    Caller must ensure `buf` has at least `nbytes` capacity and treat it as unique. -/
@[extern "tg4_fast_read_into"] opaque readInto (fd : RawFD) (buf : ByteArray) (nbytes : USize) : IO ByteArray

/-- Run an action with a raw fd opened read-only. -/
def withFile (path : String) (f : RawFD → IO α) : IO α := do
  let fd ← openRO path
  try
    f fd
  finally
    close fd

/-- Read up to `nbytes` into `buf`, returning the updated buffer. -/
def readIntoNat (fd : RawFD) (buf : ByteArray) (nbytes : Nat) : IO ByteArray :=
  readInto fd buf nbytes.toUSize

end FastFile

end TinyGrad4.Data
