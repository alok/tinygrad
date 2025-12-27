/-!
# Data Types for TinyGrad4

Mirrors tinygrad's dtype.py with Lean 4 types.
Supports:
- Scalar types: bool, int8/16/32/64, uint8/16/32/64, float16/32/64
- Vector types: vec base count
- Pointer types: ptr base addrspace (for future backend integration)

## Type Promotion
Implements JAX-style type promotion lattice for binary operations.
-/

namespace TinyGrad4

/-- Address space for pointer types -/
inductive AddrSpace where
  | global  -- Global memory (GPU VRAM, main RAM)
  | local_  -- Local/shared memory
  | reg     -- Registers
  deriving DecidableEq, Repr, Hashable

/-- Data types for tensor elements -/
inductive DType where
  | void      -- No type (for ops like STORE that don't produce values)
  | bool      -- Boolean
  | int8      -- Signed 8-bit integer
  | uint8     -- Unsigned 8-bit integer
  | int16     -- Signed 16-bit integer
  | uint16    -- Unsigned 16-bit integer
  | int32     -- Signed 32-bit integer
  | uint32    -- Unsigned 32-bit integer
  | int64     -- Signed 64-bit integer
  | uint64    -- Unsigned 64-bit integer
  | float16   -- Half precision float (IEEE 754)
  | bfloat16  -- Brain float (truncated float32)
  | float32   -- Single precision float
  | float64   -- Double precision float
  | index     -- Special index type for symbolic integers
  | vec (base : DType) (count : Nat)  -- SIMD vector type
  | ptr (base : DType) (addrspace : AddrSpace)  -- Pointer type
  deriving DecidableEq, Repr, Hashable

namespace DType

/-- Size in bytes for scalar types -/
def itemsize : DType → Nat
  | .void => 0
  | .bool => 1
  | .int8 | .uint8 => 1
  | .int16 | .uint16 | .float16 | .bfloat16 => 2
  | .int32 | .uint32 | .float32 => 4
  | .int64 | .uint64 | .float64 | .index => 8
  | .vec base count => base.itemsize * count
  | .ptr _ _ => 8  -- 64-bit pointers

/-- Priority for type promotion (higher = wider type) -/
def priority : DType → Int
  | .void => -1
  | .index => -1
  | .bool => 0
  | .int8 => 1 | .uint8 => 2
  | .int16 => 3 | .uint16 => 4
  | .int32 => 5 | .uint32 => 6
  | .int64 => 7 | .uint64 => 8
  | .float16 => 10 | .bfloat16 => 11
  | .float32 => 12
  | .float64 => 13
  | .vec base _ => base.priority
  | .ptr base _ => base.priority

/-- Is this a floating point type? -/
def isFloat : DType → Bool
  | .float16 | .bfloat16 | .float32 | .float64 => true
  | .vec base _ => base.isFloat
  | _ => false

/-- Is this a signed integer type? -/
def isSigned : DType → Bool
  | .int8 | .int16 | .int32 | .int64 => true
  | .vec base _ => base.isSigned
  | _ => false

/-- Is this an unsigned integer type? -/
def isUnsigned : DType → Bool
  | .uint8 | .uint16 | .uint32 | .uint64 | .bool => true
  | .vec base _ => base.isUnsigned
  | _ => false

/-- Is this any integer type? -/
def isInt : DType → Bool
  | dt => dt.isSigned || dt.isUnsigned

/-- Get the scalar type (for vectors) -/
def scalar : DType → DType
  | .vec base _ => base.scalar
  | dt => dt

/-- Get vector count (1 for scalars) -/
def count : DType → Nat
  | .vec _ n => n
  | _ => 1

/-- Create a vector type -/
def toVec (dt : DType) (n : Nat) : DType :=
  if n == 1 then dt else .vec dt.scalar n

/-- Create a pointer type -/
def toPtr (dt : DType) (addrspace : AddrSpace := .global) : DType :=
  .ptr dt addrspace

/-- Type promotion for binary operations (JAX-style) -/
def promote (dt1 dt2 : DType) : DType :=
  -- Same type: no promotion needed
  if dt1 == dt2 then dt1
  -- Float wins over int
  else if dt1.isFloat && !dt2.isFloat then dt1
  else if dt2.isFloat && !dt1.isFloat then dt2
  -- Higher priority wins
  else if dt1.priority >= dt2.priority then dt1
  else dt2

/-- Can losslessly cast from `from_` to `to`? (more precise than `priority`-only). -/
def canLosslessCast (from_ to : DType) : Bool :=
  let f := from_.scalar
  let t := to.scalar
  if f == t || f == .bool then
    true
  else
    match t with
    | .index => f.isInt
    | .float64 =>
      f == .float32 || f == .float16 || f == .bfloat16 ||
      f == .uint32 || f == .uint16 || f == .uint8 ||
      f == .int32 || f == .int16 || f == .int8
    | .float32 =>
      f == .float16 || f == .bfloat16 ||
      f == .uint16 || f == .uint8 || f == .int16 || f == .int8
    | .float16 =>
      f == .uint8 || f == .int8
    | .bfloat16 =>
      f == .uint8 || f == .int8
    | .uint64 => f == .uint32 || f == .uint16 || f == .uint8
    | .uint32 => f == .uint16 || f == .uint8
    | .uint16 => f == .uint8
    | .int64 => f == .uint32 || f == .uint16 || f == .uint8 || f == .int32 || f == .int16 || f == .int8
    | .int32 => f == .uint16 || f == .uint8 || f == .int16 || f == .int8
    | .int16 => f == .uint8 || f == .int8
    | .bool => f == .bool
    | _ => false

/-- Can safely cast from dt1 to dt2 without losing precision? -/
def canSafeCast (from_ to : DType) : Bool :=
  canLosslessCast from_ to

end DType

/-- Common dtype aliases -/
abbrev f16 := DType.float16
abbrev bf16 := DType.bfloat16
abbrev f32 := DType.float32
abbrev f64 := DType.float64
abbrev i8 := DType.int8
abbrev i16 := DType.int16
abbrev i32 := DType.int32
abbrev i64 := DType.int64
abbrev u8 := DType.uint8
abbrev u16 := DType.uint16
abbrev u32 := DType.uint32
abbrev u64 := DType.uint64

end TinyGrad4
