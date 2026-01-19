import Float64
import TinyGrad4.Data.Slice
/-!
# Accelerate FFI - SIMD operations via Apple's Accelerate framework

Provides vectorized operations using vDSP and BLAS.
These are the same primitives numpy uses on macOS.

## Usage
```lean
let sum ← Accel.sumU8 byteArray 0 byteArray.size
let normalized ← Accel.normalizeU8ToF32 byteArray 0 byteArray.size
let result ← Accel.matmulF32 a 0 b 0 M K N
```
-/

namespace TinyGrad4.Backend.Accel

-- ============================================================================
-- FFI declarations
-- ============================================================================

/-- Sum all bytes in a ByteArray slice. Uses vDSP_sve (vectorized). -/
@[extern "lean_accel_sum_u8"]
private opaque sumU8Impl (ba : @& ByteArray) (offset len : USize) : UInt64

/-- Sum all floats in a ByteArray (interpreted as float32). -/
@[extern "lean_accel_sum_f32"]
private opaque sumF32Impl (ba : @& ByteArray) (offset numFloats : USize) : Float64

/-- Convert uint8 to float32 and normalize by 255. Returns new ByteArray. -/
@[extern "lean_accel_normalize_u8_to_f32"]
private opaque normalizeU8ToF32Impl (ba : @& ByteArray) (offset len : USize) : ByteArray

/-- Sum normalized values (u8 -> f32/255 -> sum) in one fused pass. -/
@[extern "lean_accel_normalize_sum_u8"]
private opaque normalizeSumU8Impl (ba : @& ByteArray) (offset len : USize) : Float64

/-- Matrix multiply C = A @ B. All float32, row-major. -/
@[extern "lean_accel_matmul_f32"]
private opaque matmulF32Impl (a : @& ByteArray) (aOffset : USize)
                      (b : @& ByteArray) (bOffset : USize)
                      (M K N : USize) : ByteArray

/-- Check if Accelerate framework is available. -/
@[extern "lean_accel_available"]
private opaque isAvailableImpl : Unit → UInt8

/-- Get optimal batch size for current hardware. -/
@[extern "lean_accel_optimal_batch_size"]
private opaque optimalBatchSizeImpl : Unit → USize

-- ============================================================================
-- Safety helpers
-- ============================================================================

private def toUSizeChecked (name : String) (n : Nat) : USize :=
  let u := n.toUSize
  if u.toNat == n then
    u
  else
    panic! s!"{name}: {n} exceeds USize"

private def checkBounds (name : String) (size offset len : Nat) : Unit :=
  if offset + len ≤ size then
    ()
  else
    panic! s!"{name}: offset+len out of bounds (size={size}, offset={offset}, len={len})"

private def checkF32Bounds (name : String) (sizeBytes offset numFloats : Nat) : Unit :=
  let byteOffset := offset * 4
  let byteLen := numFloats * 4
  checkBounds name sizeBytes byteOffset byteLen

-- ============================================================================
-- Lean-friendly wrappers
-- ============================================================================

/-- Check if Accelerate is available (true on macOS). -/
def isAvailable : Bool := isAvailableImpl () != 0

/-- Get optimal batch size based on cache sizes. -/
def optimalBatchSize : Nat := (optimalBatchSizeImpl ()).toNat

/-- Sum all bytes in a ByteArray slice. O(n) with SIMD. -/
def sumU8 (ba : ByteArray) (offset : Nat := 0) (len : Nat := ba.size) : UInt64 :=
  let _ := checkBounds "sumU8" ba.size offset len
  sumU8Impl ba (toUSizeChecked "sumU8.offset" offset)
    (toUSizeChecked "sumU8.len" len)

/-- Sum all floats in a ByteArray. Interprets bytes as float32. -/
def sumF32 (ba : ByteArray) (offset : Nat := 0) (numFloats : Nat := ba.size / 4) : Float64 :=
  let _ := checkF32Bounds "sumF32" ba.size offset numFloats
  sumF32Impl ba (toUSizeChecked "sumF32.offset" offset)
    (toUSizeChecked "sumF32.numFloats" numFloats)

/-- Sum all floats in a ByteSlice (byte-aligned to 4). -/
def sumF32Slice (s : TinyGrad4.Data.ByteSlice) : Float64 :=
  if s.offset % 4 == 0 && s.length % 4 == 0 then
    sumF32 s.parent (s.offset / 4) (s.length / 4)
  else
    panic! "sumF32Slice: ByteSlice must be 4-byte aligned"

/-- Convert uint8 array to normalized float32 array.
    Each byte b becomes (b / 255.0) as float32.
    Returns ByteArray of len * 4 bytes. -/
def normalizeU8ToF32 (ba : ByteArray) (offset : Nat := 0) (len : Nat := ba.size) : ByteArray :=
  let _ := checkBounds "normalizeU8ToF32" ba.size offset len
  normalizeU8ToF32Impl ba (toUSizeChecked "normalizeU8ToF32.offset" offset)
    (toUSizeChecked "normalizeU8ToF32.len" len)

/-- Fused normalize + sum. More efficient than separate calls. -/
def normalizeSumU8 (ba : ByteArray) (offset : Nat := 0) (len : Nat := ba.size) : Float64 :=
  let _ := checkBounds "normalizeSumU8" ba.size offset len
  normalizeSumU8Impl ba (toUSizeChecked "normalizeSumU8.offset" offset)
    (toUSizeChecked "normalizeSumU8.len" len)

/-- Convert a ByteSlice of uint8 to normalized float32 array. -/
def normalizeU8ToF32Slice (s : TinyGrad4.Data.ByteSlice) : ByteArray :=
  normalizeU8ToF32 s.parent s.offset s.length

/-- Fused normalize + sum for a ByteSlice. -/
def normalizeSumU8Slice (s : TinyGrad4.Data.ByteSlice) : Float64 :=
  normalizeSumU8 s.parent s.offset s.length

/-- Sum bytes in a ByteSlice using SIMD. -/
def sumU8Slice (s : TinyGrad4.Data.ByteSlice) : UInt64 :=
  sumU8 s.parent s.offset s.length

/-- Matrix multiply C = A @ B.
    A: [M, K] float32, B: [K, N] float32, C: [M, N] float32.
    All row-major. Returns ByteArray of M * N * 4 bytes. -/
def matmulF32 (a : ByteArray) (aOffset : Nat)
              (b : ByteArray) (bOffset : Nat)
              (M K N : Nat) : ByteArray :=
  let _ := checkF32Bounds "matmulF32.a" a.size aOffset (M * K)
  let _ := checkF32Bounds "matmulF32.b" b.size bOffset (K * N)
  matmulF32Impl a (toUSizeChecked "matmulF32.aOffset" aOffset)
    b (toUSizeChecked "matmulF32.bOffset" bOffset)
    (toUSizeChecked "matmulF32.M" M)
    (toUSizeChecked "matmulF32.K" K)
    (toUSizeChecked "matmulF32.N" N)

-- ============================================================================
-- ByteSlice integration
-- ============================================================================

/-- Sum bytes in a ByteSlice using SIMD. -/
def sumSlice (s : TinyGrad4.Data.ByteSlice) : UInt64 :=
  sumU8Slice s

/-- Normalize ByteSlice to float32 array. -/
def normalizeSlice (s : TinyGrad4.Data.ByteSlice) : ByteArray :=
  normalizeU8ToF32Slice s

/-- Fused normalize + sum for ByteSlice. -/
def normalizeSumSlice (s : TinyGrad4.Data.ByteSlice) : Float64 :=
  normalizeSumU8Slice s

end TinyGrad4.Backend.Accel
