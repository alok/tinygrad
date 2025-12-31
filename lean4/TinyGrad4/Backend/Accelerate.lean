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
opaque sumU8Impl (ba : @& ByteArray) (offset len : USize) : UInt64

/-- Sum all floats in a ByteArray (interpreted as float32). -/
@[extern "lean_accel_sum_f32"]
opaque sumF32Impl (ba : @& ByteArray) (offset numFloats : USize) : Float

/-- Convert uint8 to float32 and normalize by 255. Returns new ByteArray. -/
@[extern "lean_accel_normalize_u8_to_f32"]
opaque normalizeU8ToF32Impl (ba : @& ByteArray) (offset len : USize) : ByteArray

/-- Sum normalized values (u8 -> f32/255 -> sum) in one fused pass. -/
@[extern "lean_accel_normalize_sum_u8"]
opaque normalizeSumU8Impl (ba : @& ByteArray) (offset len : USize) : Float

/-- Matrix multiply C = A @ B. All float32, row-major. -/
@[extern "lean_accel_matmul_f32"]
opaque matmulF32Impl (a : @& ByteArray) (aOffset : USize)
                      (b : @& ByteArray) (bOffset : USize)
                      (M K N : USize) : ByteArray

/-- Check if Accelerate framework is available. -/
@[extern "lean_accel_available"]
opaque isAvailableImpl : Unit → UInt8

/-- Get optimal batch size for current hardware. -/
@[extern "lean_accel_optimal_batch_size"]
opaque optimalBatchSizeImpl : Unit → USize

-- ============================================================================
-- Lean-friendly wrappers
-- ============================================================================

/-- Check if Accelerate is available (true on macOS). -/
def isAvailable : Bool := isAvailableImpl () != 0

/-- Get optimal batch size based on cache sizes. -/
def optimalBatchSize : Nat := (optimalBatchSizeImpl ()).toNat

/-- Sum all bytes in a ByteArray slice. O(n) with SIMD. -/
def sumU8 (ba : ByteArray) (offset : Nat := 0) (len : Nat := ba.size) : UInt64 :=
  sumU8Impl ba offset.toUSize len.toUSize

/-- Sum all floats in a ByteArray. Interprets bytes as float32. -/
def sumF32 (ba : ByteArray) (offset : Nat := 0) (numFloats : Nat := ba.size / 4) : Float :=
  sumF32Impl ba offset.toUSize numFloats.toUSize

/-- Convert uint8 array to normalized float32 array.
    Each byte b becomes (b / 255.0) as float32.
    Returns ByteArray of len * 4 bytes. -/
def normalizeU8ToF32 (ba : ByteArray) (offset : Nat := 0) (len : Nat := ba.size) : ByteArray :=
  normalizeU8ToF32Impl ba offset.toUSize len.toUSize

/-- Fused normalize + sum. More efficient than separate calls. -/
def normalizeSumU8 (ba : ByteArray) (offset : Nat := 0) (len : Nat := ba.size) : Float :=
  normalizeSumU8Impl ba offset.toUSize len.toUSize

/-- Matrix multiply C = A @ B.
    A: [M, K] float32, B: [K, N] float32, C: [M, N] float32.
    All row-major. Returns ByteArray of M * N * 4 bytes. -/
def matmulF32 (a : ByteArray) (aOffset : Nat)
              (b : ByteArray) (bOffset : Nat)
              (M K N : Nat) : ByteArray :=
  matmulF32Impl a aOffset.toUSize b bOffset.toUSize M.toUSize K.toUSize N.toUSize

-- ============================================================================
-- ByteSlice integration
-- ============================================================================

/-- Sum bytes in a ByteSlice using SIMD. -/
def sumSlice (parent : ByteArray) (offset len : Nat) : UInt64 :=
  sumU8 parent offset len

/-- Normalize ByteSlice to float32 array. -/
def normalizeSlice (parent : ByteArray) (offset len : Nat) : ByteArray :=
  normalizeU8ToF32 parent offset len

/-- Fused normalize + sum for ByteSlice. -/
def normalizeSumSlice (parent : ByteArray) (offset len : Nat) : Float :=
  normalizeSumU8 parent offset len

end TinyGrad4.Backend.Accel
