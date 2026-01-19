import Float64
import TinyGrad4.Backend.Device
import TinyGrad4.Backend.MetalRenderer
import TinyGrad4.Data.Slice

/-!
# Metal Backend

Type-safe Metal GPU backend via FFI to Objective-C.

## Architecture

```
Metal.lean (Lean FFI declarations)
    ↓
tg4_metal.m (Objective-C FFI)
    ↓
Metal.framework (Apple GPU)
```

All buffer and program handles are opaque to Lean,
ensuring type safety and preventing misuse.
-/

namespace TinyGrad4.Backend.Metal

open TinyGrad4.Backend

/-! ## Opaque FFI Types -/

/-- Opaque Metal buffer handle -/
opaque MetalBufferImpl : NonemptyType
def MetalBuffer : Type := MetalBufferImpl.type
instance : Nonempty MetalBuffer := MetalBufferImpl.property

/-- Opaque Metal program handle -/
opaque MetalProgramImpl : NonemptyType
def MetalProgram : Type := MetalProgramImpl.type
instance : Nonempty MetalProgram := MetalProgramImpl.property

/-! ## FFI Declarations (float32-based, legacy API)
    These use FloatArray for backwards compatibility with existing benchmarks.
    New code should use the byte-based API with RawBuffer. -/

section LegacyFloatAPI
set_option linter.useRawBuffer false

/-- Allocate a Metal buffer for n float32 elements -/
@[extern "tg4_metal_alloc"]
opaque metalAlloc : @& Nat → IO MetalBuffer

/-- Free a Metal buffer -/
@[extern "tg4_metal_free"]
opaque metalFree : @& MetalBuffer → IO Unit

/-- Copy FloatArray to Metal buffer -/
@[extern "tg4_metal_copy_in"]
opaque metalCopyIn : @& MetalBuffer → @& FloatArray → IO Unit

/-- Copy Metal buffer to FloatArray -/
@[extern "tg4_metal_copy_out"]
opaque metalCopyOut : @& MetalBuffer → IO FloatArray

/-- Get buffer size in elements -/
@[extern "tg4_metal_size"]
opaque metalSize : @& MetalBuffer → IO Nat

/-- Compile Metal shader source to program -/
@[extern "tg4_metal_compile"]
opaque metalCompile : @& String → @& String → IO MetalProgram

/-- Launch Metal kernel with 3D grid and threadgroup -/
@[extern "tg4_metal_launch"]
opaque metalLaunch : @& MetalProgram → @& (Array MetalBuffer) →
    @& Nat → @& Nat → @& Nat → @& Nat → @& Nat → @& Nat → IO Unit

end LegacyFloatAPI

/-! ## FFI Declarations (byte-based, dtype-generic) -/

/-- Allocate a Metal buffer with n bytes -/
@[extern "tg4_metal_alloc_bytes"]
opaque metalAllocBytes : @& Nat → IO MetalBuffer

/-- Copy raw bytes from host ByteArray to Metal buffer -/
@[extern "tg4_metal_copy_in_bytes"]
opaque metalCopyInBytes : @& MetalBuffer → @& ByteArray → IO Unit

/-- Copy raw bytes from Metal buffer to host ByteArray -/
@[extern "tg4_metal_copy_out_bytes"]
opaque metalCopyOutBytes : @& MetalBuffer → @& Nat → IO ByteArray

/-- Launch a Metal kernel with 2D grid -/
@[extern "tg4_metal_launch_2d"]
opaque metalLaunch2D : @& MetalProgram → @& (Array MetalBuffer) →
    @& Nat → @& Nat → @& Nat → @& Nat → IO Unit

/-- Wait for all GPU work to complete -/
@[extern "tg4_metal_sync"]
opaque metalSync : IO Unit

/-- Get Metal device name -/
@[extern "tg4_metal_device_name"]
opaque metalDeviceName : IO String

/-! ## Pipeline Cache -/

/-- Pipeline cache statistics -/
structure CacheStats where
  hits : Nat
  misses : Nat
  size : Nat
  deriving Repr, Inhabited

/-- Get pipeline cache hit count -/
@[extern "tg4_metal_cache_hits"]
opaque metalCacheHits : IO Nat

/-- Get pipeline cache miss count -/
@[extern "tg4_metal_cache_misses"]
opaque metalCacheMisses : IO Nat

/-- Get pipeline cache size -/
@[extern "tg4_metal_cache_size"]
opaque metalCacheSize : IO Nat

/-- Clear the pipeline cache -/
@[extern "tg4_metal_cache_clear"]
opaque metalCacheClear : IO Unit

/-- Get pipeline cache statistics with named fields -/
def metalCacheStats : IO CacheStats := do
  let hits ← metalCacheHits
  let misses ← metalCacheMisses
  let size ← metalCacheSize
  return { hits, misses, size }

/-- Get cache hit rate (0.0 to 1.0) -/
def metalCacheHitRate : IO Float64 := do
  let hits ← metalCacheHits
  let misses ← metalCacheMisses
  let total := hits + misses
  if total == 0 then return 0.0
  else return hits.toFloat / total.toFloat

/-- Print cache statistics -/
def metalCachePrintStats : IO Unit := do
  let stats ← metalCacheStats
  let total := stats.hits + stats.misses
  let rate := if total == 0 then 0.0 else stats.hits.toFloat / total.toFloat * 100.0
  IO.println s!"Pipeline cache: {stats.hits} hits, {stats.misses} misses ({rate}% hit rate), {stats.size} cached"

/-! ## Synchronous GPU Matmul -/

/-- Execute matmul on GPU: C[m,n] = A[m,k] @ B[k,n]
    Takes float32 ByteArrays, returns float32 ByteArray.
    Falls back to zeros on GPU error.

    Note: Pure for compatibility with pure evaluator. -/
@[extern "tg4_metal_matmul_sync"]
opaque metalMatmulSync (a b : @& ByteArray) (m k n : @& Nat) : ByteArray

/-! ## Metal Renderer -/

/-- Metal renderer using MetalRenderer module -/
def metalRenderer : Renderer where
  name := "METAL"
  renderEwise := MetalRenderer.renderKernel
  renderEwiseVec := fun name nodes outId size width =>
    if width == .vec4 && size % 4 = 0 then
      MetalRenderer.renderEwiseVectorized name nodes outId size
    else
      MetalRenderer.renderKernel name nodes outId
  renderReduce name op _axes outer inner :=
    match MetalRenderer.opsToReduceOp op with
    | some reduceOp => MetalRenderer.renderReduceKernelAuto name reduceOp inner outer
    | none => s!"// Unsupported reduce op: {repr op}"
  renderMatmul m k n :=
    MetalRenderer.renderGemmKernelAuto "matmul" m k n

/-! ## High-Level API -/

/-- Check if Metal is available -/
def isAvailable : IO Bool := do
  try
    let _ ← metalDeviceName
    return true
  catch _ =>
    return false

/-- Get GPU device info -/
def deviceInfo : IO String := do
  let name ← metalDeviceName
  return s!"Metal Device: {name}"

/-! ## Zero-Copy Buffer Support -/

/-- Wrap ByteArray data in a Metal buffer without copying.
    Uses Apple Silicon unified memory - GPU reads directly from CPU memory.
    WARNING: The ByteArray MUST outlive the Metal buffer! -/
@[extern "tg4_metal_wrap_bytes_nocopy"]
private opaque metalWrapBytesNoCopyImpl : @& ByteArray → @& Nat → @& Nat → IO MetalBuffer

/-- Wrap ByteArray data in a Metal buffer without copying.
    Fails if no-copy allocation is not possible. -/
@[extern "tg4_metal_wrap_bytes_nocopy_strict"]
private opaque metalWrapBytesNoCopyStrictImpl : @& ByteArray → @& Nat → @& Nat → IO MetalBuffer

/-- Check if a ByteArray offset is page-aligned (required for zero-copy) -/
@[extern "tg4_metal_is_aligned"]
private opaque metalIsAlignedImpl : @& ByteArray → @& Nat → IO Bool

private def ensureFitsUSize (name : String) (n : Nat) : IO Unit := do
  if n.toUSize.toNat == n then
    pure ()
  else
    throw (IO.userError s!"{name}: {n} exceeds USize")

private def checkBounds (name : String) (size offset len : Nat) : IO Unit := do
  ensureFitsUSize s!"{name}.size" size
  ensureFitsUSize s!"{name}.offset" offset
  ensureFitsUSize s!"{name}.len" len
  if offset + len ≤ size then
    pure ()
  else
    throw (IO.userError s!"{name}: offset+len out of bounds (size={size}, offset={offset}, len={len})")

/-- Check if a ByteArray offset is page-aligned (required for zero-copy). -/
def metalIsAligned (ba : ByteArray) (offset : Nat := 0) : IO Bool := do
  ensureFitsUSize "metalIsAligned.offset" offset
  ensureFitsUSize "metalIsAligned.size" ba.size
  if offset ≤ ba.size then
    metalIsAlignedImpl ba offset
  else
    throw (IO.userError s!"metalIsAligned: offset {offset} out of bounds (size={ba.size})")

/-- Wrap ByteArray data in a Metal buffer without copying.
    Performs bounds checks before calling the FFI. -/
def metalWrapBytesNoCopy (ba : ByteArray) (offset len : Nat) : IO MetalBuffer := do
  checkBounds "metalWrapBytesNoCopy" ba.size offset len
  metalWrapBytesNoCopyImpl ba offset len

/-- Wrap ByteArray data in a Metal buffer without copying.
    Fails if no-copy allocation is not possible. -/
def metalWrapBytesNoCopyStrict (ba : ByteArray) (offset len : Nat) : IO MetalBuffer := do
  checkBounds "metalWrapBytesNoCopyStrict" ba.size offset len
  metalWrapBytesNoCopyStrictImpl ba offset len

/-- Create a zero-copy Metal buffer from ByteSlice.
    Falls back to copy if not page-aligned. -/
def metalFromByteSlice (s : TinyGrad4.Data.ByteSlice) : IO MetalBuffer :=
  metalWrapBytesNoCopy s.parent s.offset s.length

/-- Check if a ByteSlice is page-aligned (required for zero-copy). -/
def metalIsAlignedSlice (s : TinyGrad4.Data.ByteSlice) : IO Bool :=
  metalIsAligned s.parent s.offset

/-! ## Shared Memory Support (Multi-Process Data Loading) -/

/-- POSIX shared memory file descriptor -/
abbrev ShmFd := UInt32

/-- Create a new shared memory region with given name and size -/
@[extern "tg4_shm_create"]
private opaque shmCreateImpl : @& String → @& Nat → IO ShmFd

/-- Open an existing shared memory region -/
@[extern "tg4_shm_open"]
private opaque shmOpenImpl : @& String → IO ShmFd

/-- Map shared memory region to ByteArray (copies data) -/
@[extern "tg4_shm_map"]
private opaque shmMapImpl : @& ShmFd → @& Nat → IO ByteArray

/-- Write ByteArray to shared memory at offset -/
@[extern "tg4_shm_write"]
private opaque shmWriteImpl : @& ShmFd → @& ByteArray → @& Nat → IO Unit

/-- Read from shared memory into ByteArray -/
@[extern "tg4_shm_read"]
private opaque shmReadImpl : @& ShmFd → @& Nat → @& Nat → IO ByteArray

/-- Close shared memory file descriptor -/
@[extern "tg4_shm_close"]
private opaque shmCloseImpl : @& ShmFd → IO Unit

/-- Delete shared memory region -/
@[extern "tg4_shm_unlink"]
private opaque shmUnlinkImpl : @& String → IO Unit

/-- High-level shared memory region manager -/
structure SharedMemory where
  name : String
  fd : ShmFd
  size : Nat
  deriving Repr

namespace SharedMemory

/-- Create a new shared memory region -/
def create (name : String) (size : Nat) : IO SharedMemory := do
  ensureFitsUSize "SharedMemory.create.size" size
  let fd ← shmCreateImpl name size
  return { name, fd, size }

/-- Attach to an existing shared memory region -/
def attach (name : String) (size : Nat) : IO SharedMemory := do
  ensureFitsUSize "SharedMemory.attach.size" size
  let fd ← shmOpenImpl name
  return { name, fd, size }

/-- Write data to shared memory -/
def write (shm : SharedMemory) (data : ByteArray) (offset : Nat := 0) : IO Unit := do
  ensureFitsUSize "SharedMemory.write.offset" offset
  ensureFitsUSize "SharedMemory.write.size" data.size
  ensureFitsUSize "SharedMemory.write.shmSize" shm.size
  if offset + data.size ≤ shm.size then
    shmWriteImpl shm.fd data offset
  else
    throw (IO.userError s!"SharedMemory.write: offset+size out of bounds (size={shm.size}, offset={offset}, data={data.size})")

/-- Read data from shared memory -/
def read (shm : SharedMemory) (offset : Nat := 0) (len : Nat := shm.size) : IO ByteArray := do
  ensureFitsUSize "SharedMemory.read.offset" offset
  ensureFitsUSize "SharedMemory.read.len" len
  ensureFitsUSize "SharedMemory.read.shmSize" shm.size
  if offset + len ≤ shm.size then
    shmReadImpl shm.fd offset len
  else
    throw (IO.userError s!"SharedMemory.read: offset+len out of bounds (size={shm.size}, offset={offset}, len={len})")

/-- Close the shared memory handle -/
def close (shm : SharedMemory) : IO Unit :=
  shmCloseImpl shm.fd

/-- Delete the shared memory region -/
def unlink (shm : SharedMemory) : IO Unit :=
  shmUnlinkImpl shm.name

/-- Create a Metal buffer from shared memory (copies into device buffer). -/
def toMetalBuffer (shm : SharedMemory) : IO MetalBuffer := do
  let data ← shm.read
  let buf ← metalAllocBytes data.size
  metalCopyInBytes buf data
  pure buf

end SharedMemory

end TinyGrad4.Backend.Metal
