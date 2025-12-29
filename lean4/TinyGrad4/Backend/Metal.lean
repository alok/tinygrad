import TinyGrad4.Backend.Device
import TinyGrad4.Backend.MetalRenderer

/-!
# Metal Backend

Type-safe Metal GPU backend via FFI to Objective-C.

## Architecture

```
Metal.lean (Lean typeclasses)
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

/-! ## FFI Declarations -/

/-- Allocate a Metal buffer with n float elements -/
@[extern "tg4_metal_alloc"]
opaque metalAlloc : @& Nat → IO MetalBuffer

/-- Free a Metal buffer -/
@[extern "tg4_metal_free"]
opaque metalFree : @& MetalBuffer → IO Unit

/-- Copy data from host FloatArray to Metal buffer -/
@[extern "tg4_metal_copy_in"]
opaque metalCopyIn : @& MetalBuffer → @& FloatArray → IO Unit

/-- Copy data from Metal buffer to host FloatArray -/
@[extern "tg4_metal_copy_out"]
opaque metalCopyOut : @& MetalBuffer → IO FloatArray

/-- Get buffer size in elements -/
@[extern "tg4_metal_size"]
opaque metalSize : @& MetalBuffer → IO Nat

/-- Compile Metal shader source to executable program -/
@[extern "tg4_metal_compile"]
opaque metalCompile : @& String → @& String → IO MetalProgram

/-- Launch a Metal kernel -/
@[extern "tg4_metal_launch"]
opaque metalLaunch : @& MetalProgram → @& (Array MetalBuffer) →
    @& Nat → @& Nat → @& Nat → @& Nat → @& Nat → @& Nat → IO Unit

/-- Wait for all GPU work to complete -/
@[extern "tg4_metal_sync"]
opaque metalSync : IO Unit

/-- Get Metal device name -/
@[extern "tg4_metal_device_name"]
opaque metalDeviceName : IO String

/-! ## Byte-Based FFI (dtype-generic) -/

/-- Allocate a Metal buffer with n bytes -/
@[extern "tg4_metal_alloc_bytes"]
opaque metalAllocBytes : @& Nat → IO MetalBuffer

/-- Copy raw bytes from host ByteArray to Metal buffer (no conversion) -/
@[extern "tg4_metal_copy_in_bytes"]
opaque metalCopyInBytes : @& MetalBuffer → @& ByteArray → IO Unit

/-- Copy raw bytes from Metal buffer to host ByteArray (no conversion) -/
@[extern "tg4_metal_copy_out_bytes"]
opaque metalCopyOutBytes : @& MetalBuffer → @& Nat → IO ByteArray

/-- Launch a Metal kernel with 2D grid (for matmul) -/
@[extern "tg4_metal_launch_2d"]
opaque metalLaunch2D : @& MetalProgram → @& (Array MetalBuffer) →
    @& Nat → @& Nat → @& Nat → @& Nat → IO Unit
    -- gridX, gridY, tgSizeX, tgSizeY

/-! ## Synchronous GPU Matmul (for pure evaluator) -/

/-- Execute matmul on GPU synchronously: C[m,n] = A[m,k] @ B[k,n]
    Takes float32 ByteArrays, returns float32 ByteArray.
    Compiles shader, launches kernel, waits for completion.
    Falls back to zeros on GPU error.

    Note: Declared as pure for compatibility with pure evaluator.
    GPU side effects are invisible to Lean (referentially transparent). -/
@[extern "tg4_metal_matmul_sync"]
opaque metalMatmulSync (a b : @& ByteArray) (m k n : @& Nat) : ByteArray

/-! ## Typeclass Instances -/

instance : Allocator MetalBuffer where
  alloc := metalAlloc
  free := metalFree
  copyIn := metalCopyIn
  copyOut := metalCopyOut
  size _buf := 0  -- TODO: track size in wrapper struct

instance : Compiler MetalProgram where
  compile := metalCompile

instance : Runtime MetalProgram MetalBuffer where
  launch prog bufs params := do
    let (gx, gy, gz) := params.globalSize
    let (lx, ly, lz) := params.localSize
    metalLaunch prog bufs gx gy gz lx ly lz
  sync := metalSync

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

/-- Complete Metal device configuration -/
def metalDevice : Device MetalProgram MetalBuffer := {
  name := "METAL"
  allocator := inferInstance
  compiler := inferInstance
  runtime := inferInstance
  renderer := metalRenderer
}

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

/-- Simple test: allocate, write, read back -/
def testRoundtrip (data : FloatArray) : IO FloatArray := do
  let buf ← metalAlloc data.size
  metalCopyIn buf data
  let result ← metalCopyOut buf
  metalFree buf
  return result

/-- Compile and run a simple kernel -/
def runKernel (source : String) (kernelName : String)
    (inputs : Array FloatArray) (outputSize : Nat) : IO FloatArray := do
  -- Allocate buffers
  let mut bufs : Array MetalBuffer := #[]
  for input in inputs do
    let buf ← metalAlloc input.size
    metalCopyIn buf input
    bufs := bufs.push buf

  -- Output buffer
  let outBuf ← metalAlloc outputSize
  bufs := bufs.push outBuf

  -- Compile and launch
  let prog ← metalCompile kernelName source
  metalLaunch prog bufs outputSize 1 1 256 1 1
  metalSync

  -- Copy result
  let result ← metalCopyOut outBuf

  -- Free
  for buf in bufs do
    metalFree buf

  return result

end TinyGrad4.Backend.Metal
