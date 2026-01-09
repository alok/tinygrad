import TinyGrad4.Backend.Metal
import TinyGrad4.Backend.Buffer

/-!
# Unified Device Buffer Abstraction

`DeviceBuffer` represents data that can live on either CPU or GPU.
The key insight: avoid copying until absolutely necessary.

## Lifecycle
1. **Creation**: Data starts where it's produced (GPU kernel → GPU, random init → CPU)
2. **Consumption**: Operations check if inputs are on-device, copy only if needed
3. **Materialization**: Copy to CPU only when user explicitly needs the data

## Buffer Locations
- `cpu`: Data lives in RawBuffer (ByteArray)
- `gpu`: Data lives on Metal GPU, referenced by opaque handle

## Design Principles
- Lazy evaluation: defer CPU copies until .toCPU is called
- Affinity tracking: remember where data was last used
- Reference counting: GPU buffers freed when no longer referenced
-/

namespace TinyGrad4.Backend.DeviceBuffer

open TinyGrad4
open TinyGrad4.Backend.Metal

/-! ## GPU Buffer ID System -/

/-- Unique identifier for a GPU-resident buffer -/
structure GPUBufferId where
  id : Nat
  deriving BEq, Hashable, Repr, Inhabited

/-- Metadata for a GPU buffer -/
structure GPUBufferEntry where
  buffer : MetalBuffer
  byteSize : Nat
  dtype : DType
  refCount : Nat := 1  -- Reference counting for shared buffers
  external : Bool := false  -- External buffers are not freed by the registry

/-- Global GPU buffer registry -/
initialize gpuRegistry : IO.Ref (Std.HashMap GPUBufferId GPUBufferEntry) ← IO.mkRef ∅

/-- Counter for generating unique buffer IDs -/
initialize gpuIdCounter : IO.Ref Nat ← IO.mkRef 0

/-! ## Device Location -/

/-- Where the authoritative copy of data lives -/
inductive DeviceLocation where
  | cpu      -- Data on CPU only
  | gpu      -- Data on GPU only
  | both     -- Data on both (GPU is authoritative)
  deriving BEq, Repr

/-! ## Device Buffer Type -/

/-- A buffer that can live on CPU, GPU, or both.
    The GPU copy is always authoritative when present. -/
structure DeviceBuffer where
  dtype : DType
  byteSize : Nat
  location : DeviceLocation
  /-- CPU data (populated if location is cpu or both) -/
  cpuData : Option ByteArray
  /-- GPU buffer ID (populated if location is gpu or both) -/
  gpuId : Option GPUBufferId
  deriving Repr

namespace DeviceBuffer

/-! ## GPU Registry Operations -/

/-- Allocate a new GPU buffer in the registry -/
def allocGPU (byteSize : Nat) (dtype : DType) : IO GPUBufferId := do
  let buf ← metalAllocBytes byteSize
  let id ← gpuIdCounter.modifyGet fun n => (⟨n⟩, n + 1)
  gpuRegistry.modify (·.insert id { buffer := buf, byteSize, dtype, external := false })
  return id

/-- Register an externally-owned Metal buffer (will not be freed by registry). -/
def registerExternal (buf : MetalBuffer) (byteSize : Nat) (dtype : DType) : IO GPUBufferId := do
  let id ← gpuIdCounter.modifyGet fun n => (⟨n⟩, n + 1)
  gpuRegistry.modify (·.insert id { buffer := buf, byteSize, dtype, external := true })
  return id

/-- Get Metal buffer handle from ID -/
def getGPUBuffer (id : GPUBufferId) : IO MetalBuffer := do
  let registry ← gpuRegistry.get
  match registry[id]? with
  | some entry => return entry.buffer
  | none => throw (IO.userError s!"GPU buffer {repr id} not found in registry")

/-- Free a GPU buffer (decrements ref count, frees when 0) -/
def freeGPU (id : GPUBufferId) : IO Unit := do
  let registry ← gpuRegistry.get
  match registry[id]? with
  | some entry =>
    if entry.refCount <= 1 then
      if !entry.external then
        metalFree entry.buffer
      gpuRegistry.modify (·.erase id)
    else
      gpuRegistry.modify (·.insert id { entry with refCount := entry.refCount - 1 })
  | none => pure ()

/-- Increment reference count for a GPU buffer -/
def retainGPU (id : GPUBufferId) : IO Unit := do
  let registry ← gpuRegistry.get
  if let some entry := registry[id]? then
    gpuRegistry.modify (·.insert id { entry with refCount := entry.refCount + 1 })

/-! ## Constructors -/

/-- Create a CPU-only buffer from RawBuffer -/
def fromCPU (raw : RawBuffer) : DeviceBuffer :=
  { dtype := raw.dtype
    byteSize := raw.data.size
    location := .cpu
    cpuData := some raw.data
    gpuId := none }

/-- Create a GPU-only buffer (data already on GPU) -/
def fromGPU (id : GPUBufferId) (dtype : DType) (byteSize : Nat) : DeviceBuffer :=
  { dtype, byteSize
    location := .gpu
    cpuData := none
    gpuId := some id }

/-- Upload CPU data to GPU, creating a GPU-resident buffer -/
def uploadToGPU (raw : RawBuffer) : IO DeviceBuffer := do
  let id ← allocGPU raw.data.size raw.dtype
  let buf ← getGPUBuffer id
  metalCopyInBytes buf raw.data
  return { dtype := raw.dtype
           byteSize := raw.data.size
           location := .gpu
           cpuData := none
           gpuId := some id }

/-- Allocate an uninitialized GPU buffer -/
def allocOnGPU (byteSize : Nat) (dtype : DType := .float32) : IO DeviceBuffer := do
  let id ← allocGPU byteSize dtype
  return { dtype, byteSize
           location := .gpu
           cpuData := none
           gpuId := some id }

/-! ## Access Methods -/

/-- Ensure data is on CPU, copying from GPU if needed -/
def toCPU (b : DeviceBuffer) : IO RawBuffer := do
  match b.location with
  | .cpu =>
    match b.cpuData with
    | some data => return { dtype := b.dtype, data }
    | none => throw (IO.userError "CPU buffer missing data")
  | .gpu | .both =>
    match b.gpuId with
    | some id =>
      let buf ← getGPUBuffer id
      let data ← metalCopyOutBytes buf b.byteSize
      return { dtype := b.dtype, data }
    | none => throw (IO.userError "GPU buffer missing ID")

/-- Ensure data is on GPU, uploading from CPU if needed.
    Returns the GPU buffer ID. -/
def toGPU (b : DeviceBuffer) : IO (DeviceBuffer × GPUBufferId) := do
  match b.location with
  | .gpu | .both =>
    match b.gpuId with
    | some id => return (b, id)
    | none => throw (IO.userError "GPU buffer missing ID")
  | .cpu =>
    match b.cpuData with
    | some data =>
      let id ← allocGPU b.byteSize b.dtype
      let buf ← getGPUBuffer id
      metalCopyInBytes buf data
      let newBuf := { b with location := .both, gpuId := some id }
      return (newBuf, id)
    | none => throw (IO.userError "CPU buffer missing data")

/-- Get GPU buffer ID if on GPU, or upload if on CPU -/
def ensureGPU (b : DeviceBuffer) : IO GPUBufferId := do
  let (_, id) ← b.toGPU
  return id

/-- Check if buffer is on GPU -/
def isOnGPU (b : DeviceBuffer) : Bool :=
  b.location == .gpu || b.location == .both

/-- Get numel (number of elements) -/
def numel (b : DeviceBuffer) : Nat :=
  b.byteSize / b.dtype.itemsize

/-! ## Lifecycle Management -/

/-- Release GPU resources (call when buffer is no longer needed) -/
def release (b : DeviceBuffer) : IO Unit := do
  if let some id := b.gpuId then
    freeGPU id

end DeviceBuffer

/-! ## Kernel Execution with Device Buffers -/

/-- Execute a kernel that takes DeviceBuffer inputs and produces DeviceBuffer output.
    Inputs are uploaded to GPU if needed; output stays on GPU. -/
def runKernelOnDevice (inputs : Array DeviceBuffer) (outputSize : Nat)
    (outputDtype : DType) (kernel : Array MetalBuffer → MetalBuffer → IO Unit)
    : IO DeviceBuffer := do
  -- Ensure all inputs are on GPU
  let mut gpuBufs : Array MetalBuffer := #[]
  for input in inputs do
    let id ← input.ensureGPU
    gpuBufs := gpuBufs.push (← DeviceBuffer.getGPUBuffer id)

  -- Allocate output on GPU
  let outId ← DeviceBuffer.allocGPU outputSize outputDtype
  let outBuf ← DeviceBuffer.getGPUBuffer outId

  -- Run kernel
  kernel gpuBufs outBuf

  return DeviceBuffer.fromGPU outId outputDtype outputSize

end TinyGrad4.Backend.DeviceBuffer
