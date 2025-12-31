import TinyGrad4.Backend.Metal
import TinyGrad4.Backend.MetalRenderer
import TinyGrad4.Backend.Buffer
import TinyGrad4.Backend.Native
import TinyGrad4.Backend.DeviceBuffer

/-!
# GPU Matmul Execution via Metal

Executes matrix multiplication on Metal GPU, falling back to CPU when unavailable.

## Architecture

```
Interpreter CONTRACT op
    ↓
MetalMatmul.matmul2D / matmulBatched
    ↓
Metal FFI (alloc, compile, launch, sync)
    ↓
Metal GPU execution
```

## Usage

Called from Interpreter.lean when processing CONTRACT operations.
GPU is the default when Metal is available.
-/

namespace TinyGrad4.Backend.MetalMatmul

open TinyGrad4
open TinyGrad4.Backend.Metal
open TinyGrad4.Backend.MetalRenderer
open TinyGrad4.Backend
open TinyGrad4.Backend.DeviceBuffer (GPUBufferId DeviceBuffer)
open TinyGrad4.Backend.DeviceBuffer.DeviceBuffer

/-! ## Local Helpers -/

/-- Create zero-filled RawBuffer (local version to avoid circular dep with Interpreter) -/
private def zerosRaw (dtype : DType) (numel : Nat) : RawBuffer :=
  { dtype, data := ByteArray.mk (Array.replicate (numel * dtype.itemsize) 0) }

/-! ## GPU Availability -/

/-- Check if Metal GPU is available for matmul -/
def isAvailable : IO Bool := Metal.isAvailable

/-- Cached availability (lazily initialized on first check) -/
initialize metalAvailableCache : IO.Ref (Option Bool) ← IO.mkRef none

/-- Check availability with caching (avoids repeated FFI calls) -/
def checkAvailable : IO Bool := do
  match ← metalAvailableCache.get with
  | some v => return v
  | none =>
    let available ← Metal.isAvailable
    metalAvailableCache.set (some available)
    return available

/-! ## 2D Matrix Multiplication -/

/-- Execute 2D matmul on GPU: C[m,n] = A[m,k] @ B[k,n]

    Input buffers must be float32 RawBuffers.
    Returns float32 RawBuffer with result.

    GPU execution:
    1. Allocate Metal buffers
    2. Copy input bytes to GPU
    3. Compile and launch GEMM kernel
    4. Copy result back
    5. Free GPU buffers
-/
def matmul2D (a : RawBuffer) (b : RawBuffer) (m k n : Nat) : IO RawBuffer := do
  -- Generate tiled GEMM shader
  let shader := renderGemmKernelAuto "matmul" m k n

  -- Calculate byte sizes (float32 = 4 bytes)
  let aBytes := m * k * 4
  let bBytes := k * n * 4
  let cBytes := m * n * 4

  -- Allocate Metal buffers
  let aBuf ← metalAllocBytes aBytes
  let bBuf ← metalAllocBytes bBytes
  let cBuf ← metalAllocBytes cBytes

  -- Copy input bytes to GPU (no conversion needed for float32)
  metalCopyInBytes aBuf a.data
  metalCopyInBytes bBuf b.data

  -- Compile shader
  let prog ← metalCompile "matmul" shader

  -- Calculate grid dimensions based on kernel type
  -- Small matrices (≤64): simple kernel, 1 thread per element
  -- Large matrices: tiled kernel, 8x8 threads compute 32x32 tile
  let (gridX, gridY) :=
    if m ≤ 64 && k ≤ 64 && n ≤ 64 then
      -- Simple kernel: each thread computes 1 element
      -- Grid of threadgroups, each 8x8 threads
      ((n + 7) / 8, (m + 7) / 8)
    else
      -- Tiled kernel: 8x8 threads compute 32x32 tile
      let tileM := 32
      let tileN := 32
      ((n + tileN - 1) / tileN, (m + tileM - 1) / tileM)

  -- Launch kernel
  metalLaunch2D prog #[aBuf, bBuf, cBuf] gridX gridY 8 8
  metalSync

  -- Copy result back
  let result ← metalCopyOutBytes cBuf cBytes

  -- Free GPU buffers
  metalFree aBuf
  metalFree bBuf
  metalFree cBuf

  return { dtype := .float32, data := result }

/-! ## Batched Matrix Multiplication -/

/-- Execute batched matmul on GPU via host loop over batches.

    For now, uses simple host-side loop calling matmul2D for each batch.
    Future optimization: single kernel with 3D dispatch.

    Input: A with shape [..., m, k], B with shape [..., k, n]
    Output: C with shape [..., m, n]
-/
def matmulBatched (a : RawBuffer) (b : RawBuffer) (m k n : Nat)
    (aStarts bStarts : Array Nat) : IO RawBuffer := do
  let numBatches := aStarts.size
  if numBatches == 0 then
    return zerosRaw .float32 0

  -- Allocate output buffer
  let outputSize := numBatches * m * n
  let mut outputBytes := ByteArray.emptyWithCapacity (outputSize * 4)

  -- For small batch counts, use host loop
  -- For large batches, could use batched kernel with 3D dispatch
  for i in [:numBatches] do
    let aStart := aStarts[i]!
    let bStart := bStarts[i]!

    -- Extract batch slices
    let aSlice := a.data.extract (aStart * 4) ((aStart + m * k) * 4)
    let bSlice := b.data.extract (bStart * 4) ((bStart + k * n) * 4)

    let aBuffer : RawBuffer := { dtype := .float32, data := aSlice }
    let bBuffer : RawBuffer := { dtype := .float32, data := bSlice }

    -- GPU matmul for this batch
    let result ← matmul2D aBuffer bBuffer m k n

    -- Append to output
    outputBytes := outputBytes ++ result.data

  return { dtype := .float32, data := outputBytes }

/-! ## Fallback to CPU -/

/-- CPU fallback for matmul when Metal unavailable -/
def matmul2DCPU (a : RawBuffer) (b : RawBuffer) (m k n : Nat) : RawBuffer :=
  { dtype := .float32, data := TinyGrad4.Backend.Native.matmulF32 a.data b.data m k n }

def matmulBatchedCPU (a : RawBuffer) (b : RawBuffer) (m k n : Nat)
    (aStarts bStarts : Array Nat) : RawBuffer :=
  { dtype := .float32, data := TinyGrad4.Backend.Native.matmulBatchedF32 a.data b.data aStarts bStarts m k n }

/-! ## Unified Entry Points (GPU with CPU fallback) -/

/-- Execute 2D matmul, preferring GPU when available -/
def runMatmul2D (a : RawBuffer) (b : RawBuffer) (m k n : Nat) : IO RawBuffer := do
  let available ← checkAvailable
  if available then
    try
      matmul2D a b m k n
    catch _ =>
      -- GPU failed, fall back to CPU
      return matmul2DCPU a b m k n
  else
    return matmul2DCPU a b m k n

/-- Execute batched matmul, preferring GPU when available -/
def runMatmulBatched (a : RawBuffer) (b : RawBuffer) (m k n : Nat)
    (aStarts bStarts : Array Nat) : IO RawBuffer := do
  let available ← checkAvailable
  if available then
    try
      matmulBatched a b m k n aStarts bStarts
    catch _ =>
      -- GPU failed, fall back to CPU
      return matmulBatchedCPU a b m k n aStarts bStarts
  else
    return matmulBatchedCPU a b m k n aStarts bStarts

/-! ## GPU-Resident Matmul (DeviceBuffer API) -/

/-- Execute 2D matmul on GPU with DeviceBuffer inputs/output.
    Avoids CPU-GPU copies when inputs are already on GPU.
    Result stays on GPU until explicitly downloaded.

    C[m,n] = A[m,k] @ B[k,n]
-/
def matmul2DDevice (a : DeviceBuffer) (b : DeviceBuffer) (m k n : Nat) : IO DeviceBuffer := do
  -- Ensure inputs are on GPU
  let aGpuId ← ensureGPU a
  let bGpuId ← ensureGPU b
  let aBuf ← getGPUBuffer aGpuId
  let bBuf ← getGPUBuffer bGpuId

  -- Allocate output on GPU
  let cBytes := m * n * 4
  let cGpuId ← allocGPU cBytes .float32
  let cBuf ← getGPUBuffer cGpuId

  -- Generate and compile shader
  let shader := renderGemmKernelAuto "matmul_device" m k n
  let prog ← metalCompile "matmul_device" shader

  -- Calculate grid dimensions for tiled GEMM
  let tileM := 32
  let tileN := 32
  let gridX := (n + tileM - 1) / tileM
  let gridY := (m + tileN - 1) / tileN

  -- Launch kernel
  metalLaunch2D prog #[aBuf, bBuf, cBuf] gridX gridY 8 8
  metalSync

  return fromGPU cGpuId .float32 cBytes

/-- Execute batched matmul on GPU with DeviceBuffer API.
    Uses host loop over batches with GPU-resident intermediates.
-/
def matmulBatchedDevice (a : DeviceBuffer) (b : DeviceBuffer) (m k n : Nat)
    (aStarts bStarts : Array Nat) : IO DeviceBuffer := do
  let numBatches := aStarts.size
  if numBatches == 0 then
    return fromCPU (zerosRaw .float32 0)

  -- For now, fall back to RawBuffer implementation and re-upload result
  -- TODO: Implement true batched GPU kernel with 3D dispatch
  let aCPU ← toCPU a
  let bCPU ← toCPU b
  let result ← matmulBatched aCPU bCPU m k n aStarts bStarts
  uploadToGPU result

/-- Run matmul with DeviceBuffer API, preferring GPU when available -/
def runMatmul2DDevice (a : DeviceBuffer) (b : DeviceBuffer) (m k n : Nat) : IO DeviceBuffer := do
  let available ← checkAvailable
  if available && a.isOnGPU && b.isOnGPU then
    -- Both inputs on GPU, use GPU-native path
    try
      matmul2DDevice a b m k n
    catch _ =>
      -- GPU failed, fall back to CPU
      let aCPU ← toCPU a
      let bCPU ← toCPU b
      return fromCPU (matmul2DCPU aCPU bCPU m k n)
  else if available then
    -- At least one input on CPU, upload and run on GPU
    try
      matmul2DDevice a b m k n
    catch _ =>
      let aCPU ← toCPU a
      let bCPU ← toCPU b
      return fromCPU (matmul2DCPU aCPU bCPU m k n)
  else
    -- No GPU, use CPU
    let aCPU ← toCPU a
    let bCPU ← toCPU b
    return fromCPU (matmul2DCPU aCPU bCPU m k n)

/-! ## Async API (for pipelining) -/

/-- Async result type for GPU matmul -/
abbrev MatmulTask := Task (Except IO.Error RawBuffer)

/-- Spawn async 2D matmul on GPU -/
def matmul2DAsync (a : RawBuffer) (b : RawBuffer) (m k n : Nat) : IO MatmulTask :=
  IO.asTask (runMatmul2D a b m k n)

/-- Spawn async batched matmul on GPU -/
def matmulBatchedAsync (a : RawBuffer) (b : RawBuffer) (m k n : Nat)
    (aStarts bStarts : Array Nat) : IO MatmulTask :=
  IO.asTask (runMatmulBatched a b m k n aStarts bStarts)

/-- Await matmul result, falling back to zeros on error -/
def awaitMatmul (task : MatmulTask) (dtype : DType) (numel : Nat) : IO RawBuffer := do
  match task.get with
  | .ok buf => return buf
  | .error _ => return zerosRaw dtype numel

/-- Run multiple matmuls in parallel and collect results -/
def matmul2DParallel (ops : Array (RawBuffer × RawBuffer × Nat × Nat × Nat)) : IO (Array RawBuffer) := do
  -- Spawn all tasks with their shapes
  let tasksWithShape ← ops.mapM fun (a, b, m, k, n) => do
    let task ← matmul2DAsync a b m k n
    pure (task, m, n)
  -- Await all results
  tasksWithShape.mapM fun (task, m, n) => awaitMatmul task .float32 (m * n)

end TinyGrad4.Backend.MetalMatmul
