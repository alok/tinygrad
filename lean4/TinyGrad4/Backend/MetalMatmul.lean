import TinyGrad4.Backend.Metal
import TinyGrad4.Backend.MetalRenderer
import TinyGrad4.Backend.Buffer

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

open TinyGrad4.Backend.Metal
open TinyGrad4.Backend.MetalRenderer
open TinyGrad4.Backend

/-! ## GPU Availability -/

/-- Check if Metal GPU is available for matmul -/
def isAvailable : IO Bool := Metal.isAvailable

/-- Cached availability check (avoids repeated FFI calls) -/
initialize metalAvailable : IO.Ref Bool ← do
  let available ← Metal.isAvailable
  IO.mkRef available

def checkAvailable : IO Bool := metalAvailable.get

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

  -- Calculate grid dimensions for tiled GEMM
  -- Uses 32x32 tiles with 8x8 threads per tile
  let tileM := 32
  let tileN := 32
  let gridX := (n + tileM - 1) / tileM
  let gridY := (m + tileN - 1) / tileN

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
    return RawBuffer.zeros .float32 0

  -- Allocate output buffer
  let outputSize := numBatches * m * n
  let mut outputBytes := ByteArray.mkEmpty (outputSize * 4)

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
  { dtype := .float32, data := Native.matmulF32 a.data b.data m k n }

def matmulBatchedCPU (a : RawBuffer) (b : RawBuffer) (m k n : Nat)
    (aStarts bStarts : Array Nat) : RawBuffer :=
  { dtype := .float32, data := Native.matmulBatchedF32 a.data b.data aStarts bStarts m k n }

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

end TinyGrad4.Backend.MetalMatmul
