import Std
import TinyGrad4.Backend.Device

/-!
# CUDA Backend

Type-safe CUDA GPU backend via FFI to CUDA Driver API.

## Architecture

```
Cuda.lean (Lean FFI declarations)
    ↓
tg4_cuda.cu (C/CUDA FFI)
    ↓
CUDA Driver API + NVRTC
    ↓
NVIDIA GPU
```

All buffer and program handles are opaque to Lean,
ensuring type safety and preventing misuse.
-/

namespace TinyGrad4.Backend.Cuda

open TinyGrad4.Backend

/-! ## Opaque FFI Types -/

/-- Opaque CUDA buffer handle -/
opaque CUDABufferImpl : NonemptyType
def CUDABuffer : Type := CUDABufferImpl.type
instance : Nonempty CUDABuffer := CUDABufferImpl.property

/-- Opaque CUDA program handle -/
opaque CUDAProgramImpl : NonemptyType
def CUDAProgram : Type := CUDAProgramImpl.type
instance : Nonempty CUDAProgram := CUDAProgramImpl.property

/-! ## FFI Declarations (byte-based, dtype-generic) -/

/-- Allocate a CUDA buffer with n bytes -/
@[extern "tg4_cuda_alloc_bytes"]
opaque cudaAllocBytes : @& Nat → IO CUDABuffer

/-- Free a CUDA buffer -/
@[extern "tg4_cuda_free"]
opaque cudaFree : @& CUDABuffer → IO Unit

/-- Copy raw bytes from host ByteArray to CUDA buffer -/
@[extern "tg4_cuda_copy_in_bytes"]
opaque cudaCopyInBytes : @& CUDABuffer → @& ByteArray → IO Unit

/-- Copy raw bytes from CUDA buffer to host ByteArray -/
@[extern "tg4_cuda_copy_out_bytes"]
opaque cudaCopyOutBytes : @& CUDABuffer → @& Nat → IO ByteArray

/-- Compile CUDA kernel source to executable program via NVRTC -/
@[extern "tg4_cuda_compile"]
opaque cudaCompile : @& String → @& String → IO CUDAProgram

/-- Launch a CUDA kernel with 2D grid -/
@[extern "tg4_cuda_launch_2d"]
opaque cudaLaunch2D : @& CUDAProgram → @& (Array CUDABuffer) →
    @& Nat → @& Nat → @& Nat → @& Nat → IO Unit

/-- Wait for all GPU work to complete -/
@[extern "tg4_cuda_sync"]
opaque cudaSync : IO Unit

/-- Get CUDA device name -/
@[extern "tg4_cuda_device_name"]
opaque cudaDeviceName : IO String

/-- Get CUDA device count -/
@[extern "tg4_cuda_device_count"]
opaque cudaDeviceCount : IO Nat

/-- Set CUDA device for the current thread -/
@[extern "tg4_cuda_set_device"]
opaque cudaSetDevice : @& Nat → IO Unit

/-! ## Synchronous GPU Matmul -/

/-- Execute matmul on GPU: {lit}`C[m,n] = A[m,k] @ B[k,n]`
    Takes float32 ByteArrays, returns float32 ByteArray.
    Falls back to zeros on GPU error.

    Note: Pure for compatibility with pure evaluator. -/
@[extern "tg4_cuda_matmul_sync"]
opaque cudaMatmulSync (a b : @& ByteArray) (m k n : @& Nat) : ByteArray

/-! ## CUDA Kernel Renderer -/

/-- Render a simple ewise CUDA kernel (simplified for now) -/
def renderCUDAKernel (name : String) (_nodes : List Unit) (_outId : Nat) : String :=
  "extern \"C\" __global__ void " ++ name ++ "(\n" ++
  "    const float* __restrict__ a,\n" ++
  "    const float* __restrict__ b,\n" ++
  "    float* __restrict__ out\n" ++
  ") {\n" ++
  "    int gid = blockIdx.x * blockDim.x + threadIdx.x;\n" ++
  "    out[gid] = a[gid] + b[gid];\n" ++
  "}\n"

/-! ## CUDA Renderer -/

/-- CUDA renderer for ewise kernels -/
def cudaRenderer : Renderer where
  name := "CUDA"
  renderEwise name nodes outId := some (renderCUDAKernel name (nodes.map fun _ => ()) outId.id)
  renderReduce name _op axes _outer _inner :=
    s!"// TODO: CUDA reduce kernel: {name}, axes={repr axes}"
  renderMatmul m k n :=
    s!"// TODO: CUDA matmul [{m}x{k}] @ [{k}x{n}]"

/-! ## High-Level API -/

/-- Fallback device count for environments without CUDA FFI. -/
private def fallbackDeviceCount : IO Nat := do
  let env ← IO.getEnv "CUDA_VISIBLE_DEVICES"
  match env with
  | some raw =>
      let s := raw.trimAscii.toString
      if s.isEmpty || s == "none" || s == "void" || s == "NoDevFiles" || s == "-1" then
        pure 0
      else
        let parts := s.splitOn "," |>.filter (fun p => !(p.trimAscii.toString.isEmpty))
        pure parts.length
  | none =>
      let mut count := 0
      try
        for entry in (← (System.FilePath.mk "/dev").readDir) do
          let name : String := entry.fileName
          if name.startsWith "nvidia" then
            let suffix : String := (name.drop 6).toString
            if !suffix.isEmpty && suffix.all Char.isDigit then
              count := count + 1
        pure count
      catch _ =>
        pure 0

/-- Check if CUDA is available -/
def isAvailable : IO Bool := do
  try
    let n ← cudaDeviceCount
    return decide (n > 0)
  catch _ =>
    let n ← fallbackDeviceCount
    return decide (n > 0)

/-- Get CUDA device count (0 if unavailable). -/
def deviceCount : IO Nat := do
  try
    cudaDeviceCount
  catch _ =>
    fallbackDeviceCount

/-- Set CUDA device for this thread. -/
def setDevice (idx : Nat) : IO Unit := do
  cudaSetDevice idx

/-- Get GPU device info -/
def deviceInfo : IO String := do
  let name ← cudaDeviceName
  return s!"CUDA Device: {name}"

/-- Get GPU device info for a specific device index. -/
def deviceInfoFor (idx : Nat) : IO String := do
  try
    cudaSetDevice idx
    let name ← cudaDeviceName
    return s!"CUDA Device: {name}"
  catch _ =>
    deviceInfo

end TinyGrad4.Backend.Cuda
