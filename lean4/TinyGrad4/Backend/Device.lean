import TinyGrad4.Ops
import TinyGrad4.DType
import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import TinyGrad4.Backend.Buffer

/-!
# Device Abstraction Layer

Principled backend architecture with typeclass-based pluggability.

## Architecture

```
Tensor API → Scheduler → CodeGen → Renderer → Compiler → Runtime
                                                 ↓
                                            Allocator
```

Each layer has a single responsibility and clean interface.
Backends are swapped at compile time via typeclass resolution.
-/

namespace TinyGrad4.Backend

/-! ## Core Types -/

/-- Kernel launch parameters -/
structure LaunchParams where
  globalSize : Nat × Nat × Nat := (1, 1, 1)
  localSize : Nat × Nat × Nat := (1, 1, 1)
  deriving Repr, Inhabited

/-- A compiled kernel ready for execution -/
structure CompiledKernel where
  name : String
  source : String
  launchParams : LaunchParams
  bufferCount : Nat
  deriving Repr

/-! ## Allocator Typeclass -/

/-- Memory allocator interface for GPU buffers (byte-based, dtype-generic) -/
class Allocator (Buf : Type) where
  /-- Allocate a buffer of n bytes -/
  allocBytes : Nat → IO Buf
  /-- Free a buffer -/
  free : Buf → IO Unit
  /-- Copy raw bytes from host to device -/
  copyInBytes : Buf → ByteArray → IO Unit
  /-- Copy raw bytes from device to host -/
  copyOutBytes : Buf → Nat → IO ByteArray
  /-- Get buffer size in bytes -/
  sizeBytes : Buf → Nat

/-! ## Compiler Typeclass -/

/-- Source code to binary compiler interface -/
class Compiler (Prog : Type) where
  /-- Compile shader source to executable program -/
  compile : String → String → IO Prog  -- (name, source) → handle

/-! ## Runtime Typeclass -/

/-- Kernel execution runtime interface -/
class Runtime (Prog Buf : Type) where
  /-- Launch a compiled program with buffers -/
  launch : Prog → Array Buf → LaunchParams → IO Unit
  /-- Wait for all GPU work to complete -/
  sync : IO Unit := pure ()

/-! ## Renderer Typeclass -/

/-- Vector width for SIMD operations -/
inductive VecWidth where
  | scalar : VecWidth
  | vec2 : VecWidth
  | vec4 : VecWidth
  | vec8 : VecWidth
  deriving Repr, DecidableEq, Inhabited

def VecWidth.toNat : VecWidth → Nat
  | .scalar => 1
  | .vec2 => 2
  | .vec4 => 4
  | .vec8 => 8

/-- UOp graph to source code renderer -/
class Renderer where
  /-- Target name (e.g., "METAL", "CUDA") -/
  name : String
  /-- Render an elementwise kernel (scalar) -/
  renderEwise : String → List UOp → UOpId → Option String
  /-- Render a vectorized elementwise kernel -/
  renderEwiseVec : String → List UOp → UOpId → Nat → VecWidth → Option String := fun _ _ _ _ _ => none
  /-- Render a reduction kernel -/
  renderReduce : String → Ops → List Nat → Nat → Nat → String
  /-- Render a matmul kernel -/
  renderMatmul : Nat → Nat → Nat → String

/-! ## Device Bundle -/

/-- Device bundles all backend components together -/
structure Device (Prog Buf : Type) where
  name : String
  allocator : Allocator Buf
  compiler : Compiler Prog
  runtime : Runtime Prog Buf
  renderer : Renderer

/-! ## Execution Plan -/

/-- A kernel to be executed with its buffer bindings -/
structure KernelExec where
  kernel : CompiledKernel
  /-- Buffer indices for each parameter -/
  bufferBindings : Array Nat
  deriving Repr

/-- Complete execution plan (byte-based) -/
structure ExecPlan where
  /-- Buffer sizes in bytes -/
  bufferSizes : Array Nat
  /-- Input buffer indices and their data -/
  inputs : Array (Nat × ByteArray)
  /-- Kernels to execute in order -/
  kernels : Array KernelExec
  /-- Output buffer index -/
  outputIdx : Nat

/-! ## CPU Backend (Reference Implementation) -/

/-- CPU buffer is just a ByteArray -/
structure CPUBuffer where
  data : ByteArray

/-- CPU "program" is just source code (interpreted) -/
structure CPUProgram where
  name : String
  source : String
  deriving Repr

instance : Allocator CPUBuffer where
  allocBytes n := pure { data := ByteArray.mk (Array.replicate n 0) }
  free _ := pure ()
  copyInBytes _buf _arr := pure ()  -- No-op since we're on CPU
  copyOutBytes buf _ := pure buf.data
  sizeBytes buf := buf.data.size

instance : Compiler CPUProgram where
  compile name source := pure { name, source }

instance : Runtime CPUProgram CPUBuffer where
  launch _ _ _ := pure ()  -- Would interpret the program
  sync := pure ()

/-- CPU renderer just generates C-like pseudocode -/
def cpuRenderer : Renderer where
  name := "CPU"
  renderEwise name _nodes _outId := some s!"// CPU ewise kernel: {name}"
  renderReduce name _op _axes _outer _inner := s!"// CPU reduce kernel: {name}"
  renderMatmul m k n := s!"// CPU matmul: [{m}x{k}] @ [{k}x{n}]"

def cpuDevice : Device CPUProgram CPUBuffer where
  name := "CPU"
  allocator := inferInstance
  compiler := inferInstance
  runtime := inferInstance
  renderer := cpuRenderer

/-! ## Device Selection -/

inductive DeviceType where
  | CPU
  | METAL
  | CUDA
  deriving Repr, DecidableEq, Inhabited

/-- Parse device string like "CPU", "METAL", "CUDA:0" -/
def parseDeviceType (s : String) : DeviceType :=
  if s.startsWith "METAL" then .METAL
  else if s.startsWith "CUDA" then .CUDA
  else .CPU

end TinyGrad4.Backend
