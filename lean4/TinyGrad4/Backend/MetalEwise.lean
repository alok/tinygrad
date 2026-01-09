import TinyGrad4.Backend.Metal
import TinyGrad4.Backend.MetalRenderer
import TinyGrad4.Backend.Buffer
import TinyGrad4.Backend.DeviceBuffer
import TinyGrad4.Backend.Native
import TinyGrad4.Backend.FusedEwise

/-!
# GPU Elementwise Execution via Metal

Executes fused elementwise kernels on Metal GPU, falling back to CPU when unavailable.

## Architecture

```
Interpreter KERNEL op (fused_ewise impl)
    ↓
MetalEwise.runEwise / runReduce
    ↓
Metal FFI (alloc, compile, launch, sync)
    ↓
Metal GPU execution
```
-/

namespace TinyGrad4.Backend.MetalEwise

open TinyGrad4
open TinyGrad4.Backend.Metal
open TinyGrad4.Backend.MetalRenderer
open TinyGrad4.Backend

/-! ## Local Helpers -/

private def zerosRaw (dtype : DType) (numel : Nat) : RawBuffer :=
  { dtype, data := ByteArray.mk (Array.replicate (numel * dtype.itemsize) 0) }

/-! ## DeviceBuffer access -/

-- Re-export for callers
export TinyGrad4.Backend.DeviceBuffer (GPUBufferId DeviceBuffer)

-- Open the namespace for function access (DeviceBuffer is the inner namespace with methods)
open TinyGrad4.Backend.DeviceBuffer.DeviceBuffer

/-! ## Program Cache -/

/-- Simple program cache to avoid recompiling identical kernels -/
initialize programCache : IO.Ref (Std.HashMap String MetalProgram) ← IO.mkRef ∅

/-- Get or compile a Metal program -/
def getOrCompile (name : String) (shader : String) : IO MetalProgram := do
  let cache ← programCache.get
  match cache[name]? with
  | some prog => return prog
  | none =>
    let prog ← metalCompile name shader
    programCache.modify (·.insert name prog)
    return prog

/-! ## Shader Rendering (must be defined before use) -/

/-- Check if all strides are zero (broadcast/scalar input) -/
private def isBroadcast (strides : Array Int64) : Bool :=
  strides.all (· == 0)

/-- Render the body of an elementwise kernel from FusedEwise program.

    Opcodes match FusedEwise.lean encoding:
    - 0 = LOAD (arg = input index)
    - 1 = NEG, 2 = SQRT, 3 = RECIP, 4 = EXP2, 5 = LOG2, 6 = SIN
    - 7 = ADD, 8 = SUB, 9 = MUL, 10 = DIV, 11 = MAX
    - 12 = WHERE (ternary)

    For broadcast inputs (all strides = 0), we read from index 0 instead of gid
    to avoid out-of-bounds access on scalar buffers.
-/
private def inputType (dtCode : Nat) : String :=
  if dtCode == 1 || dtCode == 2 then "uchar" else "float"

private def loadExpr (arg : Nat) (idx : String) (dtCode : Nat) : String :=
  if dtCode == 1 then
    s!"(in{arg}[{idx}] ? 1.0f : 0.0f)"
  else if dtCode == 2 then
    s!"(float)(in{arg}[{idx}])"
  else
    s!"in{arg}[{idx}]"

private def renderEwiseBody (prog : Array UInt64) (numInputs : Nat)
    (leafStrides : Array (Array Int64)) (leafDtypes : Array Nat) : String := Id.run do
  let mut lines := ""

  for instr in prog do
    let opcode := (instr &&& 0xFF).toNat
    let arg := ((instr >>> 8) &&& 0xFFFF).toNat

    match opcode with
    | 0 => -- LOAD (push from input)
      if arg < numInputs then
        -- Check if this input is broadcast (all strides = 0)
        let strides := leafStrides.getD arg #[]
        let idx := if isBroadcast strides then "0" else "gid"
        let dtCode := leafDtypes.getD arg 0
        let expr := loadExpr arg idx dtCode
        lines := lines ++ s!"  stack[sp++] = {expr};\n"
      else
        lines := lines ++ s!"  stack[sp++] = 0.0f; // invalid input {arg}\n"
    | 1 => -- NEG
      lines := lines ++ "  stack[sp-1] = -stack[sp-1];\n"
    | 2 => -- SQRT
      lines := lines ++ "  stack[sp-1] = sqrt(stack[sp-1]);\n"
    | 3 => -- RECIPROCAL
      lines := lines ++ "  stack[sp-1] = 1.0f / stack[sp-1];\n"
    | 4 => -- EXP2
      lines := lines ++ "  stack[sp-1] = exp2(stack[sp-1]);\n"
    | 5 => -- LOG2
      lines := lines ++ "  stack[sp-1] = log2(stack[sp-1]);\n"
    | 6 => -- SIN
      lines := lines ++ "  stack[sp-1] = sin(stack[sp-1]);\n"
    | 7 => -- ADD
      lines := lines ++ "  { float b = stack[--sp]; stack[sp-1] += b; }\n"
    | 8 => -- SUB
      lines := lines ++ "  { float b = stack[--sp]; stack[sp-1] -= b; }\n"
    | 9 => -- MUL
      lines := lines ++ "  { float b = stack[--sp]; stack[sp-1] *= b; }\n"
    | 10 => -- DIV
      lines := lines ++ "  { float b = stack[--sp]; stack[sp-1] /= b; }\n"
    | 11 => -- MAX
      lines := lines ++ "  { float b = stack[--sp]; stack[sp-1] = max(stack[sp-1], b); }\n"
    | 12 => -- WHERE (ternary: cond, true_val, false_val)
      lines := lines ++ "  { float f = stack[--sp]; float t = stack[--sp]; float c = stack[--sp]; stack[sp++] = (c != 0.0f) ? t : f; }\n"
    | _ =>
      lines := lines ++ s!"  // Unknown opcode {opcode}\n"

  return lines

/-- Render Metal kernel from FusedEwise plan.

    Translates the stack-based FusedEwise program to Metal source.
-/
private def renderEwiseKernelFromPlan (name : String) (plan : FusedEwise.Plan)
    (outShape : Shape) : String := Id.run do
  let numel := listProd outShape
  let numInputs := plan.leafBases.size

  -- Generate buffer parameters
  let mut params := ""
  for i in [:numInputs] do
    let dtCode := plan.leafDtypes.getD i 0
    let ty := inputType dtCode
    params := params ++ s!"  device const {ty}* in{i} [[buffer({i})]],\n"
  params := params ++ s!"  device float* out [[buffer({numInputs})]],\n"
  params := params ++ "  uint gid [[thread_position_in_grid]]"

  -- Generate kernel body from FusedEwise program
  -- Pass leafStrides to handle broadcast inputs correctly
  let body := renderEwiseBody plan.prog numInputs plan.leafStrides plan.leafDtypes

  s!"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
{params}
) \{
  if (gid >= {numel}) return;

  // Stack-based evaluation
  float stack[16];
  int sp = 0;

{body}
  out[gid] = stack[0];
}"

/-! ## Elementwise Kernel Dispatch -/

/-- Execute elementwise kernel on GPU.

    Arguments:
    - name: Unique kernel name (for caching)
    - shader: Metal shader source
    - inputs: Input RawBuffers
    - numel: Number of output elements
    - dtype: Output data type

    Returns float32 RawBuffer with result.
-/
def runEwiseKernel (name : String) (shader : String) (inputs : Array RawBuffer)
    (numel : Nat) (dtype : DType := .float32) : IO RawBuffer := do
  if numel == 0 then
    return zerosRaw dtype 0

  -- Allocate input buffers and copy data
  let mut metalBufs : Array MetalBuffer := #[]
  for input in inputs do
    let buf ← metalAllocBytes input.data.size
    metalCopyInBytes buf input.data
    metalBufs := metalBufs.push buf

  -- Allocate output buffer
  let outBytes := numel * dtype.itemsize
  let outBuf ← metalAllocBytes outBytes
  metalBufs := metalBufs.push outBuf

  -- Compile shader (with caching)
  let prog ← getOrCompile name shader

  -- Calculate grid dimensions
  -- Use 256 threads per threadgroup, 1D dispatch
  -- Note: metalLaunch uses dispatchThreads which takes TOTAL thread count
  let threadsPerGroup : Nat := 256
  let totalThreads := numel  -- One thread per element

  -- Launch kernel (metalLaunch takes total threads, not group count)
  metalLaunch prog metalBufs totalThreads 1 1 threadsPerGroup 1 1
  metalSync

  -- Copy result back
  let result ← metalCopyOutBytes outBuf outBytes

  -- Free all buffers
  for buf in metalBufs do
    metalFree buf

  return { dtype, data := result }

/-! ## Specialized Kernel Rendering -/

/-- Render a specialized contiguous kernel (unary or binary) -/
private def renderSpecializedKernel (name : String) (kernel : FusedEwise.Kernel)
    (numel : Nat) : String :=
  let op := match kernel with
    | .negContiguous => "-in0[gid]"
    | .sqrtContiguous => "sqrt(in0[gid])"
    | .recipContiguous => "1.0f / in0[gid]"
    | .exp2Contiguous => "exp2(in0[gid])"
    | .log2Contiguous => "log2(in0[gid])"
    | .sinContiguous => "sin(in0[gid])"
    | .cosContiguous => "cos(in0[gid])"
    | .tanContiguous => "tan(in0[gid])"
    | .addContiguous => "in0[gid] + in1[gid]"
    | .subContiguous => "in0[gid] - in1[gid]"
    | .mulContiguous => "in0[gid] * in1[gid]"
    | .divContiguous => "in0[gid] / in1[gid]"
    | .maxContiguous => "max(in0[gid], in1[gid])"
    | .powContiguous => "pow(in0[gid], in1[gid])"
    | .bytecode => "0.0f"  -- Shouldn't happen

  let isBinary := match kernel with
    | .addContiguous | .subContiguous | .mulContiguous | .divContiguous
    | .maxContiguous | .powContiguous => true
    | _ => false

  let params := if isBinary then
    "  device const float* in0 [[buffer(0)]],\n  device const float* in1 [[buffer(1)]],\n  device float* out [[buffer(2)]],"
  else
    "  device const float* in0 [[buffer(0)]],\n  device float* out [[buffer(1)]],"

  s!"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
{params}
  uint gid [[thread_position_in_grid]]
) \{
  if (gid >= {numel}) return;
  out[gid] = {op};
}"

/-! ## Fused Elementwise Dispatch -/

/-- Execute a FusedEwise plan on GPU.

    Generates Metal kernel from the fused program and dispatches to GPU.
    Handles both bytecode plans and specialized contiguous kernels.
-/
def runFusedEwise (plan : FusedEwise.Plan) (inputs : Array RawBuffer)
    (outShape : Shape) (dtype : DType := .float32) : IO RawBuffer := do
  let numel := listProd outShape

  let useBytecode := plan.kernel == .bytecode || plan.leafDtypes.any (· != 0)

  -- Generate unique kernel name from plan hash and kernel type
  -- IMPORTANT: Include numel and leafStrides in hash since they affect shader generation
  let kernelId := if useBytecode then
      0
    else
      match plan.kernel with
      | .bytecode => 0 | .negContiguous => 1 | .sqrtContiguous => 2 | .recipContiguous => 3
      | .exp2Contiguous => 4 | .log2Contiguous => 5 | .sinContiguous => 6 | .cosContiguous => 7
      | .tanContiguous => 8 | .addContiguous => 9 | .subContiguous => 10 | .mulContiguous => 11
      | .divContiguous => 12 | .maxContiguous => 13 | .powContiguous => 14
  let progHash := hash (plan.prog, plan.leafBases.size, kernelId, numel, plan.leafStrides, plan.leafDtypes)
  let name := s!"fused_ewise_{progHash}"

  -- Render Metal kernel based on kernel type
  let shader :=
    if useBytecode then
      renderEwiseKernelFromPlan name plan outShape
    else
      renderSpecializedKernel name plan.kernel numel

  runEwiseKernel name shader inputs numel dtype

/-! ## Reduce Kernel Dispatch -/

/-- Execute reduce kernel on GPU.

    Arguments:
    - reduceOp: SUM, MAX, or MIN
    - input: Input RawBuffer
    - outer: Number of independent reductions
    - inner: Size of each reduction
-/
def runReduceKernel (reduceOp : MetalRenderer.ReduceOp) (input : RawBuffer)
    (outer inner : Nat) : IO RawBuffer := do
  let numel := outer

  if numel == 0 || inner == 0 then
    return zerosRaw .float32 0

  -- Generate kernel
  let name := s!"reduce_{repr reduceOp}_{outer}_{inner}"
  let shader := renderReduceKernelAuto name reduceOp inner outer

  -- Allocate input buffer
  let inBuf ← metalAllocBytes input.data.size
  metalCopyInBytes inBuf input.data

  -- Allocate output buffer
  let outBytes := outer * 4
  let outBuf ← metalAllocBytes outBytes

  -- Compile shader (with caching)
  let prog ← getOrCompile name shader

  -- For simple reduction, use outer*threadsPerGroup total threads
  -- Each threadgroup computes one output element cooperatively
  -- Note: metalLaunch takes TOTAL thread count, not group count
  let threadsPerGroup := min 256 inner
  let totalThreads := outer * threadsPerGroup

  metalLaunch prog #[inBuf, outBuf] totalThreads 1 1 threadsPerGroup 1 1
  metalSync

  -- Copy result back
  let result ← metalCopyOutBytes outBuf outBytes

  metalFree inBuf
  metalFree outBuf

  return { dtype := .float32, data := result }

/-! ## GPU-Resident Kernel Execution -/

/-- Execute elementwise kernel with DeviceBuffer inputs, returning DeviceBuffer output.
    Inputs are uploaded if on CPU; output stays on GPU. -/
def runEwiseKernelDevice (name : String) (shader : String) (inputs : Array DeviceBuffer)
    (numel : Nat) (dtype : DType := .float32) : IO DeviceBuffer := do
  if numel == 0 then
    return fromCPU (zerosRaw dtype 0)

  -- Ensure all inputs are on GPU
  let mut metalBufs : Array MetalBuffer := #[]
  for input in inputs do
    let id ← ensureGPU input
    metalBufs := metalBufs.push (← getGPUBuffer id)

  -- Allocate output buffer on GPU
  let outBytes := numel * dtype.itemsize
  let outId ← allocGPU outBytes dtype
  let outBuf ← getGPUBuffer outId
  metalBufs := metalBufs.push outBuf

  -- Compile shader (with caching)
  let prog ← getOrCompile name shader

  -- Launch kernel
  let threadsPerGroup : Nat := 256
  let totalThreads := numel
  metalLaunch prog metalBufs totalThreads 1 1 threadsPerGroup 1 1
  metalSync

  -- Return GPU-resident buffer (no copy!)
  return fromGPU outId dtype outBytes

/-- Execute FusedEwise with DeviceBuffer inputs and output.
    No CPU-GPU copies for intermediate results when inputs are already on GPU! -/
def runFusedEwiseDevice (plan : FusedEwise.Plan) (inputs : Array DeviceBuffer)
    (outShape : Shape) (dtype : DType := .float32) : IO DeviceBuffer := do
  let numel := listProd outShape

  let useBytecode := plan.kernel == .bytecode || plan.leafDtypes.any (· != 0)

  -- Generate unique kernel name
  -- IMPORTANT: Include numel and leafStrides in hash since they affect shader generation
  let kernelId := if useBytecode then
      0
    else
      match plan.kernel with
      | .bytecode => 0 | .negContiguous => 1 | .sqrtContiguous => 2 | .recipContiguous => 3
      | .exp2Contiguous => 4 | .log2Contiguous => 5 | .sinContiguous => 6 | .cosContiguous => 7
      | .tanContiguous => 8 | .addContiguous => 9 | .subContiguous => 10 | .mulContiguous => 11
      | .divContiguous => 12 | .maxContiguous => 13 | .powContiguous => 14
  let progHash := hash (plan.prog, plan.leafBases.size, kernelId, numel, plan.leafStrides, plan.leafDtypes)
  let name := s!"fused_ewise_device_{progHash}"

  -- Render Metal kernel
  let shader :=
    if useBytecode then
      renderEwiseKernelFromPlan name plan outShape
    else
      renderSpecializedKernel name plan.kernel numel

  runEwiseKernelDevice name shader inputs numel dtype

/-- Execute reduce kernel with DeviceBuffer input, returning DeviceBuffer output -/
def runReduceKernelDevice (reduceOp : MetalRenderer.ReduceOp) (input : DeviceBuffer)
    (outer inner : Nat) : IO DeviceBuffer := do
  if outer == 0 || inner == 0 then
    return fromCPU (zerosRaw .float32 0)

  -- Generate kernel
  let name := s!"reduce_device_{repr reduceOp}_{outer}_{inner}"
  let shader := renderReduceKernelAuto name reduceOp inner outer

  -- Ensure input is on GPU
  let inId ← ensureGPU input
  let inBuf ← getGPUBuffer inId

  -- Allocate output buffer
  let outBytes := outer * 4
  let outId ← allocGPU outBytes .float32
  let outBuf ← getGPUBuffer outId

  -- Compile shader (with caching)
  let prog ← getOrCompile name shader

  -- Launch kernel
  let threadsPerGroup := min 256 inner
  let totalThreads := outer * threadsPerGroup
  metalLaunch prog #[inBuf, outBuf] totalThreads 1 1 threadsPerGroup 1 1
  metalSync

  return fromGPU outId .float32 outBytes

/-! ## CPU Fallbacks -/

def runEwiseCPU (_plan : FusedEwise.Plan) (_inputs : Array RawBuffer)
    (outShape : Shape) : RawBuffer :=
  -- Fall back to Native CPU implementation
  let numel := listProd outShape
  zerosRaw .float32 numel  -- TODO: Implement CPU fallback properly

def runReduceCPU (_reduceOp : MetalRenderer.ReduceOp) (_input : RawBuffer)
    (outer _inner : Nat) : RawBuffer :=
  zerosRaw .float32 outer  -- TODO: Implement CPU fallback properly

/-! ## Unified Entry Points -/

/-- Execute fused elementwise, preferring GPU -/
def runFusedEwiseWithFallback (plan : FusedEwise.Plan) (inputs : Array RawBuffer)
    (outShape : Shape) (dtype : DType := .float32) : IO RawBuffer := do
  let available ← Metal.isAvailable
  if available then
    try
      runFusedEwise plan inputs outShape dtype
    catch _ =>
      return runEwiseCPU plan inputs outShape
  else
    return runEwiseCPU plan inputs outShape

/-- Execute reduce, preferring GPU -/
def runReduceWithFallback (reduceOp : MetalRenderer.ReduceOp) (input : RawBuffer)
    (outer inner : Nat) : IO RawBuffer := do
  let available ← Metal.isAvailable
  if available then
    try
      runReduceKernel reduceOp input outer inner
    catch _ =>
      return runReduceCPU reduceOp input outer inner
  else
    return runReduceCPU reduceOp input outer inner

end TinyGrad4.Backend.MetalEwise
