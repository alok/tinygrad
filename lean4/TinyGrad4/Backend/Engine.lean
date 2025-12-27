import TinyGrad4.Backend.Device

/-!
# Generic Execution Engine

Polymorphic execution engine that works with any backend via typeclasses.
Separates the execution logic from the specific device implementation.
-/

namespace TinyGrad4.Backend.Engine

open TinyGrad4.Backend

/-! ## Core Execution Functions -/

/-- Execute a single kernel on a device -/
def executeKernel [Compiler Prog] [Runtime Prog Buf]
    (compiler : Compiler Prog)
    (runtime : Runtime Prog Buf)
    (kernel : CompiledKernel)
    (buffers : Array Buf) : IO Unit := do
  let prog ← compiler.compile kernel.name kernel.source
  let mut boundBuffers : Array Buf := #[]
  for i in [:kernel.bufferCount] do
    if h : i < buffers.size then
      boundBuffers := boundBuffers.push buffers[i]
  runtime.launch prog boundBuffers kernel.launchParams

/-- Execute a complete plan on a device -/
def execute [Allocator Buf] [Compiler Prog] [Runtime Prog Buf]
    (device : Device Prog Buf)
    (plan : ExecPlan) : IO FloatArray := do
  -- 1. Allocate all buffers
  let mut buffers : Array Buf := #[]
  for size in plan.bufferSizes do
    let buf ← device.allocator.alloc size
    buffers := buffers.push buf

  -- 2. Copy input data
  for (idx, data) in plan.inputs do
    if h : idx < buffers.size then
      device.allocator.copyIn buffers[idx] data

  -- 3. Execute kernels in order
  for kexec in plan.kernels do
    -- Gather buffers for this kernel
    let kernelBufs := kexec.bufferBindings.foldl (init := #[]) fun acc idx =>
      if h : idx < buffers.size then acc.push buffers[idx] else acc
    executeKernel device.compiler device.runtime kexec.kernel kernelBufs

  -- 4. Sync and copy output
  device.runtime.sync
  let result ←
    if h : plan.outputIdx < buffers.size then
      device.allocator.copyOut buffers[plan.outputIdx]
    else
      pure ⟨#[]⟩

  -- 5. Free all buffers
  for buf in buffers do
    device.allocator.free buf

  return result

/-! ## Plan Building Helpers -/

/-- Build an elementwise execution plan -/
def buildEwisePlan (device : Device Prog Buf) (name : String)
    (nodes : List UOp) (outputUid : UOpId)
    (inputData : Array FloatArray) (outputSize : Nat) : Option ExecPlan := do
  -- Render the kernel
  let source ← device.renderer.renderEwise name nodes outputUid

  let kernel : CompiledKernel := {
    name := name
    source := source
    launchParams := { globalSize := (outputSize, 1, 1), localSize := (256, 1, 1) }
    bufferCount := inputData.size + 1  -- inputs + output
  }

  let kexec : KernelExec := {
    kernel := kernel
    bufferBindings := (List.range (inputData.size + 1)).toArray
  }

  -- Buffer sizes: inputs + output
  let bufferSizes := inputData.map (·.size) |>.push outputSize

  -- Inputs are at indices 0..n-1
  let inputs := Id.run do
    let mut arr : Array (Nat × FloatArray) := #[]
    for i in [:inputData.size] do
      if h : i < inputData.size then
        arr := arr.push (i, inputData[i])
    return arr

  pure {
    bufferSizes := bufferSizes
    inputs := inputs
    kernels := #[kexec]
    outputIdx := inputData.size  -- Output is last buffer
  }

/-! ## High-Level API -/

/-- Run an elementwise operation on a device -/
def runEwise [Allocator Buf] [Compiler Prog] [Runtime Prog Buf]
    (device : Device Prog Buf)
    (name : String)
    (nodes : List UOp)
    (outputUid : UOpId)
    (inputs : Array FloatArray)
    (outputSize : Nat) : IO (Option FloatArray) := do
  match buildEwisePlan device name nodes outputUid inputs outputSize with
  | some plan => return some (← execute device plan)
  | none => return none

/-- Simple benchmark wrapper -/
def benchmark (name : String) (iterations : Nat) (action : IO α) : IO (α × Float) := do
  -- Warmup
  for _ in [:3] do
    let _ ← action

  -- Benchmark
  let startTime ← IO.monoNanosNow
  let mut result : Option α := none
  for _ in [:iterations] do
    result := some (← action)
  let endTime ← IO.monoNanosNow

  let totalMs := (endTime - startTime).toFloat / 1e6
  let avgMs := totalMs / iterations.toFloat

  match result with
  | some r => return (r, avgMs)
  | none => throw (IO.Error.userError s!"benchmark {name}: no iterations")

end TinyGrad4.Backend.Engine
