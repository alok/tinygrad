import TinyGrad4.Backend.Schedule
import TinyGrad4.Backend.Device
import Std.Data.HashMap

/-!
# JIT Compilation System

Ported from tinygrad's `engine/jit.py`.

The JIT captures repeated execution patterns and batches them for efficient replay.
This reduces kernel launch overhead by:
1. Caching compiled kernels
2. Reusing buffer allocations
3. Batching kernel submissions

## Usage

```
-- Create a JIT-wrapped function
let jit ← JIT.create fun inputs => do
  let a ← inputs[0]!
  let b ← inputs[1]!
  pure (a + b)

-- First call: normal execution
let _ ← jit.call #[tensorA, tensorB]

-- Second call: captures execution
let _ ← jit.call #[tensorA, tensorB]

-- Subsequent calls: fast replay
let _ ← jit.call #[tensorC, tensorD]  -- Uses cached execution
```
-/

namespace TinyGrad4.Backend.JIT

open TinyGrad4.Backend Std

/-! ## Core Types -/

/-- Buffer slot in the captured execution.
    Tracks which buffers are inputs (need replacement) vs intermediates. -/
inductive BufferSlot where
  | input : Nat → BufferSlot       -- Input buffer at index
  | intermediate : Nat → BufferSlot -- Intermediate buffer at index
  deriving Repr, BEq, Hashable

/-- A captured kernel execution -/
structure CapturedKernel where
  /-- Kernel source code -/
  source : String
  /-- Kernel name -/
  name : String
  /-- Launch parameters -/
  launchParams : LaunchParams
  /-- Buffer binding slots -/
  bufferSlots : Array BufferSlot
  deriving Repr

/-- A captured JIT execution -/
structure CapturedExec where
  /-- Sequence of kernel executions -/
  kernels : Array CapturedKernel
  /-- Number of input buffers -/
  inputCount : Nat
  /-- Sizes of intermediate buffers -/
  intermediateSizes : Array Nat
  /-- Output buffer slot -/
  outputSlot : BufferSlot
  deriving Repr

/-- JIT execution state -/
inductive JITState where
  | idle       -- Not currently JIT-ing
  | ignore     -- First run: ignore JIT, run normally
  | capturing  -- Capturing execution
  | replaying  -- Replaying captured execution
  deriving Repr, BEq

/-- Mutable JIT context -/
structure JITContext where
  /-- Current state -/
  state : IO.Ref JITState
  /-- Captured execution (after capture phase) -/
  captured : IO.Ref (Option CapturedExec)
  /-- Kernel being captured -/
  capturingKernels : IO.Ref (Array CapturedKernel)
  /-- Input buffer tracking during capture -/
  inputBufferIds : IO.Ref (Array UOpId)
  /-- Run counter -/
  runCount : IO.Ref Nat
  /-- Debug mode -/
  debug : Bool := false

/-! ## JIT Context Management -/

/-- Create a new JIT context -/
def JITContext.new (debug : Bool := false) : IO JITContext := do
  let state ← IO.mkRef JITState.idle
  let captured ← IO.mkRef none
  let capturingKernels ← IO.mkRef #[]
  let inputBufferIds ← IO.mkRef #[]
  let runCount ← IO.mkRef 0
  return { state, captured, capturingKernels, inputBufferIds, runCount, debug }

/-- Get current state -/
def JITContext.getState (ctx : JITContext) : IO JITState :=
  ctx.state.get

/-- Check if we have a captured execution -/
def JITContext.hasCaptured (ctx : JITContext) : IO Bool := do
  return (← ctx.captured.get).isSome

/-- Get the captured execution -/
def JITContext.getCaptured (ctx : JITContext) : IO (Option CapturedExec) :=
  ctx.captured.get

/-- Reset the JIT context -/
def JITContext.reset (ctx : JITContext) : IO Unit := do
  ctx.state.set .idle
  ctx.captured.set none
  ctx.capturingKernels.set #[]
  ctx.inputBufferIds.set #[]
  ctx.runCount.set 0

/-! ## Capture Phase -/

/-- Start capturing execution -/
def JITContext.startCapture (ctx : JITContext) (inputIds : Array UOpId) : IO Unit := do
  ctx.state.set .capturing
  ctx.capturingKernels.set #[]
  ctx.inputBufferIds.set inputIds
  if ctx.debug then
    IO.println s!"JIT: Starting capture with {inputIds.size} inputs"

/-- Record a kernel execution during capture -/
def JITContext.captureKernel (ctx : JITContext)
    (name : String) (source : String)
    (bufferIds : Array UOpId) (launchParams : LaunchParams) : IO Unit := do
  let inputIds ← ctx.inputBufferIds.get

  -- Map buffer IDs to slots
  let bufferSlots := bufferIds.map fun uid =>
    match inputIds.findIdx? (· == uid) with
    | some idx => .input idx
    | none => .intermediate 0  -- TODO: track intermediates properly

  let kernel : CapturedKernel := {
    source
    name
    launchParams
    bufferSlots
  }

  ctx.capturingKernels.modify (·.push kernel)
  if ctx.debug then
    IO.println s!"JIT: Captured kernel '{name}' with {bufferSlots.size} buffers"

/-- Finish capturing and create CapturedExec -/
def JITContext.finishCapture (ctx : JITContext) (outputSlot : BufferSlot) : IO Unit := do
  let kernels ← ctx.capturingKernels.get
  let inputIds ← ctx.inputBufferIds.get

  let captured : CapturedExec := {
    kernels
    inputCount := inputIds.size
    intermediateSizes := #[]  -- TODO: track intermediate sizes
    outputSlot
  }

  ctx.captured.set (some captured)
  ctx.state.set .idle
  ctx.runCount.modify (· + 1)

  if ctx.debug then
    IO.println s!"JIT: Finished capture with {kernels.size} kernels"

/-! ## Replay Phase -/

/-- Check if ready for replay -/
def JITContext.canReplay (ctx : JITContext) : IO Bool := do
  let count ← ctx.runCount.get
  let hasCap ← ctx.hasCaptured
  return count >= 2 && hasCap

/-- Start replay mode -/
def JITContext.startReplay (ctx : JITContext) : IO Unit := do
  ctx.state.set .replaying
  ctx.runCount.modify (· + 1)
  if ctx.debug then
    IO.println "JIT: Starting replay"

/-! ## High-Level JIT API -/

/-- A JIT-wrapped computation.
    Wraps an IO computation and caches its execution for replay. -/
structure JIT (α : Type) where
  /-- The computation to JIT -/
  computation : Array RawBuffer → IO α
  /-- JIT context -/
  context : JITContext
  /-- Expected number of inputs -/
  inputCount : Nat

/-- Create a new JIT wrapper -/
def JIT.create (f : Array RawBuffer → IO α) (inputCount : Nat)
    (debug : Bool := false) : IO (JIT α) := do
  let ctx ← JITContext.new debug
  return { computation := f, context := ctx, inputCount }

/-- Execute the JIT-wrapped computation -/
def JIT.call (jit : JIT α) (inputs : Array RawBuffer) : IO α := do
  let ctx := jit.context
  let count ← ctx.runCount.get

  -- Validate input count
  if inputs.size != jit.inputCount then
    throw (IO.userError s!"JIT: Expected {jit.inputCount} inputs, got {inputs.size}")

  if count == 0 then
    -- First run: ignore JIT, run normally
    ctx.state.set .ignore
    if ctx.debug then IO.println "JIT: First run (ignore)"
    let result ← jit.computation inputs
    ctx.runCount.modify (· + 1)
    return result
  else if count == 1 then
    -- Second run: capture execution
    -- Note: In a real implementation, we'd hook into the execution path
    -- For now, just run and increment counter
    if ctx.debug then IO.println "JIT: Second run (would capture)"
    let result ← jit.computation inputs
    ctx.runCount.modify (· + 1)
    return result
  else
    -- Subsequent runs: replay
    if ctx.debug then IO.println s!"JIT: Run {count + 1} (replay mode)"
    -- In a full implementation, we'd replay the captured execution
    -- For now, just execute normally
    let result ← jit.computation inputs
    ctx.runCount.modify (· + 1)
    return result

/-- Reset the JIT state -/
def JIT.reset (jit : JIT α) : IO Unit :=
  jit.context.reset

/-- Get run count -/
def JIT.getRunCount (jit : JIT α) : IO Nat :=
  jit.context.runCount.get

/-- Check if JIT is in replay mode -/
def JIT.isReplaying (jit : JIT α) : IO Bool := do
  let state ← jit.context.getState
  return state == .replaying

/-! ## Integration with Metal/CUDA -/

/-- Captured Metal execution for replay -/
structure MetalCapturedExec where
  /-- Compiled programs (cached) -/
  programs : Array (String × String)  -- (name, source) pairs
  /-- Execution sequence -/
  sequence : Array (Nat × Array Nat × LaunchParams)  -- (program_idx, buffer_indices, launch)
  /-- Input buffer count -/
  inputCount : Nat
  /-- Intermediate buffer sizes -/
  intermediateSizes : Array Nat
  deriving Repr

/-- Statistics for JIT performance -/
structure JITStats where
  /-- Total runs -/
  totalRuns : Nat := 0
  /-- Captured runs -/
  capturedRuns : Nat := 0
  /-- Replayed runs -/
  replayedRuns : Nat := 0
  /-- Time saved by replay (estimated, ns) -/
  timeSavedNs : Nat := 0
  deriving Repr

namespace JITStats

/-- Format stats as human-readable string -/
def format (s : JITStats) : String :=
  let replayPct := if s.totalRuns > 0
    then s.replayedRuns.toFloat / s.totalRuns.toFloat * 100.0
    else 0.0
  let savedMs := s.timeSavedNs.toFloat / 1e6
  s!"total={s.totalRuns} captured={s.capturedRuns} replayed={s.replayedRuns} ({replayPct}%) saved={savedMs}ms"

end JITStats

end TinyGrad4.Backend.JIT
