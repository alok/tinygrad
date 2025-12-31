import TinyGrad4.Backend.JIT
import TinyGrad4.Backend.Device

/-!
# JIT Smoke Tests

Tests the JIT compilation system for execution capture and replay.
-/

namespace TinyGrad4.Test.JITSmoke

open TinyGrad4.Backend
open TinyGrad4.Backend.JIT

set_option linter.useRawBuffer false

/-- Check if string contains substring -/
def containsSubstr (s sub : String) : Bool := Id.run do
  let subData := sub.toList
  let sData := s.toList
  if subData.isEmpty then return true
  if subData.length > sData.length then return false
  for i in [:sData.length - subData.length + 1] do
    let slice := sData.drop i |>.take subData.length
    if slice == subData then return true
  return false

/-- Test basic JIT creation -/
def testJITCreate : IO Unit := do
  IO.println "Testing JIT creation..."

  let jit ← JIT.create (fun inputs => do
    IO.println s!"  Executing with {inputs.size} inputs"
    return inputs.size
  ) 2 (debug := true)

  IO.println "  JIT created successfully"
  IO.println "✓ JIT creation"

/-- Test JIT run counting -/
def testJITRunCount : IO Unit := do
  IO.println "Testing JIT run counting..."

  let counter ← IO.mkRef (0 : Nat)

  let jit ← JIT.create (fun inputs => do
    counter.modify (· + 1)
    return inputs.size
  ) 1 (debug := true)

  -- First run (ignore mode)
  let _ ← jit.call #[RawBuffer.emptyFloat32]
  let count1 ← jit.getRunCount
  if count1 != 1 then
    throw (IO.userError s!"Expected run count 1, got {count1}")

  -- Second run (capture mode)
  let _ ← jit.call #[RawBuffer.emptyFloat32]
  let count2 ← jit.getRunCount
  if count2 != 2 then
    throw (IO.userError s!"Expected run count 2, got {count2}")

  -- Third run (replay mode)
  let _ ← jit.call #[RawBuffer.emptyFloat32]
  let count3 ← jit.getRunCount
  if count3 != 3 then
    throw (IO.userError s!"Expected run count 3, got {count3}")

  -- Verify all runs executed the computation
  let execCount ← counter.get
  if execCount != 3 then
    throw (IO.userError s!"Expected 3 executions, got {execCount}")

  IO.println "✓ JIT run counting"

/-- Test JIT context states -/
def testJITStates : IO Unit := do
  IO.println "Testing JIT context states..."

  let ctx ← JITContext.new (debug := true)

  -- Initial state
  let state0 ← ctx.getState
  if state0 != .idle then
    throw (IO.userError s!"Expected idle state, got {repr state0}")

  -- Start capture
  ctx.startCapture #[⟨0⟩, ⟨1⟩]
  let state1 ← ctx.getState
  if state1 != .capturing then
    throw (IO.userError s!"Expected capturing state, got {repr state1}")

  -- Finish capture
  ctx.finishCapture (.input 0)
  let state2 ← ctx.getState
  if state2 != .idle then
    throw (IO.userError s!"Expected idle state after capture, got {repr state2}")

  -- Check captured exists
  let hasCap ← ctx.hasCaptured
  if !hasCap then
    throw (IO.userError "Expected captured execution to exist")

  IO.println "✓ JIT context states"

/-- Test JIT reset -/
def testJITReset : IO Unit := do
  IO.println "Testing JIT reset..."

  let jit ← JIT.create (fun inputs => pure inputs.size) 1

  -- Do some runs
  let _ ← jit.call #[RawBuffer.emptyFloat32]
  let _ ← jit.call #[RawBuffer.emptyFloat32]

  let countBefore ← jit.getRunCount
  if countBefore != 2 then
    throw (IO.userError s!"Expected run count 2, got {countBefore}")

  -- Reset
  jit.reset

  let countAfter ← jit.getRunCount
  if countAfter != 0 then
    throw (IO.userError s!"Expected run count 0 after reset, got {countAfter}")

  IO.println "✓ JIT reset"

/-- Test JIT input validation -/
def testJITInputValidation : IO Unit := do
  IO.println "Testing JIT input validation..."

  let jit ← JIT.create (fun inputs => pure inputs.size) 2

  -- Should fail with wrong input count
  try
    let _ ← jit.call #[RawBuffer.emptyFloat32]  -- Only 1 input, expected 2
    throw (IO.userError "Should have thrown for wrong input count")
  catch e =>
    let msg := toString e
    if !containsSubstr msg "Expected 2 inputs" then
      throw (IO.userError s!"Wrong error message: {msg}")
    IO.println "  Correctly rejected wrong input count"

  IO.println "✓ JIT input validation"

/-- Test CapturedKernel structure -/
def testCapturedKernel : IO Unit := do
  IO.println "Testing CapturedKernel structure..."

  let kernel : CapturedKernel := {
    source := "kernel void test() {}"
    name := "test"
    launchParams := { globalSize := (1024, 1, 1), localSize := (256, 1, 1) }
    bufferSlots := #[.input 0, .input 1, .intermediate 0]
  }

  if kernel.name != "test" then
    throw (IO.userError "Kernel name mismatch")

  if kernel.bufferSlots.size != 3 then
    throw (IO.userError "Buffer slots size mismatch")

  IO.println "✓ CapturedKernel structure"

/-- Test JITStats formatting -/
def testJITStats : IO Unit := do
  IO.println "Testing JITStats formatting..."

  let stats : JITStats := {
    totalRuns := 100
    capturedRuns := 1
    replayedRuns := 99
    timeSavedNs := 50_000_000  -- 50ms
  }

  let formatted := stats.format
  IO.println s!"  Stats: {formatted}"

  if !containsSubstr formatted "total=100" then
    throw (IO.userError "Stats format missing total")

  if !containsSubstr formatted "99" then
    throw (IO.userError "Stats format missing replay percentage")

  IO.println "✓ JITStats formatting"

/-- Run all tests -/
def runAll : IO Unit := do
  IO.println "=== JIT Smoke Tests ==="

  testJITCreate
  testJITRunCount
  testJITStates
  testJITReset
  testJITInputValidation
  testCapturedKernel
  testJITStats

  IO.println "=== All JIT Smoke Tests Passed ==="

end TinyGrad4.Test.JITSmoke

def main : IO Unit := TinyGrad4.Test.JITSmoke.runAll
