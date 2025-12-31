import TinyGrad4
import TinyGrad4.Backend.Metal
import TinyGrad4.Backend.FusedEwise

/-!
Debug ReLU GPU bug - find exact corruption boundary.
-/

set_option linter.useRawBuffer false

namespace TinyGrad4.Test.ReluGPUDebug2

open TinyGrad4
open StaticTensor
open Interpreter
open Backend
open Std

/-- Find first corrupted index in GPU output -/
def findFirstCorruption (expected actual : Array Float) : Option (Nat × Float × Float) := Id.run do
  for i in [:expected.size] do
    let exp := expected[i]!
    let act := actual[i]!
    let isNaN := act != act
    let diff := if isNaN then 1000.0 else Float.abs (exp - act)
    if diff > 0.001 then
      return some (i, exp, act)
  return none

def testReluBoundary (n : Nat) : IO Unit := do
  IO.println s!"\n=== Testing ReLU boundary: n={n} ==="

  -- Create tensor with all 1s (simple values)
  let (xId, reluUop) := runTensorM do
    let xBuf ← Tensor.buffer [n] .float32
    let reluResult ← relu xBuf
    pure (xBuf.uop.uid, reluResult.uop)

  -- All 1s - ReLU should return 1s
  let xBuf := RawBuffer.ofF32 ⟨Array.replicate n 1.0⟩

  let compiled ← Interpreter.compileManyCached [reluUop]
  let env : Env := (∅ : Env) |>.insert xId xBuf

  -- GPU path
  let gpuCache ← Interpreter.evalCompiledRawIO compiled env
  let gpuResult := gpuCache.getD reluUop.uid (RawBuffer.zeros .float32 0)
  let gpuData := gpuResult.decode.data

  -- Expected: all 1s
  let expected := Array.replicate n 1.0

  -- Find first corruption
  match findFirstCorruption expected gpuData with
  | some (i, exp, act) =>
    IO.println s!"  ❌ FAIL at index {i}: expected={exp}, got={act}"
    -- Print around corruption
    let start := if i > 5 then i - 5 else 0
    let end_ := Nat.min (i + 10) n
    IO.println s!"  Context [{start}..{end_}]: {gpuData[start:end_]}"
  | none =>
    IO.println s!"  ✓ PASS - all {n} elements correct"

def testPlanShape : IO Unit := do
  IO.println "\n=== Checking FusedEwise Plan for ReLU ==="

  let (xId, reluUop) := runTensorM do
    let xBuf ← Tensor.buffer [100] .float32
    let reluResult ← relu xBuf
    pure (xBuf.uop.uid, reluResult.uop)

  IO.println s!"  ReLU UOp: {repr reluUop.op}"
  IO.println s!"  ReLU shape: {reluUop.shape}"
  IO.println s!"  Sources count: {reluUop.src.length}"

  -- Print source shapes
  for s in reluUop.src do
    IO.println s!"    src: op={repr s.op}, shape={s.shape}, dtype={repr s.dtype}"
    if s.op == .CONST then
      IO.println s!"      CONST detected - this is the scalar 0"

def run : IO Unit := do
  IO.println "=== ReLU GPU Boundary Debug ==="

  testPlanShape

  -- Binary search for corruption boundary
  -- We know n=100 works, n=65536 fails
  testReluBoundary 100
  testReluBoundary 256
  testReluBoundary 1000
  testReluBoundary 10000
  testReluBoundary 32000
  testReluBoundary 40000
  testReluBoundary 50000
  testReluBoundary 60000
  testReluBoundary 65000
  testReluBoundary 65536

  IO.println "\n=== Done ==="

end TinyGrad4.Test.ReluGPUDebug2

def main : IO Unit := TinyGrad4.Test.ReluGPUDebug2.run
