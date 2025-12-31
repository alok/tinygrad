import TinyGrad4
import TinyGrad4.Backend.Metal

/-!
Minimal debug test for 32x32 matmul CPU vs GPU path.
-/

set_option linter.useRawBuffer false

namespace TinyGrad4.Test.MatmulDebug32

open TinyGrad4
open StaticTensor
open Interpreter
open Backend
open Std

def main : IO Unit := do
  IO.println "=== Minimal 32x32 Matmul Debug ==="

  let gpuAvail ← Metal.isAvailable
  IO.println s!"Metal available: {gpuAvail}"

  let m := 32
  let k := 32
  let n := 32

  -- Create matmul UOp
  let (aId, bId, matmulUop) := runTensorM do
    let aBuf ← Tensor.buffer [m, k] .float32
    let bBuf ← Tensor.buffer [k, n] .float32
    let result ← matmul aBuf bBuf
    pure (aBuf.uop.uid, bBuf.uop.uid, result.uop)

  IO.println s!"aId={aId}, bId={bId}, matmulUop.uid={matmulUop.uid}"
  IO.println s!"matmulUop.shape={matmulUop.shape}"

  -- Input: all 1s
  let aData := Array.replicate (m * k) 1.0
  let bData := Array.replicate (k * n) 1.0
  let aBuf := RawBuffer.ofF32 ⟨aData⟩
  let bBuf := RawBuffer.ofF32 ⟨bData⟩

  IO.println s!"aBuf.size={aBuf.data.size}, bBuf.size={bBuf.data.size}"

  let compiled ← Interpreter.compileManyCached [matmulUop]
  IO.println s!"Schedule has {compiled.schedule.length} items"

  for (i, item) in compiled.schedule.toArray.mapIdx (·, ·) do
    IO.println s!"  Item {i}: op={repr item.ast.op}, shape={item.ast.shape}, impl={repr item.impl}"

  let env : Env := (∅ : Env) |>.insert aId aBuf |>.insert bId bBuf

  -- Test CPU path only (evalCompiledRaw)
  IO.println "\n--- Testing evalCompiledRaw (pure CPU) ---"
  let cpuCache := Interpreter.evalCompiledRaw compiled env
  let cpuResult := cpuCache.getD matmulUop.uid (RawBuffer.zeros .float32 0)
  let cpuData := cpuResult.decode.data
  IO.println s!"CPU result size: {cpuData.size}"
  IO.println s!"CPU first 20: {cpuData[:Nat.min 20 cpuData.size]}"

  -- Test IO path (evalCompiledRawIO)
  IO.println "\n--- Testing evalCompiledRawIO (IO path) ---"
  let gpuCache ← Interpreter.evalCompiledRawIO compiled env
  let gpuResult := gpuCache.getD matmulUop.uid (RawBuffer.zeros .float32 0)
  let gpuData := gpuResult.decode.data
  IO.println s!"IO result size: {gpuData.size}"
  IO.println s!"IO first 20: {gpuData[:Nat.min 20 gpuData.size]}"

  -- Find first difference
  let mut firstDiff : Option Nat := none
  for i in [:cpuData.size] do
    if Float.abs (cpuData[i]! - gpuData[i]!) > 0.01 then
      firstDiff := some i
      break

  match firstDiff with
  | some i =>
    IO.println s!"\n❌ First diff at index {i}: CPU={cpuData[i]!}, IO={gpuData[i]!}"
  | none =>
    IO.println s!"\n✓ All {cpuData.size} elements match"

end TinyGrad4.Test.MatmulDebug32

def main : IO Unit := TinyGrad4.Test.MatmulDebug32.main
