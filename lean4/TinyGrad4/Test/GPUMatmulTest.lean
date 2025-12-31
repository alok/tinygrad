import TinyGrad4
import TinyGrad4.Backend.Metal

/-!
Test GPU matmul operations for correctness and performance.
-/

set_option linter.useRawBuffer false

namespace TinyGrad4.Test.GPUMatmulTest

open TinyGrad4
open StaticTensor
open Interpreter
open Backend
open Std

/-- Simple matmul test with known values -/
def testMatmul (m k n : Nat) (name : String) : IO Unit := do
  IO.println s!"\n=== Matmul test: {name} [{m}, {k}] @ [{k}, {n}] ==="

  -- Create matmul
  let (aId, bId, matmulUop) := runTensorM do
    let aBuf ← Tensor.buffer [m, k] .float32
    let bBuf ← Tensor.buffer [k, n] .float32
    let result ← matmul aBuf bBuf
    pure (aBuf.uop.uid, bBuf.uop.uid, result.uop)

  -- Input: all 1s (result should be k for each element)
  let aData := Array.replicate (m * k) 1.0
  let bData := Array.replicate (k * n) 1.0
  let aBuf := RawBuffer.ofF32 ⟨aData⟩
  let bBuf := RawBuffer.ofF32 ⟨bData⟩

  let compiled ← Interpreter.compileManyCached [matmulUop]
  let env : Env := (∅ : Env) |>.insert aId aBuf |>.insert bId bBuf

  -- CPU path
  let cpuStart ← IO.monoMsNow
  let cpuCache := Interpreter.evalCompiledRaw compiled env
  let cpuEnd ← IO.monoMsNow
  let cpuResult := cpuCache.getD matmulUop.uid (RawBuffer.zeros .float32 0)
  let cpuData := cpuResult.decode.data

  -- GPU path
  let gpuStart ← IO.monoMsNow
  let gpuCache ← Interpreter.evalCompiledRawIO compiled env
  let gpuEnd ← IO.monoMsNow
  let gpuResult := gpuCache.getD matmulUop.uid (RawBuffer.zeros .float32 0)
  let gpuData := gpuResult.decode.data

  IO.println s!"  CPU time: {cpuEnd - cpuStart} ms"
  IO.println s!"  GPU time: {gpuEnd - gpuStart} ms"
  IO.println s!"  Output size: {cpuData.size}"
  IO.println s!"  CPU first 5: {cpuData[:Nat.min 5 cpuData.size]}"
  IO.println s!"  GPU first 5: {gpuData[:Nat.min 5 gpuData.size]}"

  -- Expected: each element = k (sum of 1*1 k times)
  let expected := Float.ofNat k
  let mut cpuOk := true
  let mut gpuOk := true
  let mut maxDiff : Float := 0.0
  let mut firstDiffIdx : Option Nat := none

  for i in [:cpuData.size] do
    let cpuV := cpuData[i]!
    let gpuV := gpuData[i]!
    let diff := Float.abs (cpuV - gpuV)
    if diff > maxDiff then
      maxDiff := diff
      if firstDiffIdx.isNone && diff > 0.01 then
        firstDiffIdx := some i
    if Float.abs (cpuV - expected) > 0.01 then cpuOk := false
    if Float.abs (gpuV - expected) > 0.01 then gpuOk := false

  IO.println s!"  Max CPU-GPU diff: {maxDiff}"
  if let some idx := firstDiffIdx then
    IO.println s!"  First diff at index {idx}: CPU={cpuData[idx]!}, GPU={gpuData[idx]!}"
    -- Show context around the diff
    let start := if idx > 5 then idx - 5 else 0
    let end_ := Nat.min (idx + 10) cpuData.size
    IO.println s!"  CPU [{start}..{end_}]: {cpuData[start:end_]}"
    IO.println s!"  GPU [{start}..{end_}]: {gpuData[start:end_]}"

  if cpuOk then
    IO.println "  ✓ CPU correct"
  else
    IO.println "  ❌ CPU wrong"

  if gpuOk then
    IO.println "  ✓ GPU correct"
  else
    IO.println "  ❌ GPU wrong"

  if maxDiff < 0.01 then
    IO.println "  ✓ CPU/GPU match"
  else
    IO.println s!"  ❌ CPU/GPU mismatch (diff={maxDiff})"

/-- Performance benchmark with random-like values -/
def benchMatmul (m k n : Nat) (name : String) (iterations : Nat := 5) : IO Unit := do
  IO.println s!"\n=== Matmul benchmark: {name} [{m}, {k}] @ [{k}, {n}] ==="

  -- Create matmul
  let (aId, bId, matmulUop) := runTensorM do
    let aBuf ← Tensor.buffer [m, k] .float32
    let bBuf ← Tensor.buffer [k, n] .float32
    let result ← matmul aBuf bBuf
    pure (aBuf.uop.uid, bBuf.uop.uid, result.uop)

  -- Use index-based values for reproducibility
  let mut aData : Array Float := #[]
  for i in [:m * k] do
    aData := aData.push (Float.ofNat (i % 100) / 100.0)
  let mut bData : Array Float := #[]
  for j in [:k * n] do
    bData := bData.push (Float.ofNat ((j * 7) % 100) / 100.0)

  let aBuf := RawBuffer.ofF32 ⟨aData⟩
  let bBuf := RawBuffer.ofF32 ⟨bData⟩

  let compiled ← Interpreter.compileManyCached [matmulUop]
  let env : Env := (∅ : Env) |>.insert aId aBuf |>.insert bId bBuf

  -- Warmup
  let _ := Interpreter.evalCompiledRaw compiled env
  let _ ← Interpreter.evalCompiledRawIO compiled env

  -- CPU benchmark
  let cpuStart ← IO.monoMsNow
  for _ in [:iterations] do
    let _ := Interpreter.evalCompiledRaw compiled env
  let cpuEnd ← IO.monoMsNow
  let cpuAvg := (cpuEnd - cpuStart).toFloat / iterations.toFloat

  -- GPU benchmark
  let gpuStart ← IO.monoMsNow
  for _ in [:iterations] do
    let _ ← Interpreter.evalCompiledRawIO compiled env
  let gpuEnd ← IO.monoMsNow
  let gpuAvg := (gpuEnd - gpuStart).toFloat / iterations.toFloat

  IO.println s!"  CPU avg: {cpuAvg} ms"
  IO.println s!"  GPU avg: {gpuAvg} ms"

  if gpuAvg < cpuAvg then
    IO.println s!"  GPU faster by: {cpuAvg / gpuAvg}x"
  else
    IO.println s!"  CPU faster by: {gpuAvg / cpuAvg}x"

def run : IO Unit := do
  IO.println "=== GPU Matmul Test ==="

  let gpuAvail ← Metal.isAvailable
  IO.println s!"Metal available: {gpuAvail}"

  -- Small correctness tests
  testMatmul 4 4 4 "tiny (4x4)"
  testMatmul 32 32 32 "small (32x32)"
  testMatmul 64 64 64 "medium (64x64)"

  -- MNIST-like sizes
  testMatmul 256 784 256 "MNIST forward (256x784 @ 784x256)"
  testMatmul 256 256 10 "MNIST output (256x256 @ 256x10)"

  -- Larger sizes
  testMatmul 512 512 512 "512x512"

  -- Performance benchmarks
  benchMatmul 256 784 256 "MNIST forward"
  benchMatmul 512 512 512 "512x512"
  benchMatmul 1024 1024 1024 "1024x1024"

  IO.println "\n=== Done ==="

end TinyGrad4.Test.GPUMatmulTest

def main : IO Unit := TinyGrad4.Test.GPUMatmulTest.run
