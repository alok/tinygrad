import TinyGrad4
import TinyGrad4.Backend.Metal
import TinyGrad4.Backend.MetalMatmul

/-!
Debug NaN in GPU matmul at larger sizes.
-/

set_option linter.useRawBuffer false
-- Disable IO.monoNanosNow linter: benchmark timing uses raw monotonic clocks.
set_option linter.monoNanosNow false

namespace TinyGrad4.Test.MatmulNaNDebug

open TinyGrad4
open TinyGrad4.Backend
open TinyGrad4.Backend.Metal

/-- Check if array contains NaN -/
def hasNaN (arr : Array Float) : Bool :=
  arr.any fun x => x != x  -- NaN != NaN

/-- Count NaN values -/
def countNaN (arr : Array Float) : Nat :=
  arr.foldl (fun acc x => if x != x then acc + 1 else acc) 0

def testMatmulSize (m k n : Nat) (desc : String) : IO Unit := do
  IO.println s!"\n=== Testing {desc}: [{m}, {k}] @ [{k}, {n}] = [{m}, {n}] ==="

  -- Create test data: simple pattern that shouldn't overflow
  let aSize := m * k
  let bSize := k * n
  let cSize := m * n

  -- Fill with small values (0.01)
  let aData := Native.fullF32Bits aSize ((0.01 : Float32).toBits)
  let bData := Native.fullF32Bits bSize ((0.01 : Float32).toBits)

  let aBuf : RawBuffer := { dtype := .float32, data := aData }
  let bBuf : RawBuffer := { dtype := .float32, data := bData }

  IO.println s!"  Input A: {aSize} elements, first few: {(aBuf.decode.data[:min 5 aSize])}"
  IO.println s!"  Input B: {bSize} elements, first few: {(bBuf.decode.data[:min 5 bSize])}"

  -- CPU matmul
  let cpuStart ← IO.monoNanosNow
  let cpuResult := MetalMatmul.matmul2DCPU aBuf bBuf m k n
  let cpuEnd ← IO.monoNanosNow
  let cpuMs := Float.ofNat (cpuEnd - cpuStart) / 1000000.0

  let cpuDecoded := cpuResult.decode.data
  let cpuNaN := countNaN cpuDecoded
  IO.println s!"  CPU: {cpuMs} ms, NaN count: {cpuNaN}"
  IO.println s!"  CPU first few: {cpuDecoded[:min 10 cSize]}"

  -- Expected value: each output element = k * 0.01 * 0.01 = k * 0.0001
  let expected := Float.ofNat k * 0.0001
  IO.println s!"  Expected value per element: {expected}"

  -- GPU matmul
  let gpuAvail ← Metal.isAvailable
  if !gpuAvail then
    IO.println "  GPU: skipped (not available)"
    return

  let gpuStart ← IO.monoNanosNow
  let gpuResult ← MetalMatmul.matmul2D aBuf bBuf m k n
  let gpuEnd ← IO.monoNanosNow
  let gpuMs := Float.ofNat (gpuEnd - gpuStart) / 1000000.0

  let gpuDecoded := gpuResult.decode.data
  let gpuNaN := countNaN gpuDecoded
  IO.println s!"  GPU: {gpuMs} ms, NaN count: {gpuNaN}/{cSize}"

  if gpuNaN > 0 then
    -- Find first NaN position
    for i in [:gpuDecoded.size] do
      let v := gpuDecoded[i]!
      if v != v then
        IO.println s!"  First NaN at index {i}"
        break
    IO.println s!"  GPU first few: {gpuDecoded[:min 10 cSize]}"
    IO.println s!"  ❌ NaN DETECTED in GPU output"
  else
    IO.println s!"  GPU first few: {gpuDecoded[:min 10 cSize]}"
    -- Check accuracy
    let mut maxDiff : Float := 0.0
    for i in [:min 100 cSize] do
      let diff := Float.abs (gpuDecoded[i]! - cpuDecoded[i]!)
      if diff > maxDiff then maxDiff := diff
    IO.println s!"  Max diff (first 100): {maxDiff}"
    if maxDiff > 0.01 then
      IO.println s!"  ⚠️ ACCURACY issue"
    else
      IO.println s!"  ✓ OK"

def run : IO Unit := do
  IO.println "=== Matmul NaN Debug ==="

  -- Start small, increase size
  testMatmulSize 64 784 256 "small (64x784x256)"
  testMatmulSize 128 784 256 "medium (128x784x256)"
  testMatmulSize 256 784 256 "large (256x784x256) - where NaN appears"
  testMatmulSize 256 256 10 "logits (256x256x10)"

  -- Test individual dimensions
  testMatmulSize 256 100 100 "square-ish (256x100x100)"
  testMatmulSize 256 784 100 "wide (256x784x100)"
  testMatmulSize 256 100 256 "tall (256x100x256)"

  -- Backward pass matmul dimensions (where NaN appears in MNIST)
  IO.println "\n--- Backward Pass Matmul Dimensions ---"
  -- gradW1 = X.T @ gradH = [784, 256] @ [256, 256]
  testMatmulSize 784 256 256 "gradW1 backward (784x256x256)"
  -- gradHRelu = gradLogits @ W2.T = [256, 10] @ [10, 256]
  testMatmulSize 256 10 256 "gradHRelu backward (256x10x256)"
  -- gradW2 = hRelu.T @ gradLogits = [256, 256] @ [256, 10]
  testMatmulSize 256 256 10 "gradW2 backward (256x256x10)"

  IO.println "\n=== Done ==="

end TinyGrad4.Test.MatmulNaNDebug

def main : IO Unit := TinyGrad4.Test.MatmulNaNDebug.run
