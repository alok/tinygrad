import TinyGrad4

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

/-!
# WeightInitSmoke

Smoke tests for weight initialization utilities.
-/

namespace TinyGrad4.Test.WeightInitSmoke

open TinyGrad4

private def assertSize (arr : FloatArray) (expected : Nat) (label : String) : IO Unit := do
  if arr.size != expected then
    throw (IO.userError s!"{label}: size mismatch {arr.size} != {expected}")

private def assertRange (arr : FloatArray) (lo hi : Float) (label : String) : IO Unit := do
  let mut ok := true
  for i in [:arr.size] do
    let v := arr[i]!
    if v < lo || v > hi then
      ok := false
  if !ok then
    throw (IO.userError s!"{label}: values out of range")

def runAll : IO Unit := do
  IO.println "=== WeightInitSmoke Tests ==="
  let shape := [4, 8]
  let (fanIn, fanOut) := Init.fanInOut shape .inOutLast
  let denom := Float.ofNat (max 1 (fanIn + fanOut))
  let bound := Float.sqrt (6.0 / denom)
  let w := Init.xavierUniformF32 shape 123
  let wArr := w.decodeF32
  assertSize wArr (Shape.numel shape) "xavier size"
  assertRange wArr (-bound - 1.0e-6) (bound + 1.0e-6) "xavier range"

  let kshape := [2, 3]
  let (fanInK, _) := Init.fanInOut kshape .inOutLast
  let denomK := Float.ofNat (max 1 fanInK)
  let gain := Float.sqrt (2.0 / 1.0)
  let kbound := gain * Float.sqrt (3.0 / denomK)
  let w2 := Init.kaimingUniformF32 kshape 321
  let w2Arr := w2.decodeF32
  assertSize w2Arr (Shape.numel kshape) "kaiming size"
  assertRange w2Arr (-kbound - 1.0e-6) (kbound + 1.0e-6) "kaiming range"
  IO.println "=== WeightInitSmoke OK ==="

end TinyGrad4.Test.WeightInitSmoke

