import TinyGrad4

/-!
# Fused Softmax Kernel Tests

Directly exercises the portable C softmax/log-softmax kernels:
- stable last-axis softmax
- stable last-axis log-softmax
-/

namespace TinyGrad4.Test.FusedSoftmax

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

open TinyGrad4
open Interpreter
open Backend

/-- Pack float64 array to float32 bytes -/
private def packF32 (data : Array Float) : ByteArray :=
  Native.packF32FromF64 ⟨data⟩

private def assertAllClose (arr : Array Float) (expected : Array Float) (tol : Float) (label : String) : IO Unit := do
  if arr.size != expected.size then
    throw (IO.userError s!"{label}: size {arr.size} != {expected.size}")
  for i in [:arr.size] do
    let v := arr[i]!
    let e := expected[i]!
    let diff := Float.abs (v - e)
    if diff > tol then
      throw (IO.userError s!"{label}: idx {i} value {v} expected {e} diff {diff} > {tol}")

private def testSoftmax2x3 : IO Unit := do
  let x : Array Float := #[1.0, 2.0, 3.0,  4.0, 5.0, 6.0]
  let xb := packF32 x
  -- Scale = 1.0 for standard softmax (the C impl uses expf, not exp2f)
  let scaleBits := (1.0 : Float).toFloat32.toBits
  let outBytes := Native.softmaxLastF32 xb 2 3 scaleBits
  let outRaw : RawBuffer := { dtype := .float32, data := outBytes }
  let out := outRaw.toFloatArray.data
  let p0 : Float := 0.0900306
  let p1 : Float := 0.244728
  let p2 : Float := 0.665241
  assertAllClose out #[p0, p1, p2,  p0, p1, p2] 0.0003 "softmax 2x3"

private def testLogSoftmax2x3 : IO Unit := do
  let x : Array Float := #[1.0, 2.0, 3.0,  4.0, 5.0, 6.0]
  let xb := packF32 x
  -- Scale = 1.0 for standard log-softmax
  -- ln2 = 1.0 to get natural log (the C code divides by ln2, so 1.0 = no conversion)
  let scaleBits := (1.0 : Float).toFloat32.toBits
  let ln2Bits := (1.0 : Float).toFloat32.toBits
  let outBytes := Native.logSoftmaxLastF32 xb 2 3 scaleBits ln2Bits
  let outRaw : RawBuffer := { dtype := .float32, data := outBytes }
  let out := outRaw.toFloatArray.data
  let a0 : Float := -2.407606
  let a1 : Float := -1.407606
  let a2 : Float := -0.407606
  assertAllClose out #[a0, a1, a2,  a0, a1, a2] 0.001 "logsoftmax 2x3"

def runAll : IO Unit := do
  IO.println "=== FusedSoftmax Tests ==="
  testSoftmax2x3
  IO.println "✓ softmax last axis"
  testLogSoftmax2x3
  IO.println "✓ logsoftmax last axis"
  IO.println "=== FusedSoftmax OK ==="

end TinyGrad4.Test.FusedSoftmax

#eval! TinyGrad4.Test.FusedSoftmax.runAll

