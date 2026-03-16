import Float64
import TinyGrad4

/-!
# SGDUpdate Smoke Test

Tests the ByteArray-level SGD update kernel:
`w <- w - lr * grad`
-/

namespace TinyGrad4.Test.SGDUpdateSmoke

-- Disable RawBuffer linter for test files that need Array Float64 literals
set_option linter.useRawBuffer false

open TinyGrad4
open Backend
open Interpreter

/-- Pack float64 array to float32 bytes -/
private def packF32 (data : Array Float64) : ByteArray :=
  Native.packF32FromF64 ⟨data⟩

private def assertAllClose (arr : Array Float64) (expected : Array Float64) (tol : Float64) (label : String) : IO Unit := do
  if arr.size != expected.size then
    throw (IO.userError s!"{label}: size {arr.size} != {expected.size}")
  for i in [:arr.size] do
    let v := arr[i]!
    let e := expected[i]!
    let diff := Float64.abs (v - e)
    if diff > tol then
      throw (IO.userError s!"{label}: idx {i} value {v} expected {e} diff {diff} > {tol}")

private def testSgdUpdate : IO Unit := do
  let w : Array Float64 := #[1.0, 2.0, 3.0]
  let g : Array Float64 := #[0.1, 0.2, -0.3]
  let lr : Float64 := 0.5

  let wb := packF32 w
  let gb := packF32 g
  let outb := Native.sgdUpdateF32 wb gb lr
  let out := (RawBuffer.decode { dtype := .float32, data := outb }).data

  assertAllClose out #[0.95, 1.9, 3.15] 0.002 "sgdUpdateF32"

def runAll : IO Unit := do
  IO.println "=== SGDUpdateSmoke Tests ==="
  testSgdUpdate
  IO.println "✓ sgd update f32"
  IO.println "=== SGDUpdateSmoke OK ==="

end TinyGrad4.Test.SGDUpdateSmoke

#eval! TinyGrad4.Test.SGDUpdateSmoke.runAll

