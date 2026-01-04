import TinyGrad4
import TinyGrad4.Data.ArrayN
import TinyGrad4.Data.Loader

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

/-!
# DataLoaderSmoke

Smoke tests for the `DataArrayN` and `DataLoader` interfaces.
-/

namespace TinyGrad4.Test.DataLoaderSmoke

open TinyGrad4

private def assertAllClose (arr : FloatArray) (expected : Array Float) (tol : Float) (label : String) : IO Unit := do
  if arr.size != expected.size then
    throw (IO.userError s!"{label}: size mismatch {arr.size} != {expected.size}")
  let mut ok := true
  for i in [:arr.size] do
    let diff := Float.abs (arr[i]! - expected[i]!)
    if diff > tol then
      ok := false
  if !ok then
    throw (IO.userError s!"{label}: values not close")

structure ToyLoader (batch : Nat) where
  xs : Array (DataArrayN (batch :: [2]) .float32)
  ys : Array (DataArrayN (batch :: [1]) .float32)

instance : DataLoader (ToyLoader batch) batch [2] [1] .float32 .float32 where
  numBatches := fun l => min l.xs.size l.ys.size
  getBatch := fun l i => do
    if i < l.xs.size && i < l.ys.size then
      pure { x := l.xs[i]!, y := l.ys[i]! }
    else
      throw (IO.userError s!"ToyLoader: batch {i} out of range")

def runAll : IO Unit := do
  IO.println "=== DataLoaderSmoke Tests ==="
  let x0 : DataArrayN [2, 2] .float32 := DataArrayN.ofArrayF32 [2, 2] #[1.0, 2.0, 3.0, 4.0]
  let y0 : DataArrayN [2, 1] .float32 := DataArrayN.ofArrayF32 [2, 1] #[0.0, 1.0]
  let x1 : DataArrayN [2, 2] .float32 := DataArrayN.ofArrayF32 [2, 2] #[5.0, 6.0, 7.0, 8.0]
  let y1 : DataArrayN [2, 1] .float32 := DataArrayN.ofArrayF32 [2, 1] #[1.0, 0.0]
  let loader : ToyLoader 2 := { xs := #[x0, x1], ys := #[y0, y1] }
  let b0 : Batch 2 [2] [1] .float32 .float32 ← DataLoader.getBatch loader 0
  assertAllClose (DataArrayN.decodeF32 b0.x) #[1.0, 2.0, 3.0, 4.0] 0.0001 "loader batch0 x"
  assertAllClose (DataArrayN.decodeF32 b0.y) #[0.0, 1.0] 0.0001 "loader batch0 y"
  let b1 : Batch 2 [2] [1] .float32 .float32 ← DataLoader.getBatch loader 1
  assertAllClose (DataArrayN.decodeF32 b1.x) #[5.0, 6.0, 7.0, 8.0] 0.0001 "loader batch1 x"
  assertAllClose (DataArrayN.decodeF32 b1.y) #[1.0, 0.0] 0.0001 "loader batch1 y"
  IO.println "=== DataLoaderSmoke OK ==="

end TinyGrad4.Test.DataLoaderSmoke
