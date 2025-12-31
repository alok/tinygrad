import TinyGrad4
import TinyGrad4.Spec

/-!
# SpecBroadcastIndexingSmoke

Quick checks for the spec layer's broadcast/stride rules and basic indexing shape inference.
-/

namespace TinyGrad4.Test.SpecBroadcastIndexingSmoke

open TinyGrad4
open TinyGrad4.Spec

private def assertEq {α} [DecidableEq α] (got expected : α) (msg : String) : IO Unit := do
  if got != expected then
    throw (IO.userError msg)

private def testBroadcastStrides : IO Unit := do
  assertEq (broadcastStrides [] [] []) (some ([] : List Int)) "broadcastStrides empty"
  assertEq (broadcastStrides [5] [8] [5]) (some ([8] : List Int)) "broadcastStrides same"
  assertEq (broadcastStrides [5] [8] [10, 5]) (some ([0, 8] : List Int)) "broadcastStrides prefix"
  assertEq (broadcastStrides [5] [8] [15, 10, 5]) (some ([0, 0, 8] : List Int)) "broadcastStrides multi"
  assertEq (broadcastStrides [1] [0] []) none "broadcastStrides rank mismatch"
  assertEq (broadcastStrides [1, 2, 3] [0, 2, 8] [2, 3]) none "broadcastStrides invalid dims"
  assertEq (broadcastStrides [2, 2, 3] [0, 2, 8] [1, 2, 3]) none "broadcastStrides invalid expand"

private def testBasicIndexShape : IO Unit := do
  let shape : Shape := [10, 20, 30]
  let items : List BasicIndexItem := [.ellipsis, .int 2, .slice Slice.all]
  assertEq (inferBasicIndexShape shape items) (some [10, 30]) "ellipsis + int + slice"

  let items2 : List BasicIndexItem := [.newaxis, .slice Slice.all, .int 1]
  assertEq (inferBasicIndexShape [2, 3] items2) (some [1, 2]) "newaxis + slice + int"

  let items3 : List BasicIndexItem := [.int (-1)]
  assertEq (inferBasicIndexShape [5] items3) (some []) "negative int index"

  let sl : Slice := { start? := some 1, stop? := some 5, step? := some 2 }
  let items4 : List BasicIndexItem := [.slice sl]
  assertEq (inferBasicIndexShape [5] items4) (some [2]) "slice size"

  let items5 : List BasicIndexItem := [.int 1]
  assertEq (inferBasicIndexShape [2, 3, 4] items5) (some [3, 4]) "implicit trailing slices"

def runAll : IO Unit := do
  IO.println "=== SpecBroadcastIndexingSmoke Tests ==="
  testBroadcastStrides
  IO.println "✓ broadcastStrides"
  testBasicIndexShape
  IO.println "✓ basic indexing shape"
  IO.println "=== SpecBroadcastIndexingSmoke OK ==="

end TinyGrad4.Test.SpecBroadcastIndexingSmoke

#eval! TinyGrad4.Test.SpecBroadcastIndexingSmoke.runAll
