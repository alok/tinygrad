import Float64
import TinyGrad4
import TinyGrad4.Spec

/-!
# UOpSpecSmoke

Smoke checks for the lower-level `TinyGrad4.Spec.UOpSpec` layer.
-/

namespace TinyGrad4.Test.UOpSpecSmoke

open TinyGrad4
open TinyGrad4.Spec
open TinyGrad4.Spec.UOpSpec

private def assertEq {α} [DecidableEq α] (got expected : α) (msg : String) : IO Unit := do
  if got != expected then
    throw (IO.userError msg)

private def assertSome (got : Option UOpSig) (expected : UOpSig) (msg : String) : IO Unit := do
  match got with
  | some sig => assertEq sig expected msg
  | none => throw (IO.userError s!"{msg}: expected some, got none")

private def testHighLevelGraph : IO Unit := do
  let outU := runTensorM do
    let q ← Tensor.buffer [2, 3, 2] .float32
    let k ← Tensor.buffer [2, 3, 2] .float32
    let v ← Tensor.buffer [2, 3, 2] .float32
    let kT ← StaticTensor.permuteUnsafe k [0, 2, 1]
    let probs ← UOp.contract2D q.uop kT.uop
    UOp.contract2D probs v.uop
  assertSome (check? outU)
    { dtype := .float32, shape := [2, 3, 2], device? := some .CPU }
    "uop spec high-level contract"

private def testManualLowLevelNodes : IO Unit := do
  let (buf, idx, special, src, lin, prog) := runUOpM do
    let buf ← UOp.buffer .float32 [4]
    let end_ ← UOp.constNat .index 4
    let idx ← UOp.range end_ 0
    let special ← UOp.special end_ "gidx0"
    let src ← UOp.source "kernel void test() {}"
    let lin ← UOp.linear [buf]
    let sink : UOp := { uid := ⟨9000⟩, op := .SINK, dtype := .void, src := [buf], arg := .empty, shape := [] }
    let prog ← UOp.program sink
    pure (buf, idx, special, src, lin, prog)

  assertSome (check? buf)
    { dtype := .float32, shape := [4], device? := some .CPU }
    "buffer"
  assertSome (check? idx)
    { dtype := .index, shape := [], device? := none }
    "range"
  assertSome (check? special)
    { dtype := .index, shape := [], device? := none }
    "special"
  assertSome (check? src)
    { dtype := .void, shape := [], device? := none }
    "source"
  assertSome (check? lin)
    { dtype := .void, shape := [], device? := none }
    "linear"
  assertSome (check? prog)
    { dtype := .void, shape := [], device? := none }
    "program"

private def testBackendFacingNodes : IO Unit := do
  let (_, _, load, store, after, copyU) := runUOpM do
    let buf ← UOp.buffer .float32 [4]
    let end_ ← UOp.constNat .index 4
    let idx ← UOp.range end_ 0
    let load : UOp := { uid := ⟨9100⟩, op := .LOAD, dtype := .float32, src := [buf, idx], arg := .empty, shape := [] }
    let store : UOp := { uid := ⟨9101⟩, op := .STORE, dtype := .void, src := [buf, idx, load], arg := .empty, shape := [4] }
    let after : UOp := { uid := ⟨9102⟩, op := .AFTER, dtype := .float32, src := [load, store], arg := .empty, shape := [] }
    let copyU : UOp := { uid := ⟨9103⟩, op := .COPY, dtype := .float32, src := [buf, idx], arg := .device "CUDA", shape := [4] }
    pure (buf, idx, load, store, after, copyU)

  assertSome (check? load)
    { dtype := .float32, shape := [], device? := some .CPU }
    "load"
  assertSome (check? store)
    { dtype := .void, shape := [4], device? := some .CPU }
    "store"
  assertSome (check? after)
    { dtype := .float32, shape := [], device? := some .CPU }
    "after"
  assertSome (check? copyU)
    { dtype := .float32, shape := [4], device? := some .CUDA }
    "copy"

private def testExplicitTags : IO Unit := do
  let tagged := runTensorM do
    let x ← Tensor.buffer [8] .float32
    pure (axis[2] (device["CUDA"] (fusion["ewise"] (cost[7] x.uop))))
  assertEq tagged.metaInfo.fusionTag? (some "ewise") "fusion tag"
  assertEq tagged.metaInfo.costTag? (some 7) "cost tag"
  assertEq tagged.metaInfo.deviceTag? (some .CUDA) "device tag"
  assertEq tagged.metaInfo.shardAxis? (some 2) "axis tag"
  assertSome (check? tagged)
    { dtype := .float32, shape := [8], device? := some .CUDA, shardAxis? := some 2 }
    "tagged signature"

def runAll : IO Unit := do
  IO.println "=== UOpSpecSmoke Tests ==="
  testHighLevelGraph
  IO.println "✓ high-level graph"
  testManualLowLevelNodes
  IO.println "✓ manual low-level nodes"
  testBackendFacingNodes
  IO.println "✓ backend-facing nodes"
  testExplicitTags
  IO.println "✓ explicit tags"
  IO.println "=== UOpSpecSmoke OK ==="

end TinyGrad4.Test.UOpSpecSmoke

#eval! TinyGrad4.Test.UOpSpecSmoke.runAll
