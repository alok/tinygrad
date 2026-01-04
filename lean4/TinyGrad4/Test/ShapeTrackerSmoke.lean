import TinyGrad4

/-!
# ShapeTrackerSmoke

Checks view-stack construction for a permute→reshape chain.
-/

namespace TinyGrad4.Test.ShapeTrackerSmoke

open TinyGrad4
open TinyGrad4.Backend

private def assertEq {α} [DecidableEq α] (got expected : α) (msg : String) : IO Unit := do
  if got != expected then
    throw (IO.userError msg)

private def testPermuteReshapeStack : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [2, 3, 4]
      let p ← UOp.permute x [1, 0, 2]
      let r ← UOp.reshape p [6, 4]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed")
  | some (base, st) =>
    assertEq base baseId "base id mismatch"
    assertEq st.views.size 2 "expected view stack size 2"
    let info := Backend.ShapeTracker.stackInfo st
    assertEq info.shapes.size 2 "expected stack shape size 2"
    assertEq info.shapes[0]! #[3, 2, 4] "permute view shape"
    assertEq info.shapes[1]! #[6, 4] "reshape view shape"
    assertEq (Backend.ShapeTracker.kernelShape st) #[6, 4] "kernel shape"
    assertEq (Backend.ShapeTracker.needsStack st) true "needsStack"
    assertEq (Backend.ShapeTracker.indexOps st) 8 "indexOps"

private def testContiguousStack : IO Unit := do
  let st := Backend.ShapeTracker.contiguous [2, 3]
  assertEq st.views.size 1 "contiguous stack size"
  assertEq (Backend.ShapeTracker.needsStack st) false "contiguous needsStack"
  assertEq (Backend.ShapeTracker.indexOps st) 0 "contiguous indexOps"

private def testUnitDimReshapeNoStack : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [2, 3, 1]
      let p ← UOp.permute x [1, 0, 2]
      let r ← UOp.reshape p [3, 2]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (unit-dim reshape)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (unit-dim reshape)"
    assertEq st.views.size 1 "expected unit-dim reshape to stay in one view"
    assertEq (Backend.ShapeTracker.top st).shape #[3, 2] "unit-dim reshape shape"
    assertEq (Backend.ShapeTracker.needsStack st) false "unit-dim reshape should not need stack"

private def testUnitDimUnsqueezeNoStack : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [2, 3]
      let p ← UOp.permute x [1, 0]
      let r ← UOp.reshape p [1, 3, 2]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (unit-dim unsqueeze)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (unit-dim unsqueeze)"
    assertEq st.views.size 1 "expected unit-dim unsqueeze to stay in one view"
    assertEq (Backend.ShapeTracker.top st).shape #[1, 3, 2] "unit-dim unsqueeze shape"
    assertEq (Backend.ShapeTracker.needsStack st) false "unit-dim unsqueeze should not need stack"

private def testReshapeFoldNonContig : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [2, 3, 4]
      let s ← UOp.shrink x [(0, 2), (0, 2), (0, 4)]
      let r ← UOp.reshape s [2, 8]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (reshape fold non-contig)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (reshape fold non-contig)"
    assertEq st.views.size 1 "expected reshape fold to stay in one view"
    assertEq (Backend.ShapeTracker.top st).shape #[2, 8] "reshape fold shape"
    assertEq (Backend.ShapeTracker.top st).strides #[(12 : Int64), (1 : Int64)] "reshape fold strides"

private def testReshapeNeedsStackForGap : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [2, 3, 4]
      let s ← UOp.shrink x [(0, 2), (0, 2), (0, 4)]
      let r ← UOp.reshape s [4, 4]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (reshape gap)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (reshape gap)"
    assertEq st.views.size 2 "expected reshape gap to require stack"
    assertEq (Backend.ShapeTracker.top st).shape #[4, 4] "reshape gap shape"
    assertEq (Backend.ShapeTracker.needsStack st) true "reshape gap needsStack"

private def testReshapeSimplifyBack : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [2, 3, 4]
      let p ← UOp.permute x [1, 0, 2]
      let r0 ← UOp.reshape p [6, 4]
      let r1 ← UOp.reshape r0 [3, 2, 4]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r1)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (reshape simplify back)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (reshape simplify back)"
    assertEq st.views.size 1 "expected reshape simplify to collapse stack"
    assertEq (Backend.ShapeTracker.top st).shape #[3, 2, 4] "reshape simplify shape"
    assertEq (Backend.ShapeTracker.top st).strides #[(4 : Int64), (12 : Int64), (1 : Int64)] "reshape simplify strides"

private def testPadReshapeNeedsStack : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [2, 3]
      let p ← UOp.pad x [(1, 1), (0, 0)]
      let r ← UOp.reshape p [2, 6]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (pad→reshape)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (pad→reshape)"
    assertEq st.views.size 2 "expected pad→reshape to require stack"
    assertEq (Backend.ShapeTracker.top st).shape #[2, 6] "pad→reshape shape"
    assertEq (Backend.ShapeTracker.needsStack st) true "pad→reshape needsStack"

private def testPadReshapeFoldInnerDims : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [2, 3, 4]
      let p ← UOp.pad x [(1, 1), (0, 0), (0, 0)]
      let r ← UOp.reshape p [4, 12]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (pad→reshape fold inner dims)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (pad→reshape fold inner dims)"
    assertEq st.views.size 1 "expected pad→reshape (inner dims) to fold"
    assertEq (Backend.ShapeTracker.top st).shape #[4, 12] "pad→reshape fold shape"
    assertEq (Backend.ShapeTracker.top st).strides #[(12 : Int64), (1 : Int64)] "pad→reshape fold strides"
    assertEq (Backend.ShapeTracker.top st).maskStart #[1, 0] "pad→reshape fold maskStart"
    assertEq (Backend.ShapeTracker.top st).maskEnd #[3, 12] "pad→reshape fold maskEnd"

private def testExpandReshapeNeedsStack : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [1, 3]
      let e ← UOp.expand x [2, 3]
      let r ← UOp.reshape e [6]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (expand→reshape)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (expand→reshape)"
    assertEq st.views.size 2 "expected expand→reshape to require stack"
    assertEq (Backend.ShapeTracker.top st).shape #[6] "expand→reshape shape"
    assertEq (Backend.ShapeTracker.needsStack st) true "expand→reshape needsStack"

private def testFlipReshapeNeedsStack : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [2, 3]
      let f ← UOp.flip x [1]
      let r ← UOp.reshape f [3, 2]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (flip→reshape)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (flip→reshape)"
    assertEq st.views.size 2 "expected flip→reshape to require stack"
    assertEq (Backend.ShapeTracker.top st).shape #[3, 2] "flip→reshape shape"
    assertEq (Backend.ShapeTracker.needsStack st) true "flip→reshape needsStack"

private def testNoopReshapeAfterPad : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [2, 3]
      let p ← UOp.pad x [(1, 1), (0, 0)]
      let r ← UOp.reshape p [4, 3]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (pad→reshape noop)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (pad→reshape noop)"
    assertEq st.views.size 1 "expected pad→reshape noop to drop extra layer"
    assertEq (Backend.ShapeTracker.top st).shape #[4, 3] "pad→reshape noop shape"

private def testNoopReshapeAfterFlip : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [2, 3]
      let f ← UOp.flip x [1]
      let r ← UOp.reshape f [2, 3]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (flip→reshape noop)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (flip→reshape noop)"
    assertEq st.views.size 1 "expected flip→reshape noop to drop extra layer"
    assertEq (Backend.ShapeTracker.top st).shape #[2, 3] "flip→reshape noop shape"


def runAll : IO Unit := do
  IO.println "=== ShapeTrackerSmoke Tests ==="
  testPermuteReshapeStack
  IO.println "✓ permute→reshape stack"
  testContiguousStack
  IO.println "✓ contiguous stack"
  testUnitDimReshapeNoStack
  IO.println "✓ unit-dim reshape no stack"
  testUnitDimUnsqueezeNoStack
  IO.println "✓ unit-dim unsqueeze no stack"
  testReshapeFoldNonContig
  IO.println "✓ reshape fold non-contig"
  testReshapeNeedsStackForGap
  IO.println "✓ reshape gap needs stack"
  testReshapeSimplifyBack
  IO.println "✓ reshape simplify back"
  testPadReshapeNeedsStack
  IO.println "✓ pad→reshape needs stack"
  testPadReshapeFoldInnerDims
  IO.println "✓ pad→reshape fold inner dims"
  testExpandReshapeNeedsStack
  IO.println "✓ expand→reshape needs stack"
  testFlipReshapeNeedsStack
  IO.println "✓ flip→reshape needs stack"
  testNoopReshapeAfterPad
  IO.println "✓ pad→reshape noop simplifies"
  testNoopReshapeAfterFlip
  IO.println "✓ flip→reshape noop simplifies"
  IO.println "=== ShapeTrackerSmoke OK ==="

end TinyGrad4.Test.ShapeTrackerSmoke

