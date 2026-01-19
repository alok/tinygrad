import Float64
import TinyGrad4

/-!
# ShapeTrackerSmoke

Checks view-stackUnsafe construction for a permuteUnsafe→reshapeUnsafe chain.
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
    assertEq st.views.size 2 "expected view stackUnsafe size 2"
    let info := Backend.ShapeTracker.stackInfo st
    assertEq info.shapes.size 2 "expected stackUnsafe shape size 2"
    assertEq info.shapes[0]! #[3, 2, 4] "permuteUnsafe view shape"
    assertEq info.shapes[1]! #[6, 4] "reshapeUnsafe view shape"
    assertEq (Backend.ShapeTracker.kernelShape st) #[6, 4] "kernel shape"
    assertEq (Backend.ShapeTracker.needsStack st) true "needsStack"
    assertEq (Backend.ShapeTracker.indexOps st) 8 "indexOps"

private def testContiguousStack : IO Unit := do
  let st := Backend.ShapeTracker.contiguous [2, 3]
  assertEq st.views.size 1 "contiguous stackUnsafe size"
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
    throw (IO.userError "ShapeTracker.ofUOp? failed (unit-dim reshapeUnsafe)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (unit-dim reshapeUnsafe)"
    assertEq st.views.size 1 "expected unit-dim reshapeUnsafe to stay in one view"
    assertEq (Backend.ShapeTracker.top st).shape #[3, 2] "unit-dim reshapeUnsafe shape"
    assertEq (Backend.ShapeTracker.needsStack st) false "unit-dim reshapeUnsafe should not need stackUnsafe"

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
    assertEq (Backend.ShapeTracker.needsStack st) false "unit-dim unsqueeze should not need stackUnsafe"

private def testReshapeFoldNonContig : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [2, 3, 4]
      let s ← UOp.shrink x [(0, 2), (0, 2), (0, 4)]
      let r ← UOp.reshape s [2, 8]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (reshapeUnsafe fold non-contig)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (reshapeUnsafe fold non-contig)"
    assertEq st.views.size 1 "expected reshapeUnsafe fold to stay in one view"
    assertEq (Backend.ShapeTracker.top st).shape #[2, 8] "reshapeUnsafe fold shape"
    assertEq (Backend.ShapeTracker.top st).strides #[(12 : Int64), (1 : Int64)] "reshapeUnsafe fold strides"

private def testReshapeNeedsStackForGap : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [2, 3, 4]
      let s ← UOp.shrink x [(0, 2), (0, 2), (0, 4)]
      let r ← UOp.reshape s [4, 4]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (reshapeUnsafe gap)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (reshapeUnsafe gap)"
    assertEq st.views.size 2 "expected reshapeUnsafe gap to require stackUnsafe"
    assertEq (Backend.ShapeTracker.top st).shape #[4, 4] "reshapeUnsafe gap shape"
    assertEq (Backend.ShapeTracker.needsStack st) true "reshapeUnsafe gap needsStack"

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
    throw (IO.userError "ShapeTracker.ofUOp? failed (reshapeUnsafe simplify back)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (reshapeUnsafe simplify back)"
    assertEq st.views.size 1 "expected reshapeUnsafe simplify to collapse stackUnsafe"
    assertEq (Backend.ShapeTracker.top st).shape #[3, 2, 4] "reshapeUnsafe simplify shape"
    assertEq (Backend.ShapeTracker.top st).strides #[(4 : Int64), (12 : Int64), (1 : Int64)] "reshapeUnsafe simplify strides"

private def testPadReshapeNeedsStack : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [2, 3]
      let p ← UOp.pad x [(1, 1), (0, 0)]
      let r ← UOp.reshape p [2, 6]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (padUnsafe→reshapeUnsafe)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (padUnsafe→reshapeUnsafe)"
    assertEq st.views.size 2 "expected padUnsafe→reshapeUnsafe to require stackUnsafe"
    assertEq (Backend.ShapeTracker.top st).shape #[2, 6] "padUnsafe→reshapeUnsafe shape"
    assertEq (Backend.ShapeTracker.needsStack st) true "padUnsafe→reshapeUnsafe needsStack"

private def testPadReshapeFoldInnerDims : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [2, 3, 4]
      let p ← UOp.pad x [(1, 1), (0, 0), (0, 0)]
      let r ← UOp.reshape p [4, 12]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (padUnsafe→reshapeUnsafe fold inner dims)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (padUnsafe→reshapeUnsafe fold inner dims)"
    assertEq st.views.size 1 "expected padUnsafe→reshapeUnsafe (inner dims) to fold"
    assertEq (Backend.ShapeTracker.top st).shape #[4, 12] "padUnsafe→reshapeUnsafe fold shape"
    assertEq (Backend.ShapeTracker.top st).strides #[(12 : Int64), (1 : Int64)] "padUnsafe→reshapeUnsafe fold strides"
    assertEq (Backend.ShapeTracker.top st).maskStart #[1, 0] "padUnsafe→reshapeUnsafe fold maskStart"
    assertEq (Backend.ShapeTracker.top st).maskEnd #[3, 12] "padUnsafe→reshapeUnsafe fold maskEnd"

private def testExpandReshapeNeedsStack : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [1, 3]
      let e ← UOp.expand x [2, 3]
      let r ← UOp.reshape e [6]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (expandUnsafe→reshapeUnsafe)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (expandUnsafe→reshapeUnsafe)"
    assertEq st.views.size 2 "expected expandUnsafe→reshapeUnsafe to require stackUnsafe"
    assertEq (Backend.ShapeTracker.top st).shape #[6] "expandUnsafe→reshapeUnsafe shape"
    assertEq (Backend.ShapeTracker.needsStack st) true "expandUnsafe→reshapeUnsafe needsStack"

private def testFlipReshapeNeedsStack : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [2, 3]
      let f ← UOp.flip x [1]
      let r ← UOp.reshape f [3, 2]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (flipUnsafe→reshapeUnsafe)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (flipUnsafe→reshapeUnsafe)"
    assertEq st.views.size 2 "expected flipUnsafe→reshapeUnsafe to require stackUnsafe"
    assertEq (Backend.ShapeTracker.top st).shape #[3, 2] "flipUnsafe→reshapeUnsafe shape"
    assertEq (Backend.ShapeTracker.needsStack st) true "flipUnsafe→reshapeUnsafe needsStack"

private def testNoopReshapeAfterPad : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [2, 3]
      let p ← UOp.pad x [(1, 1), (0, 0)]
      let r ← UOp.reshape p [4, 3]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (padUnsafe→reshapeUnsafe noop)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (padUnsafe→reshapeUnsafe noop)"
    assertEq st.views.size 1 "expected padUnsafe→reshapeUnsafe noop to drop extra layer"
    assertEq (Backend.ShapeTracker.top st).shape #[4, 3] "padUnsafe→reshapeUnsafe noop shape"

private def testNoopReshapeAfterFlip : IO Unit := do
  let (baseId, st?) := Id.run do
    runTensorM do
      let x ← UOp.buffer .float32 [2, 3]
      let f ← UOp.flip x [1]
      let r ← UOp.reshape f [2, 3]
      pure (x.uid, Backend.ShapeTracker.ofUOp? r)
  match st? with
  | none =>
    throw (IO.userError "ShapeTracker.ofUOp? failed (flipUnsafe→reshapeUnsafe noop)")
  | some (base, st) =>
    assertEq base baseId "base id mismatch (flipUnsafe→reshapeUnsafe noop)"
    assertEq st.views.size 1 "expected flipUnsafe→reshapeUnsafe noop to drop extra layer"
    assertEq (Backend.ShapeTracker.top st).shape #[2, 3] "flipUnsafe→reshapeUnsafe noop shape"


def runAll : IO Unit := do
  IO.println "=== ShapeTrackerSmoke Tests ==="
  testPermuteReshapeStack
  IO.println "✓ permuteUnsafe→reshapeUnsafe stackUnsafe"
  testContiguousStack
  IO.println "✓ contiguous stackUnsafe"
  testUnitDimReshapeNoStack
  IO.println "✓ unit-dim reshapeUnsafe no stackUnsafe"
  testUnitDimUnsqueezeNoStack
  IO.println "✓ unit-dim unsqueeze no stackUnsafe"
  testReshapeFoldNonContig
  IO.println "✓ reshapeUnsafe fold non-contig"
  testReshapeNeedsStackForGap
  IO.println "✓ reshapeUnsafe gap needs stackUnsafe"
  testReshapeSimplifyBack
  IO.println "✓ reshapeUnsafe simplify back"
  testPadReshapeNeedsStack
  IO.println "✓ padUnsafe→reshapeUnsafe needs stackUnsafe"
  testPadReshapeFoldInnerDims
  IO.println "✓ padUnsafe→reshapeUnsafe fold inner dims"
  testExpandReshapeNeedsStack
  IO.println "✓ expandUnsafe→reshapeUnsafe needs stackUnsafe"
  testFlipReshapeNeedsStack
  IO.println "✓ flipUnsafe→reshapeUnsafe needs stackUnsafe"
  testNoopReshapeAfterPad
  IO.println "✓ padUnsafe→reshapeUnsafe noop simplifies"
  testNoopReshapeAfterFlip
  IO.println "✓ flipUnsafe→reshapeUnsafe noop simplifies"
  IO.println "=== ShapeTrackerSmoke OK ==="

end TinyGrad4.Test.ShapeTrackerSmoke

#eval! TinyGrad4.Test.ShapeTrackerSmoke.runAll
