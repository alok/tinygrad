import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import TinyGrad4.Backend.View
import TinyGrad4.Backend.ShapeTracker
import Std.Data.HashMap

namespace TinyGrad4.Backend.FusedContract

open Std

/-!
# Fused Contract (Matrix Multiplication) Pattern Matching

Matches the CONTRACT operation which performs tensor contraction:
  C = A @ B  (matrix multiplication)

For 2D matmul: A[m,k] @ B[k,n] = C[m,n]
For batched: A[...,m,k] @ B[...,k,n] = C[...,m,n]

The contraction happens along the last dimension of A and second-to-last of B.
-/

structure Plan where
  aBase : UOpId
  bBase : UOpId
  aNumel : Nat
  bNumel : Nat
  aStackShapes : Array (Array Nat)
  aStackStrides : Array (Array Int64)
  aStackOffsets : Array Int64
  aStackMaskStarts : Array (Array Nat)
  aStackMaskEnds : Array (Array Nat)
  bStackShapes : Array (Array Nat)
  bStackStrides : Array (Array Int64)
  bStackOffsets : Array Int64
  bStackMaskStarts : Array (Array Nat)
  bStackMaskEnds : Array (Array Nat)
  aStrides : Array Int64
  aOffset : Int64
  aMaskStarts : Array Nat
  aMaskEnds : Array Nat
  bStrides : Array Int64
  bOffset : Int64
  bMaskStarts : Array Nat
  bMaskEnds : Array Nat
  needsStack : Bool
  k : Nat
  deriving Repr

namespace Plan

def mapIds (p : Plan) (f : UOpId → UOpId) : Plan :=
  { p with aBase := f p.aBase, bBase := f p.bBase }

end Plan

/-- Compute number of elements from shape -/
private def numel (shape : Shape) : Nat :=
  shape.foldl (· * ·) 1

/-- Empty stack info -/
private def emptyStackInfo : Backend.StackInfo :=
  { shapes := #[], strides := #[], offsets := #[], maskStarts := #[], maskEnds := #[] }

/--
Pattern match for CONTRACT operation (matrix multiplication).
CONTRACT(a, b) computes tensor contraction along last dim of a and second-to-last of b.
-/
def compile (u : UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : Option Plan := Id.run do
  -- Only handle float32
  if u.dtype != .float32 then return none

  -- Ignore keep/refCnt for this pattern (could use for optimization later)
  let _ := (keep, refCnt)

  -- Root must be CONTRACT
  if u.op != .CONTRACT then return none

  -- Get two sources
  let [a, b] := u.src | return none

  -- Try to get views for a and b using ShapeTracker
  let aResult? := Backend.ShapeTracker.ofUOp? a
  let bResult? := Backend.ShapeTracker.ofUOp? b

  let some (aBase, aSt) := aResult? | return none
  let some (bBase, bSt) := bResult? | return none

  -- Get top views
  let aView := Backend.ShapeTracker.top aSt
  let bView := Backend.ShapeTracker.top bSt

  -- Compute k (contraction dimension)
  -- For matmul A[...,m,k] @ B[...,k,n], k is last dim of A
  let aShape := a.shape
  if aShape.isEmpty then return none
  let k := aShape.getLast!

  -- Compute numels
  let aNumel := numel a.shape
  let bNumel := numel b.shape

  -- Determine if stack-based access is needed
  let aNeedsStack := Backend.ShapeTracker.needsStack aSt
  let bNeedsStack := Backend.ShapeTracker.needsStack bSt
  let needsStack := aNeedsStack || bNeedsStack

  -- Get stack info
  let aStackInfo := if aNeedsStack then Backend.ShapeTracker.stackInfo aSt else emptyStackInfo
  let bStackInfo := if bNeedsStack then Backend.ShapeTracker.stackInfo bSt else emptyStackInfo

  return some {
    aBase
    bBase
    aNumel
    bNumel
    aStackShapes := aStackInfo.shapes
    aStackStrides := aStackInfo.strides
    aStackOffsets := aStackInfo.offsets
    aStackMaskStarts := aStackInfo.maskStarts
    aStackMaskEnds := aStackInfo.maskEnds
    bStackShapes := bStackInfo.shapes
    bStackStrides := bStackInfo.strides
    bStackOffsets := bStackInfo.offsets
    bStackMaskStarts := bStackInfo.maskStarts
    bStackMaskEnds := bStackInfo.maskEnds
    aStrides := aView.strides
    aOffset := aView.offset
    aMaskStarts := aView.maskStart
    aMaskEnds := aView.maskEnd
    bStrides := bView.strides
    bOffset := bView.offset
    bMaskStarts := bView.maskStart
    bMaskEnds := bView.maskEnd
    needsStack
    k
  }

/-- Return array of plan variants (currently just 0 or 1) -/
def compileVariants (u : UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : Array Plan :=
  match compile u keep refCnt with
  | some p => #[p]
  | none => #[]

end TinyGrad4.Backend.FusedContract
