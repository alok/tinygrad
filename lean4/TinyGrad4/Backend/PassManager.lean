import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import Std.Data.HashMap

namespace TinyGrad4.Backend

open Std

/-!
# Lightweight Pass Manager for UOp Fusion

Provides a priority-based pass system for selecting fusion implementations.
Inspired by Lean's LCNF pass manager patterns but specialized for UOp graph optimization.

## Design Principles
1. **Priority-based**: Higher priority passes tried first
2. **First-match wins**: Once a pass succeeds, stop trying others
3. **Coverage tracking**: Each pass returns nodes it covers to avoid conflicts
4. **Configurable**: Passes can be enabled/disabled via PassConfig
-/

/-- Configuration for pass execution -/
structure PassConfig where
  /-- Enable matmul fusion -/
  enableMatmul : Bool := true
  /-- Enable softmax fusion -/
  enableSoftmax : Bool := true
  /-- Enable reduce fusion -/
  enableReduce : Bool := true
  /-- Enable element-wise fusion (fallback) -/
  enableEwise : Bool := true
  /-- Enable tracing/logging -/
  trace : Bool := false
  deriving Repr, Inhabited

/-- Result of a pass attempt -/
structure PassResult (Impl : Type) where
  /-- The implementation if successful -/
  impl : Impl
  /-- Nodes covered by this implementation -/
  cover : UOpIdSet
  deriving Repr

/-- A fusion pass with priority -/
structure Pass (Impl : Type) where
  /-- Human-readable name for logging -/
  name : String
  /-- Priority (higher = tried earlier) -/
  priority : Nat := 0
  /-- Quick check if pass could apply -/
  canApply : UOp → Bool
  /-- Try to apply the pass -/
  apply : UOp → UOpIdSet → HashMap UOpId Nat → Option (PassResult Impl)

namespace Pass

/-- Create a pass from a compile function -/
def ofCompile (name : String) (priority : Nat) (canApply : UOp → Bool)
    (compile : UOp → UOpIdSet → HashMap UOpId Nat → Option (Impl × UOpIdSet)) : Pass Impl :=
  { name, priority, canApply
    apply := fun u keep refCnt =>
      match compile u keep refCnt with
      | some (impl, cover) => some { impl, cover }
      | none => none }

end Pass

/-- Pass manager with ordered pass list -/
structure PassManager (Impl : Type) where
  /-- Passes sorted by priority (highest first) -/
  passes : Array (Pass Impl)
  /-- Configuration -/
  config : PassConfig := {}

namespace PassManager

/-- Add a pass (maintains priority ordering) -/
def addPass (pm : PassManager Impl) (p : Pass Impl) : PassManager Impl :=
  let passes := pm.passes.push p
  let sorted := passes.qsort (fun a b => a.priority > b.priority)
  { pm with passes := sorted }

/-- Create a pass manager from a list of passes -/
def ofPasses (passes : Array (Pass Impl)) (config : PassConfig := {}) : PassManager Impl :=
  let sorted := passes.qsort (fun a b => a.priority > b.priority)
  { passes := sorted, config }

/-- Check if two cover sets intersect -/
private def setIntersects (a b : UOpIdSet) : Bool :=
  a.fold (init := false) fun acc uid => acc || UOpIdSet.member b uid

/-- Union two cover sets -/
private def setUnion (a b : UOpIdSet) : UOpIdSet :=
  b.fold (init := a) fun acc uid => UOpIdSet.add acc uid

/-- Try each pass on a single node, return first success -/
def tryPassesOnNode (pm : PassManager Impl) (u : UOp)
    (keep : UOpIdSet) (refCnt : HashMap UOpId Nat)
    (covered : UOpIdSet) : Option (PassResult Impl) :=
  pm.passes.findSome? fun p =>
    if p.canApply u then
      match p.apply u keep refCnt with
      | some result =>
        -- Check for coverage conflict
        if !setIntersects result.cover covered then
          some result
        else
          none
      | none => none
    else
      none

/-- Run passes on entire graph in reverse topological order -/
def run (pm : PassManager Impl) (nodes : List UOp)
    (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : HashMap UOpId Impl := Id.run do
  let mut implMap : HashMap UOpId Impl := ∅
  let mut covered : UOpIdSet := UOpIdSet.mkEmpty

  for u in nodes.reverse do
    if UOpIdSet.member covered u.uid then
      continue
    match pm.tryPassesOnNode u keep refCnt covered with
    | some result =>
      implMap := implMap.insert u.uid result.impl
      covered := setUnion covered result.cover
    | none => pure ()

  implMap

end PassManager

end TinyGrad4.Backend
