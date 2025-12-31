import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import TinyGrad4.Backend.Pattern
import TinyGrad4.Backend.FusedEwise
import TinyGrad4.Backend.FusedReduce
import TinyGrad4.Backend.FusedContract
import TinyGrad4.Backend.FusedSGD
import TinyGrad4.Backend.FusedMatmul
import TinyGrad4.Backend.FusedSoftmax
import TinyGrad4.Backend.FusedLayerNorm
import TinyGrad4.Backend.FusedGELU
import TinyGrad4.Backend.PassManager
import TinyGrad4.Backend.Cost
import Std.Data.HashMap

namespace TinyGrad4.Backend

open Std

/-!
## Pattern Registry

Extensible pattern-based fusion selection. Patterns are registered with priorities
and tried in priority order (highest first).

### Adding New Patterns

1. Add pattern matcher in `Pattern.lean`: `def myPattern? (u : UOp) : Option MyInfo`
2. Add Impl variant: `| fusedMyPattern (plan : MyPattern.Plan)`
3. Register in `defaultPatterns` with appropriate priority
-/

-- CostModel and defaultCostModel now imported from TinyGrad4.Backend.Cost

namespace Fusion

inductive Impl where
  | node (deps : List UOpId)
  | fusedEwise (plan : FusedEwise.Plan)
  | fusedReduce (plan : FusedReduce.Plan)
  | fusedContract (plan : FusedContract.Plan)
  | fusedSGD (plan : FusedSGD.Plan)
  | fusedMatmul (plan : FusedMatmul.Plan)
  | fusedSoftmax (plan : FusedSoftmax.Plan)
  | fusedLayerNorm (plan : FusedLayerNorm.Plan)
  | fusedGELU (plan : FusedGELU.Plan)
  deriving Repr

namespace Impl

def tag : Impl → String
  | .node _ => "node"
  | .fusedEwise _ => "fused_ewise"
  | .fusedReduce _ => "fused_reduce"
  | .fusedContract _ => "fused_contract"
  | .fusedSGD _ => "fused_sgd"
  | .fusedMatmul _ => "fused_matmul"
  | .fusedSoftmax _ => "fused_softmax"
  | .fusedLayerNorm _ => "fused_layernorm"
  | .fusedGELU _ => "fused_gelu"


def deps : Impl → List UOpId
  | .node ds => ds
  | .fusedEwise plan => plan.leafBases.toList
  | .fusedReduce plan => plan.ewise.leafBases.toList
  | .fusedContract plan => [plan.aBase, plan.bBase]
  | .fusedSGD plan => [plan.w, plan.grad]
  | .fusedMatmul plan =>
      let base := [plan.aBase, plan.bBase, plan.bias]
      match plan.bias2 with
      | some b2 => base ++ [b2]
      | none => base
  | .fusedSoftmax plan => [plan.input]
  | .fusedLayerNorm plan =>
      let base := [plan.input]
      let withGamma := match plan.gamma with | some g => base ++ [g] | none => base
      match plan.beta with | some b => withGamma ++ [b] | none => withGamma
  | .fusedGELU plan => [plan.input]


def score (_impl : Impl) : Nat :=
  0


def mapIds (impl : Impl) (f : UOpId → UOpId) : Impl :=
  match impl with
  | .node ds => .node (ds.map f)
  | .fusedEwise plan => .fusedEwise (plan.mapIds f)
  | .fusedReduce plan => .fusedReduce (plan.mapIds f)
  | .fusedContract plan => .fusedContract (plan.mapIds f)
  | .fusedSGD plan => .fusedSGD (plan.mapIds f)
  | .fusedMatmul plan => .fusedMatmul (plan.mapIds f)
  | .fusedSoftmax plan => .fusedSoftmax (plan.mapIds f)
  | .fusedLayerNorm plan => .fusedLayerNorm (plan.mapIds f)
  | .fusedGELU plan => .fusedGELU (plan.mapIds f)

end Impl

/-! ## Pattern Registry -/

/-- Result of a pattern match: the implementation and its coverage set -/
structure PatternResult where
  impl : Impl
  cover : UOpIdSet
  deriving Repr

/-- A registered pattern with priority and matching function -/
structure PatternEntry where
  /-- Name for debugging -/
  name : String
  /-- Higher priority patterns are tried first -/
  priority : Nat
  /-- Root ops this pattern can match (empty = try all) -/
  rootOps : List Ops := []
  /-- The pattern matching function -/
  match_ : UOp → UOpIdSet → HashMap UOpId Nat → Option PatternResult

/-- Check if pattern should be tried for this op -/
def PatternEntry.appliesTo (p : PatternEntry) (op : Ops) : Bool :=
  p.rootOps.isEmpty || p.rootOps.contains op

/-- Compare patterns by priority (for sorting) -/
def PatternEntry.cmpPriority (a b : PatternEntry) : Ordering :=
  compare b.priority a.priority  -- Descending order

private def setUnion (a b : UOpIdSet) : UOpIdSet :=
  b.fold (init := a) fun acc uid => UOpIdSet.add acc uid

private def setIntersects (a b : UOpIdSet) : Bool :=
  a.fold (init := false) fun acc uid => acc || UOpIdSet.member b uid

private def allNonneg (arr : Array Int64) : Bool := Id.run do
  let mut ok := true
  for i in [:arr.size] do
    if arr[i]! < 0 then
      ok := false
  return ok

private def planScore (p : FusedMatmul.Plan) : Nat :=
  let fast :=
    (if p.aFast then 1 else 0) +
    (if p.bFast then 1 else 0) +
    (if p.biasFast then 1 else 0) +
    (if p.bias2Fast then 1 else 0)
  let stack := if p.needsStack then 0 else 4
  stack + fast

private def pickBestPlan (plans : Array FusedMatmul.Plan) : Option FusedMatmul.Plan :=
  match plans.toList with
  | [] => none
  | init :: rest =>
    let best := rest.foldl (fun best p => if planScore p > planScore best then p else best) init
    some best

private def planOk (p : FusedMatmul.Plan) : Bool :=
  !p.needsStack

/-! ## Cost Functions for Fusion Decisions -/

/-- Count of covered nodes from a plan's cover set. -/
private def coverCount (cover : UOpIdSet) : Nat :=
  cover.fold (init := 0) fun acc _ => acc + 1

/-- Extract cost features from an implementation plan. -/
def implCostFeatures (u : UOp) (impl : Impl) : CostFeatures :=
  let numel := listProd u.shape
  let itemsize := u.dtype.itemsize
  match impl with
  | .node _ =>
    -- Single node: 1 launch, read inputs, write output
    let readBytes := u.src.foldl (fun acc s => acc + listProd s.shape * s.dtype.itemsize) 0
    { launches := 1
      memRead := readBytes
      memWrite := numel * itemsize
      elemOps := numel }
  | .fusedEwise plan =>
    -- Fused elementwise: 1 launch, read all leaves, write output
    let readBytes := plan.leafBases.size * numel * 4  -- approximate
    { launches := 1
      memRead := readBytes
      memWrite := numel * itemsize
      elemOps := numel * (plan.prog.size / 2 + 1) }  -- rough op count
  | .fusedReduce plan =>
    let inNumel := plan.fullShape.foldl (· * ·) 1
    { launches := 1
      memRead := inNumel * 4
      memWrite := numel * itemsize
      reduceElems := inNumel }
  | .fusedMatmul plan =>
    -- M×K @ K×N matmul: M*N*K multiply-adds
    let mulAdds := plan.aKernelNumel * plan.bKernelNumel / (max plan.aKernelNumel 1)
    { launches := 1
      memRead := (plan.aNumel + plan.bNumel + plan.biasNumel) * 4
      memWrite := numel * itemsize
      matmulMulAdds := if plan.needsStack then 0 else mulAdds
      matmulViewMulAdds := if plan.needsStack then mulAdds else 0 }
  | .fusedSoftmax plan =>
    let inNumel := plan.outer * plan.inner
    { launches := 1
      memRead := inNumel * 4
      memWrite := inNumel * 4
      reduceElems := inNumel  -- max reduction
      elemOps := inNumel * 3 }  -- sub, exp, div
  | .fusedContract plan =>
    let mulAdds := plan.aNumel * plan.bNumel / (max plan.aNumel 1)
    { launches := 1
      memRead := (plan.aNumel + plan.bNumel) * 4
      memWrite := numel * itemsize
      matmulMulAdds := if plan.needsStack then 0 else mulAdds
      matmulViewMulAdds := if plan.needsStack then mulAdds else 0 }
  | .fusedSGD _ =>
    { launches := 1
      memRead := numel * 8  -- read w and grad
      memWrite := numel * 4
      elemOps := numel * 2 }  -- mul and sub
  | .fusedLayerNorm _ =>
    { launches := 1
      memRead := numel * 4
      memWrite := numel * 4
      reduceElems := numel * 2  -- mean and var
      elemOps := numel * 4 }
  | .fusedGELU _ =>
    { launches := 1
      memRead := numel * 4
      memWrite := numel * 4
      elemOps := numel * 5 }  -- tanh approximation

/-- Estimate cost of fusing vs not fusing based on kernel overhead.
    Returns true if fusion is worthwhile (saves more in launch overhead than costs in extra compute).

    fusedCost = 1 launch + fused compute
    unfusedCost = coverSize launches + individual compute (approximated as similar to fused)

    Key insight: fusion saves (coverSize - 1) kernel launches but may add overhead
    from view traversal in complex patterns. For very small tensors, this overhead
    can exceed the launch savings.
-/
def shouldFuse (cm : CostModel) (fusedFeatures : CostFeatures) (coverSize : Nat) : Bool :=
  -- For single-node patterns, check if kernelization overhead is worth it
  -- based on the compute cost. Very small tensors don't benefit from KERNEL wrapping.
  let fusedCompute := fusedFeatures.time cm
  let minWorthwhileCompute := cm.kernelOverhead * 2  -- need at least 2x overhead to be worth it

  if coverSize <= 1 then
    -- Single node: only kernelize if compute is substantial
    fusedCompute >= minWorthwhileCompute
  else
    -- Multi-node: balance launch savings vs fusion overhead
    let launchSavings := cm.kernelOverhead * (coverSize - 1)
    let fusionOverhead := fusedCompute / 10  -- 10% overhead for view traversal
    launchSavings > fusionOverhead

/-! ## Default Pattern Registry

Patterns are tried in priority order. Higher priority = tried first.
Priority guidelines:
- 100+: Composite patterns (softmax, attention, etc.)
- 80-99: Specialized compute (matmul variants)
- 60-79: Reduce patterns
- 40-59: Elementwise fusions
- 0-39: Fallback patterns
-/

/-- Softmax pattern (priority 100) -/
private def softmaxPatternEntry : PatternEntry :=
  { name := "softmax"
    priority := 100
    rootOps := [.FDIV]  -- Softmax root is FDIV
    match_ := fun u keep refCnt =>
      match FusedSoftmax.compile u keep refCnt with
      | some p => some { impl := .fusedSoftmax p, cover := p.cover }
      | none => none }

/-- Fused matmul pattern (priority 90) -/
private def matmulPatternEntry : PatternEntry :=
  { name := "matmul"
    priority := 90
    rootOps := [.ADD, .MAX]  -- Matmul patterns have ADD/MAX root
    match_ := fun u keep refCnt =>
      let plans0 := FusedMatmul.compileVariants u keep refCnt
      let plans := plans0.filter planOk
      match pickBestPlan plans with
      | some p => some { impl := .fusedMatmul p, cover := p.cover }
      | none => none }

/-- Fused reduce pattern (priority 70) -/
private def reducePatternEntry : PatternEntry :=
  { name := "reduce"
    priority := 70
    rootOps := [.REDUCE_AXIS]
    match_ := fun u keep refCnt =>
      match FusedReduce.compile u keep refCnt with
      | some p => some { impl := .fusedReduce p, cover := p.ewise.cover }
      | none => none }

/-- Fused elementwise pattern (priority 50) - fallback -/
private def ewisePatternEntry : PatternEntry :=
  { name := "ewise"
    priority := 50
    rootOps := []  -- Try for any op
    match_ := fun u keep refCnt =>
      match FusedEwise.compile u keep refCnt with
      | some p => some { impl := .fusedEwise p, cover := p.cover }
      | none => none }

/-- LayerNorm pattern (priority 95) - higher than softmax since it's more specific -/
private def layerNormPatternEntry : PatternEntry :=
  { name := "layernorm"
    priority := 95
    rootOps := [.FDIV]  -- LayerNorm root is FDIV (x/sqrt(var+eps))
    match_ := fun u keep refCnt =>
      match FusedLayerNorm.compile u keep refCnt with
      | some p => some { impl := .fusedLayerNorm p, cover := p.cover }
      | none => none }

/-- GELU pattern (priority 85) -/
private def geluPatternEntry : PatternEntry :=
  { name := "gelu"
    priority := 85
    rootOps := [.MUL]  -- GELU variants often end with MUL
    match_ := fun u keep refCnt =>
      match FusedGELU.compile u keep refCnt with
      | some p => some { impl := .fusedGELU p, cover := p.cover }
      | none => none }

/-- Default pattern registry -/
def defaultPatterns : Array PatternEntry := #[
  layerNormPatternEntry,
  softmaxPatternEntry,
  matmulPatternEntry,
  geluPatternEntry,
  reducePatternEntry,
  ewisePatternEntry
]

/-- Pattern-based fusion selection using the registry -/
def selectByPatterns (nodes : List UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat)
    (patterns : Array PatternEntry := defaultPatterns) : HashMap UOpId Impl := Id.run do
  -- Sort patterns by priority (highest first)
  let sortedPatterns := patterns.qsort (fun a b => a.priority > b.priority)

  let mut implMap : HashMap UOpId Impl := ∅
  let mut covered : UOpIdSet := UOpIdSet.mkEmpty

  for u in nodes.reverse do
    if UOpIdSet.member covered u.uid then
      continue

    -- Try each pattern in priority order
    for pat in sortedPatterns do
      if !pat.appliesTo u.op then
        continue
      match pat.match_ u keep refCnt with
      | some result =>
        if setIntersects result.cover covered then
          continue
        implMap := implMap.insert u.uid result.impl
        covered := setUnion covered result.cover
        break  -- Found a match, move to next node
      | none => continue

  return implMap

/-- Greedy fusion selection with cost-aware decisions.
    Uses cost model to decide whether fusion is worthwhile for each pattern. -/
def selectPhaseC (nodes : List UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat)
    (cm : CostModel := defaultCostModel) : HashMap UOpId Impl := Id.run do
  let mut implMap : HashMap UOpId Impl := ∅
  let mut covered : UOpIdSet := UOpIdSet.mkEmpty

  for u in nodes.reverse do
    if UOpIdSet.member covered u.uid then
      continue
    if u.op == .ADD || u.op == .MAX then
      let plans0 := FusedMatmul.compileVariants u keep refCnt
      let plans := plans0.filter planOk
      match pickBestPlan plans with
      | some p =>
        if setIntersects p.cover covered then
          continue
        let impl := Impl.fusedMatmul p
        let features := implCostFeatures u impl
        let cSize := coverCount p.cover
        if shouldFuse cm features cSize then
          implMap := implMap.insert p.root impl
          covered := setUnion covered p.cover
          continue
      | none => pure ()
    -- Try softmax pattern (FDIV root)
    if u.op == .FDIV then
      match FusedSoftmax.compile u keep refCnt with
      | some p =>
        if setIntersects p.cover covered then
          continue
        let impl := Impl.fusedSoftmax p
        let features := implCostFeatures u impl
        let cSize := coverCount p.cover
        if shouldFuse cm features cSize then
          implMap := implMap.insert p.root impl
          covered := setUnion covered p.cover
          continue
      | none => pure ()
    if u.op == .REDUCE_AXIS then
      match FusedReduce.compile u keep refCnt with
      | some p =>
        if setIntersects p.ewise.cover covered then
          continue
        let impl := Impl.fusedReduce p
        let features := implCostFeatures u impl
        let cSize := coverCount p.ewise.cover
        if shouldFuse cm features cSize then
          implMap := implMap.insert p.root impl
          covered := setUnion covered p.ewise.cover
          continue
      | none => pure ()
    match FusedEwise.compile u keep refCnt with
    | some p =>
      if setIntersects p.cover covered then
        continue
      let impl := Impl.fusedEwise p
      let features := implCostFeatures u impl
      let cSize := coverCount p.cover
      if shouldFuse cm features cSize then
        implMap := implMap.insert p.root impl
        covered := setUnion covered p.cover
    | none => pure ()

  return implMap

end Fusion

end TinyGrad4.Backend
