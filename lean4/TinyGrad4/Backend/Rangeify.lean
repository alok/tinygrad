import TinyGrad4.Ops
import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import TinyGrad4.Backend.Schedule
import Std.Data.HashMap

/-!
# Rangeification Pipeline

Ported from tinygrad's `schedule/rangeify.py` and `schedule/indexing.py`.

The rangeification pipeline converts shape-level movement operations (reshape,
permute, expand, pad, shrink, flip) into index-level RANGE operations. This is
a key step in converting the high-level tensor DAG into executable kernels.

## Key Concepts

1. **RANGE** - A loop variable over a dimension: `RANGE(size, id, type)`
2. **Movement Ops → Index Transforms** - Each movement op becomes an index formula
3. **Realize Map** - Tracks which ops need to be materialized (written to memory)
4. **Range Propagation** - Ranges flow from outputs toward inputs through the DAG

## Pipeline Stages

1. **Generate Realize Map** - Identify which ops need buffers
2. **Run Rangeify** - Convert movement ops to ranges via reverse toposort
3. **Apply Bufferization** - Insert BUFFERIZE nodes for realized values
4. **Split Kernels** - Group operations into individual kernels

## Example

```
# Input: reshape(expand(x, [1, 4, 8]), [4, 8])
# After rangeification:
#   RANGE(4, r0, LOOP) * 8 + RANGE(8, r1, LOOP)
# The expand dimension (size 1) becomes a CONST 0, not a range.
```
-/

namespace TinyGrad4.Backend.Rangeify

open TinyGrad4 Std

/-! ## Core Types -/

/-- A range expression representing a loop variable -/
structure RangeExpr where
  /-- Unique range identifier -/
  rangeId : Nat
  /-- Size of the range (loop bound) -/
  size : Nat
  /-- Axis type (LOOP, REDUCE, etc.) -/
  axisType : AxisType
  /-- Optional validity condition (for PAD) -/
  valid : Option UOp := none
  deriving Repr

/-- An index expression (sum of terms) -/
inductive IndexExpr where
  | const : Int → IndexExpr                    -- Constant offset
  | range : RangeExpr → IndexExpr              -- Single range variable
  | scaled : Int → IndexExpr → IndexExpr       -- c * expr
  | add : IndexExpr → IndexExpr → IndexExpr    -- expr + expr
  | mod : IndexExpr → Nat → IndexExpr          -- expr % n
  | div : IndexExpr → Nat → IndexExpr          -- expr / n
  | invalid : IndexExpr                        -- Invalid index (from PAD)
  deriving Repr

namespace IndexExpr

/-- Create zero index -/
def zero : IndexExpr := .const 0

/-- Create a constant index -/
def ofNat (n : Nat) : IndexExpr := .const (Int.ofNat n)

/-- Check if expression is a simple constant -/
def isConst : IndexExpr → Bool
  | .const _ => true
  | _ => false

/-- Get constant value if present -/
def toConst? : IndexExpr → Option Int
  | .const v => some v
  | _ => none

/-- Simplify index expression -/
partial def simplify : IndexExpr → IndexExpr
  | .const c => .const c
  | .range r => .range r
  | .invalid => .invalid
  | .scaled 0 _ => .const 0
  | .scaled 1 e => simplify e
  | .scaled c e => .scaled c (simplify e)
  | .add e1 e2 =>
    let s1 := simplify e1
    let s2 := simplify e2
    match s1, s2 with
    | .const 0, _ => s2
    | _, .const 0 => s1
    | .const c1, .const c2 => .const (c1 + c2)
    | _, _ => .add s1 s2
  | .mod e n =>
    let se := simplify e
    match se with
    | .const c => .const (c % (Int.ofNat n))
    | _ => .mod se n
  | .div e n =>
    let se := simplify e
    match se with
    | .const c => .const (c / (Int.ofNat n))
    | _ => .div se n

end IndexExpr

/-! ## Indexing Context -/

/-- Ops that are always considered contiguous (no movement needed) -/
def ALWAYS_CONTIGUOUS : List Ops :=
  [.CONTIGUOUS, .ASSIGN, .COPY, .BUFFER, .BUFFER_VIEW,
   .CONST, .BIND, .DEVICE, .DEFINE_GLOBAL, .DEFINE_LOCAL,
   .DEFINE_REG, .LOAD, .KERNEL]

/-- Check if op is always contiguous -/
def isAlwaysContiguous (op : Ops) : Bool :=
  op ∈ ALWAYS_CONTIGUOUS

/-- Context for rangeification -/
structure RangeContext where
  /-- Map from UOp to whether it needs realization (Some axes = partial, None = full) -/
  realizeMap : HashMap UOpId (Option (List Nat)) := {}
  /-- Map from UOp to (input ranges, output ranges) -/
  rangeMap : HashMap UOpId (Array IndexExpr × Array IndexExpr) := {}
  /-- Counter for unique range IDs -/
  nextRangeId : IO.Ref Nat
  deriving Nonempty

/-- Create a new range context -/
def RangeContext.new : IO RangeContext := do
  let nextId ← IO.mkRef 0
  return { nextRangeId := nextId }

/-- Create a new range with unique ID -/
def RangeContext.newRange (ctx : RangeContext) (size : Nat) (axisType : AxisType := .LOOP)
    : IO RangeExpr := do
  let id ← ctx.nextRangeId.modifyGet fun n => (n, n + 1)
  return { rangeId := id, size, axisType }

/-! ## Movement Op Application -/

/-- Apply SHRINK movement op to ranges.
    SHRINK[(start, end)] shifts the range by start. -/
def applyShrink (_inShape : Shape) (arg : List (Nat × Nat)) (ranges : Array IndexExpr)
    : Array IndexExpr := Id.run do
  let mut result : Array IndexExpr := #[]
  for i in [:ranges.size] do
    let r := ranges.getD i .invalid
    let (start, _) := arg.getD i (0, 0)
    if start == 0 then
      result := result.push r
    else
      result := result.push (IndexExpr.add r (.const (Int.ofNat start)))
  return result

/-- Apply PERMUTE movement op to ranges.
    PERMUTE[p] reorders the ranges by inverse permutation. -/
def applyPermute (arg : List Nat) (ranges : Array IndexExpr) : Array IndexExpr := Id.run do
  -- Compute inverse permutation (argsort)
  -- For each position i, find j such that arg[j] = i
  let n := arg.length
  let mut invPerm : Array Nat := #[]
  for _ in [:n] do
    invPerm := invPerm.push 0
  for j in [:n] do
    let v := arg.getD j 0
    if v < n then
      invPerm := invPerm.set! v j
  invPerm.map fun i => ranges.getD i .invalid

/-- Apply FLIP movement op to ranges.
    FLIP[f] reverses range r_i if f[i] is true: (size-1) - r_i -/
def applyFlip (inShape : Shape) (arg : List Bool) (ranges : Array IndexExpr) : Array IndexExpr := Id.run do
  let mut result : Array IndexExpr := #[]
  for i in [:ranges.size] do
    let r := ranges.getD i .invalid
    let shouldFlip := arg.getD i false
    let size := inShape.getD i 1
    if shouldFlip then
      result := result.push (IndexExpr.add (.const (Int.ofNat size - 1)) (.scaled (-1) r))
    else
      result := result.push r
  return result

/-- Apply EXPAND movement op to ranges.
    EXPAND[outShape] zeros out ranges for expanded dimensions (in_size=1, out_size>1). -/
def applyExpand (inShape : Shape) (arg : Shape) (ranges : Array IndexExpr) : Array IndexExpr := Id.run do
  let mut result : Array IndexExpr := #[]
  for i in [:ranges.size] do
    let r := ranges.getD i .invalid
    let inSize := inShape.getD i 1
    let outSize := arg.getD i 1
    if inSize == outSize then
      result := result.push r
    else
      result := result.push (.const 0)  -- Expanded dimension: always 0
  return result

/-- Apply PAD movement op to ranges.
    PAD[(start, end)] shifts ranges and adds validity conditions. -/
def applyPad (_inShape : Shape) (arg : List (Nat × Nat)) (ranges : Array IndexExpr)
    : Array IndexExpr := Id.run do
  let mut result : Array IndexExpr := #[]
  for i in [:ranges.size] do
    let r := ranges.getD i .invalid
    let (padStart, _padEnd) := arg.getD i (0, 0)
    if padStart == 0 then
      result := result.push r
    else
      -- Shift range: r - padStart, but mark as potentially invalid
      -- In full impl, we'd wrap with validity check
      result := result.push (IndexExpr.add r (.const (-(Int.ofNat padStart))))
  return result

/-- Compute strides from shape (e.g., [2,3,4] → [12,4,1]) -/
def computeStrides (shape : Shape) : List Nat := Id.run do
  let mut strides : List Nat := []
  let mut acc := 1
  for dim in shape.reverse do
    strides := acc :: strides
    acc := acc * dim
  return strides

/-- Apply RESHAPE movement op to ranges.
    This is the complex case: linearize output indices, then de-linearize to input shape. -/
def applyReshape (inShape : Shape) (outShape : Shape) (ranges : Array IndexExpr) : Array IndexExpr := Id.run do
  -- Linearize: sum of r_i * stride_i for output shape
  let outStrides := computeStrides outShape
  let mut linearized := IndexExpr.zero
  for i in [:ranges.size] do
    let r := ranges.getD i .invalid
    let s := outStrides.getD i 1
    linearized := IndexExpr.add linearized (IndexExpr.scaled (Int.ofNat s) r)

  -- De-linearize: for each input dimension, extract with mod/div
  let mut result : Array IndexExpr := #[]
  let mut remaining := linearized
  for dim in inShape do
    if dim == 1 then
      result := result.push (.const 0)
    else
      let idx := IndexExpr.mod remaining dim
      result := result.push idx
      remaining := IndexExpr.div remaining dim
  return result.map IndexExpr.simplify

/-- Apply a movement op to transform ranges.
    Returns the input ranges given the output ranges. -/
def applyMovementOp (op : Ops) (inShape : Shape) (arg : UArg) (ranges : Array IndexExpr)
    : Array IndexExpr :=
  match op with
  | .SHRINK =>
    match arg with
    | .bounds b => applyShrink inShape b ranges
    | _ => ranges
  | .PERMUTE =>
    match arg with
    | .permutation p => applyPermute p ranges
    | _ => ranges
  | .FLIP =>
    match arg with
    -- FLIP uses axes as boolean flags
    | .axes flipAxes =>
      let flags := (List.range inShape.length).map fun i => flipAxes.contains i
      applyFlip inShape flags ranges
    | _ => ranges
  | .EXPAND =>
    match arg with
    | .shape s => applyExpand inShape s ranges
    | _ => ranges
  | .PAD =>
    match arg with
    | .padding p => applyPad inShape p ranges
    | _ => ranges
  | .RESHAPE =>
    match arg with
    | .shape s => applyReshape inShape s ranges
    | _ => ranges
  | _ => ranges

/-! ## Realize Map Generation -/

/-- Ops that force realization -/
def ALWAYS_RUN_OPS : List Ops :=
  [.CONTIGUOUS, .COPY, .ASSIGN]

/-- Generate the realize map: which UOps need to be written to buffers -/
def generateRealizeMap (nodes : List UOp) (sink : UOp) : HashMap UOpId (Option (List Nat)) :=
  Id.run do
    let mut realizeMap : HashMap UOpId (Option (List Nat)) := {}

    -- Always realize SINK sources
    for src in sink.src do
      if !isAlwaysContiguous src.op then
        realizeMap := realizeMap.insert src.uid none

    -- Walk the graph to find ops that need realization
    for u in nodes do
      -- COPY/BUFFER_VIEW/CONTIGUOUS/STORE need realization
      if u.op ∈ [.COPY, .BUFFER_VIEW, .CONTIGUOUS, .STORE] then
        realizeMap := realizeMap.insert u.uid none

      -- REDUCE on outer ranges
      if u.op == .REDUCE_AXIS then
        -- Check if reduce has outer axis type (would need runtime info)
        -- For now, conservatively mark as realized
        realizeMap := realizeMap.insert u.uid none

      -- ASSIGN needs its source realized
      if u.op == .ASSIGN then
        match u.src with
        | [_, src] =>
          if !isAlwaysContiguous src.op then
            realizeMap := realizeMap.insert src.uid none
        | _ => pure ()

    return realizeMap

/-! ## Range Propagation -/

/-- Create initial output ranges for a realized op -/
def createOutputRanges (ctx : RangeContext) (shape : Shape) : IO (Array IndexExpr) := do
  let mut ranges : Array IndexExpr := #[]
  for s in shape do
    if s == 1 then
      ranges := ranges.push (.const 0)
    else
      let r ← ctx.newRange s
      ranges := ranges.push (.range r)
  return ranges

/-- Propagate ranges through the graph (reverse toposort order) -/
def runRangeify (ctx : RangeContext) (nodes : List UOp) (realizeMap : HashMap UOpId (Option (List Nat)))
    : IO (HashMap UOpId (Array IndexExpr × Array IndexExpr)) := do
  -- Build consumer map
  let mut consumerMap : HashMap UOpId (List UOpId) := {}
  for u in nodes do
    for src in u.src do
      let existing := consumerMap.getD src.uid []
      consumerMap := consumerMap.insert src.uid (u.uid :: existing)

  -- Build node map for lookups
  let nodeMap : HashMap UOpId UOp := nodes.foldl (init := {}) fun m u => m.insert u.uid u

  -- Process in reverse order (outputs before inputs)
  let sortedIds := (Schedule.toposort nodes).toList.reverse
  let mut rangeMap : HashMap UOpId (Array IndexExpr × Array IndexExpr) := {}

  for uid in sortedIds do
    match nodeMap.get? uid with
    | none => continue
    | some u =>
      if u.op ∈ [.DEVICE] then continue
      if u.op == .KERNEL then continue

      -- Get consumer ranges
      let consumers := consumerMap.getD uid []
      let consumerRanges := consumers.filterMap fun cid => rangeMap.get? cid

      -- Determine output ranges
      let outRanges ← do
        if realizeMap.contains uid then
          -- Realized: create new ranges
          createOutputRanges ctx u.shape
        else if consumerRanges.length == 0 then
          -- No consumers with ranges, skip
          continue
        else if consumerRanges.length == 1 then
          -- Single consumer: inherit ranges
          pure (consumerRanges.head!).1
        else
          -- Multiple consumers: for now, create new ranges
          -- (Full impl would try to merge common ranges)
          createOutputRanges ctx u.shape

      -- Compute input ranges
      let inRanges : Array IndexExpr :=
        if u.op.isMovement then
          match u.src with
          | [src] => applyMovementOp u.op src.shape u.arg outRanges
          | _ => outRanges
        else if u.op == .REDUCE_AXIS then
          -- REDUCE creates new ranges for reduced axes
          match u.arg with
          | .reduceWithAxes _ axes => Id.run do
            let srcShape := match u.src with | s :: _ => s.shape | _ => []
            let mut rngs := outRanges
            for i in [:srcShape.length] do
              let s := srcShape.getD i 1
              if i ∈ axes then
                -- Create a reduce range for this axis
                let r : RangeExpr := { rangeId := 0, size := s, axisType := .REDUCE }
                rngs := rngs.set! i (.range r)
            rngs
          | _ => outRanges
        else
          outRanges

      rangeMap := rangeMap.insert uid (inRanges, outRanges)

  return rangeMap

/-! ## Main API -/

/-- Result of rangeification -/
structure RangeifyResult where
  /-- Realize map: which ops need buffers -/
  realizeMap : HashMap UOpId (Option (List Nat))
  /-- Range map: (input ranges, output ranges) per op -/
  rangeMap : HashMap UOpId (Array IndexExpr × Array IndexExpr)
  deriving Repr

/-- Run the full rangeification pipeline -/
def rangeify (nodes : List UOp) (sink : UOp) : IO RangeifyResult := do
  let ctx ← RangeContext.new
  let realizeMap := generateRealizeMap nodes sink
  let rangeMap ← runRangeify ctx nodes realizeMap
  return { realizeMap, rangeMap }

/-! ## Debug Utilities -/

/-- Render a range expression for debugging -/
def RangeExpr.render (r : RangeExpr) : String :=
  s!"r{r.rangeId}:[0,{r.size})"

/-- Render an index expression for debugging -/
partial def IndexExpr.render : IndexExpr → String
  | .const c => toString c
  | .range r => r.render
  | .scaled c e => s!"{c}*({e.render})"
  | .add e1 e2 => s!"({e1.render}+{e2.render})"
  | .mod e n => s!"({e.render}%{n})"
  | .div e n => s!"({e.render}/{n})"
  | .invalid => "INVALID"

/-- Print range map for debugging -/
def printRangeMap (rangeMap : HashMap UOpId (Array IndexExpr × Array IndexExpr))
    (nodeMap : HashMap UOpId UOp) : IO Unit := do
  for (uid, (inRngs, outRngs)) in rangeMap.toList do
    match nodeMap.get? uid with
    | none => continue
    | some u =>
      let inStr := inRngs.map IndexExpr.render |>.toList |> String.intercalate ", "
      let outStr := outRngs.map IndexExpr.render |>.toList |> String.intercalate ", "
      let opStr := reprStr u.op
      IO.println s!"{opStr}: [{inStr}] → [{outStr}]"

end TinyGrad4.Backend.Rangeify
