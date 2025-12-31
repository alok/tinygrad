import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import TinyGrad4.Backend.Pattern
import Std.Data.HashMap

/-!
# Schedule Caching System

Ported from tinygrad's `engine/schedule.py`.

The scheduler converts a UOp DAG into a linear sequence of kernel executions.
Key optimization: cache schedules based on normalized graph keys to avoid
recomputation for repeated patterns (common in training loops).

## Key Concepts

1. **Graph Normalization** - Replace unique buffer IDs with sequential indices
   so structurally identical graphs hash the same way
2. **Topological Sort** - Order kernels by dependencies
3. **Range Expansion** - Handle RANGE/END for looping constructs
4. **Schedule Cache** - Store computed schedules keyed by normalized graphs
-/

namespace TinyGrad4.Backend.Schedule

open TinyGrad4 Std

/-! ## Normalized Graph Keys -/

/-- A normalized buffer reference (position-independent) -/
structure NormalizedBuf where
  index : Nat  -- Sequential index assigned during normalization
  deriving Repr, BEq, Hashable

/-- Context for graph normalization -/
structure NormContext where
  bufferMap : HashMap UOpId Nat := {}
  nextIndex : Nat := 0
  deriving Repr

/-- Normalize a UOp by replacing buffer IDs with sequential indices -/
def normalizeOp (ctx : NormContext) (u : UOp) : NormContext × UOp :=
  if u.op == Ops.BUFFER then
    match ctx.bufferMap.get? u.uid with
    | some idx => (ctx, { u with arg := .bufferIdx idx })
    | none =>
      let newCtx := {
        bufferMap := ctx.bufferMap.insert u.uid ctx.nextIndex
        nextIndex := ctx.nextIndex + 1
      }
      (newCtx, { u with arg := .bufferIdx ctx.nextIndex })
  else
    (ctx, u)

/-- Normalize an entire graph for cache key generation -/
def normalizeGraph (nodes : List UOp) : List UOp := Id.run do
  let mut ctx : NormContext := {}
  let mut result : List UOp := []
  for u in nodes do
    let (newCtx, normU) := normalizeOp ctx u
    ctx := newCtx
    result := result ++ [normU]
  return result

/-! ## Graph Key Generation -/

/-- Compute a cache key from a normalized graph.
    Uses a simple hash combining op types, args, and structure. -/
def graphKey (nodes : List UOp) : UInt64 := Id.run do
  let mut h : UInt64 := 5381  -- djb2 hash initial value
  for u in nodes do
    -- Mix in op hash
    h := h * 33 + hash u.op
    -- Mix in source count (structure)
    h := h * 33 + u.src.length.toUInt64
    -- Mix in UID
    h := h * 33 + u.uid.id.toUInt64
  return h

/-! ## Schedule Items -/

/-- A single execution item in the schedule -/
structure ExecItem where
  /-- The kernel AST -/
  ast : UOp
  /-- Buffer UIDs bound to this kernel -/
  bufferUids : Array UOpId
  /-- Metadata for debugging -/
  metadata : Option String := none
  /-- Fixed variable bindings (for RANGE loops) -/
  fixedVars : HashMap String Nat := {}
  deriving Repr

/-- Result of scheduling -/
structure ScheduleResult where
  /-- Linear sequence of execution items -/
  items : Array ExecItem
  /-- Buffer sink (for memory planning) -/
  bufferSink : List UOpId
  deriving Repr

/-! ## Dependency Graph -/

/-- Build dependency graph from UOp DAG -/
def buildDependencyGraph (nodes : List UOp) : HashMap UOpId (List UOpId) × HashMap UOpId Nat := Id.run do
  let mut children : HashMap UOpId (List UOpId) := {}
  let mut inDegree : HashMap UOpId Nat := {}

  for u in nodes do
    -- Initialize in-degree if not set
    if !inDegree.contains u.uid then
      inDegree := inDegree.insert u.uid 0

    -- Add edges from sources
    for src in u.src do
      let srcId := src.uid
      let existing := children.getD srcId []
      children := children.insert srcId (u.uid :: existing)
      inDegree := inDegree.insert u.uid (inDegree.getD u.uid 0 + 1)

  return (children, inDegree)

/-! ## Topological Sort -/

/-- Topologically sort nodes by dependencies -/
def toposort (nodes : List UOp) : Array UOpId := Id.run do
  let _nodeMap : HashMap UOpId UOp := nodes.foldl (init := ∅) fun m u => m.insert u.uid u
  let (children, inDegree) := buildDependencyGraph nodes

  -- Start with nodes that have no dependencies
  let mut queue : List UOpId := []
  for (uid, deg) in inDegree.toList do
    if deg == 0 then queue := uid :: queue

  let mut result : Array UOpId := #[]
  let mut remaining := inDegree

  while !queue.isEmpty do
    match queue with
    | [] => break
    | uid :: rest =>
      queue := rest
      result := result.push uid

      -- Decrement in-degree of children
      for childId in children.getD uid [] do
        let newDeg := remaining.getD childId 1 - 1
        remaining := remaining.insert childId newDeg
        if newDeg == 0 then queue := childId :: queue

  return result

/-! ## Schedule Creation -/

/-- Create schedule from UOp DAG (without caching) -/
def createScheduleRaw (nodes : List UOp) (_sinkId : UOpId) : ScheduleResult := Id.run do
  let nodeMap : HashMap UOpId UOp := nodes.foldl (init := ∅) fun m u => m.insert u.uid u
  let sortedIds := toposort nodes

  let mut items : Array ExecItem := #[]
  let mut bufferSink : List UOpId := []

  for uid in sortedIds do
    match nodeMap.get? uid with
    | none => continue
    | some u =>
      if u.op == Ops.KERNEL then
        -- Extract buffer bindings from sources
        let bufUids := u.src.filterMap fun s =>
          if s.op == Ops.BUFFER then some s.uid else none
        items := items.push {
          ast := u
          bufferUids := bufUids.toArray
          metadata := some s!"kernel_{uid.id}"
        }
        bufferSink := bufferSink ++ bufUids.map (·)
      else if u.op == Ops.BUFFER then
        bufferSink := bufferSink ++ [uid]

  return { items, bufferSink }

/-! ## Schedule Cache -/

/-- Global schedule cache (mutable state) -/
abbrev ScheduleCache := HashMap UInt64 ScheduleResult

/-- Create a new empty cache -/
def ScheduleCache.empty : ScheduleCache := ∅

/-- Cache statistics -/
structure CacheStats where
  hits : Nat := 0
  misses : Nat := 0
  deriving Repr

/-- Cached scheduler state -/
structure Scheduler where
  cache : IO.Ref ScheduleCache
  stats : IO.Ref CacheStats

/-- Create a new scheduler with empty cache -/
def Scheduler.new : IO Scheduler := do
  let cache ← IO.mkRef ScheduleCache.empty
  let stats ← IO.mkRef { hits := 0, misses := 0 : CacheStats }
  return { cache, stats }

/-- Schedule with caching -/
def Scheduler.schedule (s : Scheduler) (nodes : List UOp) (sinkId : UOpId) : IO ScheduleResult := do
  -- Normalize and compute key
  let normNodes := normalizeGraph nodes
  let key := graphKey normNodes

  -- Check cache
  let cache ← s.cache.get
  match cache.get? key with
  | some result =>
    -- Cache hit
    s.stats.modify fun st => { st with hits := st.hits + 1 }
    return result
  | none =>
    -- Cache miss - compute schedule
    s.stats.modify fun st => { st with misses := st.misses + 1 }
    let result := createScheduleRaw nodes sinkId
    s.cache.modify fun c => c.insert key result
    return result

/-- Get cache statistics -/
def Scheduler.getStats (s : Scheduler) : IO CacheStats :=
  s.stats.get

/-- Clear the cache -/
def Scheduler.clearCache (s : Scheduler) : IO Unit :=
  s.cache.set ScheduleCache.empty

/-! ## Convenience API -/

/-- Schedule without caching (for single-use) -/
def schedule (nodes : List UOp) (sinkId : UOpId) : ScheduleResult :=
  createScheduleRaw nodes sinkId

/-- Schedule with a provided cache -/
def scheduleWithCache (nodes : List UOp) (sinkId : UOpId) (cacheRef : IO.Ref ScheduleCache)
    : IO ScheduleResult := do
  let normNodes := normalizeGraph nodes
  let key := graphKey normNodes

  let cache ← cacheRef.get
  match cache.get? key with
  | some result => return result
  | none =>
    let result := createScheduleRaw nodes sinkId
    cacheRef.modify fun c => c.insert key result
    return result

end TinyGrad4.Backend.Schedule
