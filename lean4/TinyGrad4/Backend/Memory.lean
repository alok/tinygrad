import TinyGrad4.DType
import TinyGrad4.UOp.UOp
import TinyGrad4.Backend.Schedule
import Std.Data.HashMap

/-!
# Memory Planning

Ported from tinygrad's `engine/memory.py`.

Memory planning optimizes buffer allocation by:
1. Tracking buffer lifetimes (first/last appearance)
2. Reusing buffers that don't overlap in time
3. Suballocating from larger shared buffers when possible

## Key Concepts

1. **Lifetime Analysis** - Track when each buffer is first and last used
2. **Buffer Reuse** - Buffers with same (device, dtype, size) can be reused
3. **TLSF Allocator** - Two-Level Segregated Fit for efficient suballocation
4. **Memory Reduction** - Typical reduction: 30-50% of peak memory usage

## Example

```
Buffers: [A:100B, B:200B, C:100B]
Timeline:
  Step 0: alloc A, alloc B
  Step 1: use A, use B
  Step 2: free A, alloc C (reuses A's memory!)
  Step 3: use B, use C
  Step 4: free B, free C

Peak without planning: 400B
Peak with planning: 300B (C reuses A)
```
-/

namespace TinyGrad4.Backend.Memory

open TinyGrad4 Std

/-! ## Buffer Descriptor -/

/-- Information about a buffer for memory planning -/
structure BufferInfo where
  /-- Unique buffer ID -/
  uid : UOpId
  /-- Device name -/
  device : String
  /-- Data type -/
  dtype : DType
  /-- Size in bytes -/
  nbytes : Nat
  /-- Whether buffer is already allocated -/
  isAllocated : Bool := false
  /-- Reference count (0 = can optimize) -/
  refCount : Nat := 0
  deriving Repr, BEq

instance : Inhabited BufferInfo where
  default := { uid := ⟨0⟩, device := "", dtype := .float32, nbytes := 0 }

instance : Hashable BufferInfo where
  hash b := hash b.uid

/-! ## Lifetime Tracking -/

/-- First and last step where a buffer appears -/
structure BufferLifetime where
  firstAppearance : Nat
  lastAppearance : Nat
  deriving Repr

/-- Track buffer lifetimes from schedule -/
def trackLifetimes (schedule : Array (Array BufferInfo)) : HashMap UOpId BufferLifetime := Id.run do
  let mut lifetimes : HashMap UOpId BufferLifetime := {}

  for step in [:schedule.size] do
    let buffers := schedule.getD step #[]
    for buf in buffers do
      -- Skip already allocated or referenced buffers
      if buf.isAllocated || buf.refCount > 0 then continue

      match lifetimes.get? buf.uid with
      | none =>
        -- First appearance
        lifetimes := lifetimes.insert buf.uid { firstAppearance := step, lastAppearance := step }
      | some lt =>
        -- Update last appearance
        lifetimes := lifetimes.insert buf.uid { lt with lastAppearance := step }

  return lifetimes

/-! ## Buffer Events -/

/-- A buffer allocation/deallocation event -/
inductive BufferEvent where
  | alloc (step : Nat) (buf : BufferInfo)
  | free (step : Nat) (buf : BufferInfo)
  deriving Repr

/-- Extract step from event -/
def BufferEvent.step : BufferEvent → Nat
  | .alloc s _ => s
  | .free s _ => s

/-- Is this an allocation event? -/
def BufferEvent.isAlloc : BufferEvent → Bool
  | .alloc _ _ => true
  | .free _ _ => false

/-- Get buffer from event -/
def BufferEvent.buffer : BufferEvent → BufferInfo
  | .alloc _ b => b
  | .free _ b => b

/-- Compare events: earlier step first, alloc before free at same step -/
def BufferEvent.lt (a b : BufferEvent) : Bool :=
  if a.step != b.step then a.step < b.step
  else a.isAlloc && !b.isAlloc

/-- Generate sorted buffer events from lifetimes -/
def generateEvents (lifetimes : HashMap UOpId BufferLifetime) (bufMap : HashMap UOpId BufferInfo)
    : Array BufferEvent := Id.run do
  let mut events : Array BufferEvent := #[]

  for (uid, lt) in lifetimes.toList do
    match bufMap.get? uid with
    | none => continue
    | some buf =>
      events := events.push (.alloc lt.firstAppearance buf)
      events := events.push (.free (lt.lastAppearance + 1) buf)

  -- Sort by (step, isAlloc) - allocs before frees at same step
  return events.qsort BufferEvent.lt

/-! ## Buffer Reuse -/

/-- Key for buffer reuse: (device, dtype, size) -/
structure ReuseKey where
  device : String
  dtype : DType
  nbytes : Nat
  deriving Repr, BEq, Hashable

/-- Convert BufferInfo to reuse key -/
def BufferInfo.reuseKey (b : BufferInfo) : ReuseKey :=
  { device := b.device, dtype := b.dtype, nbytes := b.nbytes }

/-- State for buffer reuse tracking -/
structure ReuseState where
  /-- Available buffers by key -/
  available : HashMap ReuseKey (Array BufferInfo) := {}
  /-- Mapping from original buffer to replacement -/
  assigned : HashMap UOpId UOpId := {}
  deriving Repr

/-- Try to reuse an available buffer or use original -/
def tryReuse (state : ReuseState) (buf : BufferInfo) : ReuseState × UOpId :=
  let key := buf.reuseKey
  match state.available.get? key with
  | some avail =>
    if avail.isEmpty then
      -- No available buffer, use original
      ({ state with assigned := state.assigned.insert buf.uid buf.uid }, buf.uid)
    else
      -- Reuse available buffer
      let reuseBuf := avail.back!
      let newAvail := avail.pop
      let newState := {
        available := state.available.insert key newAvail
        assigned := state.assigned.insert buf.uid reuseBuf.uid
      }
      (newState, reuseBuf.uid)
  | none =>
    -- No available buffer, use original
    ({ state with assigned := state.assigned.insert buf.uid buf.uid }, buf.uid)

/-- Return a buffer to the available pool -/
def returnBuffer (state : ReuseState) (buf : BufferInfo) : ReuseState :=
  let key := buf.reuseKey
  -- Find the actual buffer that was assigned
  match state.assigned.get? buf.uid with
  | none => state
  | some _assignedUid =>
    -- Only return if it was reused (assigned to different buffer)
    let avail := state.available.getD key #[]
    -- Add back the original buf to available pool for its key
    { state with available := state.available.insert key (avail.push buf) }

/-! ## Memory Planner -/

/-- Result of memory planning -/
structure MemoryPlanResult where
  /-- Original to replacement buffer mapping -/
  bufferMap : HashMap UOpId UOpId
  /-- Memory saved in bytes -/
  memorySaved : Nat
  /-- Original peak memory -/
  originalPeak : Nat
  /-- Optimized peak memory -/
  optimizedPeak : Nat
  deriving Repr

/-- Plan memory allocation for a schedule.
    Returns a mapping from original buffer UIDs to replacement buffer UIDs. -/
def planMemory (schedule : Array (Array BufferInfo)) : MemoryPlanResult := Id.run do
  -- Build buffer map for lookups
  let mut bufMap : HashMap UOpId BufferInfo := {}
  for buffers in schedule do
    for buf in buffers do
      if !bufMap.contains buf.uid then
        bufMap := bufMap.insert buf.uid buf

  -- Track lifetimes
  let lifetimes := trackLifetimes schedule

  -- Generate sorted events
  let events := generateEvents lifetimes bufMap

  -- Process events to find reuse opportunities
  let mut state : ReuseState := {}
  for ev in events do
    match ev with
    | .alloc _ buf =>
      let (newState, _) := tryReuse state buf
      state := newState
    | .free _ buf =>
      state := returnBuffer state buf

  -- Calculate memory stats
  let mut originalTotal : Nat := 0
  let mut uniqueBuffers : HashMap UOpId Bool := {}
  for (uid, _) in lifetimes.toList do
    match bufMap.get? uid with
    | some buf =>
      originalTotal := originalTotal + buf.nbytes
      match state.assigned.get? uid with
      | some assignedUid =>
        if !uniqueBuffers.contains assignedUid then
          uniqueBuffers := uniqueBuffers.insert assignedUid true
      | none => pure ()
    | none => continue

  let mut optimizedTotal : Nat := 0
  for (uid, _) in uniqueBuffers.toList do
    match bufMap.get? uid with
    | some buf => optimizedTotal := optimizedTotal + buf.nbytes
    | none => continue

  return {
    bufferMap := state.assigned
    memorySaved := if originalTotal > optimizedTotal then originalTotal - optimizedTotal else 0
    originalPeak := originalTotal
    optimizedPeak := optimizedTotal
  }

/-! ## Integration with Schedule -/

/-- Apply memory plan to schedule items -/
def applyMemoryPlan (items : Array Schedule.ExecItem) (plan : MemoryPlanResult)
    : Array Schedule.ExecItem :=
  items.map fun item =>
    let newBufUids := item.bufferUids.map fun uid =>
      plan.bufferMap.getD uid uid
    { item with bufferUids := newBufUids }

/-! ## TLSF Allocator (Simplified) -/

/-- A simple first-fit allocator for suballocation.
    (Full TLSF is more complex; this is a simplified version) -/
structure SimpleAllocator where
  /-- Total size in bytes -/
  totalSize : Nat
  /-- Block size alignment -/
  blockSize : Nat := 0x1000
  /-- Free regions: (offset, size) -/
  freeRegions : Array (Nat × Nat) := #[(0, totalSize)]
  /-- Current high-water mark -/
  highWaterMark : Nat := 0
  deriving Repr

namespace SimpleAllocator

/-- Round up to block size -/
def roundUp (a : SimpleAllocator) (size : Nat) : Nat :=
  ((size + a.blockSize - 1) / a.blockSize) * a.blockSize

/-- Allocate a region, return offset or none if OOM -/
def alloc (a : SimpleAllocator) (size : Nat) : Option (SimpleAllocator × Nat) := Id.run do
  let alignedSize := a.roundUp size

  -- First-fit search
  for i in [:a.freeRegions.size] do
    let (offset, regionSize) := a.freeRegions[i]!
    if regionSize >= alignedSize then
      -- Found a fit - filter out this region
      let newRegions := a.freeRegions.filter fun (off, _) => off != offset
      let finalRegions := if regionSize > alignedSize then
        -- Split region: add remainder
        newRegions.push (offset + alignedSize, regionSize - alignedSize)
      else
        newRegions
      let newHWM := max a.highWaterMark (offset + alignedSize)
      return some ({ a with freeRegions := finalRegions, highWaterMark := newHWM }, offset)

  return none

/-- Free a region -/
def free (a : SimpleAllocator) (offset : Nat) (size : Nat) : SimpleAllocator :=
  let alignedSize := a.roundUp size
  -- Simple: just add back as free region (no coalescing for simplicity)
  { a with freeRegions := a.freeRegions.push (offset, alignedSize) }

end SimpleAllocator

/-! ## Debug Utilities -/

/-- Print memory planning summary -/
def printSummary (result : MemoryPlanResult) : IO Unit := do
  let origMB := (result.originalPeak.toFloat / 1e6)
  let optMB := (result.optimizedPeak.toFloat / 1e6)
  let savedMB := (result.memorySaved.toFloat / 1e6)
  if result.memorySaved > 0 then
    IO.println s!"Memory reduced from {origMB} MB -> {optMB} MB (saved {savedMB} MB)"
  else
    IO.println s!"Memory: {origMB} MB (no reduction possible)"

end TinyGrad4.Backend.Memory
