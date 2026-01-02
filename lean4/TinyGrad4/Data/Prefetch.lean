import TinyGrad4.Data.Dataset

/-!
# Prefetch - Asynchronous Data Loading

Uses Lean's concurrency primitives for background data loading:
- `IO.asTask (prio := .dedicated)`: Dedicated worker threads
- `IO.Promise`: For passing values between tasks
- `IO.Ref`: For shared state

## Usage
```lean
let prefetched ← prefetchDs 4 mnist
prefetched.forEach fun item => process item
```
-/

namespace TinyGrad4.Data

/-! ## Simple Prefetch Buffer

A circular buffer with producer-consumer semantics using IO.Ref.
-/

/-- Thread-safe queue using IO.Ref -/
structure IOQueue (T : Type) where
  /-- Buffer of items -/
  items : IO.Ref (Array T)
  /-- Is the queue closed? (No more items will be accepted.) -/
  done : IO.Ref Bool
  /-- Max buffer size for backpressure -/
  maxSize : Nat

namespace IOQueue

/-- Create a new IO queue -/
def new (maxSize : Nat := 16) : IO (IOQueue T) := do
  let items ← IO.mkRef #[]
  let done ← IO.mkRef false
  pure { items, done, maxSize }

/-- Add item to queue (may spin if full for backpressure) -/
def push (q : IOQueue T) (item : T) : IO Bool := do
  -- Simple spin-wait if buffer is full. Bail if closed.
  repeat do
    if ← q.done.get then
      return false
    let arr ← q.items.get
    if arr.size < q.maxSize then
      q.items.modify (·.push item)
      return true
    IO.sleep 1  -- 1ms backoff
  return false

/-- Mark producer as done -/
def finish (q : IOQueue T) : IO Unit :=
  q.done.set true

/-- Try to pop an item without blocking. -/
def tryPop (q : IOQueue T) : IO (Option T) := do
  q.items.modifyGet fun arr =>
    if h : arr.size > 0 then
      let item := arr[0]'(by omega)
      (some item, arr.eraseIdx 0)
    else
      (none, arr)

/-- Try to pop an item (returns none if empty and done) -/
def pop (q : IOQueue T) : IO (Option T) := do
  repeat do
    let item? ← q.tryPop
    match item? with
    | some item => return some item
    | none => pure ()
    -- Empty - check if done
    if ← q.done.get then
      return none
    -- Wait for more items
    IO.sleep 1
  -- Should never reach here
  pure none

/-- Pop an item and report time spent waiting (sleeping) in nanoseconds. -/
def popWithWait (q : IOQueue T) : IO (Option T × Nat) := do
  let mut waitNs : Nat := 0
  repeat do
    let item? ← q.tryPop
    match item? with
    | some item => return (some item, waitNs)
    | none => pure ()
    if ← q.done.get then
      return (none, waitNs)
    let start ← IO.monoNanosNow
    IO.sleep 1
    let stop ← IO.monoNanosNow
    waitNs := waitNs + (stop - start)
  pure (none, waitNs)

end IOQueue

/-! ## Prefetcher -/

/-- Prefetcher for a dataset using background task -/
structure Prefetcher (T : Type) where
  /-- Item queue -/
  queue : IOQueue T
  /-- Worker task handle -/
  worker : Task (Except IO.Error Unit)
  /-- Total items to fetch -/
  totalItems : Nat

namespace Prefetcher

/-- Create a prefetcher for a dataset -/
def create [Dataset D T] (ds : D) (bufferSize : Nat := 8) : IO (Prefetcher T) := do
  let queue ← IOQueue.new bufferSize
  let n := Dataset.len ds

  -- Spawn dedicated worker thread for prefetching
  let worker ← IO.asTask (prio := .dedicated) do
    for i in [:n] do
      -- Check for cancellation
      if ← IO.checkCanceled then break

      -- Fetch item
      if h : i < n then
        let item ← Dataset.getItem ds i h
        let ok ← queue.push item
        if !ok then break

    queue.finish
    return ()

  pure { queue, worker, totalItems := n }

/-- Get next prefetched item (blocks if buffer empty, returns none at end) -/
def next (p : Prefetcher T) : IO (Option T) :=
  p.queue.pop

/-- Get next item and the time spent waiting for it (ns). -/
def nextWithWait (p : Prefetcher T) : IO (Option T × Nat) :=
  p.queue.popWithWait

/-- Cancel the prefetcher -/
def cancel (p : Prefetcher T) : IO Unit := do
  p.queue.finish
  IO.cancel p.worker

/-- Check if prefetcher is done -/
def isDone (p : Prefetcher T) : IO Bool :=
  IO.hasFinished p.worker

/-- Wait for prefetcher to complete -/
def wait (p : Prefetcher T) : IO Unit := do
  let _ ← IO.wait p.worker

/-- Iterate through all prefetched items -/
def forEach (p : Prefetcher T) (f : T → IO Unit) : IO Unit := do
  repeat do
    match ← p.next with
    | some item => f item
    | none => break

/-- Collect all items into array -/
def toArray (p : Prefetcher T) : IO (Array T) := do
  let mut arr := Array.mkEmpty p.totalItems
  repeat do
    match ← p.next with
    | some item => arr := arr.push item
    | none => break
  pure arr

end Prefetcher

/-! ## ForIn Instances -/

instance : ForIn IO (Prefetcher T) T where
  forIn p init f := do
    let mut acc := init
    repeat do
      match ← p.next with
      | none => break
      | some x =>
        match ← f x acc with
        | .done a =>
            p.cancel
            return a
        | .yield a => acc := a
    pure acc

/-! ## PrefetchedDataset -/

/-- A dataset that prefetches items using a background worker -/
structure PrefetchedDataset (D : Type) (T : Type) where
  inner : D
  bufferSize : Nat

/-- Create a prefetched version of a dataset -/
def prefetchDs [Dataset D T] (bufferSize : Nat := 8) (ds : D) : PrefetchedDataset D T :=
  { inner := ds, bufferSize }

/-- Start iteration with prefetching -/
def PrefetchedDataset.toPrefetcher [Dataset D T] (pds : PrefetchedDataset D T) : IO (Prefetcher T) :=
  Prefetcher.create pds.inner pds.bufferSize

/-- Iterate over prefetcher results, applying a function -/
def iteratePrefetched [Dataset D T] (ds : D) (bufferSize : Nat := 8) (f : T → IO Unit) : IO Unit := do
  let prefetcher ← Prefetcher.create ds bufferSize
  prefetcher.forEach f

/-! ## Benchmarking Utilities -/

/-- Benchmark prefetcher throughput -/
def benchmarkPrefetcher [Dataset D T] (ds : D) (bufferSize : Nat := 8) : IO Float := do
  let n := Dataset.len ds
  if n == 0 then return 0.0

  let start ← IO.monoNanosNow
  let prefetcher ← Prefetcher.create ds bufferSize

  let mut count := 0
  repeat do
    match ← prefetcher.next with
    | some _ => count := count + 1
    | none => break

  let stop ← IO.monoNanosNow
  let seconds := (stop - start).toFloat / 1e9
  pure (count.toFloat / seconds)

/-- Compare prefetched vs non-prefetched throughput -/
def benchmarkComparison [Dataset D T] (ds : D) (bufferSize : Nat := 8) : IO (Float × Float) := do
  -- Non-prefetched baseline
  let baselineRate ← benchmarkThroughput ds 1

  -- Prefetched
  let prefetchRate ← benchmarkPrefetcher ds bufferSize

  pure (baselineRate, prefetchRate)

end TinyGrad4.Data
