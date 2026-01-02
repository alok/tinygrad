import TinyGrad4.Data.Dataset
import TinyGrad4.Data.Shard

-- Disable IO.monoNanosNow linter: prefetch wait attribution uses raw monotonic timestamps.
set_option linter.monoNanosNow false

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

/-! ## IteratorPrefetcher -/

/-- Queue item with iterator state after this element. -/
structure QueuedItem (T : Type) where
  item : T
  state : IteratorState

/-- Prefetcher that preserves iterator state for deterministic resume. -/
structure IteratorPrefetcher (T : Type) where
  queue : IOQueue (QueuedItem T)
  worker : Task (Except IO.Error Unit)
  totalItems : Nat
  lastState : IO.Ref IteratorState

namespace IteratorPrefetcher

private def createFromIterator (iter : DataIterator T) (initState : IteratorState) (totalItems : Nat)
    (bufferSize : Nat := 8) : IO (IteratorPrefetcher T) := do
  let queue ← IOQueue.new bufferSize
  let lastState ← IO.mkRef initState

  let worker ← IO.asTask (prio := .dedicated) do
    repeat do
      if ← IO.checkCanceled then break
      match ← iter.next with
      | some item =>
          let state ← iter.checkpoint
          let ok ← queue.push { item, state }
          if !ok then break
      | none => break
    queue.finish

  pure { queue, worker, totalItems, lastState }

/-- Create a stateful prefetcher from an iterator config. -/
def createFromIteratorCfg [Dataset D T] (cfg : IteratorConfig D) (bufferSize : Nat := 8) :
    IO (IteratorPrefetcher T) := do
  let iter ← Dataset.toIteratorCfg cfg
  let initState ← iter.checkpoint
  let n := Dataset.len cfg.base
  let remaining := remainingItems n cfg.epochs initState
  createFromIterator iter initState remaining bufferSize

/-- Create a stateful prefetcher from an iterator config and explicit state. -/
def createFromIteratorCfgState [Dataset D T] (cfg : IteratorConfig D) (state : IteratorState)
    (bufferSize : Nat := 8) : IO (IteratorPrefetcher T) := do
  let iter ← Dataset.toIteratorCfg cfg
  iter.restore state
  let n := Dataset.len cfg.base
  let remaining := remainingItems n cfg.epochs state
  createFromIterator iter state remaining bufferSize

/-- Get next prefetched item. -/
def next (p : IteratorPrefetcher T) : IO (Option T) := do
  match ← p.queue.pop with
  | some item =>
      p.lastState.set item.state
      pure (some item.item)
  | none => pure none

/-- Get next item and time spent waiting for it (ns). -/
def nextWithWait (p : IteratorPrefetcher T) : IO (Option T × Nat) := do
  let (item?, waitNs) ← p.queue.popWithWait
  match item? with
  | some item =>
      p.lastState.set item.state
      pure (some item.item, waitNs)
  | none => pure (none, waitNs)

/-- Get checkpoint state for deterministic resume. -/
def checkpoint (p : IteratorPrefetcher T) : IO IteratorState :=
  p.lastState.get

/-- Cancel the prefetcher. -/
def cancel (p : IteratorPrefetcher T) : IO Unit := do
  p.queue.finish
  IO.cancel p.worker

/-- Check if worker is done. -/
def isDone (p : IteratorPrefetcher T) : IO Bool :=
  IO.hasFinished p.worker

/-- Wait for worker completion. -/
def wait (p : IteratorPrefetcher T) : IO Unit := do
  let _ ← IO.wait p.worker

/-- Drain any queued items. -/
def drain (p : IteratorPrefetcher T) : IO Unit := do
  repeat do
    match ← p.queue.pop with
    | some _ => pure ()
    | none => break

/-- Iterate through all items. -/
def forEach (p : IteratorPrefetcher T) (f : T → IO Unit) : IO Unit := do
  repeat do
    match ← p.next with
    | some item => f item
    | none => break

/-- Collect all remaining items. -/
def toArray (p : IteratorPrefetcher T) : IO (Array T) := do
  let mut arr := Array.mkEmpty p.totalItems
  repeat do
    match ← p.next with
    | some item => arr := arr.push item
    | none => break
  pure arr

end IteratorPrefetcher

/-! ## BatchPrefetcher -/

/-- Queue item with iterator state after this batch. -/
structure QueuedBatch (B : Type) where
  batch : B
  state : IteratorState

/-- Prefetcher that yields collated batches with checkpoint state. -/
structure BatchPrefetcher (B : Type) where
  queue : IOQueue (QueuedBatch B)
  worker : Task (Except IO.Error Unit)
  totalBatches : Nat
  lastState : IO.Ref IteratorState

namespace BatchPrefetcher

private def batchCount (totalItems batchSize : Nat) (dropLast : Bool) : Nat :=
  if batchSize == 0 then
    0
  else if dropLast then
    totalItems / batchSize
  else
    let full := totalItems / batchSize
    let rem := totalItems % batchSize
    if rem == 0 then full else full + 1

private def nextBatchFromIterator (iter : DataIterator T) (batchSize : Nat) (dropLast : Bool)
    (collate : Array T → IO B) : IO (Option (B × IteratorState)) := do
  if batchSize == 0 then
    return none
  let mut items := Array.mkEmpty batchSize
  for _ in [:batchSize] do
    match ← iter.next with
    | some item => items := items.push item
    | none =>
        if dropLast || items.isEmpty then
          return none
        else
          let state ← iter.checkpoint
          let batch ← collate items
          return some (batch, state)
  let state ← iter.checkpoint
  let batch ← collate items
  pure (some (batch, state))

private def createFromIterator (iter : DataIterator T) (initState : IteratorState) (totalItems : Nat)
    (batchSize : Nat) (collate : Array T → IO B) (dropLast : Bool := true) (bufferSize : Nat := 8) :
    IO (BatchPrefetcher B) := do
  if batchSize == 0 then
    throw (IO.userError "BatchPrefetcher: batchSize must be > 0")
  let queue ← IOQueue.new bufferSize
  let lastState ← IO.mkRef initState
  let totalBatches := batchCount totalItems batchSize dropLast

  let worker ← IO.asTask (prio := .dedicated) do
    repeat do
      if ← IO.checkCanceled then break
      match ← nextBatchFromIterator iter batchSize dropLast collate with
      | some (batch, state) =>
          let ok ← queue.push { batch, state }
          if !ok then break
      | none => break
    queue.finish

  pure { queue, worker, totalBatches, lastState }

/-- Create a stateful batch prefetcher from an iterator config. -/
def createFromIteratorCfg [Dataset D T] (cfg : IteratorConfig D) (batchSize : Nat)
    (collate : Array T → IO B) (dropLast : Bool := true) (bufferSize : Nat := 8) :
    IO (BatchPrefetcher B) := do
  let iter ← Dataset.toIteratorCfg cfg
  let initState ← iter.checkpoint
  let n := Dataset.len cfg.base
  let remaining := remainingItems n cfg.epochs initState
  createFromIterator iter initState remaining batchSize collate dropLast bufferSize

/-- Get next prefetched batch. -/
def next (p : BatchPrefetcher B) : IO (Option B) := do
  match ← p.queue.pop with
  | some item =>
      p.lastState.set item.state
      pure (some item.batch)
  | none => pure none

/-- Get next batch and its checkpoint state. -/
def nextWithState (p : BatchPrefetcher B) : IO (Option (B × IteratorState)) := do
  match ← p.queue.pop with
  | some item =>
      p.lastState.set item.state
      pure (some (item.batch, item.state))
  | none => pure none

/-- Get next batch and time spent waiting (ns). -/
def nextWithWait (p : BatchPrefetcher B) : IO (Option B × Nat) := do
  let (item?, waitNs) ← p.queue.popWithWait
  match item? with
  | some item =>
      p.lastState.set item.state
      pure (some item.batch, waitNs)
  | none => pure (none, waitNs)

/-- Get checkpoint state for deterministic resume. -/
def checkpoint (p : BatchPrefetcher B) : IO IteratorState :=
  p.lastState.get

/-- Cancel the prefetcher. -/
def cancel (p : BatchPrefetcher B) : IO Unit := do
  p.queue.finish
  IO.cancel p.worker

/-- Check if worker is done. -/
def isDone (p : BatchPrefetcher B) : IO Bool :=
  IO.hasFinished p.worker

/-- Wait for worker completion. -/
def wait (p : BatchPrefetcher B) : IO Unit := do
  let _ ← IO.wait p.worker

/-- Drain any queued batches. -/
def drain (p : BatchPrefetcher B) : IO Unit := do
  repeat do
    match ← p.queue.pop with
    | some _ => pure ()
    | none => break

/-- Iterate through all batches. -/
def forEach (p : BatchPrefetcher B) (f : B → IO Unit) : IO Unit := do
  repeat do
    match ← p.next with
    | some batch => f batch
    | none => break

/-- Collect all remaining batches. -/
def toArray (p : BatchPrefetcher B) : IO (Array B) := do
  let mut arr := Array.mkEmpty p.totalBatches
  repeat do
    match ← p.next with
    | some batch => arr := arr.push batch
    | none => break
  pure arr

end BatchPrefetcher

/-! ## MultiIteratorPrefetcher -/

/-- Checkpoint state for multi-worker prefetch. -/
structure MultiIteratorState where
  nextWorker : Nat
  workerStates : Array IteratorState
  deriving Repr, Inhabited

/-- Multi-worker prefetcher that deterministically interleaves shard iterators. -/
structure MultiIteratorPrefetcher (T : Type) where
  workers : Array (IteratorPrefetcher T)
  nextWorker : IO.Ref Nat
  done : IO.Ref (Array Bool)
  totalItems : Nat

namespace MultiIteratorPrefetcher

private def defaultStateFromCfg (cfg : IteratorConfig D) : IteratorState :=
  { position := cfg.startPos, epoch := cfg.startEpoch, key := cfg.key }

private def shardStartPos (cfg : ShardConfig) (n globalStart : Nat) : Nat := Id.run do
  if n == 0 then
    return 0
  let shardSize := cfg.shardSize n
  let localPos :=
    match cfg.mode with
    | .interleaved =>
        if globalStart <= cfg.shardIndex then
          0
        else
          (globalStart - cfg.shardIndex + cfg.numShards - 1) / cfg.numShards
    | .contiguous =>
        let baseSize := n / cfg.numShards
        let remainder := n % cfg.numShards
        let blockStart :=
          if cfg.shardIndex < remainder then
            cfg.shardIndex * (baseSize + 1)
          else
            remainder * (baseSize + 1) + (cfg.shardIndex - remainder) * baseSize
        if globalStart <= blockStart then 0 else globalStart - blockStart
  if localPos > shardSize then shardSize else localPos

private def shardIteratorCfg [Dataset D T] (cfg : IteratorConfig D) (scfg : ShardConfig) (n : Nat) :
    IteratorConfig (ShardedDataset D T) :=
  {
    base := shardWithConfig scfg cfg.base
    startPos := shardStartPos scfg n cfg.startPos
    startEpoch := cfg.startEpoch
    epochs := cfg.epochs
    key := cfg.key
    updateKey := cfg.updateKey
    datasetAtEpoch := fun ds k epoch =>
      shardWithConfig scfg (cfg.datasetAtEpoch ds.inner k epoch)
  }

/-- Build a per-worker iterator config for multi-worker prefetch. -/
def workerConfig [Dataset D T] (cfg : IteratorConfig D) (numWorkers workerIdx : Nat)
    (mode : ShardMode := .interleaved) (dropRemainder : Bool := true) :
    IteratorConfig (ShardedDataset D T) :=
  let n := Dataset.len cfg.base
  let scfg : ShardConfig := { shardIndex := workerIdx, numShards := numWorkers, mode, dropRemainder }
  shardIteratorCfg cfg scfg n

/-- Create a multi-worker prefetcher from a config and explicit state. -/
def createFromIteratorCfgState [Dataset D T] (cfg : IteratorConfig D) (numWorkers : Nat)
    (state : MultiIteratorState) (bufferSize : Nat := 8) (mode : ShardMode := .interleaved)
    (dropRemainder : Bool := true) : IO (MultiIteratorPrefetcher T) := do
  if numWorkers == 0 then
    throw (IO.userError "MultiIteratorPrefetcher: numWorkers must be > 0")
  let n := Dataset.len cfg.base
  let mut workers := Array.mkEmpty numWorkers
  let mut doneFlags := Array.mkEmpty numWorkers
  let mut totalItems := 0
  for i in [:numWorkers] do
    let scfg : ShardConfig := { shardIndex := i, numShards := numWorkers, mode, dropRemainder }
    let cfg' := shardIteratorCfg cfg scfg n
    let st := state.workerStates.getD i (defaultStateFromCfg cfg')
    let pref ← IteratorPrefetcher.createFromIteratorCfgState cfg' st bufferSize
    totalItems := totalItems + pref.totalItems
    doneFlags := doneFlags.push (pref.totalItems == 0)
    workers := workers.push pref
  let nextWorker ← IO.mkRef (state.nextWorker % numWorkers)
  let done ← IO.mkRef doneFlags
  pure { workers, nextWorker, done, totalItems }

/-- Create a multi-worker prefetcher from a config. -/
def createFromIteratorCfg [Dataset D T] (cfg : IteratorConfig D) (numWorkers : Nat) (bufferSize : Nat := 8)
    (mode : ShardMode := .interleaved) (dropRemainder : Bool := true) :
    IO (MultiIteratorPrefetcher T) := do
  if numWorkers == 0 then
    throw (IO.userError "MultiIteratorPrefetcher: numWorkers must be > 0")
  let n := Dataset.len cfg.base
  let mut initStates := Array.mkEmpty numWorkers
  for i in [:numWorkers] do
    let scfg : ShardConfig := { shardIndex := i, numShards := numWorkers, mode, dropRemainder }
    let cfg' := shardIteratorCfg cfg scfg n
    initStates := initStates.push (defaultStateFromCfg cfg')
  createFromIteratorCfgState cfg numWorkers { nextWorker := 0, workerStates := initStates }
    bufferSize mode dropRemainder

private def advanceIndex (idx count : Nat) : Nat :=
  if count == 0 then 0 else (idx + 1) % count

/-- Get next item with wait time. -/
partial def nextWithWait (p : MultiIteratorPrefetcher T) : IO (Option T × Nat) := do
  let n := p.workers.size
  if n == 0 then
    return (none, 0)
  let mut idx := (← p.nextWorker.get) % n
  let mut tried := 0
  let mut waitTotal := 0
  let mut doneFlags ← p.done.get
  while tried < n do
    if doneFlags[idx]! then
      idx := advanceIndex idx n
      tried := tried + 1
    else
      if hIdx : idx < n then
        let w := p.workers[idx]'hIdx
        let (item?, waitNs) ← w.nextWithWait
        waitTotal := waitTotal + waitNs
        match item? with
        | some item =>
            p.nextWorker.set (advanceIndex idx n)
            return (some item, waitTotal)
        | none =>
            doneFlags := doneFlags.set! idx true
            p.done.set doneFlags
            idx := advanceIndex idx n
            tried := tried + 1
      else
        idx := advanceIndex idx n
        tried := tried + 1
  p.nextWorker.set idx
  pure (none, waitTotal)

/-- Get next item (blocks in round-robin order). -/
def next (p : MultiIteratorPrefetcher T) : IO (Option T) := do
  let (item?, _) ← nextWithWait p
  pure item?

/-- Get checkpoint state for deterministic resume. -/
def checkpoint (p : MultiIteratorPrefetcher T) : IO MultiIteratorState := do
  let nextWorker ← p.nextWorker.get
  let mut states := Array.mkEmpty p.workers.size
  for w in p.workers do
    states := states.push (← w.checkpoint)
  pure { nextWorker, workerStates := states }

/-- Cancel all workers. -/
def cancel (p : MultiIteratorPrefetcher T) : IO Unit := do
  for w in p.workers do
    w.cancel

/-- Wait for all workers to finish. -/
def wait (p : MultiIteratorPrefetcher T) : IO Unit := do
  for w in p.workers do
    w.wait

/-- Check if all workers are done (as observed by the consumer). -/
def isDone (p : MultiIteratorPrefetcher T) : IO Bool := do
  let doneFlags ← p.done.get
  pure (doneFlags.all id)

/-- Iterate through all items. -/
def forEach (p : MultiIteratorPrefetcher T) (f : T → IO Unit) : IO Unit := do
  repeat do
    match ← p.next with
    | some item => f item
    | none => break

/-- Collect all remaining items. -/
def toArray (p : MultiIteratorPrefetcher T) : IO (Array T) := do
  let mut arr := Array.mkEmpty p.totalItems
  repeat do
    match ← p.next with
    | some item => arr := arr.push item
    | none => break
  pure arr

end MultiIteratorPrefetcher

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

instance : ForIn IO (IteratorPrefetcher T) T where
  forIn p init f := do
    let mut acc := init
    repeat do
      match ← IteratorPrefetcher.next p with
      | none => break
      | some x =>
          match ← f x acc with
          | .done a =>
              IteratorPrefetcher.cancel p
              return a
          | .yield a => acc := a
    pure acc

instance : ForIn IO (BatchPrefetcher B) B where
  forIn p init f := do
    let mut acc := init
    repeat do
      match ← BatchPrefetcher.next p with
      | none => break
      | some x =>
          match ← f x acc with
          | .done a =>
              BatchPrefetcher.cancel p
              return a
          | .yield a => acc := a
    pure acc

instance : ForIn IO (MultiIteratorPrefetcher T) T where
  forIn p init f := do
    let mut acc := init
    repeat do
      match ← MultiIteratorPrefetcher.next p with
      | none => break
      | some x =>
          match ← f x acc with
          | .done a =>
              MultiIteratorPrefetcher.cancel p
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
