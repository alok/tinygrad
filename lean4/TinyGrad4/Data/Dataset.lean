import TinyGrad4.DType
import TinyGrad4.Data.ArrayN

/-!
# Dataset - Core Data Loading Abstractions

Grain-style composable data loading for distributed training.

## Features
- `RandKey`: JAX-style splittable PRNG for reproducible shuffling
- `Dataset`: Random-access dataset typeclass
- `DataIterator`: Checkpointable sequential iteration

## Usage
```lean
let mnist ← MNIST.load "data"
let key := RandKey.new 42
let trainDs := mnist.train
  |> shard (rank := 0) (worldSize := 4)
  |> shuffle key
  |> batch 64
  |> prefetch 4

for batch in trainDs.toIterator do
  let loss ← model.forward batch.x
```
-/

namespace TinyGrad4.Data

/-! ## RandKey - Reproducible Randomness -/

/-- JAX-style splittable PRNG key for reproducible randomness.
    Uses a simple LCG with good mixing properties. -/
structure RandKey where
  state : UInt64
  deriving Repr, BEq, Hashable, Inhabited

namespace RandKey

/-- LCG multiplier (same as PCG) -/
private def lcgMult : UInt64 := 6364136223846793005

/-- LCG increment -/
private def lcgInc : UInt64 := 1442695040888963407

/-- Create a new key from a seed -/
def new (seed : Nat := 0) : RandKey :=
  { state := UInt64.ofNat seed ||| 1 }  -- Ensure odd (better LCG behavior)

/-- Advance the state and return a random UInt64 -/
def next (k : RandKey) : UInt64 × RandKey :=
  let s := k.state * lcgMult + lcgInc
  (s, { state := s })

/-- Generate a random Nat in [0, bound) -/
def randNat (k : RandKey) (bound : Nat) : Nat × RandKey :=
  if bound == 0 then (0, k)
  else
    let (r, k') := k.next
    (r.toNat % bound, k')

/-- Split a key into two independent keys.
    Uses different constants to ensure independence. -/
def split (k : RandKey) : RandKey × RandKey :=
  let s1 := k.state * lcgMult + lcgInc
  let s2 := s1 * lcgMult + 0xDEADBEEF
  ({ state := s1 }, { state := s2 })

/-- Fold additional data into the key (for per-sample determinism) -/
def fold (k : RandKey) (data : UInt64) : RandKey :=
  { state := k.state ^^^ data * lcgMult + lcgInc }

/-- Generate n random UInt64 values -/
def randoms (k : RandKey) (n : Nat) : Array UInt64 × RandKey := Id.run do
  let mut state := k.state
  let mut out := Array.mkEmpty n
  for _ in [:n] do
    state := state * lcgMult + lcgInc
    out := out.push state
  (out, { state })

/-- Fisher-Yates shuffle of indices [0, n) -/
def shuffleIndices (k : RandKey) (n : Nat) : Array Nat × RandKey := Id.run do
  let mut arr := Array.range n
  let mut state := k.state
  for i in [:n] do
    -- Generate random index j in [i, n)
    state := state * lcgMult + lcgInc
    let j := i + (state.toNat % (n - i))
    -- Swap arr[i] and arr[j]
    let tmp := arr[i]!
    arr := arr.set! i arr[j]!
    arr := arr.set! j tmp
  (arr, { state })

end RandKey

/-! ## Dataset Typeclass -/

/-- Random-access dataset: finite collection with indexed access.
    This is the core abstraction for data sources.

    Follows GetElem pattern: safe access requires proof, unsafe is available.
-/
class Dataset (D : Type) (T : outParam Type) where
  /-- Number of elements in the dataset -/
  len : D → Nat
  /-- Get element at index with proof of bounds (may perform IO for lazy loading) -/
  getItem : (ds : D) → (idx : Nat) → idx < len ds → IO T

namespace Dataset

/-- Check if dataset is empty -/
def isEmpty [Dataset D T] (ds : D) : Bool := len ds == 0

/-- Get item with bounds check, returning Option -/
def getItem? [Dataset D T] (ds : D) (idx : Nat) : IO (Option T) := do
  if h : idx < len ds then
    some <$> getItem ds idx h
  else
    pure none

/-- Get item unsafely (panics if out of bounds) -/
def getItem! [Dataset D T] [Inhabited T] (ds : D) (idx : Nat) : IO T := do
  if h : idx < len ds then
    getItem ds idx h
  else
    panic! s!"Dataset index {idx} out of bounds (len = {len ds})"

/-- Convert entire dataset to array (for small datasets) -/
def toArray [Dataset D T] (ds : D) : IO (Array T) := do
  let n := len ds
  let mut arr := Array.mkEmpty n
  for i in [:n] do
    if h : i < n then
      arr := arr.push (← getItem ds i h)
  pure arr

end Dataset

/-! ## Dataset GetElem instance -/

/-- Enable bracket notation: `ds[i]` with proof, `ds[i]?` without -/
instance [Dataset D T] : GetElem D Nat (IO T) (fun ds i => i < Dataset.len ds) where
  getElem ds i h := Dataset.getItem ds i h

/-! ## DataIterator -/

/-- Checkpointable iterator state -/
structure IteratorState where
  position : Nat
  epoch : Nat
  key : RandKey
  deriving Repr, Inhabited

/-- Sequential iterator over a dataset with checkpoint support -/
structure DataIterator (T : Type) where
  private mk ::
  /-- Get next item, or none if exhausted -/
  nextFn : IO (Option T)
  /-- Get current state for checkpointing -/
  stateFn : IO IteratorState
  /-- Restore from checkpoint state -/
  restoreFn : IteratorState → IO Unit

namespace DataIterator

/-- Get next item -/
def next (iter : DataIterator T) : IO (Option T) := iter.nextFn

/-- Get checkpoint state -/
def checkpoint (iter : DataIterator T) : IO IteratorState := iter.stateFn

/-- Restore from checkpoint -/
def restore (iter : DataIterator T) (state : IteratorState) : IO Unit := iter.restoreFn state

/-- Iterate with a callback -/
def forEach (iter : DataIterator T) (f : T → IO Unit) : IO Unit := do
  repeat do
    match ← iter.next with
    | some x => f x
    | none => break

/-- Collect remaining items into array -/
def toArray (iter : DataIterator T) : IO (Array T) := do
  let mut arr := #[]
  repeat do
    match ← iter.next with
    | some x => arr := arr.push x
    | none => break
  pure arr

end DataIterator

/-! ## Simple Dataset Wrappers -/

/-- Dataset backed by an array -/
structure ArrayDataset (T : Type) where
  data : Array T

instance : Dataset (ArrayDataset T) T where
  len ds := ds.data.size
  getItem ds idx h := pure ds.data[idx]

/-- Create dataset from array -/
def ofArray (arr : Array T) : ArrayDataset T := { data := arr }

/-- Create iterator for an array dataset -/
def ArrayDataset.toIterator (ds : ArrayDataset T) (key : RandKey := RandKey.new 0) : IO (DataIterator T) := do
  let posRef ← IO.mkRef 0
  let epochRef ← IO.mkRef 0
  let keyRef ← IO.mkRef key

  pure {
    nextFn := do
      let pos ← posRef.get
      if h : pos < ds.data.size then
        posRef.set (pos + 1)
        pure (some ds.data[pos])
      else
        pure none

    stateFn := do
      pure {
        position := ← posRef.get
        epoch := ← epochRef.get
        key := ← keyRef.get
      }

    restoreFn := fun state => do
      posRef.set state.position
      epochRef.set state.epoch
      keyRef.set state.key
  }

/-! ## Benchmarking Utilities -/

/-- Time an IO action in nanoseconds -/
def timeNs (action : IO α) : IO (α × Nat) := do
  let start ← IO.monoNanosNow
  let result ← action
  let stop ← IO.monoNanosNow
  pure (result, stop - start)

/-- Benchmark throughput: items per second -/
def benchmarkThroughput [Dataset D T] (ds : D) (iterations : Nat := 1) : IO Float := do
  let n := Dataset.len ds
  if n == 0 then return 0.0

  let (_, elapsed) ← timeNs do
    for _ in [:iterations] do
      for i in [:n] do
        if h : i < n then
          let _ ← Dataset.getItem ds i h

  let totalItems := n * iterations
  let seconds := elapsed.toFloat / 1e9
  pure (totalItems.toFloat / seconds)

end TinyGrad4.Data
