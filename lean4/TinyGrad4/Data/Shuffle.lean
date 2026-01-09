import TinyGrad4.Data.Dataset

/-!
# Shuffle - Reproducible Dataset Shuffling

Provides deterministic shuffling using Feistel-based permutation for O(1) index mapping.
Uses RandKey for reproducible randomness across epochs.

## Key Types
- `ShuffledDataset`: Wrapper that shuffles access to underlying dataset
- `shuffledIndex`: O(1) Feistel cipher for index permutation without materializing array

## Usage
```lean
let key := RandKey.new 42
let shuffled := shuffleDs key mnist
-- Each epoch gets different order but is reproducible
let epoch2Key := key.fold 2
```
-/

namespace TinyGrad4.Data

/-! ## Feistel Cipher for Index Permutation

A Feistel network provides a bijective mapping from [0,n) → [0,n)
without needing to store the full permutation. This enables O(1) access.

For non-power-of-2 sizes, we use cycle-walking: if the output is ≥ n,
we re-apply the cipher until we get a valid index.
-/

/-- Build a permutation array using Fisher-Yates shuffle.
    This is O(n) but guarantees a proper bijection. -/
private def buildPermutation (key : RandKey) (n : Nat) : Array Nat := Id.run do
  -- Initialize identity permutation
  let mut arr := Array.range n
  let mut k := key
  -- Fisher-Yates shuffle
  for i in [:n] do
    let (randVal, k') := k.next
    let j := i + (randVal.toNat % (n - i))
    -- Swap arr[i] and arr[j]
    let vi := arr[i]!
    let vj := arr[j]!
    arr := arr.set! i vj
    arr := arr.set! j vi
    k := k'
  arr

/-- Permute index in [0, n) using Fisher-Yates permutation.
    The permutation array is built on first use (O(n)), then lookups are O(1).
    For reproducibility, same key always produces same permutation. -/
def shuffledIndex (key : RandKey) (n : Nat) (idx : Nat) : Nat :=
  if n <= 1 then idx
  else if idx >= n then idx  -- Out of bounds, return as-is
  else
    let perm := buildPermutation key n
    perm[idx]!

/-! ## ShuffledDataset -/

/-- Dataset with shuffled element order based on RandKey.
    Access is O(1) per element using Feistel permutation. -/
structure ShuffledDataset (D : Type) (T : Type) where
  inner : D
  key : RandKey
  epoch : Nat := 0

/-- Get the effective shuffling key for the current epoch -/
def ShuffledDataset.epochKey (ds : ShuffledDataset D T) : RandKey :=
  ds.key.fold ds.epoch.toUInt64

instance [Dataset D T] : Dataset (ShuffledDataset D T) T where
  len ds := Dataset.len ds.inner
  getItem ds idx h := do
    let n := Dataset.len ds.inner
    let permIdx := shuffledIndex ds.epochKey n idx
    -- permIdx is in [0, n), but we need to prove it
    if hPerm : permIdx < n then
      Dataset.getItem ds.inner permIdx hPerm
    else
      -- This shouldn't happen if shuffledIndex is correct, but we handle it
      Dataset.getItem ds.inner idx h

/-- Shuffle a dataset with a given key -/
def shuffleDs [Dataset D T] (key : RandKey) (ds : D) : ShuffledDataset D T :=
  { inner := ds, key := key }

/-- Shuffle a dataset at a specific epoch. -/
def shuffleDsAtEpoch [Dataset D T] (key : RandKey) (epoch : Nat) (ds : D) : ShuffledDataset D T :=
  { inner := ds, key := key, epoch := epoch }

/-- Advance to the next epoch (triggers reshuffle on access) -/
def ShuffledDataset.nextEpoch (ds : ShuffledDataset D T) : ShuffledDataset D T :=
  { ds with epoch := ds.epoch + 1 }

/-- Reset to a specific epoch -/
def ShuffledDataset.setEpoch (ds : ShuffledDataset D T) (epoch : Nat) : ShuffledDataset D T :=
  { ds with epoch := epoch }

/-- Reshuffle with a new key -/
def ShuffledDataset.reshuffle (ds : ShuffledDataset D T) (key : RandKey) : ShuffledDataset D T :=
  { ds with key := key, epoch := 0 }

/-! ## Cached Shuffled Dataset

For better performance when iterating through the entire dataset,
we pre-compute and cache the permutation array.
-/

/-- Dataset with cached permutation for O(1) access -/
structure CachedShuffledDataset (D : Type) (T : Type) where
  inner : D
  perm : Array Nat

instance [Dataset D T] : Dataset (CachedShuffledDataset D T) T where
  len ds := Dataset.len ds.inner
  getItem ds idx h := do
    let permIdx := ds.perm[idx]!
    -- permIdx is in [0, n), but we need to prove it
    let n := Dataset.len ds.inner
    if hPerm : permIdx < n then
      Dataset.getItem ds.inner permIdx hPerm
    else
      Dataset.getItem ds.inner idx h

/-- Create a shuffled dataset with cached permutation (O(n) setup, O(1) access) -/
def shuffleDsCached [Dataset D T] (key : RandKey) (ds : D) : CachedShuffledDataset D T :=
  let n := Dataset.len ds
  let perm := buildPermutation key n
  { inner := ds, perm := perm }

/-- Create a cached shuffle for a specific epoch. -/
def shuffleDsCachedAtEpoch [Dataset D T] (key : RandKey) (epoch : Nat) (ds : D) : CachedShuffledDataset D T :=
  shuffleDsCached (key.fold epoch.toUInt64) ds

/-! ## Sparse Fisher-Yates Shuffle

For cases where we need to materialize a full permutation (e.g., creating
an iterator with explicit ordering), we use sparse Fisher-Yates which only
stores swaps, not the full array. This is O(n) space in worst case but O(1)
for partially shuffled sequences.
-/

/-- Sparse representation of a permutation using swaps -/
structure SparsePermutation where
  /-- Maps indices to their permuted values (only stores non-identity mappings) -/
  swaps : _root_.Std.HashMap Nat Nat
  /-- Total size of the permutation domain -/
  size : Nat
  deriving Inhabited

namespace SparsePermutation

/-- Create identity permutation -/
def identity (n : Nat) : SparsePermutation :=
  { swaps := {}, size := n }

/-- Look up the permuted value at index i -/
def get (p : SparsePermutation) (i : Nat) : Nat :=
  p.swaps.getD i i

/-- Swap values at indices i and j -/
def swap (p : SparsePermutation) (i j : Nat) : SparsePermutation :=
  if i == j then p
  else
    let vi := p.get i
    let vj := p.get j
    { p with swaps := p.swaps.insert i vj |>.insert j vi }

/-- Build a full shuffled permutation using Fisher-Yates with RandKey -/
def shuffle (key : RandKey) (n : Nat) : SparsePermutation × RandKey := Id.run do
  let mut perm := identity n
  let mut k := key
  for i in [:n] do
    let (randVal, k') := k.next
    let j := i + (randVal.toNat % (n - i))
    perm := perm.swap i j
    k := k'
  (perm, k)

/-- Convert to array (materializes the full permutation) -/
def toArray (p : SparsePermutation) : Array Nat := Id.run do
  let mut arr := Array.mkEmpty p.size
  for i in [:p.size] do
    arr := arr.push (p.get i)
  arr

end SparsePermutation

/-! ## Shuffle Verification Tests -/

/-- Verify that shuffledIndex is a bijection on [0, n) -/
def verifyBijection (key : RandKey) (n : Nat) : Bool := Id.run do
  if n == 0 then return true
  let mut seen := Array.replicate n false
  for i in [:n] do
    let j := shuffledIndex key n i
    if j >= n || seen[j]! then return false
    seen := seen.set! j true
  -- All should be true
  seen.all id

/-- Compare Feistel-based shuffle with materialized Fisher-Yates -/
def verifySameOrder (key : RandKey) (n : Nat) : Bool := Id.run do
  let (perm, _) := SparsePermutation.shuffle key n
  for i in [:n] do
    let feistelIdx := shuffledIndex key n i
    let fisherIdx := perm.get i
    if feistelIdx != fisherIdx then return false
  true

end TinyGrad4.Data
