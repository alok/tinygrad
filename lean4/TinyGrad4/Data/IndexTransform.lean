import Batteries.Data.Array.Basic

/-!
# IndexTransform - Pure Index Mapping Category

Pure index transforms that can be composed and verified.
This layer is entirely pure - no IO, no ownership concerns.

## Design Principles
1. **Pure**: All transforms are pure functions on indices
2. **Composable**: Category structure with identity and composition
3. **Verifiable**: Can prove properties like bijectivity
4. **Efficient**: O(1) transform application, O(n) for shuffle construction

## Usage
```lean
-- Create transforms
let shuffle := IndexTransform.shuffle key n
let batch := IndexTransform.batch 64 n
let shard := IndexTransform.shard 0 4 n

-- Compose (right to left)
let combined := shuffle.compose batch (by rfl)

-- Apply
let originalIdx := combined.map ⟨5, by omega⟩
```
-/

namespace TinyGrad4.Data

/-! ## Core Transform Structure -/

/-- Pure index transform: maps output indices to input indices.
    This is the fundamental building block for data loading pipelines. -/
structure IndexTransform where
  /-- Number of elements in the input -/
  inputLen : Nat
  /-- Number of elements in the output -/
  outputLen : Nat
  /-- Map output index to input index -/
  map : Fin outputLen → Fin inputLen

namespace IndexTransform

/-! ## Basic Constructors -/

/-- Identity transform -/
def id (n : Nat) : IndexTransform where
  inputLen := n
  outputLen := n
  map := fun i => i

/-- Empty transform (no outputs) -/
def empty (n : Nat) : IndexTransform where
  inputLen := n
  outputLen := 0
  map := fun i => i.elim0

/-- Take first k elements -/
def take (k : Nat) (n : Nat) (h : k ≤ n := by omega) : IndexTransform where
  inputLen := n
  outputLen := k
  map := fun ⟨i, hi⟩ => ⟨i, Nat.lt_of_lt_of_le hi h⟩

/-- Drop first k elements -/
def drop (k : Nat) (n : Nat) (h : k ≤ n := by omega) : IndexTransform where
  inputLen := n
  outputLen := n - k
  map := fun ⟨i, hi⟩ => ⟨i + k, by omega⟩

/-- Reverse order -/
def reverse (n : Nat) : IndexTransform where
  inputLen := n
  outputLen := n
  map := fun ⟨i, hi⟩ => ⟨n - 1 - i, by omega⟩

/-! ## Composition -/

/-- Compose two transforms (f ∘ g means apply g then f).
    Requires that f's output length equals g's input length. -/
def compose (f : IndexTransform) (g : IndexTransform)
    (h : f.outputLen = g.inputLen) : IndexTransform where
  inputLen := f.inputLen
  outputLen := g.outputLen
  map := fun i => f.map (h ▸ g.map i)

-- Note: Compose requires proof that lengths match, so we don't define an infix operator
-- Use compose directly with the proof

/-! ## Category Laws (Statements) -/

/-- Left identity: id ∘ f = f -/
theorem id_comp (f : IndexTransform) :
    compose (id f.inputLen) f (by simp [id]) = f := by
  simp only [compose, id]

/-- Right identity: f ∘ id = f -/
theorem comp_id (f : IndexTransform) :
    compose f (id f.outputLen) rfl = f := by
  simp only [compose, id]

/-! ## Shuffle Transform -/

/-- Fisher-Yates permutation as an array -/
def fisherYatesArray (seed : UInt64) (n : Nat) : Array Nat :=
  if n == 0 then #[]
  else Id.run do
    -- Initialize with identity permutation
    let mut arr : Array Nat := Array.range n
    let mut rng := seed
    -- Fisher-Yates shuffle (Knuth shuffle)
    for i in [:n - 1] do
      -- LCG step
      rng := rng * 6364136223846793005 + 1442695040888963407
      -- Random index in [i, n)
      let j := i + (rng.toNat % (n - i))
      -- Swap arr[i] and arr[j]
      let vi := arr[i]!
      let vj := arr[j]!
      arr := arr.set! i vj
      arr := arr.set! j vi
    arr

/-- Lookup in permutation array with bounds proof -/
def permLookup (perm : Array Nat) (n : Nat) (hn : perm.size = n)
    (_h_valid : ∀ i : Fin n, perm[i.val]! < n) : Fin n → Fin n :=
  fun ⟨i, hi⟩ =>
    let idx := perm[i]?.getD 0
    if h : idx < n then ⟨idx, h⟩ else ⟨0, by omega⟩

/-- Shuffle transform with pre-computed permutation array.
    The permutation is computed once at construction time, not per-lookup. -/
structure ShuffleTransform (n : Nat) where
  /-- Pre-computed permutation array -/
  perm : Array Nat
  /-- Proof that permutation has correct size -/
  h_size : perm.size = n

/-- Create a shuffle transform - O(n) construction, O(1) lookup -/
def shuffle (seed : UInt64) (n : Nat) : IndexTransform :=
  let perm := fisherYatesArray seed n
  { inputLen := n
    outputLen := n
    map := fun ⟨i, hi⟩ =>
      -- Direct array access - perm was computed once at construction
      let idx := perm[i]?.getD 0
      if h : idx < n then ⟨idx, h⟩ else ⟨i, hi⟩ }

/-! ## Batching Transforms -/

/-- Batch transform: groups consecutive elements.
    Output length is n / batchSize (drops incomplete batch). -/
def batchIndices (batchSize : Nat) (_n : Nat) (batchIdx : Nat) (elemIdx : Nat) : Nat :=
  batchIdx * batchSize + elemIdx

/-- Number of complete batches -/
def numBatches (batchSize : Nat) (n : Nat) : Nat :=
  if batchSize == 0 then 0 else n / batchSize

/-! ## Sharding Transforms -/

/-- Compute shard size: number of elements for rank in worldSize shards of n elements -/
def shardSize (rank : Nat) (worldSize : Nat) (n : Nat) : Nat :=
  if worldSize == 0 then 0
  else if rank >= worldSize then 0
  else (n - rank + worldSize - 1) / worldSize

/-- Shard transform: selects every kth element starting at offset.
    For distributed training with `worldSize` workers, worker `rank`
    sees elements at indices rank, rank + worldSize, rank + 2*worldSize, ... -/
def shard (rank : Nat) (worldSize : Nat) (n : Nat)
    (_h_rank : rank < worldSize) (_h_world : worldSize > 0 := by omega)
    (h_n : n > 0 := by omega) : IndexTransform where
  inputLen := n
  outputLen := shardSize rank worldSize n
  map := fun ⟨i, _hi⟩ =>
    let idx := rank + i * worldSize
    -- Use decidable to handle bounds
    if h : idx < n then ⟨idx, h⟩
    else ⟨0, h_n⟩

/-! ## Verification -/

/-- A transform is injective (no two outputs map to the same input) -/
def isInjective (t : IndexTransform) : Prop :=
  ∀ i j : Fin t.outputLen, t.map i = t.map j → i = j

/-- A transform preserves size (same input and output length) -/
def sameSize (t : IndexTransform) : Prop :=
  t.inputLen = t.outputLen

/-- Identity is injective -/
theorem id_injective (n : Nat) : isInjective (id n) := by
  intro i j h
  simp only [id] at h
  exact h

/-- Reverse is injective -/
theorem reverse_injective (n : Nat) : isInjective (reverse n) := by
  intro ⟨a, ha⟩ ⟨b, hb⟩ h
  simp only [reverse, Fin.mk.injEq] at h
  have ha' : a < n := ha
  have hb' : b < n := hb
  have : a = b := by omega
  exact Fin.ext this

/-- Verify a permutation array is valid (all elements < n, all elements unique) -/
def verifyPermutation (perm : Array Nat) (n : Nat) : Bool := Id.run do
  if perm.size != n then return false
  let mut seen : Array Bool := Array.replicate n false
  for i in [:n] do
    let idx := perm[i]!
    if idx >= n then return false
    if seen[idx]! then return false
    seen := seen.set! idx true
  true

/-! ## Epoch Management -/

/-- Create a shuffle transform for a specific epoch.
    Uses seed XOR epoch to get different permutations each epoch. -/
def shuffleEpoch (seed : UInt64) (epoch : Nat) (n : Nat) : IndexTransform :=
  shuffle (seed ^^^ epoch.toUInt64) n

end IndexTransform

end TinyGrad4.Data
