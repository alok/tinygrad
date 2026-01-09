import TinyGrad4.Data.Dataset

/-!
# Transform - Composable Dataset Transformations

Grain-style composable transformations that wrap datasets.

## Transformations
- `MappedDataset`: Apply function to each element
- `BatchedDataset`: Collect elements into batches
- `TakenDataset`: Take first n elements
- `DroppedDataset`: Skip first n elements
- `FilteredDataset`: Keep elements matching predicate

## Usage
```lean
let ds := dataset
  |> mapDs (fun x => x * 2)
  |> batchDs 64
  |> takeDs 1000
```
-/

namespace TinyGrad4.Data

/-! ## MappedDataset -/

/-- Dataset with a function applied to each element -/
structure MappedDataset (D : Type) (A B : Type) where
  inner : D
  f : A → B

instance [Dataset D A] : Dataset (MappedDataset D A B) B where
  len ds := Dataset.len ds.inner
  getItem ds idx h := do
    let x ← Dataset.getItem ds.inner idx h
    pure (ds.f x)

/-- Map a function over dataset elements -/
def mapDs [Dataset D A] (f : A → B) (ds : D) : MappedDataset D A B :=
  { inner := ds, f := f }

/-- Map with IO function -/
structure MappedIODataset (D : Type) (A B : Type) where
  inner : D
  f : A → IO B

instance [Dataset D A] : Dataset (MappedIODataset D A B) B where
  len ds := Dataset.len ds.inner
  getItem ds idx h := do
    let x ← Dataset.getItem ds.inner idx h
    ds.f x

/-- Map an IO function over dataset elements -/
def mapIODs [Dataset D A] (f : A → IO B) (ds : D) : MappedIODataset D A B :=
  { inner := ds, f := f }

/-! ## BatchedDataset -/

/-- Dataset that groups elements into batches -/
structure BatchedDataset (D : Type) (T : Type) where
  inner : D
  batchSize : Nat
  dropLast : Bool := true

/-- Number of batches in a batched dataset -/
def BatchedDataset.numBatches [Dataset D T] (ds : BatchedDataset D T) : Nat :=
  let n := Dataset.len ds.inner
  if ds.batchSize == 0 then 0
  else if ds.dropLast then n / ds.batchSize
  else (n + ds.batchSize - 1) / ds.batchSize

instance [Dataset D T] : Dataset (BatchedDataset D T) (Array T) where
  len ds := ds.numBatches
  getItem ds batchIdx _ := do
    let start := batchIdx * ds.batchSize
    let n := Dataset.len ds.inner
    let endIdx := min (start + ds.batchSize) n
    let mut batch := Array.mkEmpty ds.batchSize
    for i in [start:endIdx] do
      if h : i < n then
        batch := batch.push (← Dataset.getItem ds.inner i h)
    pure batch

/-- Group elements into batches -/
def batchDs [Dataset D T] (batchSize : Nat) (ds : D) (dropLast : Bool := true) : BatchedDataset D T :=
  { inner := ds, batchSize, dropLast }

/-! ## TakenDataset -/

/-- Dataset limited to first n elements -/
structure TakenDataset (D : Type) (T : Type) where
  inner : D
  count : Nat

instance [Dataset D T] : Dataset (TakenDataset D T) T where
  len ds := min ds.count (Dataset.len ds.inner)
  getItem ds idx h := Dataset.getItem ds.inner idx (Nat.lt_of_lt_of_le h (Nat.min_le_right ..))


/-- Take first n elements -/
def takeDs [Dataset D T] (n : Nat) (ds : D) : TakenDataset D T :=
  { inner := ds, count := n }

/-! ## DroppedDataset -/

/-- Dataset with first n elements skipped -/
structure DroppedDataset (D : Type) (T : Type) where
  inner : D
  count : Nat

instance [Dataset D T] : Dataset (DroppedDataset D T) T where
  len ds :=
    let n := Dataset.len ds.inner
    if ds.count >= n then 0 else n - ds.count
  getItem ds idx h := do
    let innerLen := Dataset.len ds.inner
    have hInner : idx + ds.count < innerLen := by
      simp at h
      split at h <;> omega
    Dataset.getItem ds.inner (idx + ds.count) hInner

/-- Drop first n elements -/
def dropDs [Dataset D T] (n : Nat) (ds : D) : DroppedDataset D T :=
  { inner := ds, count := n }

/-! ## FilteredDataset -/

/-- Dataset filtered by predicate.
    Note: This requires pre-computing the valid indices for random access. -/
structure FilteredDataset (D : Type) (T : Type) [Dataset D T] where
  inner : D
  /-- Cached length of the inner dataset (for index proofs). -/
  innerLen : Nat
  /-- Proof that cached length matches the inner dataset length. -/
  lenEq : innerLen = Dataset.len inner
  /-- Indices that pass the filter (computed on construction). -/
  validIndices : Array (Fin innerLen)

/-- Filter a dataset by predicate (computes valid indices eagerly) -/
def filterDs [Dataset D T] (pred : T → Bool) (ds : D) : IO (FilteredDataset D T) := do
  let n := Dataset.len ds
  let mut indices : Array (Fin n) := Array.mkEmpty n
  for i in [:n] do
    if h : i < n then
      let x ← Dataset.getItem ds i h
      if pred x then
        indices := indices.push ⟨i, h⟩
  pure { inner := ds, innerLen := n, lenEq := rfl, validIndices := indices }

instance [Dataset D T] : Dataset (FilteredDataset D T) T where
  len ds := ds.validIndices.size
  getItem ds idx h := do
    let realIdx := ds.validIndices[idx]'(h)
    have hReal : realIdx.val < Dataset.len ds.inner := by
      simpa [ds.lenEq] using realIdx.isLt
    Dataset.getItem ds.inner realIdx.val hReal

/-! ## ConcatDataset -/

/-- Concatenation of two datasets -/
structure ConcatDataset (D1 D2 : Type) (T : Type) where
  first : D1
  second : D2

instance [Dataset D1 T] [Dataset D2 T] : Dataset (ConcatDataset D1 D2 T) T where
  len ds := Dataset.len ds.first + Dataset.len ds.second
  getItem ds idx h := do
    let n1 := Dataset.len ds.first
    if hLt : idx < n1 then
      Dataset.getItem ds.first idx hLt
    else
      have hSecond : idx - n1 < Dataset.len ds.second := by omega
      Dataset.getItem ds.second (idx - n1) hSecond

/-- Concatenate two datasets -/
def concatDs [Dataset D1 T] [Dataset D2 T] (ds1 : D1) (ds2 : D2) : ConcatDataset D1 D2 T :=
  { first := ds1, second := ds2 }

/-! ## ZippedDataset -/

/-- Zip two datasets together -/
structure ZippedDataset (D1 D2 : Type) (T1 T2 : Type) where
  first : D1
  second : D2

instance [Dataset D1 T1] [Dataset D2 T2] : Dataset (ZippedDataset D1 D2 T1 T2) (T1 × T2) where
  len ds := min (Dataset.len ds.first) (Dataset.len ds.second)
  getItem ds idx h := do
    have h1 : idx < Dataset.len ds.first := Nat.lt_of_lt_of_le h (Nat.min_le_left ..)
    have h2 : idx < Dataset.len ds.second := Nat.lt_of_lt_of_le h (Nat.min_le_right ..)
    let x ← Dataset.getItem ds.first idx h1
    let y ← Dataset.getItem ds.second idx h2
    pure (x, y)

/-- Zip two datasets -/
def zipDs [Dataset D1 T1] [Dataset D2 T2] (ds1 : D1) (ds2 : D2) : ZippedDataset D1 D2 T1 T2 :=
  { first := ds1, second := ds2 }

/-! ## EnumeratedDataset -/

/-- Dataset with indices attached -/
structure EnumeratedDataset (D : Type) (T : Type) where
  inner : D

instance [Dataset D T] : Dataset (EnumeratedDataset D T) (Nat × T) where
  len ds := Dataset.len ds.inner
  getItem ds idx h := do
    let x ← Dataset.getItem ds.inner idx h
    pure (idx, x)

/-- Enumerate elements with their indices -/
def enumerateDs [Dataset D T] (ds : D) : EnumeratedDataset D T :=
  { inner := ds }

/-! ## Pipeline operator for composability

Note: Lean/Std already provides `|>` operator, so we don't need to define it.
Use: `dataset |> mapDs f |> batchDs 64`
-/

end TinyGrad4.Data
