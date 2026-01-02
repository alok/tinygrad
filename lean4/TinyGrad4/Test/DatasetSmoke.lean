import TinyGrad4.Data.Dataset
import TinyGrad4.Data.Transform
import TinyGrad4.Data.Shuffle
import TinyGrad4.Data.Shard
import TinyGrad4.Data.Prefetch
import TinyGrad4.Data.Checkpoint

/-!
# Dataset Smoke Tests

Tests for the grain-style data loading pipeline.
-/

namespace TinyGrad4.Test.DatasetSmoke

open TinyGrad4.Data

/-! ## Helper Utilities -/

/-- Simple array dataset for testing -/
def testArray : ArrayDataset Nat := ofArray #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- Assert condition with message -/
def assert (cond : Bool) (msg : String) : IO Unit := do
  if !cond then
    throw (IO.userError s!"Assertion failed: {msg}")

/-! ## Dataset Tests -/

def testArrayDataset : IO Unit := do
  IO.println "Testing ArrayDataset..."

  let ds := testArray

  -- Test length
  assert (Dataset.len ds == 10) "Length should be 10"

  -- Test getItem with proof
  if h : 5 < Dataset.len ds then
    let item ← Dataset.getItem ds 5 h
    assert (item == 5) "Item at index 5 should be 5"

  -- Test GetElem bracket notation
  if h : 3 < Dataset.len ds then
    let item ← ds[3]
    assert (item == 3) "ds[3] should be 3"

  -- Test toArray
  let arr ← Dataset.toArray ds
  assert (arr.size == 10) "toArray should return 10 elements"
  assert (arr[0]! == 0 && arr[9]! == 9) "Array contents should match"

  IO.println "✓ ArrayDataset tests passed"

/-! ## Transform Tests -/

def testMapDs : IO Unit := do
  IO.println "Testing mapDs..."

  let ds := testArray |> mapDs (· * 2)

  assert (Dataset.len ds == 10) "Mapped dataset should have same length"

  if h : 3 < Dataset.len ds then
    let item ← Dataset.getItem ds 3 h
    assert (item == 6) "Mapped item at 3 should be 6"

  IO.println "✓ mapDs tests passed"

def testBatchDs : IO Unit := do
  IO.println "Testing batchDs..."

  let ds := testArray |> batchDs 3 (dropLast := true)

  assert (Dataset.len ds == 3) "Should have 3 batches (10/3 = 3 with dropLast)"

  if h : 0 < Dataset.len ds then
    let batch ← Dataset.getItem ds 0 h
    assert (batch.size == 3) "First batch should have 3 elements"
    assert (batch[0]! == 0 && batch[1]! == 1 && batch[2]! == 2) "First batch contents"

  IO.println "✓ batchDs tests passed"

def testTakeDs : IO Unit := do
  IO.println "Testing takeDs..."

  let ds := testArray |> takeDs 5

  assert (Dataset.len ds == 5) "Taken dataset should have 5 elements"

  if h : 4 < Dataset.len ds then
    let item ← Dataset.getItem ds 4 h
    assert (item == 4) "Item at index 4 should be 4"

  IO.println "✓ takeDs tests passed"

def testDropDs : IO Unit := do
  IO.println "Testing dropDs..."

  let ds := testArray |> dropDs 3

  assert (Dataset.len ds == 7) "Dropped dataset should have 7 elements"

  if h : 0 < Dataset.len ds then
    let item ← Dataset.getItem ds 0 h
    assert (item == 3) "First item should be 3 (after dropping 3)"

  IO.println "✓ dropDs tests passed"

def testConcatDs : IO Unit := do
  IO.println "Testing concatDs..."

  let ds1 := ofArray #[1, 2, 3]
  let ds2 := ofArray #[4, 5, 6]
  let ds := concatDs ds1 ds2

  assert (Dataset.len ds == 6) "Concatenated dataset should have 6 elements"

  -- Check elements from both halves
  if h : 2 < Dataset.len ds then
    let item ← Dataset.getItem ds 2 h
    assert (item == 3) "Item at index 2 should be 3"

  if h : 3 < Dataset.len ds then
    let item ← Dataset.getItem ds 3 h
    assert (item == 4) "Item at index 3 should be 4"

  IO.println "✓ concatDs tests passed"

def testZipDs : IO Unit := do
  IO.println "Testing zipDs..."

  let ds1 := ofArray #["a", "b", "c"]
  let ds2 := ofArray #[1, 2, 3, 4, 5]
  let ds := zipDs ds1 ds2

  assert (Dataset.len ds == 3) "Zipped dataset should have min(3, 5) = 3 elements"

  if h : 1 < Dataset.len ds then
    let (s, n) ← Dataset.getItem ds 1 h
    assert (s == "b" && n == 2) "Zipped item at 1 should be (\"b\", 2)"

  IO.println "✓ zipDs tests passed"

/-! ## Shuffle Tests -/

def testShuffledIndex : IO Unit := do
  IO.println "Testing shuffledIndex..."

  let key := RandKey.new 42
  let n := 100

  -- Verify bijection property
  let isBijection := verifyBijection key n
  assert isBijection "shuffledIndex should be a bijection"

  -- Verify same key gives same permutation
  let idx1 := shuffledIndex key n 0
  let idx2 := shuffledIndex key n 0
  assert (idx1 == idx2) "Same key should give same result"

  -- Different keys should (likely) give different results
  let key2 := RandKey.new 123
  let idx3 := shuffledIndex key2 n 0
  assert (idx1 != idx3 || n <= 1) "Different keys should give different results"

  IO.println "✓ shuffledIndex tests passed"

def testShuffleDs : IO Unit := do
  IO.println "Testing shuffleDs..."

  let key := RandKey.new 42
  let ds := testArray |> shuffleDs key

  assert (Dataset.len ds == 10) "Shuffled dataset should have same length"

  -- Access some items (shuffled order)
  let mut seen := Array.replicate 10 false
  for i in [:10] do
    if h : i < Dataset.len ds then
      let item ← Dataset.getItem ds i h
      if item < 10 then
        seen := seen.set! item true

  assert (seen.all id) "All items should be accessible after shuffling"

  IO.println "✓ shuffleDs tests passed"

/-! ## Shard Tests -/

def testShardDs : IO Unit := do
  IO.println "Testing shardDs..."

  let ds := ofArray #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  -- Shard 0 of 2 (interleaved: 0, 2, 4, 6, 8)
  let shard0 := shardDs 0 2 .interleaved true ds
  assert (Dataset.len shard0 == 5) "Shard 0 should have 5 elements"

  -- Shard 1 of 2 (interleaved: 1, 3, 5, 7, 9)
  let shard1 := shardDs 1 2 .interleaved true ds
  assert (Dataset.len shard1 == 5) "Shard 1 should have 5 elements"

  -- Verify partition property
  let partition := verifyPartition 2 10 .interleaved true
  assert partition "Shards should partition the dataset"

  IO.println "✓ shardDs tests passed"

/-! ## RandKey Tests -/

def testRandKey : IO Unit := do
  IO.println "Testing RandKey..."

  let key := RandKey.new 42

  -- Test next
  let (val1, key1) := key.next
  let (val2, _key2) := key1.next
  assert (val1 != val2) "Consecutive values should differ"

  -- Test split
  let (keyA, keyB) := key.split
  let (valA, _) := keyA.next
  let (valB, _) := keyB.next
  assert (valA != valB) "Split keys should produce different values"

  -- Test fold
  let foldedKey := key.fold 100
  let (foldedVal, _) := foldedKey.next
  let (originalVal, _) := key.next
  assert (foldedVal != originalVal) "Folded key should differ from original"

  -- Test shuffleIndices
  let (indices, _) := key.shuffleIndices 10
  assert (indices.size == 10) "Should produce 10 indices"

  -- All indices should be present
  let sorted := indices.insertionSort (·<·)
  for i in [:10] do
    assert (sorted[i]! == i) s!"Index {i} should be present"

  IO.println "✓ RandKey tests passed"

/-! ## Checkpoint Tests -/

def testCheckpoint : IO Unit := do
  IO.println "Testing Checkpoint serialization..."

  let state : IteratorState := {
    position := 12345
    epoch := 7
    key := RandKey.new 999
  }

  -- Serialize and deserialize
  let bytes := serializeIteratorState state
  let restored := deserializeIteratorState bytes

  match restored with
  | some s =>
    assert (s.position == state.position) "Position should match"
    assert (s.epoch == state.epoch) "Epoch should match"
    assert (s.key.state == state.key.state) "Key state should match"
  | none =>
    throw (IO.userError "Deserialization failed")

  IO.println "✓ Checkpoint tests passed"

/-! ## Prefetcher Tests -/

def testPrefetcher : IO Unit := do
  IO.println "Testing Prefetcher..."

  let ds := testArray

  -- Create prefetcher
  let prefetcher ← Prefetcher.create ds 4

  -- Consume all items
  let mut count := 0
  let mut sum := 0
  repeat do
    match ← prefetcher.next with
    | some item =>
      count := count + 1
      sum := sum + item
    | none => break

  assert (count == 10) "Should fetch all 10 items"
  assert (sum == 45) "Sum should be 0+1+...+9 = 45"

  IO.println "✓ Prefetcher tests passed"

def testPrefetcherEarlyStop : IO Unit := do
  IO.println "Testing Prefetcher early stop..."

  let ds := ofArray (Array.range 1000)
  let prefetcher ← Prefetcher.create ds 1
  let mut count := 0

  for _ in prefetcher do
    count := count + 1
    if count == 5 then
      break

  assert (count == 5) "Should stop after 5 items"
  prefetcher.wait
  let done ← prefetcher.isDone
  assert done "Prefetcher should stop after early break"

  IO.println "✓ Prefetcher early stop passed"

def arrayTake (arr : Array T) (n : Nat) : Array T := Id.run do
  let stop := min n arr.size
  let mut out := Array.mkEmpty stop
  for i in [:stop] do
    if h : i < arr.size then
      out := out.push (arr[i]'(h))
  out

def arrayDrop (arr : Array T) (n : Nat) : Array T := Id.run do
  let start := min n arr.size
  let mut out := Array.mkEmpty (arr.size - start)
  for i in [start:arr.size] do
    if h : i < arr.size then
      out := out.push (arr[i]'(h))
  out

def testIteratorPrefetcherResume : IO Unit := do
  IO.println "Testing IteratorPrefetcher resume..."

  let ds := ofArray (Array.range 100)
  let key := RandKey.new 123
  let cfg : IteratorConfig (ArrayDataset Nat) := {
    base := ds,
    epochs := 1,
    key := key
  }

  let baselineIter ← Dataset.toIteratorCfg cfg
  let baseline ← DataIterator.toArray baselineIter

  let prefetcher ← TinyGrad4.Data.IteratorPrefetcher.createFromIteratorCfg cfg 4
  let mut seen := #[]
  let split := 7
  for _ in [:split] do
    match ← TinyGrad4.Data.IteratorPrefetcher.next prefetcher with
    | some x => seen := seen.push x
    | none => break

  let state : IteratorState ← TinyGrad4.Data.IteratorPrefetcher.checkpoint prefetcher
  TinyGrad4.Data.IteratorPrefetcher.cancel prefetcher
  TinyGrad4.Data.IteratorPrefetcher.drain prefetcher

  let cfg' := {
    cfg with
    startPos := state.position
    startEpoch := state.epoch
    key := state.key
  }
  let iter2 ← Dataset.toIteratorCfg cfg'
  let remainder ← DataIterator.toArray iter2

  let expectedPrefix := arrayTake baseline seen.size
  let expectedRemainder := arrayDrop baseline seen.size

  assert (seen == expectedPrefix) "IteratorPrefetcher prefix mismatch"
  assert (remainder == expectedRemainder) "IteratorPrefetcher remainder mismatch"

  IO.println "✓ IteratorPrefetcher resume passed"

/-! ## Pipeline Composition Test -/

def testPipelineComposition : IO Unit := do
  IO.println "Testing pipeline composition..."

  let key := RandKey.new 42
  let ds := ofArray #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

  -- Complex pipeline: shuffle → shard → map → batch
  -- Note: shardDs needs explicit nesting due to optional args
  let shuffled := shuffleDs key ds
  let sharded := shardDs 0 2 .interleaved true shuffled
  let mapped := mapDs (· * 10) sharded
  let pipeline := batchDs 2 mapped

  let numBatches := Dataset.len pipeline
  IO.println s!"  Pipeline produces {numBatches} batches"

  -- Iterate through batches
  let mut totalItems := 0
  for i in [:numBatches] do
    if h : i < Dataset.len pipeline then
      let batch ← Dataset.getItem pipeline i h
      totalItems := totalItems + batch.size  -- batch is Array Nat

  IO.println s!"  Total items across batches: {totalItems}"

  IO.println "✓ Pipeline composition tests passed"

/-! ## Run All Tests -/

def runAll : IO Unit := do
  IO.println "=== Dataset Smoke Tests ==="

  IO.println "Running testRandKey..."
  testRandKey
  IO.println "Running testArrayDataset..."
  testArrayDataset
  IO.println "Running testMapDs..."
  testMapDs
  IO.println "Running testBatchDs..."
  testBatchDs
  IO.println "Running testTakeDs..."
  testTakeDs
  IO.println "Running testDropDs..."
  testDropDs
  IO.println "Running testConcatDs..."
  testConcatDs
  IO.println "Running testZipDs..."
  testZipDs
  IO.println "Running testShuffledIndex..."
  testShuffledIndex
  IO.println "Running testShuffleDs..."
  testShuffleDs
  IO.println "Running testShardDs..."
  testShardDs
  IO.println "Running testCheckpoint..."
  testCheckpoint
  IO.println "Running testPrefetcher..."
  testPrefetcher
  IO.println "Running testPrefetcherEarlyStop..."
  testPrefetcherEarlyStop
  IO.println "Running testIteratorPrefetcherResume..."
  testIteratorPrefetcherResume
  IO.println "Running testPipelineComposition..."
  testPipelineComposition

  IO.println "=== All Dataset Smoke Tests Passed ==="

end TinyGrad4.Test.DatasetSmoke

def main : IO Unit := TinyGrad4.Test.DatasetSmoke.runAll
