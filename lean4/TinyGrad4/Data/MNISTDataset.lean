import TinyGrad4.Data.Dataset
import TinyGrad4.Data.Transform
import TinyGrad4.Data.Shuffle
import TinyGrad4.Data.Shard
import TinyGrad4.Data.MNIST

-- Disable IO.monoNanosNow linter: dataset benchmarks use raw monotonic timing.
set_option linter.monoNanosNow false

/-!
# MNIST Dataset - grain-style Wrapper

Wraps MNIST data as a Dataset for use with the new data loading pipeline.

## Usage
```lean
let mnist ← MNISTDataset.loadTrain "data"
let trainDs := mnist
  |> shuffleDs (RandKey.new 42)
  |> batchDs 64
  |> takeDs 1000

for batch in trainDs.toArray do
  process batch
```
-/

namespace TinyGrad4.Data

set_option linter.useRawBuffer false

/-! ## MNIST as Dataset -/

/-- A single MNIST sample (image pixels + label) -/
structure MNISTSample where
  /-- Flattened 28x28 = 784 pixel values normalized to [0, 1] -/
  pixels : Array Float
  /-- Label 0-9 -/
  label : Nat
  deriving Repr, Inhabited

/-- MNIST dataset wrapping parsed ImageData and LabelData -/
structure MNISTDataset where
  images : MNIST.ImageData
  labels : MNIST.LabelData

/-- Load training dataset -/
def MNISTDataset.loadTrain (dataDir : String := "data") (maxImages? : Option Nat := none) : IO MNISTDataset := do
  let (images, labels) ← MNIST.loadTrain dataDir maxImages?
  pure { images, labels }

/-- Load test dataset -/
def MNISTDataset.loadTest (dataDir : String := "data") (maxImages? : Option Nat := none) : IO MNISTDataset := do
  let (images, labels) ← MNIST.loadTest dataDir maxImages?
  pure { images, labels }

instance : Dataset MNISTDataset MNISTSample where
  len ds := ds.images.numImages
  getItem ds idx _ := do
    let pixelsPerImage := ds.images.rows * ds.images.cols
    let startPixel := idx * pixelsPerImage
    let endPixel := startPixel + pixelsPerImage
    let pixels := ds.images.pixels.extract startPixel endPixel
    let label := ds.labels.labels[idx]!.toUInt64.toNat
    pure { pixels, label }

/-! ## Batched MNIST for Training -/

/-- A batch of MNIST samples -/
structure MNISTBatch where
  /-- Stacked pixels [batchSize, 784] as flat array -/
  pixels : Array Float
  /-- Labels as Float for compatibility -/
  labels : Array Float
  /-- One-hot encoded labels [batchSize, 10] -/
  oneHotLabels : Array Float
  /-- Actual batch size (may be smaller for last batch) -/
  size : Nat
  deriving Repr, Inhabited

/-- Convert Array MNISTSample to MNISTBatch -/
def toMNISTBatch (samples : Array MNISTSample) : MNISTBatch :=
  let pixels := samples.flatMap (·.pixels)
  let labels := samples.map (·.label.toFloat)
  let oneHotLabels := MNIST.toOneHot labels
  { pixels, labels, oneHotLabels, size := samples.size }

/-! ## Pipeline Composition Helpers -/

/-- Standard MNIST training pipeline:
    shuffle → shard → batch → (optionally) prefetch -/
def mnistTrainPipeline (ds : MNISTDataset) (key : RandKey) (batchSize : Nat := 64)
    (shardIndex : Nat := 0) (numShards : Nat := 1) :
    BatchedDataset (ShardedDataset (ShuffledDataset MNISTDataset MNISTSample) MNISTSample) MNISTSample :=
  batchDs batchSize (shardDs shardIndex numShards .interleaved true (shuffleDs key ds))

/-! ## Compatibility Bridge -/

/-- Convert new-style pipeline iteration to old-style batch access pattern.
    This enables gradual migration while maintaining API compatibility. -/
def iterateBatches (ds : MNISTDataset) (batchSize : Nat) (key : RandKey)
    (f : MNISTBatch → IO Unit) : IO Unit := do
  let pipeline := mnistTrainPipeline ds key batchSize
  let n := Dataset.len pipeline
  for i in [:n] do
    if h : i < n then
      let samples ← Dataset.getItem pipeline i h
      f (toMNISTBatch samples)

/-! ## Benchmark: New vs Old Loader -/

/-- Benchmark comparison between old DataLoader and new Dataset pipeline -/
def mnistBenchmarkComparison (dataDir : String := "data") (batchSize : Nat := 64)
    (maxImages : Nat := 10000) : IO Unit := do
  IO.println "=== MNIST Loader Benchmark ==="

  -- Load data once
  let mnist ← MNISTDataset.loadTrain dataDir (some maxImages)
  let key := RandKey.new 42

  -- New pipeline with CACHED shuffle: O(n) setup, O(1) per access
  let shuffled := shuffleDsCached key mnist
  let pipeline := batchDs batchSize shuffled
  let numBatches := Dataset.len pipeline

  -- Time new loader (with cached shuffle)
  let startNew ← IO.monoNanosNow
  let mut countNew := 0
  for i in [:numBatches] do
    if h : i < numBatches then
      let samples ← Dataset.getItem pipeline i h
      let _batch := toMNISTBatch samples
      countNew := countNew + 1
  let stopNew ← IO.monoNanosNow
  let newTimeMs := (stopNew - startNew).toFloat / 1e6

  IO.println s!"New Dataset pipeline (cached shuffle): {countNew} batches in {newTimeMs} ms"
  IO.println s!"  Throughput: {countNew.toFloat * 1000.0 / newTimeMs} batches/sec"

  -- Non-shuffled pipeline for fair comparison
  let pipelineNoShuffle := batchDs batchSize mnist
  let startNoShuffle ← IO.monoNanosNow
  let mut countNoShuffle := 0
  for i in [:numBatches] do
    if h : i < Dataset.len pipelineNoShuffle then
      let samples ← Dataset.getItem pipelineNoShuffle i h
      let _batch := toMNISTBatch samples
      countNoShuffle := countNoShuffle + 1
  let stopNoShuffle ← IO.monoNanosNow
  let noShuffleTimeMs := (stopNoShuffle - startNoShuffle).toFloat / 1e6

  IO.println s!"New Dataset pipeline (no shuffle): {countNoShuffle} batches in {noShuffleTimeMs} ms"
  IO.println s!"  Throughput: {countNoShuffle.toFloat * 1000.0 / noShuffleTimeMs} batches/sec"

  -- Old loader for comparison
  let (images, labels) ← MNIST.loadTrain dataDir (some maxImages)
  let oldLoader : MNIST.Loader batchSize := { images, labels }
  let oldNumBatches := oldLoader.numBatches

  let startOld ← IO.monoNanosNow
  let mut countOld := 0
  for i in [:oldNumBatches] do
    let _batch ← oldLoader.getBatch i
    countOld := countOld + 1
  let stopOld ← IO.monoNanosNow
  let oldTimeMs := (stopOld - startOld).toFloat / 1e6

  IO.println s!"Old DataLoader: {countOld} batches in {oldTimeMs} ms"
  IO.println s!"  Throughput: {countOld.toFloat * 1000.0 / oldTimeMs} batches/sec"

  IO.println ""
  IO.println "Summary:"
  let speedupCached := oldTimeMs / newTimeMs
  let speedupNoShuffle := oldTimeMs / noShuffleTimeMs
  IO.println s!"  New (cached shuffle) vs Old: {speedupCached}x"
  IO.println s!"  New (no shuffle) vs Old: {speedupNoShuffle}x"

end TinyGrad4.Data
