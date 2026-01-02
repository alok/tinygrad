-- Core types
import TinyGrad4.Data.Dataset
import TinyGrad4.Data.ArrayN

-- Buffer protocol (device-agnostic, zero-copy exchange)
import TinyGrad4.Data.Device
import TinyGrad4.Data.Buffer
import TinyGrad4.Data.IndexTransform

-- Transformations
import TinyGrad4.Data.Transform
import TinyGrad4.Data.Shuffle
import TinyGrad4.Data.Prefetch
import TinyGrad4.Data.Shard
import TinyGrad4.Data.Checkpoint
import TinyGrad4.Data.Profile
import TinyGrad4.Data.Timing

-- Datasets
import TinyGrad4.Data.MNIST
import TinyGrad4.Data.MNISTDataset
import TinyGrad4.Data.MNISTRaw
import TinyGrad4.Data.GPULoader
import TinyGrad4.Data.TPULoader
import TinyGrad4.Data.Iterators

/-!
# TinyGrad4 Data Loading Library

High-performance, composable data loading for ML training.
Designed to be used by downstream libraries like SciLean.

## Core Types
- `Dataset`: Random-access dataset typeclass
- `RandKey`: JAX-style splittable PRNG for reproducible shuffling
- `DataIterator`: Checkpointable sequential iteration
- `Prefetcher`: Background data loading with IO.asTask
- `Profiler`: Lightweight timing stats for pipeline stages
- `TPUDataLoader`: Host-staged loader tagged with TPU device IDs

## Buffer Protocol (New)
Device-agnostic buffer exchange inspired by DLPack:
- `Device`: Memory location (cpu, gpu, nvme, remote)
- `Allocation` / `TrackedAllocation`: Owned allocations with ref-counting
- `Buffer`: Typed zero-copy view (shape/dtype at type level)
- `TrackedBuffer`: Typed view with ownership tracking
- `RawBuffer`: FFI descriptor (runtime shape/dtype)
- `BufferExchange`: Typeclass for RawBuffer conversion
- `IndexTransform`: Pure index transforms with composition

## Transformations
All transformations are O(1) to construct and compose lazily:
- `mapDs`: Apply function to elements
- `batchDs`: Group into batches
- `takeDs` / `dropDs`: Slice datasets
- `filterDs`: Filter by predicate (O(n) construction)
- `concatDs` / `zipDs`: Combine datasets
- `enumerateDs`: Add indices
- `shuffleDs`: Lazy shuffling (uses O(n) permutation per epoch)
- `shuffleDsCached`: Pre-computed shuffle (O(n) setup, O(1) access)
- `prefetchDs`: Background prefetching

## Performance Notes
- Use `shuffleDsCached` for performance-critical training loops
- `shuffleDs` is convenient but rebuilds permutation each epoch
- `filterDs` eagerly scans dataset on construction
- All other transformations are lazy wrappers
- `Profiler` helps identify stages dominated by wait time

## Usage
```lean
import TinyGrad4.Data
open TinyGrad4.Data.Dataset TinyGrad4.Data.Transform TinyGrad4.Data.Shuffle

let mnist ← loadMNIST "data"
let key := RandKey.new 42
let trainDs := mnist.train
  |> shuffleDsCached key  -- O(n) once, O(1) per access
  |> batchDs 64

for batch in trainDs.toIterator do
  ...
```

## Namespace Structure
After `import TinyGrad4.Data`, use:
- `open TinyGrad4.Data.Dataset` for `Dataset`, `RandKey`, `ArrayDataset`, `ofArray`
- `open TinyGrad4.Data.Transform` for `mapDs`, `batchDs`, `takeDs`, `dropDs`, etc.
- `open TinyGrad4.Data.Shuffle` for `shuffleDs`, `shuffleDsCached`
- `open TinyGrad4.Data.Prefetch` for `Prefetcher`, `prefetchDs`
- `open TinyGrad4.Data.Shard` for `shardDs`, `ShardConfig`
- `open TinyGrad4.Data` for buffer protocol types
-/
