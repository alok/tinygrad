import TinyGrad4.Data.Dataset
import TinyGrad4.Data.Transform
import TinyGrad4.Data.Shuffle
import TinyGrad4.Data.Shard
import TinyGrad4.Data.Profile

/-!
# IterDataset - Configured, resumable iteration

Grain-style "generalized iterable" wrapper over `IteratorConfig` that keeps
epoch/key/resume logic alongside dataset transformations.

Use this when you want to build a pipeline once and iterate it many times,
including resume from checkpoints.
-/

namespace TinyGrad4.Data

/-- Configured dataset iterator pipeline. -/
structure IterDataset (D : Type) (T : Type) where
  cfg : IteratorConfig D

namespace IterDataset

/-- Wrap an iterator config. -/
def fromCfg (cfg : IteratorConfig D) : IterDataset D T :=
  { cfg }

/-- Build an iterator dataset from a dataset with a key/epoch count. -/
def ofDataset [Dataset D T] (ds : D) (key : RandKey := RandKey.new 0) (epochs : Nat := 1) :
    IterDataset D T :=
  { cfg := { base := ds, key := key, epochs := epochs } }

/-- Override epoch count. -/
def withEpochs (ds : IterDataset D T) (epochs : Nat) : IterDataset D T :=
  { cfg := { ds.cfg with epochs := epochs } }

/-- Override resume state. -/
def withStart (ds : IterDataset D T) (startPos : Nat) (startEpoch : Nat) (key : RandKey) :
    IterDataset D T :=
  { cfg := { ds.cfg with startPos, startEpoch, key } }

/-- Create a fresh iterator. -/
def toIterator [Dataset D T] (ds : IterDataset D T) : IO (DataIterator T) :=
  Dataset.toIteratorCfg ds.cfg

/-- Create an iterator starting from a saved state. -/
def toIteratorFrom [Dataset D T] (ds : IterDataset D T) (state : IteratorState) : IO (DataIterator T) := do
  let cfg := { ds.cfg with startPos := state.position, startEpoch := state.epoch, key := state.key }
  Dataset.toIteratorCfg cfg

/-- Create a stateful prefetcher from this dataset. -/
def toPrefetcher [Dataset D T] (ds : IterDataset D T) (bufferSize : Nat := 8) :
    IO (IteratorPrefetcher T) :=
  IteratorPrefetcher.createFromIteratorCfg ds.cfg bufferSize

/-- Create a multi-worker prefetcher from this dataset. -/
def toMultiPrefetcher [Dataset D T] (ds : IterDataset D T) (numWorkers : Nat) (bufferSize : Nat := 8)
    (mode : ShardMode := .interleaved) (dropRemainder : Bool := true) :
    IO (MultiIteratorPrefetcher T) :=
  MultiIteratorPrefetcher.createFromIteratorCfg ds.cfg numWorkers bufferSize mode dropRemainder

/-- Create a multi-worker prefetcher from this dataset and explicit state. -/
def toMultiPrefetcherFrom [Dataset D T] (ds : IterDataset D T) (state : MultiIteratorState) (numWorkers : Nat)
    (bufferSize : Nat := 8) (mode : ShardMode := .interleaved) (dropRemainder : Bool := true) :
    IO (MultiIteratorPrefetcher T) :=
  MultiIteratorPrefetcher.createFromIteratorCfgState ds.cfg numWorkers state bufferSize mode dropRemainder

/-- Create a stateful batch prefetcher from this dataset. -/
def toBatchPrefetcher [Dataset D T] (ds : IterDataset D T) (batchSize : Nat)
    (collate : Array T → IO B) (dropLast : Bool := true) (bufferSize : Nat := 8) :
    IO (BatchPrefetcher B) :=
  BatchPrefetcher.createFromIteratorCfg ds.cfg batchSize collate dropLast bufferSize

private def mapCfg (cfg : IteratorConfig D) (f : D → D2) : IteratorConfig D2 :=
  {
    cfg with
    base := f cfg.base
    datasetAtEpoch := fun _ k e => f (cfg.datasetAtEpoch cfg.base k e)
  }

/-- Map over elements (pure). -/
def map [Dataset D A] (f : A → B) (ds : IterDataset D A) : IterDataset (MappedDataset D A B) B :=
  { cfg := mapCfg ds.cfg (mapDs f) }

/-- Map over elements (IO). -/
def mapIO [Dataset D A] (f : A → IO B) (ds : IterDataset D A) :
    IterDataset (MappedIODataset D A B) B :=
  { cfg := mapCfg ds.cfg (mapIODs f) }

/-- Batch elements. -/
def batch [Dataset D T] (batchSize : Nat) (dropLast : Bool := true) (ds : IterDataset D T) :
    IterDataset (BatchedDataset D T) (Array T) :=
  { cfg := mapCfg ds.cfg (fun d => batchDs batchSize d dropLast) }

/-- Take first n elements. -/
def take [Dataset D T] (n : Nat) (ds : IterDataset D T) :
    IterDataset (TakenDataset D T) T :=
  { cfg := mapCfg ds.cfg (takeDs n) }

/-- Drop first n elements. -/
def drop [Dataset D T] (n : Nat) (ds : IterDataset D T) :
    IterDataset (DroppedDataset D T) T :=
  { cfg := mapCfg ds.cfg (dropDs n) }

/-- Filter elements (precomputes indices). -/
def filter [Dataset D T] [Inhabited T] (pred : T → Bool) (ds : IterDataset D T) :
    IO (IterDataset (FilteredDataset D T) T) := do
  let filtered ← filterDs pred ds.cfg.base
  let cfg := { ds.cfg with base := filtered, datasetAtEpoch := fun _ _ _ => filtered }
  pure { cfg := cfg }

/-- Concatenate two datasets. -/
def concat [Dataset D1 T] [Dataset D2 T] (ds1 : IterDataset D1 T) (ds2 : IterDataset D2 T) :
    IterDataset (ConcatDataset D1 D2 T) T :=
  {
    cfg := {
      ds1.cfg with
      base := concatDs ds1.cfg.base ds2.cfg.base
      datasetAtEpoch := fun _ k epoch =>
        concatDs (ds1.cfg.datasetAtEpoch ds1.cfg.base k epoch)
                 (ds2.cfg.datasetAtEpoch ds2.cfg.base k epoch)
    }
  }

/-- Zip two datasets. -/
def zip [Dataset D1 T1] [Dataset D2 T2] (ds1 : IterDataset D1 T1) (ds2 : IterDataset D2 T2) :
    IterDataset (ZippedDataset D1 D2 T1 T2) (T1 × T2) :=
  {
    cfg := {
      ds1.cfg with
      base := zipDs ds1.cfg.base ds2.cfg.base
      datasetAtEpoch := fun _ k epoch =>
        zipDs (ds1.cfg.datasetAtEpoch ds1.cfg.base k epoch)
              (ds2.cfg.datasetAtEpoch ds2.cfg.base k epoch)
    }
  }

/-- Enumerate elements. -/
def enumerate [Dataset D T] (ds : IterDataset D T) :
    IterDataset (EnumeratedDataset D T) (Nat × T) :=
  { cfg := mapCfg ds.cfg enumerateDs }

/-- Apply cached shuffle per epoch (deterministic). -/
def shuffleCached [Dataset D T] (key : RandKey) (ds : IterDataset D T) :
    IterDataset (CachedShuffledDataset D T) T :=
  {
    cfg := {
      ds.cfg with
      key := key
      base := shuffleDsCachedAtEpoch key 0 ds.cfg.base
      datasetAtEpoch := fun d k epoch => shuffleDsCachedAtEpoch k epoch d.inner
    }
  }

/-- Apply on-the-fly shuffle per epoch. -/
def shuffle [Dataset D T] (key : RandKey) (ds : IterDataset D T) :
    IterDataset (ShuffledDataset D T) T :=
  {
    cfg := {
      ds.cfg with
      key := key
      base := shuffleDsAtEpoch key 0 ds.cfg.base
      datasetAtEpoch := fun d k epoch => shuffleDsAtEpoch k epoch d.inner
    }
  }

/-- Shard dataset for a specific worker. -/
def shard [Dataset D T] (config : ShardConfig) (ds : IterDataset D T) :
    IterDataset (ShardedDataset D T) T :=
  { cfg := mapCfg ds.cfg (fun d => shardWithConfig config d) }

/-- Profile `getItem` timing for this dataset. -/
def profile [Dataset D T] (profiler : Profiler) (name : String) (ds : IterDataset D T) :
    IterDataset (ProfiledDataset D T) T :=
  { cfg := mapCfg ds.cfg (profileDs profiler name) }

end IterDataset

instance [Dataset D T] : ForIn IO (IterDataset D T) T where
  forIn ds init f := do
    let it ← IterDataset.toIterator ds
    forIn it init f

end TinyGrad4.Data
