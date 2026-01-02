import TinyGrad4.Data.Dataset
import TinyGrad4.Data.Shuffle
import TinyGrad4.Data.IterDataset
import TinyGrad4.Data.Prefetch
import TinyGrad4.Data.Checkpoint

/-!
# Multi-Prefetch Resume Smoke Test

Checks deterministic resume for multi-worker prefetch with interleaved sharding.
- Interrupt mid-stream
- Save/load checkpoint
- Resume and validate output order matches baseline
- Exercises non-even shard sizes (dropRemainder=false)
-/

namespace TinyGrad4.Test.MultiPrefetchResumeSmoke

open TinyGrad4.Data

def assert (cond : Bool) (msg : String) : IO Unit := do
  if !cond then
    throw (IO.userError s!"Assertion failed: {msg}")

def arrayDrop (arr : Array T) (n : Nat) : Array T := Id.run do
  let start := min n arr.size
  let mut out := Array.mkEmpty (arr.size - start)
  for i in [start:arr.size] do
    if h : i < arr.size then
      out := out.push (arr[i]'(h))
  out

def arrayTake (arr : Array T) (n : Nat) : Array T := Id.run do
  let stop := min n arr.size
  let mut out := Array.mkEmpty stop
  for i in [:stop] do
    if h : i < arr.size then
      out := out.push (arr[i]'(h))
  out

def buildCfg (n epochs : Nat) (key : RandKey) :
    IteratorConfig (CachedShuffledDataset (ArrayDataset Nat) Nat) := by
  let baseRaw := ofArray (Array.range n)
  let base := shuffleDsCachedAtEpoch key 0 baseRaw
  exact {
    base := base,
    epochs := epochs,
    key := key,
    datasetAtEpoch := fun ds k epoch => shuffleDsCachedAtEpoch k epoch ds.inner
  }

def resumeAtSplit (cfg : IteratorConfig D) (baseline : Array T) (split : Nat) (workers : Nat)
    [Dataset D T] [BEq T] : IO Unit := do
  let prefetcher ← MultiIteratorPrefetcher.createFromIteratorCfg cfg workers 4 .interleaved false
  let mut seen := (#[] : Array T)
  for _ in [:split] do
    match ← prefetcher.next with
    | some x => seen := seen.push x
    | none => break

  let state ← prefetcher.checkpoint
  prefetcher.cancel

  let path : System.FilePath := "/tmp/tg4_multi_resume_checkpoint.bin"
  saveMultiCheckpoint path state
  let loaded ← loadMultiCheckpoint path
  IO.FS.removeFile path

  let prefetcher2 ← MultiIteratorPrefetcher.createFromIteratorCfgState cfg workers loaded 4 .interleaved false
  let remainder ← prefetcher2.toArray
  prefetcher2.cancel

  let expectedPrefix := arrayTake baseline seen.size
  let expectedRemainder := arrayDrop baseline seen.size
  assert (seen == expectedPrefix) s!"Prefix mismatch at split={split}"
  assert (remainder == expectedRemainder) s!"Remainder mismatch at split={split}"

def runAll : IO Unit := do
  IO.println "Multi-prefetch resume smoke test..."

  let key := RandKey.new 42
  let cfg := buildCfg 257 2 key
  let baselinePrefetcher ← MultiIteratorPrefetcher.createFromIteratorCfg cfg 3 4 .interleaved false
  let baseline ← baselinePrefetcher.toArray
  baselinePrefetcher.cancel
  assert (baseline.size == 257 * 2) "Baseline size mismatch"

  let splits := #[0, 1, 3, 17, 63, 128, 255, 256, 257, 300, baseline.size]
  for split in splits do
    resumeAtSplit cfg baseline split 3

  IO.println "✓ Multi-prefetch resume smoke test passed"

end TinyGrad4.Test.MultiPrefetchResumeSmoke

def main : IO Unit := TinyGrad4.Test.MultiPrefetchResumeSmoke.runAll
