import TinyGrad4.Data.Dataset
import TinyGrad4.Data.Shuffle
import TinyGrad4.Data.Checkpoint

/-!
# Checkpoint Resume Smoke Test

Stress test deterministic resume by interrupting iteration mid-epoch and
at epoch boundaries.
-/

namespace TinyGrad4.Test.CheckpointResumeSmoke

open TinyGrad4.Data

/-- Assert condition with message. -/
def assert (cond : Bool) (msg : String) : IO Unit := do
  if !cond then
    throw (IO.userError s!"Assertion failed: {msg}")

/-- Drop first n elements from an array. -/
def arrayDrop (arr : Array T) (n : Nat) : Array T := Id.run do
  let start := min n arr.size
  let mut out := Array.mkEmpty (arr.size - start)
  for i in [start:arr.size] do
    if h : i < arr.size then
      out := out.push (arr[i]'(h))
  out

/-- Take first n elements from an array. -/
def arrayTake (arr : Array T) (n : Nat) : Array T := Id.run do
  let stop := min n arr.size
  let mut out := Array.mkEmpty stop
  for i in [:stop] do
    if h : i < arr.size then
      out := out.push (arr[i]'(h))
  out

/-- Build iterator config for a cached shuffled dataset. -/
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

/-- Run a deterministic resume check at a split point. -/
def resumeAtSplit (cfg : IteratorConfig D) (baseline : Array T) (split : Nat)
    [Dataset D T] [BEq T] : IO Unit := do
  let iter ← Dataset.toIteratorCfg cfg
  let mut seen := (#[] : Array T)
  for _ in [:split] do
    match ← iter.next with
    | some x => seen := seen.push x
    | none => break

  let state ← iter.checkpoint

  -- File roundtrip to simulate interruption
  let path : System.FilePath := "/tmp/tg4_resume_checkpoint.bin"
  saveCheckpoint path state
  let loaded ← loadCheckpoint path
  IO.FS.removeFile path

  let iter2 ← Dataset.toIteratorCfg cfg
  iter2.restore loaded
  let remainder ← DataIterator.toArray iter2

  let expectedPrefix := arrayTake baseline seen.size
  let expectedRemainder := arrayDrop baseline seen.size
  assert (seen == expectedPrefix) s!"Prefix mismatch at split={split}"
  assert (remainder == expectedRemainder) s!"Remainder mismatch at split={split}"

/-- Run all checks. -/
def runAll : IO Unit := do
  IO.println "Checkpoint resume smoke test..."

  let key := RandKey.new 123
  let cfg := buildCfg 256 3 key
  let iter ← Dataset.toIteratorCfg cfg
  let baseline ← DataIterator.toArray iter
  assert (baseline.size == 256 * 3) "Baseline size mismatch"

  let splits := #[0, 1, 5, 127, 255, 256, 300, 511, baseline.size]
  for split in splits do
    resumeAtSplit cfg baseline split

  IO.println "✓ Checkpoint resume smoke test passed"

end TinyGrad4.Test.CheckpointResumeSmoke

def main : IO Unit := TinyGrad4.Test.CheckpointResumeSmoke.runAll
