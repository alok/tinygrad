import TinyGrad4.Data.Dataset
import TinyGrad4.Data.IterDataset
import TinyGrad4.Data.Prefetch
import TinyGrad4.Data.Shard

/-!
# Multi-Prefetch Best-Effort Smoke Test

Checks that best-effort ordering respects the max-lead bound across workers.
- Uses interleaved sharding (no shuffle) so we can infer worker id from item value.
- Adds artificial delay to one worker to force imbalance.
- Asserts per-step lead bound and final counts.
- Does not assume deterministic ordering.
-/

namespace TinyGrad4.Test.MultiPrefetchBestEffortSmoke

open TinyGrad4.Data

def assert (cond : Bool) (msg : String) : IO Unit := do
  if !cond then
    throw (IO.userError s!"Assertion failed: {msg}")

def expectedCounts (n workers : Nat) (dropRemainder : Bool := false) : Array Nat := Id.run do
  let mut out := Array.mkEmpty workers
  for i in [:workers] do
    let scfg : ShardConfig := { shardIndex := i, numShards := workers, mode := .interleaved, dropRemainder }
    out := out.push (scfg.shardSize n)
  out

def minActive (counts expected : Array Nat) : Nat := Id.run do
  let mut minVal : Option Nat := none
  for i in [:counts.size] do
    if counts[i]! < expected.getD i 0 then
      let v := counts[i]!
      minVal := match minVal with
        | none => some v
        | some m => some (Nat.min m v)
  minVal.getD 0

def checkMaxLead (counts expected : Array Nat) (maxLead : Nat) : Bool := Id.run do
  let minVal := minActive counts expected
  let mut ok := true
  for i in [:counts.size] do
    if counts[i]! < expected.getD i 0 then
      if counts[i]! > minVal + maxLead then
        ok := false
  ok

def runCase (n workers : Nat) (maxLead : Nat) (maxSkips : Nat) (delayMs : Nat) : IO Unit := do
  if workers == 0 then
    throw (IO.userError "workers must be > 0")
  let base := ofArray (Array.range n)
  let iterDs := IterDataset.ofDataset base (RandKey.new 0) 1
  let delayed := iterDs.mapIO (fun x => do
    if x % workers == 0 then
      IO.sleep (UInt32.ofNat delayMs)
    pure x)
  let policy := MultiIteratorPrefetcher.OrderingPolicy.bestEffortCfg maxSkips maxLead
  let prefetcher ← delayed.toMultiPrefetcher workers 4 .interleaved false policy
  let expected := expectedCounts n workers false
  let mut counts := Array.replicate workers 0
  repeat do
    match ← prefetcher.next with
    | none => break
    | some x =>
        let idx := x % workers
        counts := counts.set! idx (counts[idx]! + 1)
        let ok := checkMaxLead counts expected maxLead
        assert ok s!"lead bound violated (maxLead={maxLead}) counts={counts}"
  prefetcher.cancel
  assert (counts == expected) s!"counts mismatch: expected={expected}, got={counts}"

def runAll : IO Unit := do
  IO.println "Multi-prefetch best-effort smoke test..."
  runCase 101 4 2 8 2
  runCase 257 5 3 10 1
  IO.println "✓ Multi-prefetch best-effort smoke test passed"

end TinyGrad4.Test.MultiPrefetchBestEffortSmoke

def main : IO Unit := TinyGrad4.Test.MultiPrefetchBestEffortSmoke.runAll
