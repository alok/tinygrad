import Std
import Std.Sync.Mutex
import TinyGrad4.Data.Dataset
import TinyGrad4.Data.Prefetch
import TinyGrad4.Data.Transform
import TinyGrad4.Data.Shuffle
import TinyGrad4.Data.GPULoader
import TinyGrad4.Data.TPULoader

/-!
# Profile - Lightweight pipeline profiling

Collects per-stage timing to spot bottlenecks in dataset pipelines.
Designed to mirror Grain's "wait time" signals without heavy dependencies.
-/

namespace TinyGrad4.Data

-- Disable IO.monoNanosNow linter: profiling needs raw monotonic timestamps.
set_option linter.monoNanosNow false

/-! ## StageStat -/

/-- Aggregate timing stats for a pipeline stage. -/
structure StageStat where
  count : Nat := 0
  totalNs : Nat := 0
  minNs : Nat := 0
  maxNs : Nat := 0
  waitNs : Nat := 0
  deriving Repr, Inhabited

namespace StageStat

/-- Record a new sample into stats. -/
def recordSample (s : StageStat) (totalNs : Nat) (waitNs : Nat := 0) : StageStat :=
  let count := s.count + 1
  let minNs := if s.count == 0 then totalNs else Nat.min s.minNs totalNs
  let maxNs := if s.count == 0 then totalNs else Nat.max s.maxNs totalNs
  {
    count
    totalNs := s.totalNs + totalNs
    minNs
    maxNs
    waitNs := s.waitNs + waitNs
  }

/-- Average time per element (ns). -/
def avgNs (s : StageStat) : Nat :=
  if s.count == 0 then 0 else s.totalNs / s.count

/-- Fraction of time spent waiting (0-1). -/
def waitRatio (s : StageStat) : Float :=
  if s.totalNs == 0 then 0.0 else s.waitNs.toFloat / s.totalNs.toFloat

/-- Pretty-ish formatting for nanoseconds. -/
def formatNs (ns : Nat) : String :=
  if ns < 1000 then s!"{ns}ns"
  else if ns < 1000 * 1000 then s!"{ns / 1000}us"
  else if ns < 1000 * 1000 * 1000 then s!"{ns / 1000 / 1000}ms"
  else s!"{ns / 1000 / 1000 / 1000}s"

/-- Render a one-line summary. -/
def summaryLine (name : String) (s : StageStat) : String :=
  let avg := formatNs s.avgNs
  let total := formatNs s.totalNs
  let wait := formatNs s.waitNs
  let ratio := s.waitRatio * 100.0
  s!"{name}: count={s.count} avg={avg} total={total} wait={wait} wait%={ratio}"

end StageStat

/-! ## StageSpan -/

/-- Timing span for concurrency analysis. -/
structure StageSpan where
  name : String
  startNs : Nat
  endNs : Nat
  deriving Repr, Inhabited

/-! ## ConcurrencyStats -/

/-- Aggregate concurrency statistics across recorded spans. -/
structure ConcurrencyStats where
  wallNs : Nat := 0
  busyNs : Nat := 0
  idleNs : Nat := 0
  avgConcurrency : Float := 0.0
  peakConcurrency : Nat := 0
  deriving Repr, Inhabited

namespace StageStat

/-- Render a one-line summary with concurrency stats. -/
def summaryLineWithConcurrency (name : String) (s : StageStat) (c : ConcurrencyStats) : String :=
  let base := summaryLine name s
  let wall := formatNs c.wallNs
  let busy := formatNs c.busyNs
  let idle := formatNs c.idleNs
  s!"{base} conc_avg={c.avgConcurrency} conc_peak={c.peakConcurrency} conc_wall={wall} conc_busy={busy} conc_idle={idle}"

end StageStat

/-! ## Profiler -/

/-- Collects stage timing statistics. -/
structure Profiler where
  stats : Std.Mutex (Std.HashMap String StageStat)
  spans : Std.Mutex (Array StageSpan)

namespace Profiler

/-- Insert into a list keeping order under `lt`. -/
private def insertBy (lt : α → α → Bool) (x : α) : List α → List α
  | [] => [x]
  | y :: ys => if lt x y then x :: y :: ys else y :: insertBy lt x ys

/-- Simple insertion sort for small lists. -/
private def sortBy (lt : α → α → Bool) (xs : List α) : List α :=
  xs.foldl (fun acc x => insertBy lt x acc) []

/-- Create a new profiler. -/
def new : IO Profiler := do
  let stats ← Std.Mutex.new ({} : Std.HashMap String StageStat)
  let spans ← Std.Mutex.new (#[] : Array StageSpan)
  pure { stats, spans }

/-- Reset all collected stats. -/
def reset (p : Profiler) : IO Unit := do
  p.stats.atomically do
    set ({} : Std.HashMap String StageStat)
  p.spans.atomically do
    set (#[] : Array StageSpan)

/-- Record a sample. -/
def recordSample (p : Profiler) (name : String) (totalNs : Nat) (waitNs : Nat := 0) : IO Unit := do
  p.stats.atomically do
    let m ← get
    let stat : StageStat :=
      match m.get? name with
      | some v => v
      | none => default
    set (m.insert name (StageStat.recordSample stat totalNs waitNs))

/-- Record a sample with span timing for concurrency analysis. -/
def recordSampleSpan (p : Profiler) (name : String) (startNs endNs : Nat) (waitNs : Nat := 0) : IO Unit := do
  let totalNs := endNs - startNs
  p.recordSample name totalNs waitNs
  p.spans.atomically do
    let spans ← get
    set (spans.push { name, startNs, endNs })

/-- Record timing for an IO action. -/
def record (p : Profiler) (name : String) (action : IO α) : IO α := do
  let start ← IO.monoNanosNow
  let result ← action
  let stop ← IO.monoNanosNow
  p.recordSampleSpan name start stop
  pure result

/-- Snapshot all stats as an array of (name, stat). -/
def snapshot (p : Profiler) : IO (Array (String × StageStat)) := do
  p.stats.atomically do
    let m ← get
    pure m.toList.toArray

/-- Snapshot all recorded spans. -/
def snapshotSpans (p : Profiler) : IO (Array StageSpan) := do
  p.spans.atomically do
    get

private structure SpanEvent where
  time : Nat
  delta : Int
  deriving Inhabited

private def eventLt (a b : SpanEvent) : Bool :=
  if a.time == b.time then a.delta > b.delta else a.time < b.time

private def concurrencyStatsFromSpans (spans : Array StageSpan) : ConcurrencyStats :=
  if spans.isEmpty then
    default
  else Id.run do
    let events := spans.foldl (fun acc s =>
      acc.push { time := s.startNs, delta := 1 } |>.push { time := s.endNs, delta := -1 }) #[]
    let sorted := events.qsort eventLt
    let wallStart := sorted[0]!.time
    let wallEnd := sorted[sorted.size - 1]!.time
    let mut prev := wallStart
    let mut active : Int := 0
    let mut busyWeighted : Nat := 0
    let mut idle : Nat := 0
    let mut peak : Nat := 0
    for e in sorted do
      let dt := e.time - prev
      if dt > 0 then
        if active <= 0 then
          idle := idle + dt
        else
          busyWeighted := busyWeighted + dt * active.toNat
          if active.toNat > peak then
            peak := active.toNat
      active := active + e.delta
      prev := e.time
    let wallNs := wallEnd - wallStart
    let avg :=
      if wallNs == 0 then 0.0 else busyWeighted.toFloat / wallNs.toFloat
    pure { wallNs, busyNs := busyWeighted, idleNs := idle, avgConcurrency := avg, peakConcurrency := peak }

private def concurrencyStatsByStageFromSpans (spans : Array StageSpan) : Array (String × ConcurrencyStats) :=
  Id.run do
    let mut grouped : Std.HashMap String (Array StageSpan) := {}
    for s in spans do
      let current :=
        match grouped.get? s.name with
        | some v => v
        | none => #[]
      grouped := grouped.insert s.name (current.push s)
    let entries := grouped.toList
    let sorted := sortBy (fun a b => a.1 < b.1) entries
    pure (sorted.map (fun (name, sps) => (name, concurrencyStatsFromSpans sps)) |>.toArray)

/-- Compute concurrency stats across all recorded spans. -/
def concurrencyStats (p : Profiler) : IO ConcurrencyStats := do
  let spans ← p.snapshotSpans
  pure (concurrencyStatsFromSpans spans)

/-- Compute concurrency stats per stage name. -/
def concurrencyStatsByStage (p : Profiler) : IO (Array (String × ConcurrencyStats)) := do
  let spans ← p.snapshotSpans
  pure (concurrencyStatsByStageFromSpans spans)

/-- Snapshot stats plus concurrency summary. -/
def snapshotWithConcurrency (p : Profiler) : IO (Array (String × StageStat) × ConcurrencyStats) := do
  let stats ← p.snapshot
  let concurrency ← p.concurrencyStats
  pure (stats, concurrency)

/-- Snapshot stats plus concurrency summary and per-stage concurrency. -/
def snapshotWithConcurrencyByStage (p : Profiler) :
    IO (Array (String × StageStat) × ConcurrencyStats × Array (String × ConcurrencyStats)) := do
  let stats ← p.snapshot
  let spans ← p.snapshotSpans
  let concurrency := concurrencyStatsFromSpans spans
  let byStage := concurrencyStatsByStageFromSpans spans
  pure (stats, concurrency, byStage)

/-- Render a summary string (unsorted). -/
def summary (p : Profiler) : IO String := do
  let entries ← p.snapshot
  let lines := entries.map (fun (name, stat) => StageStat.summaryLine name stat)
  pure <| String.intercalate "\n" lines.toList

def concurrencySummary (p : Profiler) : IO String := do
  let s ← p.concurrencyStats
  let wall := StageStat.formatNs s.wallNs
  let busy := StageStat.formatNs s.busyNs
  let idle := StageStat.formatNs s.idleNs
  pure s!"concurrency: wall={wall} busy={busy} idle={idle} avg={s.avgConcurrency} peak={s.peakConcurrency}"

/-- Render a summary sorted by wait ratio (descending). -/
def summaryByWaitRatio (p : Profiler) : IO String := do
  let entries ← p.snapshot
  let sorted := sortBy (fun a b => a.2.waitRatio > b.2.waitRatio) entries.toList
  let lines := sorted.map (fun (name, stat) => StageStat.summaryLine name stat)
  pure <| String.intercalate "\n" lines

/-- Render a summary sorted by total time (descending). -/
def summaryByTotal (p : Profiler) : IO String := do
  let entries ← p.snapshot
  let sorted := sortBy (fun a b => a.2.totalNs > b.2.totalNs) entries.toList
  let lines := sorted.map (fun (name, stat) => StageStat.summaryLine name stat)
  pure <| String.intercalate "\n" lines

/-- Render a summary with per-stage concurrency stats. -/
def summaryWithConcurrencyByStage (p : Profiler) : IO String := do
  let (stats, _total, byStage) ← p.snapshotWithConcurrencyByStage
  let mut statMap : Std.HashMap String StageStat := {}
  for (name, stat) in stats do
    statMap := statMap.insert name stat
  let mut lines : Array String := #[]
  lines := lines.push (← p.concurrencySummary)
  for (name, conc) in byStage do
    let stat := statMap.getD name default
    lines := lines.push (StageStat.summaryLineWithConcurrency name stat conc)
  pure <| String.intercalate "\n" lines.toList

end Profiler

/-! ## Profiling Wrappers -/

/-- Dataset wrapper that records `getItem` timings. -/
structure ProfiledDataset (D : Type) (T : Type) where
  inner : D
  profiler : Profiler
  name : String

instance [Dataset D T] : Dataset (ProfiledDataset D T) T where
  len ds := Dataset.len ds.inner
  getItem ds idx h := do
    ds.profiler.record ds.name (Dataset.getItem ds.inner idx h)

/-- Wrap a dataset with profiling. -/
def profileDs [Dataset D T] (profiler : Profiler) (name : String) (ds : D) : ProfiledDataset D T :=
  { inner := ds, profiler, name }

/-- Wrap an iterator to profile `next`. -/
def profileIter (profiler : Profiler) (name : String) (iter : DataIterator T) : DataIterator T :=
  DataIterator.wrap iter (fun next => profiler.record name next)

/-- Profile a prefetcher's `next` with wait time attribution. -/
def profilePrefetcherNext (profiler : Profiler) (name : String) (p : Prefetcher T) : IO (Option T) := do
  let start ← IO.monoNanosNow
  let (result, waitNs) ← p.nextWithWait
  let stop ← IO.monoNanosNow
  profiler.recordSampleSpan name start stop waitNs
  pure result

/-- Profile an IteratorPrefetcher `next` with wait time attribution. -/
def profileIteratorPrefetcherNext (profiler : Profiler) (name : String)
    (p : IteratorPrefetcher T) : IO (Option T) := do
  let start ← IO.monoNanosNow
  let (result, waitNs) ← p.nextWithWait
  let stop ← IO.monoNanosNow
  profiler.recordSampleSpan name start stop waitNs
  pure result

/-- Profile a BatchPrefetcher `next` with wait time attribution. -/
def profileBatchPrefetcherNext (profiler : Profiler) (name : String)
    (p : BatchPrefetcher B) : IO (Option B) := do
  let start ← IO.monoNanosNow
  let (result, waitNs) ← p.nextWithWait
  let stop ← IO.monoNanosNow
  profiler.recordSampleSpan name start stop waitNs
  pure result

/-- Profile a GPU loader's `next` with wait time attribution. -/
def profileGPULoaderNext (profiler : Profiler) (name : String)
    (loader : GPULoader.GPUDataLoader) : IO (Option GPULoader.GPUBuffer) := do
  let start ← IO.monoNanosNow
  let (result, waitNs) ← loader.nextWithWait
  let stop ← IO.monoNanosNow
  profiler.recordSampleSpan name start stop waitNs
  pure result

/-- Profile a TPU loader's `next` with wait time attribution. -/
def profileTPULoaderNext (profiler : Profiler) (name : String)
    (loader : TPULoader.TPUDataLoader) : IO (Option TPULoader.TPUBuffer) := do
  let start ← IO.monoNanosNow
  let (result, waitNs) ← loader.nextWithWait
  let stop ← IO.monoNanosNow
  profiler.recordSampleSpan name start stop waitNs
  pure result

/-- Profile a MultiGPULoader `nextAll` with aggregated wait time attribution. -/
def profileMultiGPUNextAll (profiler : Profiler) (name : String)
    (pool : GPULoader.MultiGPULoader) :
    IO (Array (GPULoader.DeviceId × Option GPULoader.GPUBuffer)) := do
  let start ← IO.monoNanosNow
  let (result, waitNs) ← pool.nextAllWithWait
  let stop ← IO.monoNanosNow
  profiler.recordSampleSpan name start stop waitNs
  pure result

/-! ## Transform Helpers -/

def profileMapDs [Dataset D A] (profiler : Profiler) (name : String := "map")
    (f : A → B) (ds : D) : ProfiledDataset (MappedDataset D A B) B :=
  profileDs profiler name (mapDs f ds)

def profileMapIODs [Dataset D A] (profiler : Profiler) (name : String := "mapIO")
    (f : A → IO B) (ds : D) : ProfiledDataset (MappedIODataset D A B) B :=
  profileDs profiler name (mapIODs f ds)

def profileBatchDs [Dataset D T] (profiler : Profiler) (name : String := "batch")
    (batchSize : Nat) (ds : D) (dropLast : Bool := true) :
    ProfiledDataset (BatchedDataset D T) (Array T) :=
  profileDs profiler name (batchDs batchSize ds dropLast)

def profileTakeDs [Dataset D T] (profiler : Profiler) (name : String := "take")
    (n : Nat) (ds : D) : ProfiledDataset (TakenDataset D T) T :=
  profileDs profiler name (takeDs n ds)

def profileDropDs [Dataset D T] (profiler : Profiler) (name : String := "drop")
    (n : Nat) (ds : D) : ProfiledDataset (DroppedDataset D T) T :=
  profileDs profiler name (dropDs n ds)

def profileFilterDs [Dataset D T] [Inhabited T]
    (profiler : Profiler) (name : String := "filter")
    (pred : T → Bool) (ds : D) : IO (ProfiledDataset (FilteredDataset D T) T) := do
  let filtered ← filterDs pred ds
  pure (profileDs profiler name filtered)

def profileConcatDs [Dataset D1 T] [Dataset D2 T]
    (profiler : Profiler) (name : String := "concat")
    (ds1 : D1) (ds2 : D2) : ProfiledDataset (ConcatDataset D1 D2 T) T :=
  profileDs profiler name (concatDs ds1 ds2)

def profileZipDs [Dataset D1 T1] [Dataset D2 T2]
    (profiler : Profiler) (name : String := "zip")
    (ds1 : D1) (ds2 : D2) : ProfiledDataset (ZippedDataset D1 D2 T1 T2) (T1 × T2) :=
  profileDs profiler name (zipDs ds1 ds2)

def profileEnumerateDs [Dataset D T]
    (profiler : Profiler) (name : String := "enumerate") (ds : D) :
    ProfiledDataset (EnumeratedDataset D T) (Nat × T) :=
  profileDs profiler name (enumerateDs ds)

def profileShuffleDs [Dataset D T]
    (profiler : Profiler) (name : String := "shuffle") (key : RandKey) (ds : D) :
    ProfiledDataset (ShuffledDataset D T) T :=
  profileDs profiler name (shuffleDs key ds)

def profileShuffleDsCached [Dataset D T]
    (profiler : Profiler) (name : String := "shuffleCached") (key : RandKey) (ds : D) :
    ProfiledDataset (CachedShuffledDataset D T) T :=
  profileDs profiler name (shuffleDsCached key ds)

end TinyGrad4.Data
