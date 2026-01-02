import Lean.Data.Json
import LeanBench
import TinyGrad4.Data.MNISTDataset
import TinyGrad4.Data.Profile
import TinyGrad4.Data.Prefetch
import TinyGrad4.Data.Shard
import TinyGrad4.Data.Shuffle
import TinyGrad4.Data.Transform

namespace TinyGrad4Bench

open LeanBench
open TinyGrad4.Data

structure BenchParams where
  dataDir : String := "../data"
  maxImages : Nat := 10000
  batchSize : Nat := 64
  prefetchBuffer : Nat := 8
  warmup : Nat := 1
  samples : Nat := 3
  seed : Nat := 42
  useCachedShuffle : Bool := true
  deriving Inhabited

structure ProfileSnapshot where
  stages : Array (String × StageStat)
  concurrency : ConcurrencyStats
  stageConcurrency : Array (String × ConcurrencyStats)
  deriving Inhabited

private def getEnvNat (key : String) (default : Nat) : IO Nat := do
  match (← IO.getEnv key) with
  | some v =>
    match v.toNat? with
    | some n => pure n
    | none => pure default
  | none => pure default

private def getEnvString (key : String) (default : String) : IO String := do
  match (← IO.getEnv key) with
  | some v => pure v
  | none => pure default

private def getEnvBool (key : String) (default : Bool) : IO Bool := do
  match (← IO.getEnv key) with
  | some v =>
    let v' := v.trimAscii.toString.toLower
    pure (v' == "1" || v' == "true" || v' == "yes")
  | none => pure default

private def readParams : IO BenchParams := do
  let dataDir := ← getEnvString "TG4_DATA_DIR" "../data"
  let maxImages := ← getEnvNat "TG4_MAX_IMAGES" 10000
  let batchSize := ← getEnvNat "TG4_BATCH" 64
  let prefetchBuffer := ← getEnvNat "TG4_PREFETCH_BUF" 8
  let warmup := ← getEnvNat "TG4_BENCH_WARMUP" 1
  let samples := ← getEnvNat "TG4_BENCH_SAMPLES" 3
  let seed := ← getEnvNat "TG4_BENCH_SEED" 42
  let useCachedShuffle := ← getEnvBool "TG4_BENCH_CACHED_SHUFFLE" true
  pure { dataDir, maxImages, batchSize, prefetchBuffer, warmup, samples, seed, useCachedShuffle }

initialize mnistCache : IO.Ref (Option MNISTDataset) <- IO.mkRef none

private def getMnist (cfg : BenchParams) : IO MNISTDataset := do
  match (← mnistCache.get) with
  | some ds => pure ds
  | none =>
      let ds ← MNISTDataset.loadTrain cfg.dataDir (some cfg.maxImages)
      mnistCache.set (some ds)
      pure ds

private def batchCount (cfg : BenchParams) : Nat :=
  if cfg.batchSize == 0 then 0 else cfg.maxImages / cfg.batchSize

private def baseConfig (cfg : BenchParams) (tags : List String := []) : BenchConfig :=
  { suite := some "data"
    tags := tags
    warmup := cfg.warmup
    samples := cfg.samples
    items := some (batchCount cfg) }

private def jsonNumOfFloat (x : Float) : Lean.Json :=
  match Lean.JsonNumber.fromFloat? x with
  | .inr n => Lean.Json.num n
  | .inl s => Lean.Json.str s

private def stageStatJson (name : String) (s : StageStat) : Lean.Json :=
  Lean.Json.mkObj [
    ("name", Lean.Json.str name),
    ("count", Lean.Json.num s.count),
    ("total_ns", Lean.Json.num s.totalNs),
    ("min_ns", Lean.Json.num s.minNs),
    ("max_ns", Lean.Json.num s.maxNs),
    ("wait_ns", Lean.Json.num s.waitNs),
    ("avg_ns", Lean.Json.num s.avgNs),
    ("wait_ratio", jsonNumOfFloat s.waitRatio)
  ]

private def concurrencyJson (s : ConcurrencyStats) : Lean.Json :=
  Lean.Json.mkObj [
    ("wall_ns", Lean.Json.num s.wallNs),
    ("busy_ns", Lean.Json.num s.busyNs),
    ("idle_ns", Lean.Json.num s.idleNs),
    ("avg_concurrency", jsonNumOfFloat s.avgConcurrency),
    ("peak_concurrency", Lean.Json.num s.peakConcurrency)
  ]

private def stageConcurrencyJson (entries : Array (String × ConcurrencyStats)) : Lean.Json :=
  Lean.Json.arr <|
    entries.map (fun (name, stat) =>
      Lean.Json.mkObj [
        ("name", Lean.Json.str name),
        ("concurrency", concurrencyJson stat)
      ])

private def extrasJson (cfg : BenchParams) (snap : ProfileSnapshot) : Lean.Json :=
  let stageArr := Lean.Json.arr (snap.stages.map (fun (name, stat) => stageStatJson name stat))
  let configObj := Lean.Json.mkObj [
    ("data_dir", Lean.Json.str cfg.dataDir),
    ("max_images", Lean.Json.num cfg.maxImages),
    ("batch_size", Lean.Json.num cfg.batchSize),
    ("prefetch_buffer", Lean.Json.num cfg.prefetchBuffer),
    ("cached_shuffle", Lean.Json.bool cfg.useCachedShuffle)
  ]
  Lean.Json.mkObj [
    ("config", configObj),
    ("concurrency", concurrencyJson snap.concurrency),
    ("stage_concurrency", stageConcurrencyJson snap.stageConcurrency),
    ("stages", stageArr)
  ]

private def mkShuffled (cfg : BenchParams) (key : RandKey)
    (base : ProfiledDataset MNISTDataset MNISTSample) :
    Sigma (fun D => (Dataset D MNISTSample) × D) :=
  if cfg.useCachedShuffle then
    ⟨CachedShuffledDataset (ProfiledDataset MNISTDataset MNISTSample) MNISTSample,
      (inferInstance, shuffleDsCached key base)⟩
  else
    ⟨ShuffledDataset (ProfiledDataset MNISTDataset MNISTSample) MNISTSample,
      (inferInstance, shuffleDs key base)⟩

private def runBaseline (cfg : BenchParams) : IO ProfileSnapshot := do
  let profiler ← Profiler.new
  let mnist ← getMnist cfg
  let key := RandKey.new cfg.seed
  let base := profileDs profiler "base" mnist
  let ⟨D, inst, ds⟩ := mkShuffled cfg key base
  let _ : Dataset D MNISTSample := inst
  let shuffled := profileDs profiler "shuffle" ds
  let sharded := profileDs profiler "shard" (shardDs 0 1 .interleaved true shuffled)
  let batched := profileDs profiler "batch" (batchDs cfg.batchSize sharded)
  let n := Dataset.len batched
  for i in [:n] do
    if h : i < n then
      let _ ← Dataset.getItem batched i h
  let (stages, concurrency, stageConcurrency) ← profiler.snapshotWithConcurrencyByStage
  pure { stages, concurrency, stageConcurrency }

private def runPrefetch (cfg : BenchParams) : IO ProfileSnapshot := do
  let profiler ← Profiler.new
  let mnist ← getMnist cfg
  let key := RandKey.new cfg.seed
  let base := profileDs profiler "base" mnist
  let ⟨D, inst, ds⟩ := mkShuffled cfg key base
  let _ : Dataset D MNISTSample := inst
  let shuffled := profileDs profiler "shuffle" ds
  let sharded := profileDs profiler "shard" (shardDs 0 1 .interleaved true shuffled)
  let batched := profileDs profiler "batch" (batchDs cfg.batchSize sharded)
  let prefetcher ← Prefetcher.create batched cfg.prefetchBuffer
  repeat do
    match ← profilePrefetcherNext profiler "prefetch_next" prefetcher with
    | some _ => pure ()
    | none => break
  let (stages, concurrency, stageConcurrency) ← profiler.snapshotWithConcurrencyByStage
  pure { stages, concurrency, stageConcurrency }

private def runPrefetchItems (cfg : BenchParams) : IO ProfileSnapshot := do
  let profiler ← Profiler.new
  let mnist ← getMnist cfg
  let key := RandKey.new cfg.seed
  let base := profileDs profiler "base" mnist
  let ⟨D, inst, ds⟩ := mkShuffled cfg key base
  let _ : Dataset D MNISTSample := inst
  let shuffled := profileDs profiler "shuffle" ds
  let sharded := profileDs profiler "shard" (shardDs 0 1 .interleaved true shuffled)
  let n := Dataset.len sharded
  let totalBatches := if cfg.batchSize == 0 then 0 else n / cfg.batchSize
  let prefetcher ← Prefetcher.create sharded cfg.prefetchBuffer
  let mut batches := 0
  while batches < totalBatches do
    let startBatch ← IO.monoNanosNow
    let mut taken := 0
    while taken < cfg.batchSize do
      match ← profilePrefetcherNext profiler "prefetch_item" prefetcher with
      | some _ =>
          taken := taken + 1
      | none =>
          taken := cfg.batchSize
    let stopBatch ← IO.monoNanosNow
    profiler.recordSample "batch" (stopBatch - startBatch)
    batches := batches + 1
  let (stages, concurrency, stageConcurrency) ← profiler.snapshotWithConcurrencyByStage
  pure { stages, concurrency, stageConcurrency }

initialize do
  let cfg ← readParams
  let baselineStats ← IO.mkRef (default : ProfileSnapshot)
  let prefetchStats ← IO.mkRef (default : ProfileSnapshot)
  let prefetchItemStats ← IO.mkRef (default : ProfileSnapshot)
  LeanBench.register {
    name := "dataloader/mnist/baseline"
    action := do
      let snap ← runBaseline cfg
      baselineStats.set snap
    report? := some do
      extrasJson cfg <$> baselineStats.get
    config := baseConfig cfg ["data", "mnist", "baseline"]
  }
  LeanBench.register {
    name := "dataloader/mnist/prefetch"
    action := do
      let snap ← runPrefetch cfg
      prefetchStats.set snap
    report? := some do
      extrasJson cfg <$> prefetchStats.get
    config := baseConfig cfg ["data", "mnist", "prefetch"]
  }
  LeanBench.register {
    name := "dataloader/mnist/prefetch-items"
    action := do
      let snap ← runPrefetchItems cfg
      prefetchItemStats.set snap
    report? := some do
      extrasJson cfg <$> prefetchItemStats.get
    config := baseConfig cfg ["data", "mnist", "prefetch", "items"]
  }

end TinyGrad4Bench
