import Lean.Data.Json
import LeanBench
import Std.Data.HashMap
import TinyGrad4.Data.MNISTDataset
import TinyGrad4.Data.MNISTRaw
import TinyGrad4.Data.Profile
import TinyGrad4.Data.Prefetch
import TinyGrad4.Data.Shard
import TinyGrad4.Data.Shuffle
import TinyGrad4.Data.Transform
import TinyGrad4.Backend.FusedGather
import TinyGrad4.Backend.Interpreter
import TinyGrad4.Tensor.Math
import TinyGrad4.UOp.Graph
import TinyGrad4.Benchmark.Instrumentation
-- Disable IO.monoNanosNow linter: benchmark timing uses raw monotonic clocks.
set_option linter.monoNanosNow false

namespace TinyGrad4Bench

open LeanBench
open TinyGrad4.Data
open TinyGrad4

structure BenchParams where
  dataDir : String := "../data"
  maxImages : Nat := 10000
  batchSize : Nat := 64
  indexSelectMaxImages : Nat := 256
  indexSelectBatchSize : Nat := 32
  indexSelectValidate : Bool := true
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

structure IndexSelectSnapshot where
  batches : Nat
  total : Float
  expectedTotal : Float
  mismatchCount : Nat
  maxAbsErr : Float
  outDType : String
  fusedGatherCount : Nat
  fusedGatherMatchCount : Nat
  validated : Bool
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
  let maxImages0 := ← getEnvNat "TG4_MAX_IMAGES" 10000
  let batchSize0 := ← getEnvNat "TG4_BATCH" 64
  let indexSelectMaxImages0 := ← getEnvNat "TG4_INDEXSELECT_MAX_IMAGES" 256
  let indexSelectBatchSize0 := ← getEnvNat "TG4_INDEXSELECT_BATCH" 32
  let indexSelectValidate := ← getEnvBool "TG4_INDEXSELECT_VALIDATE" true
  let prefetchBuffer0 := ← getEnvNat "TG4_PREFETCH_BUF" 8
  let warmup0 := ← getEnvNat "TG4_BENCH_WARMUP" 1
  let samples0 := ← getEnvNat "TG4_BENCH_SAMPLES" 3
  let seed := ← getEnvNat "TG4_BENCH_SEED" 42
  let useCachedShuffle := ← getEnvBool "TG4_BENCH_CACHED_SHUFFLE" true
  let quick := ← getEnvBool "TINYGRAD4_BENCH_QUICK" false
  let maxImages := if quick then min maxImages0 1024 else maxImages0
  let batchSize := if quick then min batchSize0 32 else batchSize0
  let indexSelectMaxImages := if quick then min indexSelectMaxImages0 64 else indexSelectMaxImages0
  let indexSelectBatchSize := if quick then min indexSelectBatchSize0 16 else indexSelectBatchSize0
  let prefetchBuffer := if quick then min prefetchBuffer0 2 else prefetchBuffer0
  let warmup := if quick then 0 else warmup0
  let samples := if quick then min samples0 1 else samples0
  pure {
    dataDir, maxImages, batchSize, indexSelectMaxImages, indexSelectBatchSize,
    indexSelectValidate, prefetchBuffer, warmup, samples, seed, useCachedShuffle
  }

initialize mnistCache : IO.Ref (Option MNISTDataset) <- IO.mkRef none
initialize mnistRawCache : IO.Ref (Option TinyGrad4.Data.MNISTRaw.MNISTRaw) <- IO.mkRef none

private def getMnist (cfg : BenchParams) : IO MNISTDataset := do
  match (← mnistCache.get) with
  | some ds => pure ds
  | none =>
      let ds ← MNISTDataset.loadTrain cfg.dataDir (some cfg.maxImages)
      mnistCache.set (some ds)
      pure ds

private def getMnistRaw (cfg : BenchParams) : IO TinyGrad4.Data.MNISTRaw.MNISTRaw := do
  match (← mnistRawCache.get) with
  | some ds => pure ds
  | none =>
      let ds ← TinyGrad4.Data.MNISTRaw.loadTrain cfg.dataDir (some cfg.indexSelectMaxImages)
      mnistRawCache.set (some ds)
      pure ds

private def batchCount (cfg : BenchParams) : Nat :=
  if cfg.batchSize == 0 then 0 else cfg.maxImages / cfg.batchSize

private def indexSelectBatchCount (cfg : BenchParams) : Nat :=
  if cfg.indexSelectBatchSize == 0 then 0 else cfg.indexSelectMaxImages / cfg.indexSelectBatchSize

private def baseConfig (cfg : BenchParams) (tags : List String := []) : BenchConfig :=
  { suite := some "data"
    tags := tags
    warmup := cfg.warmup
    samples := cfg.samples
    items := some (batchCount cfg) }

private def indexSelectConfig (cfg : BenchParams) (tags : List String := []) : BenchConfig :=
  { suite := some "data"
    tags := tags
    warmup := cfg.warmup
    samples := cfg.samples
    items := some (indexSelectBatchCount cfg) }

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

private def indexSelectJson (cfg : BenchParams) (snap : IndexSelectSnapshot) : Lean.Json :=
  let configObj := Lean.Json.mkObj [
    ("data_dir", Lean.Json.str cfg.dataDir),
    ("max_images", Lean.Json.num cfg.indexSelectMaxImages),
    ("batch_size", Lean.Json.num cfg.indexSelectBatchSize),
    ("validate", Lean.Json.bool cfg.indexSelectValidate)
  ]
  Lean.Json.mkObj [
    ("config", configObj),
    ("batches", Lean.Json.num snap.batches),
    ("total", jsonNumOfFloat snap.total),
    ("expected_total", jsonNumOfFloat snap.expectedTotal),
    ("mismatch_count", Lean.Json.num snap.mismatchCount),
    ("max_abs_err", jsonNumOfFloat snap.maxAbsErr),
    ("out_dtype", Lean.Json.str snap.outDType),
    ("fused_gather_count", Lean.Json.num snap.fusedGatherCount),
    ("fused_gather_match_count", Lean.Json.num snap.fusedGatherMatchCount),
    ("validated", Lean.Json.bool snap.validated)
  ]

private def bytesFromUInt32 (v : UInt32) : Array UInt8 :=
  let b0 : UInt8 := UInt8.ofNat (UInt32.toNat (v &&& 0xFF))
  let b1 : UInt8 := UInt8.ofNat (UInt32.toNat ((v >>> 8) &&& 0xFF))
  let b2 : UInt8 := UInt8.ofNat (UInt32.toNat ((v >>> 16) &&& 0xFF))
  let b3 : UInt8 := UInt8.ofNat (UInt32.toNat ((v >>> 24) &&& 0xFF))
  #[b0, b1, b2, b3]

private def pushBytes (out : ByteArray) (bytes : Array UInt8) : ByteArray := Id.run do
  let mut acc := out
  for b in bytes do
    acc := acc.push b
  return acc

private def packI32 (vals : Array Nat) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity (vals.size * 4)
  for v in vals do
    out := pushBytes out (bytesFromUInt32 (UInt32.ofNat v))
  return out

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
  prefetcher.cancel
  let (stages, concurrency, stageConcurrency) ← profiler.snapshotWithConcurrencyByStage
  pure { stages, concurrency, stageConcurrency }

private def runIndexSelectSmall (cfg : BenchParams) : IO IndexSelectSnapshot := do
  let batchSize := cfg.indexSelectBatchSize
  let mnist ← getMnistRaw cfg
  let batches := indexSelectBatchCount cfg
  if batchSize == 0 || batches == 0 then
    return {
      batches := 0, total := 0.0, expectedTotal := 0.0, mismatchCount := 0,
      maxAbsErr := 0.0, outDType := "none", fusedGatherCount := 0, fusedGatherMatchCount := 0,
      validated := cfg.indexSelectValidate
    }
  let (sumT, imgBuf, idxBuf) := runTensorM do
    let img ← Tensor.buffer [mnist.images.numImages, 784] .uint8
    let idx ← Tensor.buffer [batchSize] .int32
    let gathered ← StaticTensor.indexSelect img 0 idx
    let gatheredF ← StaticTensor.cast gathered .float32
    let total ← StaticTensor.sum gatheredF
    pure (total, img.uop, idx.uop)
  let compiled ← TinyGrad4.Interpreter.compileManyCached [sumT.uop]
  let fusedGatherCount := compiled.implMap.fold (init := 0) fun acc _ impl =>
    match impl with
    | TinyGrad4.Backend.Fusion.Impl.fusedGather _ => acc + 1
    | _ => acc
  let mut fusedGatherMatchCount := 0
  let keep : UOpIdSet := UOpIdSet.mkEmpty
  let refCnt : Std.HashMap UOpId Nat := ∅
  for u in UOp.toposort sumT.uop do
    if u.op == .REDUCE_AXIS then
      match TinyGrad4.Backend.FusedGather.compile u keep refCnt with
      | some _ => fusedGatherMatchCount := fusedGatherMatchCount + 1
      | none => pure ()
  let imgRaw : RawBuffer := { dtype := .uint8, data := mnist.images.data.toByteArray }
  let baseEnv := TinyGrad4.Interpreter.setBuffer (∅ : Env) imgBuf imgRaw
  let key := RandKey.new cfg.seed
  let (indices, _) := key.shuffleIndices mnist.images.numImages
  let mut total : Float := 0.0
  let mut expectedTotal : Float := 0.0
  let mut mismatchCount : Nat := 0
  let mut maxAbsErr : Float := 0.0
  let mut outDType := "unknown"
  for i in [:batches] do
    let startIdx := i * batchSize
    let endIdx := startIdx + batchSize
    let idxSlice := indices.extract startIdx endIdx
    let idxBytes := packI32 idxSlice
    let idxRaw : RawBuffer := { dtype := .int32, data := idxBytes }
    let env := TinyGrad4.Interpreter.setBuffer baseEnv idxBuf idxRaw
    let out ← TinyGrad4.Interpreter.evalTensorCached sumT env
    if i == 0 then
      outDType := toString (repr out.dtype)
    let v := RawBuffer.decodeScalarF32 out
    total := total + v
    if cfg.indexSelectValidate then
      let mut expectedBatch : Float32 := 0.0
      for j in [:batchSize] do
        let idx := idxSlice[j]!
        for p in [:TinyGrad4.Data.MNISTRaw.ImageBuffer.pixelsPerImage] do
          let px := mnist.images.getPixel idx p
          expectedBatch := expectedBatch + (Float32.ofNat px.toNat)
      let expected := expectedBatch.toFloat
      expectedTotal := expectedTotal + expected
      let err := Float.abs (v - expected)
      if err > maxAbsErr then
        maxAbsErr := err
      if err > 1.0 then
        mismatchCount := mismatchCount + 1
  pure {
    batches, total, expectedTotal, mismatchCount, maxAbsErr, outDType,
    fusedGatherCount, fusedGatherMatchCount, validated := cfg.indexSelectValidate
  }

initialize do
  let cfg ← readParams
  let baselineStats ← IO.mkRef (default : ProfileSnapshot)
  let prefetchStats ← IO.mkRef (default : ProfileSnapshot)
  let prefetchItemStats ← IO.mkRef (default : ProfileSnapshot)
  let indexSelectStats ← IO.mkRef (default : IndexSelectSnapshot)
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
  LeanBench.register {
    name := "dataloader/mnist/indexselect-small"
    action := do
      let snap ← runIndexSelectSmall cfg
      indexSelectStats.set snap
      -- Print profile events (gather kernel timing) when PROFILE=1
      TinyGrad4.Benchmark.printProfileEvents
      TinyGrad4.Benchmark.resetProfileEvents
    report? := some do
      indexSelectJson cfg <$> indexSelectStats.get
    config := indexSelectConfig cfg ["data", "mnist", "indexselect", "small"]
  }

end TinyGrad4Bench
