import TinyGrad4.Shape
import TinyGrad4.Data.MNISTRaw
import TinyGrad4.Data.MNISTDataset
import TinyGrad4.Data.Prefetch
import TinyGrad4.Data.IterDataset
import TinyGrad4.Data.Profile
import TinyGrad4.Data.GPULoader
import TinyGrad4.Data.Shuffle
import TinyGrad4.Data.Transform
-- Disable IO.monoNanosNow linter: benchmark timing uses raw monotonic clocks.
set_option linter.monoNanosNow false

open TinyGrad4

def main : IO Unit := do
  let dataDir := "../data"
  let batchSize := 64
  let maxImages := 10000

  IO.println "=== Lean Data Loader Bench Suite ==="
  IO.println s!"dataDir={dataDir} batch={batchSize} maxImages={maxImages}"
  IO.println ""

  -- 1) Raw ByteSlice pipeline (matches Python bench_data_loading.py)
  TinyGrad4.Data.MNISTRaw.benchmarkComprehensive dataDir batchSize maxImages 10
  IO.println ""

  -- 2) Grain-style dataset pipeline vs old loader
  TinyGrad4.Data.mnistBenchmarkComparison dataDir batchSize maxImages
  IO.println ""

  -- 3) Prefetch throughput on batched MNIST pipeline
  let mnist ← TinyGrad4.Data.MNISTDataset.loadTrain dataDir (some maxImages)
  let key := TinyGrad4.Data.RandKey.new 42
  let pipeline := TinyGrad4.Data.mnistTrainPipeline mnist key batchSize
  let (baseline, prefetched) ← TinyGrad4.Data.benchmarkComparison pipeline 8
  IO.println s!"Prefetch throughput: baseline={baseline} items/s, prefetched={prefetched} items/s"

  -- 3b) Multi-worker prefetch throughput (interleaved shards)
  let workers :=
    match (← IO.getEnv "TG4_MULTI_PREFETCH_WORKERS") with
    | some v => v.trimAscii.toString.toNat?.getD 4
    | none => 4
  let iterDs := TinyGrad4.Data.IterDataset.ofDataset pipeline (TinyGrad4.Data.RandKey.new 7) 1
  let multiPrefetcher ← iterDs.toMultiPrefetcher workers 8 .interleaved true
  let startMulti ← IO.monoNanosNow
  let mut countMulti := 0
  repeat do
    match ← multiPrefetcher.next with
    | some _ => countMulti := countMulti + 1
    | none => break
  let stopMulti ← IO.monoNanosNow
  multiPrefetcher.cancel
  let secondsMulti := (stopMulti - startMulti).toFloat / 1e9
  let rateMulti := if secondsMulti == 0.0 then 0.0 else countMulti.toFloat / secondsMulti
  IO.println s!"Multi-prefetch throughput: workers={workers} rate={rateMulti} items/s"

  -- 3c) Multi-worker prefetch resume benchmark (strict round-robin)
  let interruptAtRaw :=
    match (← IO.getEnv "TG4_MULTI_PREFETCH_INTERRUPT") with
    | some v => v.trimAscii.toString.toNat?.getD 1024
    | none => 1024
  let policy := TinyGrad4.Data.MultiIteratorPrefetcher.OrderingPolicy.strict
  let prefetcherResume ← iterDs.toMultiPrefetcher workers 8 .interleaved true policy
  let totalResume := prefetcherResume.totalItems
  let interruptAt := min interruptAtRaw totalResume
  let mut seen := 0
  for _ in [:interruptAt] do
    match ← prefetcherResume.next with
    | some _ => seen := seen + 1
    | none => break
  let ckptStart ← IO.monoNanosNow
  let state ← prefetcherResume.checkpoint
  let ckptStop ← IO.monoNanosNow
  prefetcherResume.cancel
  let resumeStart ← IO.monoNanosNow
  let prefetcherResume2 ← iterDs.toMultiPrefetcherFrom state workers 8 .interleaved true policy
  let mut remainder := 0
  repeat do
    match ← prefetcherResume2.next with
    | some _ => remainder := remainder + 1
    | none => break
  let resumeStop ← IO.monoNanosNow
  prefetcherResume2.cancel
  if seen + remainder != totalResume then
    throw (IO.userError s!"Multi-prefetch resume mismatch: total={totalResume} seen={seen} remainder={remainder}")
  let ckptMs := (ckptStop - ckptStart).toFloat / 1e6
  let resumeSeconds := (resumeStop - resumeStart).toFloat / 1e9
  let resumeRate := if resumeSeconds == 0.0 then 0.0 else remainder.toFloat / resumeSeconds
  IO.println s!"Multi-prefetch resume: workers={workers} interruptAt={interruptAt} total={totalResume} " ++
    s!"ckptMs={ckptMs} resumeRate={resumeRate} items/s policy=strict"

  -- 4) Stagewise profiling: iterator/prefetch/transfer/compute
  IO.println ""
  IO.println "=== Stagewise Profiling (iterator/prefetch/transfer/compute) ==="

  let concatByteArrays (chunks : Array ByteArray) : ByteArray := Id.run do
    let total := chunks.foldl (fun acc b => acc + b.size) 0
    let mut out := ByteArray.emptyWithCapacity total
    let mut offset := 0
    for chunk in chunks do
      out := ByteArray.copySlice chunk 0 out offset chunk.size false
      offset := offset + chunk.size
    out

  let profiler ← TinyGrad4.Data.Profiler.new
  let sampleSize := 4096
  let sampleCount := 512
  let sample := ByteArray.mk (Array.replicate sampleSize 1)
  let ds := TinyGrad4.Data.ofArray (Array.replicate sampleCount sample)
  let iterDs := TinyGrad4.Data.IterDataset.ofDataset ds (TinyGrad4.Data.RandKey.new 7) 1
  let profiled := TinyGrad4.Data.IterDataset.profile profiler "iterator" iterDs
  let batchSize := 32
  let batchShape : Shape := [batchSize, sampleSize]
  let prefetcher ← profiled.toBatchPrefetcher batchSize (fun chunks => pure (concatByteArrays chunks)) true 8

  let devices ← TinyGrad4.Data.GPULoader.discoverDevices
  let gpuDevice? :=
    devices.find? fun d => d != .cpu && (match d with | .tpu _ => false | _ => true)

  let checksum (ba : ByteArray) : UInt64 := Id.run do
    let mut acc : UInt64 := 0
    for i in [:ba.size] do
      acc := acc + ba[i]!.toUInt64
    acc

  let mut count := 0
  repeat do
    match ← TinyGrad4.Data.profileBatchPrefetcherNext profiler "prefetch" prefetcher with
    | none => break
    | some bytes =>
        let start ← IO.monoNanosNow
        match gpuDevice? with
        | some dev =>
            let buf ← TinyGrad4.Data.GPULoader.ByteArray.toGPUBuffer bytes dev batchShape .uint8
            let stop ← IO.monoNanosNow
            profiler.recordSampleSpan "transfer" start stop
            buf.free
        | none =>
            let stop ← IO.monoNanosNow
            profiler.recordSampleSpan "transfer" start stop
        let computeStart ← IO.monoNanosNow
        let _ := checksum bytes
        let computeStop ← IO.monoNanosNow
        profiler.recordSampleSpan "compute" computeStart computeStop
        count := count + 1

  prefetcher.cancel
  IO.println s!"  batches={count}"
  IO.println (← profiler.summaryWithConcurrencyByStage)
