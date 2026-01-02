import TinyGrad4.Data.MNISTRaw
import TinyGrad4.Data.MNISTDataset
import TinyGrad4.Data.Prefetch
import TinyGrad4.Data.IterDataset
import TinyGrad4.Data.Profile
import TinyGrad4.Data.GPULoader
import TinyGrad4.Data.Shuffle
import TinyGrad4.Data.Transform

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

  -- 4) Stagewise profiling: iterator/prefetch/transfer/compute
  IO.println ""
  IO.println "=== Stagewise Profiling (iterator/prefetch/transfer/compute) ==="

  let profiler ← TinyGrad4.Data.Profiler.new
  let sampleSize := 4096
  let sampleCount := 512
  let sample := ByteArray.mk (Array.replicate sampleSize 1)
  let ds := TinyGrad4.Data.ofArray (Array.replicate sampleCount sample)
  let iterDs := TinyGrad4.Data.IterDataset.ofDataset ds (TinyGrad4.Data.RandKey.new 7) 1
  let profiled := TinyGrad4.Data.IterDataset.profile profiler "iterator" iterDs
  let prefetcher ← profiled.toPrefetcher 8

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
    match ← TinyGrad4.Data.profileIteratorPrefetcherNext profiler "prefetch" prefetcher with
    | none => break
    | some bytes =>
        let start ← IO.monoNanosNow
        match gpuDevice? with
        | some dev =>
            let buf ← TinyGrad4.Data.GPULoader.ByteArray.toGPUBuffer bytes dev .uint8
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
  IO.println s!"  samples={count}"
  IO.println (← profiler.summaryWithConcurrencyByStage)
