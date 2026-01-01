import TinyGrad4.Data.MNISTRaw
import TinyGrad4.Data.MNISTDataset
import TinyGrad4.Data.Prefetch
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
