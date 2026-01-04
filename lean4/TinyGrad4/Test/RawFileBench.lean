import TinyGrad4.Backend.Accelerate
import TinyGrad4.Data.Prefetch
import TinyGrad4.Data.FastFile
-- Disable IO.monoNanosNow linter: benchmark timing uses raw monotonic clocks.
set_option linter.monoNanosNow false

/-!
# Raw File Bench

Streaming benchmark for large raw files (GBs). Reads sequential chunks and
measures throughput + latency distribution. Optional prefetch worker.
-/

namespace TinyGrad4.Test

open TinyGrad4.Backend.Accel
open TinyGrad4.Data

structure BenchConfig where
  path : System.FilePath
  totalBytes : Nat
  chunkBytes : Nat
  iterations : Nat
  mode : String
  usePrefetch : Bool
  useFast : Bool
  prefetchBuffer : Nat
  deriving Repr

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

private def bytesFromMB (mb : Nat) : Nat := mb * 1024 * 1024

private def modeName (m : String) : String :=
  let m' := m.trimAscii.toString.toLower
  if m' == "checksum" || m' == "accel" || m' == "normalize" || m' == "noop" then m' else "checksum"

private def sumBytesSlow (data : ByteArray) : UInt64 := Id.run do
  let mut total : UInt64 := 0
  for i in [:data.size] do
    total := total + (data.get! i).toUInt64
  total

private def sumBytesAccel (data : ByteArray) : UInt64 :=
  if isAvailable then
    sumU8 data 0 data.size
  else
    sumBytesSlow data

private def normalizeSumBytes (data : ByteArray) : Float :=
  if isAvailable then
    normalizeSumU8 data 0 data.size
  else
    Id.run do
      let mut total : Float := 0.0
      for i in [:data.size] do
        let v := (data.get! i).toFloat / 255.0
        total := total + v
      total

private def touchBytes (data : ByteArray) : UInt64 :=
  if data.size == 0 then 0
  else
    let b0 := data.get! 0
    let b1 := data.get! (data.size - 1)
    (UInt64.ofNat b0.toNat) + ((UInt64.ofNat b1.toNat) <<< 8)

private def pctIndex (n : Nat) (num denom : Nat) : Nat :=
  if n == 0 then 0 else (num * (n - 1)) / denom

private def statsMs (samples : Array Nat) : (Float × Float × Float × Float) :=
  if samples.isEmpty then (0.0, 0.0, 0.0, 0.0) else
    let sorted := samples.qsort (· ≤ ·)
    let n := sorted.size
    let p50 := sorted[pctIndex n 50 100]!
    let p90 := sorted[pctIndex n 90 100]!
    let p99 := sorted[pctIndex n 99 100]!
    let mean := (samples.foldl (fun acc v => acc + v) 0).toFloat / n.toFloat
    (mean / 1.0e6, p50.toFloat / 1.0e6, p90.toFloat / 1.0e6, p99.toFloat / 1.0e6)

private structure BenchResult where
  totalBytes : Nat
  wallNs : Nat
  samples : Array Nat
  checksum : UInt64
  normTotal : Float

private def benchSync (cfg : BenchConfig) : IO BenchResult := do
  IO.FS.withFile cfg.path .read fun h => do
    let mut samples : Array Nat := #[]
    let mut checksum : UInt64 := 0
    let mut normTotal : Float := 0.0
    let mut totalBytes : Nat := 0

    let startAll ← IO.monoNanosNow
    for _ in [:cfg.iterations] do
      h.rewind
      let mut remaining := cfg.totalBytes
      repeat do
        if remaining == 0 then break
        let toRead := min cfg.chunkBytes remaining
        let t0 ← IO.monoNanosNow
        let data ← h.read toRead.toUSize
        if data.isEmpty then break
        match cfg.mode with
        | "normalize" =>
          normTotal := normTotal + normalizeSumBytes data
        | "accel" =>
          checksum := checksum + sumBytesAccel data
        | "noop" =>
          checksum := checksum + touchBytes data
        | _ =>
          checksum := checksum + sumBytesSlow data
        let t1 ← IO.monoNanosNow
        samples := samples.push (t1 - t0)
        totalBytes := totalBytes + data.size
        if data.size >= remaining then
          remaining := 0
        else
          remaining := remaining - data.size

    let stopAll ← IO.monoNanosNow
    pure { totalBytes, wallNs := stopAll - startAll, samples, checksum, normTotal }

private def benchSyncFast (cfg : BenchConfig) : IO BenchResult := do
  FastFile.withFile cfg.path.toString fun fd => do
    let mut samples : Array Nat := #[]
    let mut checksum : UInt64 := 0
    let mut normTotal : Float := 0.0
    let mut totalBytes : Nat := 0
    let mut buf := ByteArray.emptyWithCapacity cfg.chunkBytes

    let startAll ← IO.monoNanosNow
    for _ in [:cfg.iterations] do
      FastFile.rewind fd
      let mut remaining := cfg.totalBytes
      repeat do
        if remaining == 0 then break
        let toRead := min cfg.chunkBytes remaining
        let t0 ← IO.monoNanosNow
        buf ← FastFile.readIntoNat fd buf toRead
        if buf.isEmpty then break
        match cfg.mode with
        | "normalize" =>
          normTotal := normTotal + normalizeSumBytes buf
        | "accel" =>
          checksum := checksum + sumBytesAccel buf
        | "noop" =>
          checksum := checksum + touchBytes buf
        | _ =>
          checksum := checksum + sumBytesSlow buf
        let t1 ← IO.monoNanosNow
        samples := samples.push (t1 - t0)
        totalBytes := totalBytes + buf.size
        if buf.size >= remaining then
          remaining := 0
        else
          remaining := remaining - buf.size

    let stopAll ← IO.monoNanosNow
    pure { totalBytes, wallNs := stopAll - startAll, samples, checksum, normTotal }

private structure FilePrefetcher where
  queue : IOQueue ByteArray
  worker : Task (Except IO.Error Unit)
  totalBytes : Nat

private def FilePrefetcher.create (cfg : BenchConfig) : IO FilePrefetcher := do
  let queue ← IOQueue.new cfg.prefetchBuffer
  let totalBytes := cfg.totalBytes * cfg.iterations
  let worker ← IO.asTask (prio := .dedicated) do
    if cfg.useFast then
      FastFile.withFile cfg.path.toString fun fd => do
        for _ in [:cfg.iterations] do
          FastFile.rewind fd
          let mut remaining := cfg.totalBytes
          repeat do
            if remaining == 0 then break
            let toRead := min cfg.chunkBytes remaining
            let data ← FastFile.readIntoNat fd (ByteArray.emptyWithCapacity toRead) toRead
            if data.isEmpty then break
            let _ ← queue.push data
            if data.size >= remaining then
              remaining := 0
            else
              remaining := remaining - data.size
    else
      IO.FS.withFile cfg.path .read fun h => do
        for _ in [:cfg.iterations] do
          h.rewind
          let mut remaining := cfg.totalBytes
          repeat do
            if remaining == 0 then break
            let toRead := min cfg.chunkBytes remaining
            let data ← h.read toRead.toUSize
            if data.isEmpty then break
            let _ ← queue.push data
            if data.size >= remaining then
              remaining := 0
            else
              remaining := remaining - data.size
    queue.finish
    return ()
  pure { queue, worker, totalBytes }

private def benchPrefetch (cfg : BenchConfig) : IO BenchResult := do
  let prefetcher ← FilePrefetcher.create cfg
  let mut samples : Array Nat := #[]
  let mut checksum : UInt64 := 0
  let mut normTotal : Float := 0.0
  let mut totalBytes : Nat := 0

  let startAll ← IO.monoNanosNow
  repeat do
    let t0 ← IO.monoNanosNow
    match ← prefetcher.queue.pop with
    | some data =>
      match cfg.mode with
      | "normalize" =>
        normTotal := normTotal + normalizeSumBytes data
      | "accel" =>
        checksum := checksum + sumBytesAccel data
      | "noop" =>
        checksum := checksum + touchBytes data
      | _ =>
        checksum := checksum + sumBytesSlow data
      let t1 ← IO.monoNanosNow
      samples := samples.push (t1 - t0)
      totalBytes := totalBytes + data.size
    | none => break
  let stopAll ← IO.monoNanosNow

  let _ ← IO.wait prefetcher.worker
  pure { totalBytes, wallNs := stopAll - startAll, samples, checksum, normTotal }

private def readConfig : IO BenchConfig := do
  let path := System.FilePath.mk (← getEnvString "TG4_RAW_PATH" "../data/raw_40gb.bin")
  let chunkMB ← getEnvNat "TG4_RAW_CHUNK_MB" 4
  let iterations ← getEnvNat "TG4_RAW_ITERS" 1
  let mode ← getEnvString "TG4_RAW_MODE" "accel"
  let usePrefetch ← getEnvBool "TG4_RAW_PREFETCH" false
  let useFast ← getEnvBool "TG4_RAW_FAST" false
  let prefetchBuffer ← getEnvNat "TG4_RAW_PREFETCH_BUF" 8

  let metaData ← path.metadata
  let fileBytes := metaData.byteSize.toNat
  let envTotal ← getEnvNat "TG4_RAW_BYTES" 0
  let totalBytes := if envTotal == 0 then fileBytes else min envTotal fileBytes

  pure {
    path
    totalBytes
    chunkBytes := bytesFromMB chunkMB
    iterations
    mode := modeName mode
    usePrefetch
    useFast
    prefetchBuffer
  }

private def printResult (cfg : BenchConfig) (res : BenchResult) : IO Unit := do
  let (meanMs, p50Ms, p90Ms, p99Ms) := statsMs res.samples
  let wallMs := res.wallNs.toFloat / 1.0e6
  let gb := res.totalBytes.toFloat / 1.0e9
  let gbps := if wallMs == 0.0 then 0.0 else gb / (wallMs / 1000.0)
  IO.println "RAW_FILE_BENCH"
  IO.println s!"path={cfg.path}"
  IO.println s!"mode={cfg.mode} prefetch={cfg.usePrefetch} fast={cfg.useFast} buf={cfg.prefetchBuffer}"
  IO.println s!"chunk_bytes={cfg.chunkBytes} iterations={cfg.iterations}"
  IO.println s!"bytes_total={res.totalBytes} wall_ms={wallMs} gbps={gbps}"
  IO.println s!"lat_mean_ms={meanMs} p50_ms={p50Ms} p90_ms={p90Ms} p99_ms={p99Ms}"
  if cfg.mode == "normalize" then
    IO.println s!"normalize_total={res.normTotal}"
  else
    IO.println s!"checksum={res.checksum}"

/-- Entry point. Environment variables:
  TG4_RAW_PATH, TG4_RAW_BYTES, TG4_RAW_CHUNK_MB, TG4_RAW_ITERS,
  TG4_RAW_MODE=checksum|accel|normalize|noop, TG4_RAW_PREFETCH, TG4_RAW_PREFETCH_BUF, TG4_RAW_FAST
-/
def main : IO Unit := do
  let cfg ← readConfig
  if cfg.totalBytes == 0 then
    IO.println s!"raw_file_bench: file empty or missing: {cfg.path}"
    IO.println "Create with:"
    IO.println "  macOS:   mkfile -n 40g ../data/raw_40gb.bin"
    IO.println "  Linux:   fallocate -l 40G ../data/raw_40gb.bin"
    return
  IO.println s!"Raw file bench config: {repr cfg}"
  let res ←
    if cfg.usePrefetch then
      benchPrefetch cfg
    else if cfg.useFast then
      benchSyncFast cfg
    else
      benchSync cfg
  printResult cfg res

end TinyGrad4.Test

-- Export main
def main := TinyGrad4.Test.main
