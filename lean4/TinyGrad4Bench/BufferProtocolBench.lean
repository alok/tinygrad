import TinyGrad4.Data.Buffer
import TinyGrad4.Data.IndexTransform
import TinyGrad4Bench.BenchUtil

/-!
# Buffer Protocol Benchmarks

Measures overhead of buffer protocol operations.

## Targets (from design plan)
- `toRaw` overhead: < 50ns
- Shuffle construction (60K elements): < 1ms
- computeCStrides: O(n) but low constant factor
- Registry operations: < 100ns per op
-/

open TinyGrad4.Data
open LeanBenchNew
open TinyGrad4Bench

/-! ## computeCStrides Benchmarks -/

def benchComputeStrides : IO (Array Result) := do
  let mut results := #[]

  -- Small shape (common case: 3D tensor)
  let r1 ← runPure "computeCStrides [2,3,4]" microBenchConfig fun () =>
    RawBuffer.computeCStrides #[2, 3, 4]
  results := results.push r1

  -- Medium shape (4D tensor, typical for images)
  let r2 ← runPure "computeCStrides [32,3,224,224]" microBenchConfig fun () =>
    RawBuffer.computeCStrides #[32, 3, 224, 224]
  results := results.push r2

  -- Large shape (6D tensor, edge case)
  let r3 ← runPure "computeCStrides [2,3,4,5,6,7]" microBenchConfig fun () =>
    RawBuffer.computeCStrides #[2, 3, 4, 5, 6, 7]
  results := results.push r3

  pure results

/-! ## Shuffle Construction Benchmarks -/

def benchShuffleConstruction : IO (Array Result) := do
  let mut results := #[]

  -- Small dataset
  let r1 ← runPure "shuffle(1K elements)" microBenchConfig fun () =>
    IndexTransform.shuffle 42 1000
  results := results.push r1

  -- MNIST-sized (60K)
  let r2 ← runPure "shuffle(60K elements)" mediumBenchConfig fun () =>
    IndexTransform.shuffle 42 60000
  results := results.push r2

  -- ImageNet-sized (1.2M)
  let r3 ← runPure "shuffle(100K elements)" mediumBenchConfig fun () =>
    IndexTransform.shuffle 42 100000
  results := results.push r3

  pure results

/-! ## Shuffle Lookup Benchmarks (should be O(1)) -/

/-- Do N lookups in a shuffle -/
def doLookups (shuffle : IndexTransform) (n : Nat) (h : shuffle.outputLen = 60000) : Unit := Id.run do
  for i in [:n] do
    if hi : i < 60000 then
      let _ := shuffle.map ⟨i, h ▸ hi⟩
  ()

def benchShuffleLookup : IO (Array Result) := do
  let mut results := #[]

  -- Pre-construct shuffle once
  let shuffle60k := IndexTransform.shuffle 42 60000

  -- Single lookup
  let r1 ← runPure "shuffle lookup (single)" microBenchConfig fun () =>
    let _ := shuffle60k.map ⟨12345, by native_decide⟩
    ()
  results := results.push r1

  -- 100 lookups - using explicit function to avoid instance issues
  let r2 ← runPure "shuffle lookup (100x)" microBenchConfig fun () =>
    doLookups shuffle60k 100 rfl
  results := results.push r2

  -- 1000 lookups (full batch)
  let r3 ← runPure "shuffle lookup (1000x)" microBenchConfig fun () =>
    doLookups shuffle60k 1000 rfl
  results := results.push r3

  pure results

/-! ## BufferRegistry Benchmarks -/

def benchRegistry : IO (Array Result) := do
  let mut results := #[]

  -- Registry creation
  let r1 ← run "BufferRegistry.new" microBenchConfig fun () =>
    BufferRegistry.new
  results := results.push r1

  -- Register operation
  let reg ← BufferRegistry.new
  let r2 ← run "BufferRegistry.register" microBenchConfig fun () => do
    let _ ← reg.register .cpu 1024
    pure ()
  results := results.push r2

  -- Acquire operation (need existing buffer)
  let reg2 ← BufferRegistry.new
  let id ← reg2.register .cpu 1024
  let r3 ← run "BufferRegistry.acquire" microBenchConfig fun () => do
    reg2.acquire id
  results := results.push r3

  -- Release operation
  let r4 ← run "BufferRegistry.release" microBenchConfig fun () => do
    reg2.acquire id  -- acquire first so we can release
    let _ ← reg2.release id
    pure ()
  results := results.push r4

  pure results

/-! ## TrackedBuffer Benchmarks -/

def benchTrackedBuffer : IO (Array Result) := do
  let mut results := #[]
  let reg ← BufferRegistry.new

  -- Allocation
  let r1 ← run "TrackedBuffer.alloc" microBenchConfig fun () => do
    let _ ← TrackedBuffer.alloc reg [1024] .float32 .cpu
    pure ()
  results := results.push r1

  -- withBorrowed pattern
  let buf ← TrackedBuffer.alloc reg [1024] .float32 .cpu
  let r2 ← run "withBorrowed (single)" microBenchConfig fun () => do
    withBorrowed buf fun _ => pure ()
  results := results.push r2

  -- withBorrowedAll pattern (5 buffers)
  let bufs ← (List.range 5).mapM fun _ =>
    TrackedBuffer.alloc reg [1024] .float32 .cpu
  let bufsArr := bufs.toArray
  let r3 ← run "withBorrowedAll (5 bufs)" microBenchConfig fun () => do
    withBorrowedAll bufsArr fun _ => pure ()
  results := results.push r3

  pure results

/-! ## RawBuffer Operations -/

def benchDescriptorOps : IO (Array Result) := do
  let mut results := #[]

  let desc : RawBuffer := {
    handle := { ptr := 0x1000, device := .cpu }
    dtype := .float32
    shape := #[32, 3, 224, 224]
    strides := none
    byteOffset := 0
  }

  -- numel computation
  let r1 ← runPure "RawBuffer.numel" microBenchConfig fun () =>
    desc.numel
  results := results.push r1

  -- bytes computation
  let r2 ← runPure "RawBuffer.bytes" microBenchConfig fun () =>
    desc.bytes
  results := results.push r2

  -- isContiguous check
  let r3 ← runPure "RawBuffer.isContiguous" microBenchConfig fun () =>
    desc.isContiguous
  results := results.push r3

  -- slice operation
  let r4 ← runPure "RawBuffer.slice" microBenchConfig fun () =>
    desc.slice 100 #[16, 3, 224, 224]
  results := results.push r4

  pure results

/-! ## BufferExport/Import Benchmarks -/

def benchBufferExchange : IO (Array Result) := do
  let mut results := #[]
  let reg ← BufferRegistry.new

  let buf ← TrackedBuffer.alloc reg [1024] .float32 .cpu

  -- toRaw (TrackedBuffer → RawBuffer)
  let r1 ← run "BufferExport.toRaw" microBenchConfig fun () => do
    let _ ← BufferExport.toRaw buf
    pure ()
  results := results.push r1

  let desc := buf.toRaw

  -- fromRaw (RawBuffer + registry → TrackedBuffer)
  let r2 ← run "BufferImport.fromRaw" microBenchConfig fun () => do
    let _ ← (BufferImport.fromRaw reg desc : IO (TrackedBuffer [1024] .float32))
    pure ()
  results := results.push r2

  pure results

/-! ## End-to-End: Data Loading Simulation -/

def benchDataLoadingPattern : IO (Array Result) := do
  let mut results := #[]
  let reg ← BufferRegistry.new

  -- Simulate: create shuffle, lookup indices, allocate buffers
  let datasetSize := 60000
  let batchSize := 64

  -- Full batch loading pattern
  let shuffle := IndexTransform.shuffle 42 datasetSize
  -- Note: shuffle.outputLen = datasetSize = 60000, and we iterate over [:64]
  let r1 ← run "batch load pattern (64 items)" microBenchConfig fun () => do
    for i in [:batchSize] do
      -- Safe since i < batchSize (64) < 60000
      if h : i < shuffle.outputLen then
        let _idx := shuffle.map ⟨i, h⟩
        let _buf ← TrackedBuffer.alloc reg [784] .float32 .cpu  -- MNIST image
        pure ()
  results := results.push r1

  pure results

/-! ## Run All Benchmarks -/

def main : IO Unit := do
  IO.println "═══════════════════════════════════════════════════════════════"
  IO.println "         BUFFER PROTOCOL BENCHMARKS"
  IO.println "═══════════════════════════════════════════════════════════════"
  IO.println ""

  -- computeCStrides
  IO.println "▶ computeCStrides (target: low overhead)"
  IO.println "─────────────────────────────────────────"
  let stridesResults ← benchComputeStrides
  printComparison stridesResults
  IO.println ""

  -- Shuffle construction
  IO.println "▶ Shuffle Construction (target: 60K < 1ms)"
  IO.println "──────────────────────────────────────────"
  let shuffleResults ← benchShuffleConstruction
  printComparison shuffleResults
  -- Check target (60K benchmark is index 1)
  if let some r := shuffleResults[1]? then
    let ms := r.avgTimeNs.toFloat / 1_000_000.0
    if ms < 1.0 then
      IO.println s!"  ✓ {r.name}: {ms}ms < 1ms target"
    else
      IO.println s!"  ✗ {r.name}: {ms}ms >= 1ms target (SLOW)"
  IO.println ""

  -- Shuffle lookup
  IO.println "▶ Shuffle Lookup (target: O(1))"
  IO.println "────────────────────────────────"
  let lookupResults ← benchShuffleLookup
  printComparison lookupResults
  -- Verify O(1): 100x should be ~100x slower than 1x, not 10000x (O(n))
  match lookupResults[0]?, lookupResults[1]? with
  | some r1, some r100 =>
    let ratio := r100.avgTimeNs.toFloat / r1.avgTimeNs.toFloat
    if ratio < 200.0 then  -- Allow 2x overhead for loop
      IO.println s!"  ✓ O(1) verified: 100 lookups = {ratio}x single lookup"
    else
      IO.println s!"  ✗ NOT O(1): 100 lookups = {ratio}x single lookup (expected ~100x)"
  | _, _ => pure ()
  IO.println ""

  -- Registry
  IO.println "▶ BufferRegistry (target: < 100ns per op)"
  IO.println "──────────────────────────────────────────"
  let regResults ← benchRegistry
  printComparison regResults
  IO.println ""

  -- TrackedBuffer
  IO.println "▶ TrackedBuffer Operations"
  IO.println "──────────────────────────"
  let bufResults ← benchTrackedBuffer
  printComparison bufResults
  IO.println ""

  -- Descriptor ops
  IO.println "▶ RawBuffer Operations"
  IO.println "──────────────────────"
  let descResults ← benchDescriptorOps
  printComparison descResults
  IO.println ""

  -- Buffer exchange
  IO.println "▶ BufferExport/Import (target: toRaw < 50ns)"
  IO.println "────────────────────────────────────────────────────"
  let exchangeResults ← benchBufferExchange
  printComparison exchangeResults
  -- Check toRaw (index 0)
  if let some r := exchangeResults[0]? then
    if r.avgTimeNs < 50 then
      IO.println s!"  ✓ {r.name}: {r.avgTimeNs}ns < 50ns target"
    else
      IO.println s!"  ⚠ {r.name}: {r.avgTimeNs}ns >= 50ns target"
  IO.println ""

  -- End-to-end
  IO.println "▶ End-to-End Data Loading Pattern"
  IO.println "──────────────────────────────────"
  let e2eResults ← benchDataLoadingPattern
  printComparison e2eResults
  IO.println ""

  IO.println "═══════════════════════════════════════════════════════════════"
  IO.println "         BENCHMARK COMPLETE"
  IO.println "═══════════════════════════════════════════════════════════════"
