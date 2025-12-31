import TinyGrad4.Benchmark.Instrumentation
import TinyGrad4.Backend.Metal

/-!
# Benchmark Instrumentation Smoke Test

Tests fine-grained timing breakdown for Metal GPU operations.
-/

namespace TinyGrad4.Test.InstrumentationSmoke

set_option linter.useRawBuffer false

open TinyGrad4.Benchmark
open TinyGrad4.Backend.Metal

def testTimedFunction : IO Unit := do
  IO.println "Testing timed() function..."
  let (result, ns) ← timed (IO.sleep 10)  -- Sleep 10ms
  IO.println s!"  Slept for result={result}, measured {ns} ns (~{ns.toFloat / 1000000.0} ms)"
  -- Should be approximately 10ms
  if ns < 5_000_000 || ns > 50_000_000 then
    IO.println "  WARNING: timing seems off (expected ~10ms)"
  else
    IO.println "  timed() ✓"

def testTimedMs : IO Unit := do
  IO.println "Testing timedMs() function..."
  let (_, ms) ← timedMs (IO.sleep 5)  -- Sleep 5ms
  IO.println s!"  Slept 5ms, measured {ms} ms"
  if ms < 2.0 || ms > 25.0 then
    IO.println "  WARNING: timing seems off (expected ~5ms)"
  else
    IO.println "  timedMs() ✓"

def testBenchmarkN : IO Unit := do
  IO.println "Testing benchmarkN()..."
  let (stats, _) ← benchmarkN 5 (IO.sleep 1)
  IO.println s!"  5 runs: {stats.format}"
  if stats.count != 5 then
    throw (IO.userError s!"expected 5 samples, got {stats.count}")
  IO.println "  benchmarkN() ✓"

def testBenchmarkWithWarmup : IO Unit := do
  IO.println "Testing benchmarkWithWarmup()..."
  let (stats, _) ← benchmarkWithWarmup 2 3 (IO.sleep 1)
  IO.println s!"  2 warmup + 3 runs: {stats.format}"
  if stats.count != 3 then
    throw (IO.userError s!"expected 3 samples (not including warmup), got {stats.count}")
  IO.println "  benchmarkWithWarmup() ✓"

def testKernelTimingFormat : IO Unit := do
  IO.println "Testing KernelTiming format..."
  let kt : KernelTiming := { dispatchNs := 100_000, kernelNs := 500_000, syncNs := 50_000 }
  IO.println s!"  KernelTiming: {kt.format}"
  IO.println s!"  Total: {kt.totalNs} ns = {kt.totalMs} ms"
  IO.println "  KernelTiming format ✓"

def testOpTimingFormat : IO Unit := do
  IO.println "Testing OpTiming format..."
  let kt : KernelTiming := { dispatchNs := 100_000, kernelNs := 500_000, syncNs := 50_000 }
  let ot : OpTiming := { compileNs := 1_000_000, kernel := kt }
  IO.println s!"  OpTiming: {ot.format}"
  IO.println s!"  Compile: {ot.compileMs} ms ({ot.compileFrac * 100.0}%)"
  IO.println s!"  Kernel: {ot.kernel.kernelUs} μs ({ot.kernelFrac * 100.0}%)"
  IO.println "  OpTiming format ✓"

def testMetalCompileTimed : IO Unit := do
  IO.println "Testing timedCompile() with real Metal shader..."
  let shader := "
#include <metal_stdlib>
using namespace metal;
kernel void add_kernel(device float* a [[buffer(0)]],
                       device float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       uint idx [[thread_position_in_grid]]) {
    c[idx] = a[idx] + b[idx];
}
"
  let (_, compileNs) ← timedCompile "add_kernel" shader
  let compileMs := compileNs.toFloat / 1000000.0
  IO.println s!"  Compiled shader in {compileMs} ms"
  -- Compilation typically takes 10-100ms
  if compileMs < 0.1 then
    IO.println "  (very fast - might be cached)"
  IO.println "  timedCompile() ✓"

def testMetalLaunchTimed : IO Unit := do
  IO.println "Testing timedLaunch() with real Metal kernel..."
  let shader := "
#include <metal_stdlib>
using namespace metal;
kernel void add_kernel(device float* a [[buffer(0)]],
                       device float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       uint idx [[thread_position_in_grid]]) {
    c[idx] = a[idx] + b[idx];
}
"
  -- Compile
  let (prog, _) ← timedCompile "add_kernel" shader

  -- Allocate buffers
  let n := 1024
  let aBuf ← metalAlloc (n * 4)  -- 4 bytes per float
  let bBuf ← metalAlloc (n * 4)
  let cBuf ← metalAlloc (n * 4)

  -- Initialize input buffers
  let aData : FloatArray := ⟨Array.replicate n 1.0⟩
  let bData : FloatArray := ⟨Array.replicate n 2.0⟩
  metalCopyIn aBuf aData
  metalCopyIn bBuf bData

  -- Launch with timing
  let (_, launchNs) ← timedLaunch prog #[aBuf, bBuf, cBuf] n 1 1 64 1 1
  let launchMs := launchNs.toFloat / 1000000.0
  IO.println s!"  Launched kernel in {launchMs} ms"

  -- Verify result
  let cData ← metalCopyOut cBuf
  let expected := 3.0  -- 1.0 + 2.0
  let sample := cData.data[0]!
  if (sample - expected).abs > 0.001 then
    throw (IO.userError s!"incorrect result: expected {expected}, got {sample}")
  IO.println s!"  Result verified: {sample} = {expected}"

  -- Cleanup
  metalFree aBuf
  metalFree bBuf
  metalFree cBuf

  IO.println "  timedLaunch() ✓"

def testPipelineCache : IO Unit := do
  IO.println "Testing pipeline cache..."

  -- Clear cache first
  metalCacheClear

  let shader := "
#include <metal_stdlib>
using namespace metal;
kernel void cache_test(device float* a [[buffer(0)]],
                       uint idx [[thread_position_in_grid]]) {
    a[idx] = a[idx] * 2.0f;
}
"
  -- First compile - should be cache miss
  let (_, firstMs) ← timedMs (metalCompile "cache_test" shader)
  IO.println s!"  First compile: {firstMs} ms (cache miss)"

  -- Second compile - should be cache hit
  let (_, secondMs) ← timedMs (metalCompile "cache_test" shader)
  IO.println s!"  Second compile: {secondMs} ms (cache hit)"

  -- Check cache stats
  let stats ← metalCacheStats
  IO.println s!"  Cache stats: hits={stats.hits}, misses={stats.misses}, size={stats.size}"

  -- Verify cache is working
  if stats.hits != 1 then
    throw (IO.userError s!"expected 1 cache hit, got {stats.hits}")
  if stats.misses != 1 then
    throw (IO.userError s!"expected 1 cache miss, got {stats.misses}")

  -- Cache hit should be much faster
  if firstMs > 0.1 && secondMs > firstMs * 0.5 then
    IO.println s!"  WARNING: cache hit not much faster than miss"
  else
    let speedup := if secondMs > 0.0001 then firstMs / secondMs else 999.0
    IO.println s!"  Cache speedup: {speedup}x"

  IO.println "  Pipeline cache ✓"

def testFullTimedOp : IO Unit := do
  IO.println "Testing timedOp() full timing breakdown..."
  let shader := "
#include <metal_stdlib>
using namespace metal;
kernel void mul_kernel(device float* a [[buffer(0)]],
                       device float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       uint idx [[thread_position_in_grid]]) {
    c[idx] = a[idx] * b[idx];
}
"
  -- Allocate buffers
  let n := 4096
  let aBuf ← metalAlloc (n * 4)
  let bBuf ← metalAlloc (n * 4)
  let cBuf ← metalAlloc (n * 4)

  -- Initialize
  let aData : FloatArray := ⟨Array.replicate n 2.0⟩
  let bData : FloatArray := ⟨Array.replicate n 3.0⟩
  metalCopyIn aBuf aData
  metalCopyIn bBuf bData

  -- Full timed operation
  let timing ← timedOp "mul_kernel" shader #[aBuf, bBuf, cBuf] n 1 1 256 1 1

  -- Print breakdown
  printTiming "mul_kernel" timing
  IO.println s!"  Compile: {timing.compileMs} ms ({timing.compileFrac * 100.0}%)"
  IO.println s!"  Dispatch: {timing.kernel.dispatchUs} μs"
  IO.println s!"  Kernel: {timing.kernel.kernelUs} μs ({timing.kernelFrac * 100.0}%)"
  IO.println s!"  Sync: {timing.kernel.syncUs} μs"
  IO.println s!"  Total: {timing.totalMs} ms"

  -- Verify result
  let cData ← metalCopyOut cBuf
  let expected := 6.0  -- 2.0 * 3.0
  let sample := cData.data[0]!
  if (sample - expected).abs > 0.001 then
    throw (IO.userError s!"incorrect result: expected {expected}, got {sample}")
  IO.println s!"  Result verified: {sample}"

  -- Cleanup
  metalFree aBuf
  metalFree bBuf
  metalFree cBuf

  IO.println "  timedOp() ✓"

def runAll : IO Unit := do
  IO.println "=== Benchmark Instrumentation Smoke Tests ==="

  -- Basic timing tests (no GPU)
  testTimedFunction
  testTimedMs
  testBenchmarkN
  testBenchmarkWithWarmup
  testKernelTimingFormat
  testOpTimingFormat

  -- Metal GPU tests
  testMetalCompileTimed
  testMetalLaunchTimed
  testPipelineCache
  testFullTimedOp

  IO.println "=== Instrumentation Smoke OK ==="

end TinyGrad4.Test.InstrumentationSmoke

def main : IO Unit := TinyGrad4.Test.InstrumentationSmoke.runAll
