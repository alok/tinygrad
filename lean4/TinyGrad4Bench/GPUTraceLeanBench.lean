import Lean.Data.Json
import Lean.Data.Json.Elab
import LeanBench
import TinyGrad4.Backend.Metal
import TinyGrad4.Benchmark.Kernels
import TinyGrad4.Benchmark.MetalDirect
import TinyGrad4.Benchmark.Trace

/-!
LeanBench GPU trace benchmarks for TinyGrad4.
These attach trace JSON for MetalDirect runs.
-/

namespace TinyGrad4Bench

open LeanBench
open TinyGrad4.Benchmark
open TinyGrad4.Benchmark.Kernels
open TinyGrad4.Benchmark.MetalDirect

private def gpuSuiteCfg : LeanBench.SuiteConfig :=
  { description := "GPU kernel benches (Metal Direct + trace)"
    tags := ["gpu"] }

bench_suite "gpu" (gpuSuiteCfg)

structure GpuBenchParams where
  name : String
  kernel : BenchKernel
  size : Nat
  iterations : Nat
  tags : List String
  bytes : Nat
  flops : Nat

private def metalAvailable : IO Bool := do
  if !System.Platform.isOSX then
    return false
  try
    let _ <- TinyGrad4.Backend.Metal.metalDeviceName
    return true
  catch _ =>
    return false

private def benchConfig (p : GpuBenchParams) : BenchConfig :=
  { suite := some "gpu"
    tags := p.tags
    warmup := 0
    samples := 1
    bytes := some p.bytes
    flops := some p.flops }

private def runMetalIfAvailable (p : GpuBenchParams) : IO Unit := do
  if !(<- metalAvailable) then
    return
  let _ <- runDirectBenchmark p.kernel p.size p.iterations
  pure ()

private def traceExtras (p : GpuBenchParams) : IO Lean.Json := do
  if !(<- metalAvailable) then
    return json% { skipped: true, reason: "metal_unavailable" }
  let run <- runDirectBenchmarkWithTrace p.kernel p.size p.iterations
  return json% {
    backend: "METAL_DIRECT",
    device: $(run.result.device),
    kernel: $(p.kernel.name),
    size: $(p.size),
    iterations: $(p.iterations),
    trace: $(Lean.toJson run.trace)
  }

initialize do
  let size := 1_000_000
  let kernel := BenchKernel.add
  let bytes := size * 4 * (kernel.numInputs + 1)
  let flops := size * kernel.flopsPerElement
  let params : GpuBenchParams := {
    name := "gpu/metal/add-1m"
    kernel := kernel
    size := size
    iterations := 10
    tags := ["gpu", "metal", "trace", "ewise"]
    bytes := bytes
    flops := flops
  }
  LeanBench.register {
    name := params.name
    action := runMetalIfAvailable params
    report? := some (traceExtras params)
    config := benchConfig params
  }

end TinyGrad4Bench
