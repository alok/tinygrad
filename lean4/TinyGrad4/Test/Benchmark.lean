import TinyGrad4

/-!
# Performance Benchmarks

Measures execution time for tensor operations at various sizes.
Inspired by tinygrad's test_microbenchmarks.py.

## Notes

- Benchmarks measure full pipeline: graph construction + interpreter evaluation
- Uses IO.Ref to ensure fresh computation graphs each iteration (prevents memoization)
- MFLOP/s reflects interpreter overhead, not raw computation speed
- For realistic kernel performance, need native backend or MLIR codegen

## Running

```
lake build benchmark && ./.lake/build/bin/benchmark
```
-/

namespace TinyGrad4.Test.Benchmark

open TinyGrad4
open StaticTensor
open Interpreter

/-! ## Timing Infrastructure -/

/-- Measure execution time, returns nanoseconds -/
def timeIONs (action : IO Unit) : IO Nat := do
  let start ← IO.monoNanosNow
  action
  let finish ← IO.monoNanosNow
  pure (finish - start)

/-- Run action N times and return average time in nanoseconds -/
def benchmarkN (n : Nat) (action : IO Unit) : IO Float := do
  -- Warmup
  for _ in [:3] do
    action
  -- Actual measurement
  let mut total : Nat := 0
  for _ in [:n] do
    let ns ← timeIONs action
    total := total + ns
  pure (total.toFloat / n.toFloat)

/-- Force FlatArray evaluation - prevents DCE -/
@[noinline] def forceEval (arr : FlatArray) : IO Unit := do
  if arr.size == 0 then
    IO.println "!" -- Never happens but prevents optimization
  pure ()

/-- Format nanoseconds as human-readable string -/
def formatTime (ns : Float) : String :=
  if ns < 1000 then
    s!"{ns} ns"
  else if ns < 1000000 then
    s!"{ns / 1000} μs"
  else if ns < 1000000000 then
    s!"{ns / 1000000} ms"
  else
    s!"{ns / 1000000000} s"

/-- Compute MFLOP/s -/
def mflops (flops : Nat) (ns : Float) : Float :=
  let seconds := ns / 1e9
  flops.toFloat / seconds / 1e6

/-- Compute GB/s -/
def gbps (bytes : Nat) (ns : Float) : Float :=
  let seconds := ns / 1e9
  bytes.toFloat / seconds / 1e9

/-- Print benchmark result -/
def printBench (name : String) (ns : Float) (flops : Nat := 0) (bytes : Nat := 0) : IO Unit := do
  let timeStr := formatTime ns
  let mut extra := ""
  if flops > 0 then
    extra := extra ++ s!" | {mflops flops ns} MFLOP/s"
  if bytes > 0 then
    extra := extra ++ s!" | {gbps bytes ns} GB/s"
  IO.println s!"{name} : {timeStr}{extra}"

/-! ## Tensor Creation Benchmarks -/

def benchTensorCreation : IO Unit := do
  IO.println "\n=== Tensor Creation ==="

  let seedRef ← IO.mkRef (0 : Nat)

  let ns1 ← benchmarkN 100 do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let val : Float32 := Float32.ofScientific seed false 3  -- seed * 10^-3
    let result ← evalTensorCached (runTensorM do Tensor.full [10, 10] .float32 val)
    forceEval result
  printBench "full [10,10]" ns1

  let ns2 ← benchmarkN 50 do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let val : Float32 := Float32.ofScientific seed false 3
    let result ← evalTensorCached (runTensorM do Tensor.full [100, 100] .float32 val)
    forceEval result
  printBench "full [100,100]" ns2

  let ns3 ← benchmarkN 5 do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let val : Float32 := Float32.ofScientific seed false 3
    let result ← evalTensorCached (runTensorM do Tensor.full [1000, 1000] .float32 val)
    forceEval result
  printBench "full [1000,1000]" ns3

  let ns4 ← benchmarkN 50 do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do Tensor.rand [100, 100] .float32 seed)
    forceEval result
  printBench "rand [100,100]" ns4

/-! ## Elementwise Operation Benchmarks -/

def benchElementwise : IO Unit := do
  IO.println "\n=== Elementwise Operations ==="

  let n := 1000
  let size := 1000
  let flops := size
  let bytes := size * 4 * 2

  -- Use IO.Ref for mutable seed across closures
  let seedRef ← IO.mkRef (0 : Nat)

  let nsNeg ← benchmarkN n do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do
      let a ← Tensor.rand [size] .float32 seed
      neg a)
    forceEval result
  printBench s!"neg [{size}]" nsNeg flops bytes

  let nsSqrt ← benchmarkN n do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do
      let a ← Tensor.rand [size] .float32 seed
      sqrt a)
    forceEval result
  printBench s!"sqrt [{size}]" nsSqrt flops bytes

  let nsExp ← benchmarkN n do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do
      let a ← Tensor.rand [size] .float32 seed
      exp a)
    forceEval result
  printBench s!"exp [{size}]" nsExp flops bytes

  let nsSin ← benchmarkN n do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do
      let a ← Tensor.rand [size] .float32 seed
      sin a)
    forceEval result
  printBench s!"sin [{size}]" nsSin flops bytes

  let nsCos ← benchmarkN n do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do
      let a ← Tensor.rand [size] .float32 seed
      cos a)
    forceEval result
  printBench s!"cos [{size}]" nsCos flops bytes

  let nsTan ← benchmarkN n do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do
      let a ← Tensor.rand [size] .float32 seed
      tan a)
    forceEval result
  printBench s!"tan [{size}]" nsTan flops bytes

  let bytesAdd := size * 4 * 3
  let nsAdd ← benchmarkN n do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do
      let a ← Tensor.rand [size] .float32 seed
      let b ← Tensor.rand [size] .float32 (seed + 1)
      add a b)
    forceEval result
  printBench s!"add [{size}]" nsAdd flops bytesAdd

  let nsMul ← benchmarkN n do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do
      let a ← Tensor.rand [size] .float32 seed
      let b ← Tensor.rand [size] .float32 (seed + 1)
      mul a b)
    forceEval result
  printBench s!"mul [{size}]" nsMul flops bytesAdd

  let nsPow ← benchmarkN n do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do
      let a ← Tensor.rand [size] .float32 seed
      let b ← Tensor.rand [size] .float32 (seed + 1)
      pow a b)
    forceEval result
  printBench s!"pow [{size}]" nsPow flops bytesAdd

/-! ## Raw Kernel Benchmarks (Pre-generated data) -/

/-- Benchmark using pre-generated contiguous buffers to isolate kernel performance from graph overhead -/
def benchRawKernel : IO Unit := do
  IO.println "\n=== Raw Kernel (Pre-cached inputs) ==="

  let size := 10000
  let n := 100

  -- Pre-generate and cache input data
  let inputData ← evalTensorCached (runTensorM do
    Tensor.rand [size] .float32 42)
  let inputData2 ← evalTensorCached (runTensorM do
    Tensor.rand [size] .float32 43)

  -- Create the graphs just once
  let negGraph := runTensorM do
    let a ← Tensor.buffer [size] .float32
    neg a

  -- Benchmark: measure only the kernel execution (graph already built)
  let nsNegRaw ← benchmarkN n do
    let result ← evalTensorCached negGraph
    forceEval result
  printBench s!"raw neg [{size}]" nsNegRaw size (size * 4 * 2)

  let addGraph := runTensorM do
    let a ← Tensor.buffer [size] .float32
    let b ← Tensor.buffer [size] .float32
    add a b

  let nsAddRaw ← benchmarkN n do
    let result ← evalTensorCached addGraph
    forceEval result
  printBench s!"raw add [{size}]" nsAddRaw size (size * 4 * 3)

  let sqrtGraph := runTensorM do
    let a ← Tensor.buffer [size] .float32
    sqrt a

  let nsSqrtRaw ← benchmarkN n do
    let result ← evalTensorCached sqrtGraph
    forceEval result
  printBench s!"raw sqrt [{size}]" nsSqrtRaw size (size * 4 * 2)

/-! ## Reduction Benchmarks -/

def benchReductions : IO Unit := do
  IO.println "\n=== Reductions ==="

  let seedRef ← IO.mkRef (0 : Nat)

  for size in #[100, 1000, 10000, 100000] do
    let n := if size > 10000 then 100 else 1000
    let flops := size

    let nsSum ← benchmarkN n do
      let seed ← seedRef.modifyGet fun s => (s, s + 1)
      let result ← evalTensorCached (runTensorM do
        let a ← Tensor.rand [size] .float32 seed
        sum a)
      forceEval result
    printBench s!"sum [{size}]" nsSum flops

    let nsMax ← benchmarkN n do
      let seed ← seedRef.modifyGet fun s => (s, s + 1)
      let result ← evalTensorCached (runTensorM do
        let a ← Tensor.rand [size] .float32 seed
        max a)
      forceEval result
    printBench s!"max [{size}]" nsMax flops

/-! ## Matrix Multiplication Benchmarks -/

def benchMatmul : IO Unit := do
  IO.println "\n=== Matrix Multiplication ==="

  let seedRef ← IO.mkRef (0 : Nat)

  -- Small: [32, 32] @ [32, 32]
  let m1 := 32
  let flops1 := 2 * m1 * m1 * m1
  let n1 := 100  -- Reduce iterations for costly matmuls

  let ns1 ← benchmarkN n1 do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do
      let a ← Tensor.rand [m1, m1] .float32 seed
      let b ← Tensor.rand [m1, m1] .float32 (seed + 1)
      matmul a b)
    forceEval result
  printBench s!"matmul {m1}x{m1}" ns1 flops1

  -- Medium: [128, 128] @ [128, 128]
  let m2 := 128
  let flops2 := 2 * m2 * m2 * m2
  let n2 := 20

  let ns2 ← benchmarkN n2 do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do
      let a ← Tensor.rand [m2, m2] .float32 seed
      let b ← Tensor.rand [m2, m2] .float32 (seed + 1)
      matmul a b)
    forceEval result
  printBench s!"matmul {m2}x{m2}" ns2 flops2

  -- Large: [256, 256] @ [256, 256]
  let m3 := 256
  let flops3 := 2 * m3 * m3 * m3
  let n3 := 5

  let ns3 ← benchmarkN n3 do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do
      let a ← Tensor.rand [m3, m3] .float32 seed
      let b ← Tensor.rand [m3, m3] .float32 (seed + 1)
      matmul a b)
    forceEval result
  printBench s!"matmul {m3}x{m3}" ns3 flops3

/-! ## Composite Operations -/

def benchComposite : IO Unit := do
  IO.println "\n=== Composite Operations ==="

  let size := 1000
  let n := 100
  let seedRef ← IO.mkRef (0 : Nat)

  let nsSoftmax ← benchmarkN n do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do
      let x ← Tensor.rand [size] .float32 seed
      softmax x)
    forceEval result
  printBench s!"softmax [{size}]" nsSoftmax (size * 4)

  let nsTanh ← benchmarkN n do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do
      let x ← Tensor.rand [size] .float32 seed
      tanh x)
    forceEval result
  printBench s!"tanh [{size}]" nsTanh (size * 4)

  let nsSigmoid ← benchmarkN n do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do
      let x ← Tensor.rand [size] .float32 seed
      sigmoid x)
    forceEval result
  printBench s!"sigmoid [{size}]" nsSigmoid (size * 3)

  let nsRelu ← benchmarkN n do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do
      let x ← Tensor.rand [size] .float32 seed
      relu x)
    forceEval result
  printBench s!"relu [{size}]" nsRelu size

  let nsGelu ← benchmarkN n do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do
      let x ← Tensor.rand [size] .float32 seed
      gelu x)
    forceEval result
  printBench s!"gelu [{size}]" nsGelu (size * 10)

/-! ## MLP Forward Pass -/

def benchMLP : IO Unit := do
  IO.println "\n=== MLP Forward Pass ==="

  let batch := 32
  let flops := 2 * batch * 784 * 128 + 2 * batch * 128 * 10
  let n := 5
  let seedRef ← IO.mkRef (0 : Nat)

  let ns ← benchmarkN n do
    let seed ← seedRef.modifyGet fun s => (s, s + 1)
    let result ← evalTensorCached (runTensorM do
      let x ← Tensor.rand [batch, 784] .float32 seed
      let w1 ← Tensor.rand [784, 128] .float32 (seed + 1)
      let w2 ← Tensor.rand [128, 10] .float32 (seed + 2)
      let h ← matmul x w1
      let h ← relu h
      matmul h w2)
    forceEval result
  printBench "MLP [32,784]->[32,10]" ns flops

/-! ## Run All -/

def runAll : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════════╗"
  IO.println "║            TinyGrad4 Performance Benchmarks                  ║"
  IO.println "╚══════════════════════════════════════════════════════════════╝"

  benchTensorCreation
  benchElementwise
  benchRawKernel
  benchReductions
  benchMatmul
  benchComposite
  benchMLP

  IO.println "\n=== Benchmark Complete ==="

end TinyGrad4.Test.Benchmark
