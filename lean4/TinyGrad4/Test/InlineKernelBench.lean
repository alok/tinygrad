import Float64
import TinyGrad4.Kernel.Inline
import TinyGrad4.Kernel.InlineMetal

/-!
# InlineKernelBench

Runtime performance of `kernel!`-generated kernels on Metal.

Measures steady-state dispatch (buffers resident, shader compiled once):
per-iteration latency and effective bandwidth. Also times the per-element
native `fn` path on CPU for scale. Copies are excluded — this is the kernel
cost the compile-time codegen actually controls.
-/

set_option linter.useRawBuffer false

namespace TinyGrad4.Test.InlineKernelBench

open TinyGrad4 Kernel Kernel.Inline
open TinyGrad4.Backend
open TinyGrad4.Backend.Metal

def saxpy := kernel! "bench_saxpy" fun a x y => a * x + y
def sigmoidish := kernel! "bench_sigmoidish" fun x => recip (1.0 + exp2 (-1.442695 * x))

private def packF32 (data : Array Float64) : ByteArray :=
  Native.packF32FromF64 ⟨data⟩

/-- Steady-state GPU benchmark: alloc+copy once, launch `iters` times, sync once. -/
def benchGpu (name : String) (source : String) (arity numel iters : Nat) : IO Unit := do
  let data := (Array.range numel).map fun i => (Float64.ofNat (i % 1000)) / 500.0 - 1.0
  let bytes := packF32 data
  let mut bufs : Array MetalBuffer := #[]
  for _ in [:arity] do
    let b ← metalAllocBytes bytes.size
    metalCopyInBytes b bytes
    bufs := bufs.push b
  let out ← metalAllocBytes bytes.size
  bufs := bufs.push out
  let prog ← MetalEwise.getOrCompile name source
  -- warmup
  metalLaunch prog bufs numel 1 1 256 1 1
  metalSync
  let start ← IO.monoNanosNow
  for _ in [:iters] do
    metalLaunch prog bufs numel 1 1 256 1 1
  metalSync
  let stop ← IO.monoNanosNow
  let usPerIter : Float64 := (stop - start).toFloat / 1000.0 / Float64.ofNat iters
  let bytesMoved : Float64 := Float64.ofNat ((arity + 1) * 4 * numel)
  let gbps : Float64 := bytesMoved / ((stop - start).toFloat / Float64.ofNat iters)
  IO.println s!"  {name}: {usPerIter} μs/iter, {gbps} GB/s ({numel} elems, {iters} iters)"
  for b in bufs do metalFree b

/-- CPU per-element `fn` loop (interpreted native path, for scale). -/
def benchCpuFn (numel iters : Nat) : IO Unit := do
  let xs := (Array.range numel).map fun i => ((Float64.ofNat (i % 1000)) / 500.0 - 1.0).toFloat32
  let start ← IO.monoNanosNow
  let mut acc : Float32 := 0.0
  for _ in [:iters] do
    for i in [:numel] do
      let x := xs[i]!
      acc := acc + sigmoidish.fn (fun _ => x)
  let stop ← IO.monoNanosNow
  let usPerIter : Float64 := (stop - start).toFloat / 1000.0 / Float64.ofNat iters
  IO.println s!"  cpu fn sigmoidish: {usPerIter} μs/iter ({numel} elems; checksum {acc.toFloat})"

def runAll : IO Unit := do
  IO.println "InlineKernelBench (steady-state dispatch, copies excluded):"
  let available ← Metal.isAvailable
  if available then
    let numel := 1 <<< 20
    benchGpu "bench_saxpy" saxpy.metal 3 numel 200
    benchGpu "bench_sigmoidish" sigmoidish.metal 1 numel 200
  else
    IO.println "  (Metal not available)"
  benchCpuFn (1 <<< 16) 10

end TinyGrad4.Test.InlineKernelBench

def main : IO Unit := TinyGrad4.Test.InlineKernelBench.runAll
