import TinyGrad4
-- Disable IO.monoNanosNow linter: benchmark timing uses raw monotonic clocks.
set_option linter.monoNanosNow false

/-!
# RuntimeChoiceBench

Portable CPU microbench to test whether Phase C's cost-based choice matches *measured* runtime:

- `*.view`: virtualize movement ops as strided/masked `View` loads
- `*.fast` / `*.fast.v`: materialize movement ops and then use fast contiguous/broadcast kernels

We compile the same graph twice with different `CostModel`s and time `Interpreter.evalCompiledRaw`.
-/

namespace TinyGrad4Bench.RuntimeChoiceBench

open TinyGrad4
open StaticTensor
open Interpreter
open Backend

private def timeIt (label : String) (iters : Nat) (act : IO Unit) : IO Unit := do
  let start ← IO.monoNanosNow
  for _ in [:iters] do
    act
  let stop ← IO.monoNanosNow
  let dtNs : Nat := stop - start
  let totalMs : Float := (Float.ofNat dtNs) / 1.0e6
  let perMs : Float := totalMs / (Float.ofNat iters)
  IO.println s!"{label}: {perMs} ms/iter ({totalMs} ms total, iters={iters})"

private def pickTag (root : UOp) (c : Interpreter.Compiled) : String :=
  match c.implMap[root.uid]? with
  | some impl => impl.tag
  | none => "node"

private def buildGraph (n : Nat) : UOpId × UOpId × UOp := Id.run do
  let (xId, bId, out) := runTensorM do
    let x ← Tensor.buffer [n, n] .float32
    let b ← Tensor.buffer [1, n] .float32

    let xp ← StaticTensor.permute x [1, 0]
    let be ← StaticTensor.expand b [n, n]

    let y ← UOp.add xp.uop be.uop
    let y ← UOp.exp2 y
    let y ← UOp.sqrt y
    let y ← UOp.recip y
    let y ← UOp.sin y
    pure (x.uop.uid, b.uop.uid, y)
  (xId, bId, out)

private def mkEnv (xId bId : UOpId) (n : Nat) : Env :=
  let x := { dtype := .float32, data := Native.fullF32Bits (n * n) ((0.01 : Float).toFloat32.toBits) }
  let b := { dtype := .float32, data := Native.fullF32Bits n ((0.02 : Float).toFloat32.toBits) }
  (∅ : Env)
    |>.insert xId x
    |>.insert bId b

def run : IO Unit := do
  let n := 256
  let iters := 200
  let (xId, bId, out) := buildGraph n
  let env := mkEnv xId bId n

  let cm0 := Backend.defaultCostModel
  let cmViewHate : Backend.CostModel :=
    { cm0 with
      memReadViewByte := cm0.memReadByte * 1000
      memWriteViewByte := cm0.memWriteByte * 1000 }

  let c0 := Interpreter.compileMany [out] (cm := cm0)
  let c1 := Interpreter.compileMany [out] (cm := cmViewHate)
  IO.println s!"selection: default={pickTag out c0}, viewHate={pickTag out c1}"

  let sink ← IO.mkRef (0 : UInt32)
  let mix (bytes : ByteArray) : IO Unit := do
    if bytes.size == 0 then
      sink.modify (· + 1)
    else
      let b0 := bytes.get! 0
      let b1 := bytes.get! (bytes.size - 1)
      let x : UInt32 := UInt32.ofNat (b0.toNat) + (UInt32.ofNat (b1.toNat) <<< 8)
      sink.modify (fun acc => acc + x + 1)

  let act0 : IO Unit := do
    let cache := Interpreter.evalCompiledRaw c0 env
    let outBuf := cache.getD out.uid (RawBuffer.zeros out.dtype (listProd out.shape))
    mix outBuf.data

  let act1 : IO Unit := do
    let cache := Interpreter.evalCompiledRaw c1 env
    let outBuf := cache.getD out.uid (RawBuffer.zeros out.dtype (listProd out.shape))
    mix outBuf.data

  IO.println s!"=== RuntimeChoiceBench n={n} ==="
  timeIt "default (compileMany defaultCostModel)" iters act0
  timeIt "viewHate (memReadViewByte*1000)" iters act1
  IO.println s!"sink: {← sink.get}"

end TinyGrad4Bench.RuntimeChoiceBench

#eval! TinyGrad4Bench.RuntimeChoiceBench.run
