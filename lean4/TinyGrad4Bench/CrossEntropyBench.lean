import TinyGrad4
-- Disable IO.monoNanosNow linter: benchmark timing uses raw monotonic clocks.
set_option linter.monoNanosNow false

/-!
# CrossEntropyBench

Micro-benchmark for log-softmax + cross-entropy (one-hot targets).

We compare:
- `fused`: `Interpreter.compileMany` (Phase C fusion selection + `.KERNEL` lowering)
- `node`: no fusion at all (every UOp evaluated as a separate node)
-/

namespace TinyGrad4Bench.CrossEntropyBench

open TinyGrad4
open StaticTensor
open Interpreter
open Backend
open Std

private structure Program where
  xId : UOpId
  yId : UOpId
  loss : UOp

private def compileNoFusion (roots : List UOp) : Interpreter.Compiled := Id.run do
  let nodes := UOp.toposortMany roots
  let keepIds := roots.foldl (fun s r => UOpIdSet.add s r.uid) UOpIdSet.mkEmpty
  let mut implMap : HashMap UOpId Fusion.Impl := ∅
  for u in nodes do
    implMap := implMap.insert u.uid (.node (u.src.map (fun s => s.uid)))
  { roots, nodes, keepIds, implMap }

private def buildProgram (batchSize classes : Nat) : Program := Id.run do
  let (xId, yId, lossUop) := runTensorM do
    let xBuf ← Tensor.buffer [batchSize, classes] .float32
    let yBuf ← Tensor.buffer [batchSize, classes] .float32
    let loss ← crossEntropyOneHot xBuf yBuf
    pure (xBuf.uop.uid, yBuf.uop.uid, loss.uop)
  { xId, yId, loss := lossUop }

private def mkOneHot (batchSize classes : Nat) : RawBuffer := Id.run do
  let mut out := FloatArray.emptyWithCapacity (batchSize * classes)
  for i in [:batchSize] do
    for j in [:classes] do
      let v : Float := if j == i % classes then 1.0 else 0.0
      out := out.push v
  RawBuffer.ofF32 out

private def mkInitEnv (p : Program) (batchSize classes : Nat) : Env :=
  let x := { dtype := .float32, data := Native.fullF32Bits (batchSize * classes) ((0.01 : Float32).toBits) }
  let y := mkOneHot batchSize classes
  (∅ : Env)
    |>.insert p.xId x
    |>.insert p.yId y

private def countKernelNodes (c : Interpreter.Compiled) : Nat :=
  c.nodes.foldl (init := 0) fun acc u => if u.op == .KERNEL then acc + 1 else acc

private def tagOf (c : Interpreter.Compiled) (u : UOp) : String :=
  match c.implMap[u.uid]? with
  | some impl => impl.tag
  | none => "node"

private def timeIt (label : String) (iters : Nat) (act : IO Unit) : IO Unit := do
  let start ← IO.monoNanosNow
  for _ in [:iters] do
    act
  let stop ← IO.monoNanosNow
  let dtNs : Nat := stop - start
  let totalMs : Float := (Float.ofNat dtNs) / 1.0e6
  let perMs : Float := totalMs / (Float.ofNat iters)
  IO.println s!"{label}: {perMs} ms/iter ({totalMs} ms total, iters={iters})"

def runWith (batchSize classes iters : Nat) : IO Unit := do
  if batchSize == 0 || classes == 0 || iters == 0 then
    throw (IO.userError "batchSize/classes/iters must be > 0")

  IO.println s!"=== CrossEntropyBench batch={batchSize} classes={classes} iters={iters} ==="
  let p := buildProgram batchSize classes

  let compiledFused := Interpreter.compileMany [p.loss]
  let compiledNode := compileNoFusion [p.loss]

  IO.println s!"selection: fused kernels={countKernelNodes compiledFused}, node kernels={countKernelNodes compiledNode}"
  IO.println s!"root tags: loss={tagOf compiledFused p.loss}"

  let sink ← IO.mkRef (0 : UInt32)
  let mix (b : RawBuffer) : IO Unit := do
    if b.data.size == 0 then
      sink.modify (· + 1)
    else
      let b0 := b.data.get! 0
      let b1 := b.data.get! (b.data.size - 1)
      let x : UInt32 := UInt32.ofNat (b0.toNat) + (UInt32.ofNat (b1.toNat) <<< 8)
      sink.modify (fun acc => acc + x + 1)

  let initEnv := mkInitEnv p batchSize classes

  let _ := Interpreter.evalCompiledRaw compiledFused initEnv
  let _ := Interpreter.evalCompiledRaw compiledNode initEnv

  let actFused : IO Unit := do
    let cache := Interpreter.evalCompiledRaw compiledFused initEnv
    let lossBuf := cache.getD p.loss.uid (RawBuffer.zeros p.loss.dtype (listProd p.loss.shape))
    mix lossBuf

  let actNode : IO Unit := do
    let cache := Interpreter.evalCompiledRaw compiledNode initEnv
    let lossBuf := cache.getD p.loss.uid (RawBuffer.zeros p.loss.dtype (listProd p.loss.shape))
    mix lossBuf

  timeIt "fused (compileMany + Phase C fusions)" iters actFused
  timeIt "node (no fusion, per-UOp kernels)" iters actNode
  IO.println s!"sink: {← sink.get}"

def run : IO Unit := do
  runWith 512 10 50

end TinyGrad4Bench.CrossEntropyBench

#eval! TinyGrad4Bench.CrossEntropyBench.run
