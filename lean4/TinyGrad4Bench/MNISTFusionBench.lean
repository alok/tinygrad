import TinyGrad4
import TinyGrad4Bench.KernelProfile
import LeanBenchWandb.Logger
import Wandb.Json
import Lean.Data.Json.Parser
-- Disable IO.monoNanosNow linter: benchmark timing uses raw monotonic clocks.
set_option linter.monoNanosNow false

/-!
# MNISTFusionBench

Measures the runtime impact of fusion on an MNIST-shaped MLP training step.

We compare:
- `fused`: `Interpreter.compileMany` (Phase C fusion selection + `.KERNEL` lowering)
- `node`: no fusion at all (every UOp evaluated as a separate node)

The benchmark uses *synthetic* data of MNIST shapes to isolate compute/fusion effects (no dataset IO).
-/

namespace TinyGrad4Bench.MNISTFusionBench

open TinyGrad4
open StaticTensor
open Interpreter
open Backend
open Std
open LeanBenchWandb
open Wandb.Json

private structure Program where
  w1Id : UOpId
  w2Id : UOpId
  xId : UOpId
  yId : UOpId
  loss : UOp
  newW1 : UOp
  newW2 : UOp

private def compileNoFusion (roots : List UOp) : Interpreter.Compiled := Id.run do
  let nodes := UOp.toposortMany roots
  let keepIds := roots.foldl (fun s r => UOpIdSet.add s r.uid) UOpIdSet.mkEmpty
  let mut implMap : HashMap UOpId Fusion.Impl := ∅
  for u in nodes do
    implMap := implMap.insert u.uid (.node (u.src.map (fun s => s.uid)))
  -- Build schedule: for no-fusion case, every node is executed directly (no KERNEL transform)
  let schedule := nodes.map fun u =>
    { ast := u, impl := none, tag := "node" : Interpreter.ExecItem }
  { roots, nodes, keepIds, implMap, schedule }

private def buildProgram (batchSize hidden : Nat) (lr : Float32) : Program := Id.run do
  let (w1Id, w2Id, xId, yId, lossUop, newW1Uop, newW2Uop) := runTensorM do
    let xBuf ← Tensor.buffer [batchSize, 784] .float32
    let yBuf ← Tensor.buffer [batchSize, 10] .float32
    let w1Buf ← Tensor.buffer [784, hidden] .float32
    let w2Buf ← Tensor.buffer [hidden, 10] .float32

    let h ← matmul xBuf w1Buf
    let hRelu ← relu h
    let logits ← matmul hRelu w2Buf
    let loss ← crossEntropyOneHot logits yBuf

    let gradMap ← backward loss [w1Buf.uop, w2Buf.uop]
    let gradW1 := gradMap.getD w1Buf.uop.uid w1Buf.uop
    let gradW2 := gradMap.getD w2Buf.uop.uid w2Buf.uop

    let lrConst ← UOp.const .float32 lr
    let stepW1 ← UOp.sub w1Buf.uop (← UOp.mul gradW1 lrConst)
    let stepW2 ← UOp.sub w2Buf.uop (← UOp.mul gradW2 lrConst)

    pure (w1Buf.uop.uid, w2Buf.uop.uid, xBuf.uop.uid, yBuf.uop.uid, loss.uop, stepW1, stepW2)

  { w1Id, w2Id, xId, yId, loss := lossUop, newW1 := newW1Uop, newW2 := newW2Uop }

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

private def mkOneHot (batchSize classes : Nat) : RawBuffer := Id.run do
  let mut out := FloatArray.emptyWithCapacity (batchSize * classes)
  for i in [:batchSize] do
    for j in [:classes] do
      let v : Float := if j == i % classes then 1.0 else 0.0
      out := out.push v
  RawBuffer.ofF32 out

private def mkInitEnv (p : Program) (batchSize hidden : Nat) : Env :=
  let x := { dtype := .float32, data := Native.fullF32Bits (batchSize * 784) ((0.01 : Float32).toBits) }
  let y := mkOneHot batchSize 10
  let w1 := { dtype := .float32, data := Native.fullF32Bits (784 * hidden) ((0.01 : Float32).toBits) }
  let w2 := { dtype := .float32, data := Native.fullF32Bits (hidden * 10) ((0.01 : Float32).toBits) }
  (∅ : Env)
    |>.insert p.xId x
    |>.insert p.yId y
    |>.insert p.w1Id w1
    |>.insert p.w2Id w2

private def getNatField (j : Lean.Json) (k : String) (fallback : Nat) : Nat :=
  match Lean.Json.getObjVal? j k with
  | Except.ok v =>
    match Lean.Json.getNat? v with
    | Except.ok n => n
    | _ => fallback
  | _ => fallback

private def costModelFromJson (j : Lean.Json) : Backend.CostModel :=
  let d := Backend.defaultCostModel
  { kernelOverhead := getNatField j "kernelOverhead" d.kernelOverhead
    elem := getNatField j "elem" d.elem
    moveElem := getNatField j "moveElem" d.moveElem
    memReadByte := getNatField j "memReadByte" d.memReadByte
    memWriteByte := getNatField j "memWriteByte" d.memWriteByte
    memReadViewByte := getNatField j "memReadViewByte" d.memReadViewByte
    memWriteViewByte := getNatField j "memWriteViewByte" d.memWriteViewByte
    reduceElem := getNatField j "reduceElem" d.reduceElem
    matmulMulAdd := getNatField j "matmulMulAdd" d.matmulMulAdd
    matmulViewMulAdd := getNatField j "matmulViewMulAdd" d.matmulViewMulAdd }

private def loadCostModelFromEnv (envVar : String := "TINYGRAD4_COST_MODEL") : IO Backend.CostModel := do
  match (← IO.getEnv envVar) with
  | none => pure Backend.defaultCostModel
  | some path =>
    let raw ← IO.FS.readFile path
    match Lean.Json.parse raw with
    | Except.error err => throw (IO.userError s!"cost model parse failed: {err}")
    | Except.ok j => pure (costModelFromJson j)

def runWith (batchSize hidden iters : Nat) (lr : Float32 := 0.01) : IO Unit := do
  if batchSize == 0 || hidden == 0 || iters == 0 then
    throw (IO.userError "batchSize/hidden/iters must be > 0")

  IO.println s!"=== MNISTFusionBench batch={batchSize} hidden={hidden} iters={iters} ==="
  let p := buildProgram batchSize hidden lr

  let cm ← loadCostModelFromEnv
  let compiledFused := Interpreter.compileMany [p.loss, p.newW1, p.newW2] (cm := cm)
  let compiledNode := compileNoFusion [p.loss, p.newW1, p.newW2]

  IO.println s!"selection: fused kernels={countKernelNodes compiledFused}, node kernels={countKernelNodes compiledNode}"
  IO.println s!"root tags: loss={tagOf compiledFused p.loss}, newW1={tagOf compiledFused p.newW1}, newW2={tagOf compiledFused p.newW2}"
  IO.println "fused kernel dump:"
  TinyGrad4Bench.dumpKernels compiledFused

  let sink ← IO.mkRef (0 : UInt32)
  let mix (b : RawBuffer) : IO Unit := do
    if b.data.size == 0 then
      sink.modify (· + 1)
    else
      let b0 := b.data.get! 0
      let b1 := b.data.get! (b.data.size - 1)
      let x : UInt32 := UInt32.ofNat (b0.toNat) + (UInt32.ofNat (b1.toNat) <<< 8)
      sink.modify (fun acc => acc + x + 1)

  let initEnv := mkInitEnv p batchSize hidden

  -- Warmup (avoid measuring first-run effects).
  let _ := Interpreter.evalCompiledRaw compiledFused initEnv
  let _ := Interpreter.evalCompiledRaw compiledNode initEnv
  let aggRuns := if iters < 5 then iters else 5
  TinyGrad4Bench.dumpKernelAggFromRuns compiledFused initEnv aggRuns
  match (← IO.getEnv "TINYGRAD4_COST_TRACE") with
  | some path =>
    TinyGrad4Bench.dumpKernelSamplesWithCost compiledFused initEnv aggRuns path
  | none => pure ()

  let w1RefFused ← IO.mkRef (initEnv.getD p.w1Id (RawBuffer.zeros .float32 0))
  let w2RefFused ← IO.mkRef (initEnv.getD p.w2Id (RawBuffer.zeros .float32 0))
  let actFused : IO Unit := do
    let env : Env := initEnv
      |>.insert p.w1Id (← w1RefFused.get)
      |>.insert p.w2Id (← w2RefFused.get)
    let cache := Interpreter.evalCompiledRaw compiledFused env
    let w1' := cache.getD p.newW1.uid (RawBuffer.zeros p.newW1.dtype (listProd p.newW1.shape))
    let w2' := cache.getD p.newW2.uid (RawBuffer.zeros p.newW2.dtype (listProd p.newW2.shape))
    w1RefFused.set w1'
    w2RefFused.set w2'
    let lossBuf := cache.getD p.loss.uid (RawBuffer.zeros p.loss.dtype (listProd p.loss.shape))
    mix lossBuf

  let w1RefNode ← IO.mkRef (initEnv.getD p.w1Id (RawBuffer.zeros .float32 0))
  let w2RefNode ← IO.mkRef (initEnv.getD p.w2Id (RawBuffer.zeros .float32 0))
  let actNode : IO Unit := do
    let env : Env := initEnv
      |>.insert p.w1Id (← w1RefNode.get)
      |>.insert p.w2Id (← w2RefNode.get)
    let cache := Interpreter.evalCompiledRaw compiledNode env
    let w1' := cache.getD p.newW1.uid (RawBuffer.zeros p.newW1.dtype (listProd p.newW1.shape))
    let w2' := cache.getD p.newW2.uid (RawBuffer.zeros p.newW2.dtype (listProd p.newW2.shape))
    w1RefNode.set w1'
    w2RefNode.set w2'
    let lossBuf := cache.getD p.loss.uid (RawBuffer.zeros p.loss.dtype (listProd p.loss.shape))
    mix lossBuf

  let trials := 5
  let fusedTimes ← TinyGrad4Bench.timeTrials "fused (compileMany + Phase C fusions)" trials iters actFused
  let nodeTimes ← TinyGrad4Bench.timeTrials "node (no fusion, per-UOp kernels)" trials iters actNode
  let fusedMed := TinyGrad4Bench.median fusedTimes
  let nodeMed := TinyGrad4Bench.median nodeTimes
  let ratio : Float := if fusedMed == 0.0 then 0.0 else nodeMed / fusedMed
  IO.println s!"median speedup: {ratio}x (node/fused)"
  let params := [
    LeanBenchWandb.paramNat "batch" batchSize,
    LeanBenchWandb.paramNat "hidden" hidden,
    LeanBenchWandb.paramNat "iters" iters
  ]
  LeanBenchWandb.logMetric "mnist_fusion" "fused_median_ms" fusedMed (unit? := some "ms") (params := params)
  LeanBenchWandb.logMetric "mnist_fusion" "node_median_ms" nodeMed (unit? := some "ms") (params := params)
  LeanBenchWandb.logMetric "mnist_fusion" "speedup" ratio (params := params)
  match (← LeanBenchWandb.fetchSummaryCurrent) with
  | some summary => IO.println s!"wandb summary: {Wandb.Json.render summary}"
  | none => pure ()
  if ratio < 0.95 then
    IO.println "WARNING: fused slower than node on median (regression candidate)"
  IO.println s!"sink: {← sink.get}"

def run : IO Unit := do
  runWith 32 128 10
  LeanBenchWandb.finishCurrent

end TinyGrad4Bench.MNISTFusionBench

-- Large benchmarks should run as executables, not compile-time #eval:
--   lake exe mnist_fusion_bench
-- The interpreter has limited stack during elaboration.
