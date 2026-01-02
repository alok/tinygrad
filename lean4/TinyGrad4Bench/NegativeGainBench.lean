import TinyGrad4
-- Disable IO.monoNanosNow linter: benchmark timing uses raw monotonic clocks.
set_option linter.monoNanosNow false

/-!
# NegativeGainBench

Empirically checks whether Phase C candidates with non-positive estimated gain (`gainTime ≤ 0`)
are actually slower at runtime.

We:
- build an MNIST-shaped MLP training step graph (synthetic data, no dataset IO),
- use `Backend.Fusion.report` to find negative-gain candidates,
- for a handful of the "most negative" ones, force-apply *only that one* candidate (everything else is `.node`),
- time the full training step vs a baseline "no fusion" run.

This is a sanity-check for the `gainTime > 0` filter in Phase C selection.
-/

namespace TinyGrad4Bench.NegativeGainBench

open TinyGrad4
open StaticTensor
open Interpreter
open Backend
open Std

private structure Program where
  w1Id : UOpId
  w2Id : UOpId
  xId : UOpId
  yId : UOpId
  loss : UOp
  newW1 : UOp
  newW2 : UOp

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

private def mkInitEnv (p : Program) (batchSize hidden : Nat) : Env :=
  let x := { dtype := .float32, data := Native.fullF32Bits (batchSize * 784) ((0.01 : Float32).toBits) }
  let y := { dtype := .float32, data := Native.fullF32Bits (batchSize * 10) ((0.0 : Float32).toBits) }
  let w1 := { dtype := .float32, data := Native.fullF32Bits (784 * hidden) ((0.01 : Float32).toBits) }
  let w2 := { dtype := .float32, data := Native.fullF32Bits (hidden * 10) ((0.01 : Float32).toBits) }
  (∅ : Env)
    |>.insert p.xId x
    |>.insert p.yId y
    |>.insert p.w1Id w1
    |>.insert p.w2Id w2

private def compileNoFusion (roots : List UOp) : Interpreter.Compiled := Id.run do
  let nodes := UOp.toposortMany roots
  let keepIds := roots.foldl (fun s r => UOpIdSet.add s r.uid) UOpIdSet.mkEmpty
  let mut implMap : HashMap UOpId Fusion.Impl := ∅
  for u in nodes do
    implMap := implMap.insert u.uid (.node (u.src.map (fun s => s.uid)))
  { roots, nodes, keepIds, implMap }

private def buildNodeMap (nodes : List UOp) : HashMap UOpId UOp :=
  nodes.foldl (init := (∅ : HashMap UOpId UOp)) fun m u => m.insert u.uid u

private def compileWithOverrides (roots : List UOp) (overrides : HashMap UOpId Fusion.Impl) : Interpreter.Compiled := Id.run do
  let nodes0 := UOp.toposortMany roots
  let keepIds := roots.foldl (fun s r => UOpIdSet.add s r.uid) UOpIdSet.mkEmpty
  let nodeMap := buildNodeMap nodes0

  let mut implMap : HashMap UOpId Fusion.Impl := ∅
  for u in nodes0 do
    let dflt : Fusion.Impl := .node (u.src.map (fun s => s.uid))
    implMap := implMap.insert u.uid (overrides.getD u.uid dflt)

  let mut kernMap : HashMap UOpId UOp := ∅
  for u in nodes0 do
    let impl := implMap.getD u.uid (.node (u.src.map (fun s => s.uid)))
    let depUids := impl.deps
    let src' : List UOp := depUids.map fun sid =>
      match kernMap[sid]? with
      | some v => v
      | none => nodeMap.getD sid default

    let u' : UOp :=
      match impl with
      | .node _ => { u with src := src' }
      | _ => { u with op := .KERNEL, src := src', arg := .empty }
    kernMap := kernMap.insert u.uid u'

  let roots' := roots.map fun r => kernMap.getD r.uid r
  let nodes := UOp.toposortMany roots'
  { roots := roots', nodes, keepIds, implMap }

private def setEq (a b : UOpIdSet) : Bool :=
  a.size == b.size && a.fold (init := true) (fun ok uid => ok && UOpIdSet.member b uid)

private def findImplFor (u : UOp) (ci : Fusion.CandidateInfo) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : Option Fusion.Impl := Id.run do
  if u.uid != ci.root then
    return none
  let imps := Fusion.candidates u keep refCnt
  let mut out : Option Fusion.Impl := none
  for imp in imps do
    let ok :=
      imp.tag == ci.tag &&
      imp.deps == ci.deps &&
      setEq (imp.cover u.uid) ci.cover &&
      imp.score == ci.score
    if ok then
      out := some imp
  return out

private def timeItNs (iters : Nat) (act : IO Unit) : IO Nat := do
  let start ← IO.monoNanosNow
  for _ in [:iters] do
    act
  let stop ← IO.monoNanosNow
  pure (stop - start)

private def benchTrainingStepNs (iters : Nat) (p : Program) (compiled : Interpreter.Compiled) (initEnv : Env) (sink : IO.Ref UInt32)
    : IO Nat := do
  let w1Init := initEnv.getD p.w1Id (RawBuffer.zeros .float32 0)
  let w2Init := initEnv.getD p.w2Id (RawBuffer.zeros .float32 0)
  let w1Ref ← IO.mkRef w1Init
  let w2Ref ← IO.mkRef w2Init

  let mix (b : RawBuffer) : IO Unit := do
    if b.data.size == 0 then
      sink.modify (· + 1)
    else
      let b0 := b.data.get! 0
      let b1 := b.data.get! (b.data.size - 1)
      let x : UInt32 := UInt32.ofNat (b0.toNat) + (UInt32.ofNat (b1.toNat) <<< 8)
      sink.modify (fun acc => acc + x + 1)

  let act : IO Unit := do
    let env : Env := initEnv
      |>.insert p.w1Id (← w1Ref.get)
      |>.insert p.w2Id (← w2Ref.get)
    let cache := Interpreter.evalCompiledRaw compiled env
    let w1' := cache.getD p.newW1.uid (RawBuffer.zeros p.newW1.dtype (listProd p.newW1.shape))
    let w2' := cache.getD p.newW2.uid (RawBuffer.zeros p.newW2.dtype (listProd p.newW2.shape))
    w1Ref.set w1'
    w2Ref.set w2'
    let lossBuf := cache.getD p.loss.uid (RawBuffer.zeros p.loss.dtype (listProd p.loss.shape))
    mix lossBuf

  -- Warmup.
  act
  timeItNs iters act

def run : IO Unit := do
  let batchSize := 32
  let hidden := 128
  let lr : Float32 := 0.01
  let iters := 30
  let maxCands := 6

  IO.println s!"=== NegativeGainBench batch={batchSize} hidden={hidden} iters={iters} ==="
  let p := buildProgram batchSize hidden lr
  let roots : List UOp := [p.loss, p.newW1, p.newW2]
  let initEnv := mkInitEnv p batchSize hidden

  let nodes := UOp.toposortMany roots
  let nodeMap := buildNodeMap nodes
  let keepIds := roots.foldl (fun s r => UOpIdSet.add s r.uid) UOpIdSet.mkEmpty
  let refCnt : HashMap UOpId Nat := Id.run do
    let mut refCnt : HashMap UOpId Nat := ∅
    for u in nodes do
      for s in u.src do
        refCnt := refCnt.insert s.uid (refCnt.getD s.uid 0 + 1)
    return refCnt

  let report := Fusion.report roots (cm := Backend.defaultCostModel)
  let neg0 := report.candidates.filter fun ci => ci.tag != "node" && ci.gainTime <= 0
  let neg := neg0.qsort (fun a b => a.gainTime < b.gainTime)

  IO.println s!"negative candidates (gainTime ≤ 0): {neg.size} (testing {min maxCands neg.size})"

  let baseline := compileNoFusion roots
  let sink ← IO.mkRef (0 : UInt32)
  let baseNs ← benchTrainingStepNs iters p baseline initEnv sink
  IO.println s!"baseline (no fusion): {(Float.ofNat baseNs) / 1.0e6 / (Float.ofNat iters)} ms/iter"

  let mut i : Nat := 0
  for ci in neg do
    if i >= maxCands then
      break
    let u := nodeMap.getD ci.root default
    match findImplFor u ci keepIds refCnt with
    | none =>
      IO.println s!"[{i}] {ci.tag}: could not reconstruct candidate impl (root={ci.root})"
    | some impl =>
      let overrides : HashMap UOpId Fusion.Impl := (∅ : HashMap UOpId Fusion.Impl) |>.insert ci.root impl
      let compiled := compileWithOverrides roots overrides
      let ns ← benchTrainingStepNs iters p compiled initEnv sink
      let baseMs : Float := (Float.ofNat baseNs) / 1.0e6 / (Float.ofNat iters)
      let thisMs : Float := (Float.ofNat ns) / 1.0e6 / (Float.ofNat iters)
      let ratio : Float := if baseMs == 0 then 0 else thisMs / baseMs
      IO.println s!"[{i}] {ci.tag} gainTime={ci.gainTime} cover={ci.cover.size} -> {thisMs} ms/iter ({ratio}x baseline)"
    i := i + 1

  IO.println s!"sink: {← sink.get}"

end TinyGrad4Bench.NegativeGainBench

#eval! TinyGrad4Bench.NegativeGainBench.run
