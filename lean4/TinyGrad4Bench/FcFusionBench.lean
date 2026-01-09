import TinyGrad4
-- Disable IO.monoNanosNow linter: benchmark timing uses raw monotonic clocks.
set_option linter.monoNanosNow false

/-!
# FcFusionBench

Fusion vs. node evaluation on a 2-layer fully connected network with bias.
Uses synthetic data to isolate compute/fusion effects.
-/

namespace TinyGrad4Bench.FcFusionBench

open TinyGrad4
open StaticTensor
open Interpreter
open Backend
open Std

private structure Program where
  w1Id : UOpId
  w2Id : UOpId
  b1Id : UOpId
  b2Id : UOpId
  xId : UOpId
  yId : UOpId
  loss : UOp
  newW1 : UOp
  newW2 : UOp
  newB1 : UOp
  newB2 : UOp

private def compileNoFusion (roots : List UOp) : Interpreter.Compiled := Id.run do
  let nodes := UOp.toposortMany roots
  let keepIds := roots.foldl (fun s r => UOpIdSet.add s r.uid) UOpIdSet.mkEmpty
  let mut implMap : HashMap UOpId Fusion.Impl := ∅
  for u in nodes do
    implMap := implMap.insert u.uid (.node (u.src.map (fun s => s.uid)))
  { roots, nodes, keepIds, implMap }

private def buildProgram (batch inDim hidden outDim : Nat) (lr : Float32) : Program := Id.run do
  let (w1Id, w2Id, b1Id, b2Id, xId, yId, lossUop, newW1Uop, newW2Uop, newB1Uop, newB2Uop) := runTensorM do
    let xBuf ← Tensor.buffer [batch, inDim] .float32
    let yBuf ← Tensor.buffer [batch, outDim] .float32
    let w1Buf ← Tensor.buffer [inDim, hidden] .float32
    let w2Buf ← Tensor.buffer [hidden, outDim] .float32
    let b1Buf ← Tensor.buffer [hidden] .float32
    let b2Buf ← Tensor.buffer [outDim] .float32

    let h ← linearBias xBuf w1Buf b1Buf
    let hRelu ← relu h
    let logits ← linearBias hRelu w2Buf b2Buf

    let diff ← sub logits yBuf
    let sq ← mul diff diff
    let loss ← mean sq

    let gradMap ← backward loss [w1Buf.uop, w2Buf.uop, b1Buf.uop, b2Buf.uop]
    let gradW1 := gradMap.getD w1Buf.uop.uid w1Buf.uop
    let gradW2 := gradMap.getD w2Buf.uop.uid w2Buf.uop
    let gradB1 := gradMap.getD b1Buf.uop.uid b1Buf.uop
    let gradB2 := gradMap.getD b2Buf.uop.uid b2Buf.uop

    let lrConst ← UOp.const .float32 lr
    let stepW1 ← UOp.sub w1Buf.uop (← UOp.mul gradW1 lrConst)
    let stepW2 ← UOp.sub w2Buf.uop (← UOp.mul gradW2 lrConst)
    let stepB1 ← UOp.sub b1Buf.uop (← UOp.mul gradB1 lrConst)
    let stepB2 ← UOp.sub b2Buf.uop (← UOp.mul gradB2 lrConst)

    pure (w1Buf.uop.uid, w2Buf.uop.uid, b1Buf.uop.uid, b2Buf.uop.uid, xBuf.uop.uid, yBuf.uop.uid,
      loss.uop, stepW1, stepW2, stepB1, stepB2)

  { w1Id, w2Id, b1Id, b2Id, xId, yId, loss := lossUop, newW1 := newW1Uop, newW2 := newW2Uop,
    newB1 := newB1Uop, newB2 := newB2Uop }

private def countKernelNodes (c : Interpreter.Compiled) : Nat :=
  c.nodes.foldl (init := 0) fun acc u => if u.op == .KERNEL then acc + 1 else acc

private def tagOf (c : Interpreter.Compiled) (u : UOp) : String :=
  match c.implMap[u.uid]? with
  | some impl => impl.tag
  | none => "node"

private def tagCounts (c : Interpreter.Compiled) : List (String × Nat) :=
  let m : HashMap String Nat := c.nodes.foldl (init := (∅ : HashMap String Nat)) fun acc u =>
    let t := tagOf c u
    acc.insert t (acc.getD t 0 + 1)
  m.toList

private def timeIt (label : String) (iters : Nat) (act : IO Unit) : IO Unit := do
  let start ← IO.monoNanosNow
  for _ in [:iters] do
    act
  let stop ← IO.monoNanosNow
  let dtNs : Nat := stop - start
  let totalMs : Float := (Float.ofNat dtNs) / 1.0e6
  let perMs : Float := totalMs / (Float.ofNat iters)
  IO.println s!"{label}: {perMs} ms/iter ({totalMs} ms total, iters={iters})"

private def mkInitEnv (p : Program) (batch inDim hidden outDim : Nat) : Env :=
  let x := { dtype := .float32, data := Native.fullF32Bits (batch * inDim) ((0.01 : Float32).toBits) }
  let y := { dtype := .float32, data := Native.fullF32Bits (batch * outDim) ((0.0 : Float32).toBits) }
  let w1 := { dtype := .float32, data := Native.fullF32Bits (inDim * hidden) ((0.01 : Float32).toBits) }
  let w2 := { dtype := .float32, data := Native.fullF32Bits (hidden * outDim) ((0.01 : Float32).toBits) }
  let b1 := { dtype := .float32, data := Native.fullF32Bits hidden ((0.0 : Float32).toBits) }
  let b2 := { dtype := .float32, data := Native.fullF32Bits outDim ((0.0 : Float32).toBits) }
  (∅ : Env)
    |>.insert p.xId x
    |>.insert p.yId y
    |>.insert p.w1Id w1
    |>.insert p.w2Id w2
    |>.insert p.b1Id b1
    |>.insert p.b2Id b2

def runWith (batch inDim hidden outDim iters : Nat) (lr : Float32 := 0.01) : IO Unit := do
  if batch == 0 || inDim == 0 || hidden == 0 || outDim == 0 || iters == 0 then
    throw (IO.userError "dims/iters must be > 0")

  IO.println s!"=== FcFusionBench batch={batch} in={inDim} hidden={hidden} out={outDim} iters={iters} ==="
  let p := buildProgram batch inDim hidden outDim lr

  let roots : List UOp := [p.loss, p.newW1, p.newW2, p.newB1, p.newB2]
  let compiledFused := Interpreter.compileMany roots
  let compiledNode := compileNoFusion roots

  IO.println s!"selection: fused kernels={countKernelNodes compiledFused}, node kernels={countKernelNodes compiledNode}"
  IO.println s!"node counts: fused={compiledFused.nodes.length}, node={compiledNode.nodes.length}"
  IO.println s!"tags (fused): {tagCounts compiledFused}"
  let rootTags :=
    s!"root tags: loss={tagOf compiledFused p.loss}, newW1={tagOf compiledFused p.newW1}, " ++
    s!"newW2={tagOf compiledFused p.newW2}, newB1={tagOf compiledFused p.newB1}, newB2={tagOf compiledFused p.newB2}"
  IO.println rootTags

  let sink ← IO.mkRef (0 : UInt32)
  let mix (b : RawBuffer) : IO Unit := do
    if b.data.size == 0 then
      sink.modify (· + 1)
    else
      let b0 := b.data.get! 0
      let b1 := b.data.get! (b.data.size - 1)
      let x : UInt32 := UInt32.ofNat (b0.toNat) + (UInt32.ofNat (b1.toNat) <<< 8)
      sink.modify (fun acc => acc + x + 1)

  let initEnv := mkInitEnv p batch inDim hidden outDim
  let _ := Interpreter.evalCompiledRaw compiledFused initEnv
  let _ := Interpreter.evalCompiledRaw compiledNode initEnv

  let w1RefFused ← IO.mkRef (initEnv.getD p.w1Id (RawBuffer.zeros .float32 0))
  let w2RefFused ← IO.mkRef (initEnv.getD p.w2Id (RawBuffer.zeros .float32 0))
  let b1RefFused ← IO.mkRef (initEnv.getD p.b1Id (RawBuffer.zeros .float32 0))
  let b2RefFused ← IO.mkRef (initEnv.getD p.b2Id (RawBuffer.zeros .float32 0))
  let actFused : IO Unit := do
    let env : Env := initEnv
      |>.insert p.w1Id (← w1RefFused.get)
      |>.insert p.w2Id (← w2RefFused.get)
      |>.insert p.b1Id (← b1RefFused.get)
      |>.insert p.b2Id (← b2RefFused.get)
    let cache := Interpreter.evalCompiledRaw compiledFused env
    let w1' := cache.getD p.newW1.uid (RawBuffer.zeros p.newW1.dtype (listProd p.newW1.shape))
    let w2' := cache.getD p.newW2.uid (RawBuffer.zeros p.newW2.dtype (listProd p.newW2.shape))
    let b1' := cache.getD p.newB1.uid (RawBuffer.zeros p.newB1.dtype (listProd p.newB1.shape))
    let b2' := cache.getD p.newB2.uid (RawBuffer.zeros p.newB2.dtype (listProd p.newB2.shape))
    w1RefFused.set w1'
    w2RefFused.set w2'
    b1RefFused.set b1'
    b2RefFused.set b2'
    let lossBuf := cache.getD p.loss.uid (RawBuffer.zeros p.loss.dtype (listProd p.loss.shape))
    mix lossBuf

  let w1RefNode ← IO.mkRef (initEnv.getD p.w1Id (RawBuffer.zeros .float32 0))
  let w2RefNode ← IO.mkRef (initEnv.getD p.w2Id (RawBuffer.zeros .float32 0))
  let b1RefNode ← IO.mkRef (initEnv.getD p.b1Id (RawBuffer.zeros .float32 0))
  let b2RefNode ← IO.mkRef (initEnv.getD p.b2Id (RawBuffer.zeros .float32 0))
  let actNode : IO Unit := do
    let env : Env := initEnv
      |>.insert p.w1Id (← w1RefNode.get)
      |>.insert p.w2Id (← w2RefNode.get)
      |>.insert p.b1Id (← b1RefNode.get)
      |>.insert p.b2Id (← b2RefNode.get)
    let cache := Interpreter.evalCompiledRaw compiledNode env
    let w1' := cache.getD p.newW1.uid (RawBuffer.zeros p.newW1.dtype (listProd p.newW1.shape))
    let w2' := cache.getD p.newW2.uid (RawBuffer.zeros p.newW2.dtype (listProd p.newW2.shape))
    let b1' := cache.getD p.newB1.uid (RawBuffer.zeros p.newB1.dtype (listProd p.newB1.shape))
    let b2' := cache.getD p.newB2.uid (RawBuffer.zeros p.newB2.dtype (listProd p.newB2.shape))
    w1RefNode.set w1'
    w2RefNode.set w2'
    b1RefNode.set b1'
    b2RefNode.set b2'
    let lossBuf := cache.getD p.loss.uid (RawBuffer.zeros p.loss.dtype (listProd p.loss.shape))
    mix lossBuf

  timeIt "fused (compileMany + Phase C fusions)" iters actFused
  timeIt "node (no fusion, per-UOp kernels)" iters actNode
  IO.println s!"sink: {← sink.get}"

end TinyGrad4Bench.FcFusionBench

#eval! TinyGrad4Bench.FcFusionBench.runWith 128 784 1024 10 5
