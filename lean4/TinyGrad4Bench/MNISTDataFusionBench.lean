import TinyGrad4
import TinyGrad4.Data.MNIST
import TinyGrad4Bench.KernelProfile
-- Disable IO.monoNanosNow linter: benchmark timing uses raw monotonic clocks.
set_option linter.monoNanosNow false

/-!
# MNISTDataFusionBench

End-to-end (CPU portable) timing on *real MNIST data* for an MNIST MLP training step.

We measure only the compute loop (data is loaded + packed to float32 buffers upfront).

Compare:
- `fused`: `Interpreter.compileMany` (Phase C fusion selection + `.KERNEL` lowering)
- `node`: no fusion (every UOp evaluated as a separate node)
-/

namespace TinyGrad4Bench.MNISTDataFusionBench

open TinyGrad4
open TinyGrad4.Data.MNIST
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

private def compileNoFusion (roots : List UOp) : Interpreter.Compiled := Id.run do
  let nodes := UOp.toposortMany roots
  let keepIds := roots.foldl (fun s r => UOpIdSet.add s r.uid) UOpIdSet.mkEmpty
  let mut implMap : HashMap UOpId Fusion.Impl := ∅
  for u in nodes do
    implMap := implMap.insert u.uid (.node (u.src.map (fun s => s.uid)))
  { roots, nodes, keepIds, implMap }

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

private def tagOf (c : Interpreter.Compiled) (u : UOp) : String :=
  match c.implMap[u.uid]? with
  | some impl => impl.tag
  | none => "node"

private def mkInitWeights (hidden : Nat) : RawBuffer × RawBuffer :=
  let w1 := { dtype := .float32, data := Native.fullF32Bits (784 * hidden) ((0.01 : Float32).toBits) }
  let w2 := { dtype := .float32, data := Native.fullF32Bits (hidden * 10) ((0.01 : Float32).toBits) }
  (w1, w2)

private def timeIt (label : String) (act : IO Unit) : IO Nat := do
  let start ← IO.monoNanosNow
  act
  let stop ← IO.monoNanosNow
  let dtNs : Nat := stop - start
  IO.println s!"{label}: {(Float.ofNat dtNs) / 1.0e6} ms"
  pure dtNs

private def runSteps (p : Program) (compiled : Interpreter.Compiled) (batches : Array (RawBuffer × RawBuffer)) (hidden : Nat) : IO Float := do
  let (w1Init, w2Init) := mkInitWeights hidden
  let w1Ref ← IO.mkRef w1Init
  let w2Ref ← IO.mkRef w2Init

  let sink ← IO.mkRef (0 : UInt32)
  let mixLoss (b : RawBuffer) : IO Unit := do
    if b.data.size == 0 then
      sink.modify (· + 1)
    else
      let b0 := b.data.get! 0
      let b1 := b.data.get! (b.data.size - 1)
      let x : UInt32 := UInt32.ofNat (b0.toNat) + (UInt32.ofNat (b1.toNat) <<< 8)
      sink.modify (fun acc => acc + x + 1)

  let act : IO Unit := do
    for (xBuf, yBuf) in batches do
      let env : Env := (∅ : Env)
        |>.insert p.xId xBuf
        |>.insert p.yId yBuf
        |>.insert p.w1Id (← w1Ref.get)
        |>.insert p.w2Id (← w2Ref.get)
      let cache := Interpreter.evalCompiledRaw compiled env
      let w1' := cache.getD p.newW1.uid (RawBuffer.zeros p.newW1.dtype (listProd p.newW1.shape))
      let w2' := cache.getD p.newW2.uid (RawBuffer.zeros p.newW2.dtype (listProd p.newW2.shape))
      w1Ref.set w1'
      w2Ref.set w2'
      let lossBuf := cache.getD p.loss.uid (RawBuffer.zeros p.loss.dtype (listProd p.loss.shape))
      mixLoss lossBuf

  -- Warmup (avoid first-run noise).
  match batches.toList with
  | [] => pure ()
  | (xBuf, yBuf) :: _ =>
    let env : Env := (∅ : Env)
      |>.insert p.xId xBuf
      |>.insert p.yId yBuf
      |>.insert p.w1Id w1Init
      |>.insert p.w2Id w2Init
    let cache := Interpreter.evalCompiledRaw compiled env
    let lossBuf := cache.getD p.loss.uid (RawBuffer.zeros p.loss.dtype (listProd p.loss.shape))
    mixLoss lossBuf

  let ns ← timeIt s!"steps={batches.size}" act
  let perMs : Float := (Float.ofNat ns) / 1.0e6 / (Float.ofNat batches.size)
  IO.println s!"  avg: {perMs} ms/step (sink={← sink.get})"
  pure perMs

private def fileExists (path : String) : IO Bool := do
  try
    IO.FS.withFile path .read (fun _ => pure ())
    pure true
  catch _ =>
    pure false

def run : IO Unit := do
  let batchSize := 32
  let hidden := 128
  let lr : Float32 := 0.01
  let numBatches := 50

  IO.println s!"=== MNISTDataFusionBench batch={batchSize} hidden={hidden} batches={numBatches} ==="

  let trainImagesPath := "data/train-images-idx3-ubyte"
  let trainLabelsPath := "data/train-labels-idx1-ubyte"
  if !(← fileExists trainImagesPath) || !(← fileExists trainLabelsPath) then
    throw (IO.userError s!"missing MNIST files in `data/`.\nexpected:\n  {trainImagesPath}\n  {trainLabelsPath}")

  let maxImages := batchSize * numBatches
  let (images, labels) ← loadTrain "data" (maxImages? := some maxImages)

  let mut batches : Array (RawBuffer × RawBuffer) := #[]
  for bi in [:numBatches] do
    let startIdx := bi * batchSize
    let xData := getBatch images startIdx batchSize
    let yLabels := getBatchLabels labels startIdx batchSize
    let yOneHot := toOneHot yLabels
    let xBuf := RawBuffer.ofF32 ⟨xData⟩
    let yBuf := RawBuffer.ofF32 ⟨yOneHot⟩
    batches := batches.push (xBuf, yBuf)

  let p := buildProgram batchSize hidden lr
  let roots : List UOp := [p.loss, p.newW1, p.newW2]

  let compiledFused := Interpreter.compileMany roots
  let compiledNode := compileNoFusion roots

  IO.println s!"root tags (fused): loss={tagOf compiledFused p.loss}, newW1={tagOf compiledFused p.newW1}, newW2={tagOf compiledFused p.newW2}"
  IO.println "fused kernel dump:"
  TinyGrad4Bench.dumpKernels compiledFused

  let fusedMs ← runSteps p compiledFused batches hidden
  let nodeMs ← runSteps p compiledNode batches hidden
  let ratio : Float := if fusedMs == 0 then 0 else nodeMs / fusedMs
  IO.println s!"speedup: {ratio}x (node/fused)"

end TinyGrad4Bench.MNISTDataFusionBench

#eval! TinyGrad4Bench.MNISTDataFusionBench.run
