import TinyGrad4
import TinyGrad4.Data.MNIST
import TinyGrad4.Test.MNISTTrain
import TinyGrad4Bench.KernelProfile
-- Disable IO.monoNanosNow linter: benchmark timing uses raw monotonic clocks.
set_option linter.monoNanosNow false

/-!
# MNISTTrainCacheBench

Compare cached vs uncached `evalMany` on MNISTTrain-style dynamic graphs.
We precompute batches to focus on graph compile/eval cost (data loading is excluded).
-/

namespace TinyGrad4Bench.MNISTTrainCacheBench

open TinyGrad4
open TinyGrad4.Data.MNIST
open TinyGrad4.Test.MNISTTrain
open StaticTensor
open Interpreter
open TinyGrad4Bench

private abbrev Model := TinyGrad4.Test.MNISTTrain.Model

private def initModelHidden (hidden : Nat) : IO Model := do
  let w1Size := 784 * hidden
  let w2Size := hidden * 10

  let mut w1 := FloatArray.emptyWithCapacity w1Size
  let mut seed : UInt64 := 42
  for _ in [:w1Size] do
    seed := seed * 1103515245 + 12345
    let val := ((seed >>> 16).toNat % 1000).toFloat / 1000.0 - 0.5
    w1 := w1.push (val * 0.1)

  let mut w2 := FloatArray.emptyWithCapacity w2Size
  for _ in [:w2Size] do
    seed := seed * 1103515245 + 12345
    let val := ((seed >>> 16).toNat % 1000).toFloat / 1000.0 - 0.5
    w2 := w2.push (val * 0.1)

  pure { w1Data := w1, w2Data := w2 }

private def resolveDataDir : IO String := do
  let cwd ← IO.currentDir
  let dataHere := cwd / "data"
  let dataParent := cwd / ".." / "data"
  if (← dataHere.pathExists) then
    pure dataHere.toString
  else if (← dataParent.pathExists) then
    pure dataParent.toString
  else
    pure "data"

private def buildBatches (images : ImageData) (labels : LabelData) (batchSize numBatches : Nat)
    : Array (Array Float × Array Float) := Id.run do
  let mut out := Array.mkEmpty numBatches
  for batchIdx in [:numBatches] do
    let startIdx := batchIdx * batchSize
    let xData := getBatch images startIdx batchSize
    let yLabels := getBatchLabels labels startIdx batchSize
    let yOneHot := toOneHot yLabels
    out := out.push (xData, yOneHot)
  return out

private structure ForwardResult where
  loss : Float
  gradW1 : FlatArray
  gradW2 : FlatArray
  compileNs : Nat
  evalNs : Nat

private structure StepResult where
  loss : Float
  model : Model
  compileNs : Nat
  evalNs : Nat

private structure BreakdownSeries where
  total : Array Float
  prep : Array Float
  compile : Array Float
  eval : Array Float
  deriving Repr

private def nsToMs (n : Nat) : Float :=
  (Float.ofNat n) / 1.0e6

private def reportSeries (label : String) (series : BreakdownSeries) : IO Unit := do
  IO.println s!"{label}: trials={series.total.size} mean={mean series.total} ms/iter median={median series.total} ms/iter"
  IO.println s!"{label}: samples={series.total.toList}"
  IO.println s!"{label}: prep mean={mean series.prep} median={median series.prep} ms/iter"
  IO.println s!"{label}: compile mean={mean series.compile} median={median series.compile} ms/iter"
  IO.println s!"{label}: eval mean={mean series.eval} median={median series.eval} ms/iter"

private def reportCacheStats (label : String) : IO Unit := do
  let stats ← Interpreter.getScheduleCacheStats
  let hitAvg :=
    if stats.hits == 0 then 0.0 else nsToMs stats.hitNs / (Float.ofNat stats.hits)
  let missAvg :=
    if stats.misses == 0 then 0.0 else nsToMs stats.missNs / (Float.ofNat stats.misses)
  IO.println s!"{label}: cache hits={stats.hits} misses={stats.misses} rebuilds={stats.rebuilds}"
  IO.println s!"{label}: cache hit_ms_total={nsToMs stats.hitNs} miss_ms_total={nsToMs stats.missNs}"
  IO.println s!"{label}: cache hit_ms_avg={hitAvg} miss_ms_avg={missAvg}"

private def timeTrialsBreakdown (label : String) (trials iters : Nat) (reset : IO Unit)
    (act : IO (Nat × Nat × Nat)) : IO BreakdownSeries := do
  let t := if trials == 0 then 1 else trials
  let iters' := if iters == 0 then 1 else iters
  let itersF : Float := Float.ofNat iters'
  let mut totalMs : Array Float := #[]
  let mut prepMs : Array Float := #[]
  let mut compileMs : Array Float := #[]
  let mut evalMs : Array Float := #[]

  for _ in [:t] do
    reset
    let mut prepNs : Nat := 0
    let mut compileNs : Nat := 0
    let mut evalNs : Nat := 0
    let start ← IO.monoNanosNow
    for _ in [:iters'] do
      let (p, c, e) ← act
      prepNs := prepNs + p
      compileNs := compileNs + c
      evalNs := evalNs + e
    let stop ← IO.monoNanosNow
    let totalNs := stop - start
    totalMs := totalMs.push (nsToMs totalNs / itersF)
    prepMs := prepMs.push (nsToMs prepNs / itersF)
    compileMs := compileMs.push (nsToMs compileNs / itersF)
    evalMs := evalMs.push (nsToMs evalNs / itersF)

  let series := { total := totalMs, prep := prepMs, compile := compileMs, eval := evalMs }
  reportSeries label series
  return series

private def forwardLoss (useCache : Bool) (model : Model) (xData : Array Float) (yOneHot : Array Float)
    (batchSize hidden : Nat) : IO ForwardResult := do
  let result := runTensorM do
    let xBuf ← Tensor.buffer [batchSize, 784] .float32
    let w1Buf ← Tensor.buffer [784, hidden] .float32
    let w2Buf ← Tensor.buffer [hidden, 10] .float32
    let yBuf ← Tensor.buffer [batchSize, 10] .float32

    let h ← matmul xBuf w1Buf
    let hNorm ← layerNorm h
    let hRelu ← relu hNorm
    let logits ← matmul hRelu w2Buf

    let loss ← crossEntropyOneHot logits yBuf
    let gradMap ← backward loss [w1Buf.uop, w2Buf.uop]

    pure (loss, xBuf.uop.uid, w1Buf.uop.uid, w2Buf.uop.uid, yBuf.uop.uid,
      loss.uop, w1Buf.uop, w2Buf.uop, gradMap)

  let (_loss, xId, w1Id, w2Id, yId, lossUop, _w1Uop, _w2Uop, gradResult) := result

  let env : Env := (∅ : Env)
    |>.insert xId (RawBuffer.ofF32 ⟨xData⟩)
    |>.insert w1Id (RawBuffer.ofF32 model.w1Data)
    |>.insert w2Id (RawBuffer.ofF32 model.w2Data)
    |>.insert yId (RawBuffer.ofF32 ⟨yOneHot⟩)

  let gradW1Uop? := gradResult[w1Id]?
  let gradW2Uop? := gradResult[w2Id]?

  let roots := [lossUop] ++ gradW1Uop?.toList ++ gradW2Uop?.toList
  let (vals, compileNs, evalNs) ←
    if useCache then
      evalManyCachedTimed roots env
    else
      evalManyTimed roots env

  let lossVal := (vals.getD lossUop.uid (zeros 1))[0]!
  let gradW1 := match gradW1Uop? with
    | some g => vals.getD g.uid model.w1Data
    | none => model.w1Data

  let gradW2 := match gradW2Uop? with
    | some g => vals.getD g.uid model.w2Data
    | none => model.w2Data

  pure { loss := lossVal, gradW1, gradW2, compileNs, evalNs }

private def trainStep (useCache : Bool) (model : Model) (xData : Array Float) (yOneHot : Array Float)
    (batchSize hidden : Nat) (lr : Float) : IO StepResult := do
  let out ← forwardLoss useCache model xData yOneHot batchSize hidden

  let newW1 := FloatArray.zipWith (fun w g => w - lr * g) model.w1Data out.gradW1
  let newW2 := FloatArray.zipWith (fun w g => w - lr * g) model.w2Data out.gradW2

  pure { loss := out.loss, model := { w1Data := newW1, w2Data := newW2 }, compileNs := out.compileNs, evalNs := out.evalNs }

private def runTrialsPrecomputed (label : String) (useCache : Bool) (batches : Array (Array Float × Array Float))
    (batchSize hidden : Nat) (lr : Float) (trials : Nat) : IO BreakdownSeries := do
  if batches.isEmpty then
    IO.println s!"{label}: empty batches"
    return { total := #[], prep := #[], compile := #[], eval := #[] }

  let modelInit ← initModelHidden hidden
  let modelRef ← IO.mkRef modelInit
  let idxRef ← IO.mkRef 0
  let sink ← IO.mkRef (0.0 : Float)

  let reset : IO Unit := do
    idxRef.set 0
    modelRef.set modelInit
    sink.set 0.0

  let act : IO (Nat × Nat × Nat) := do
    let idx ← idxRef.get
    if idx == 0 then
      modelRef.set modelInit
    let (xData, yOneHot) := batches[idx]!
    let out ← trainStep useCache (← modelRef.get) xData yOneHot batchSize hidden lr
    modelRef.set out.model
    idxRef.set ((idx + 1) % batches.size)
    sink.modify (· + out.loss)
    pure (0, out.compileNs, out.evalNs)

  let t0 ← IO.monoNanosNow
  let (p0, c0, e0) ← act
  let t1 ← IO.monoNanosNow
  IO.println s!"{label}: cold total_ms={nsToMs (t1 - t0)} prep_ms={nsToMs p0} compile_ms={nsToMs c0} eval_ms={nsToMs e0}"

  let series ← timeTrialsBreakdown label trials batches.size reset act
  IO.println s!"{label}: sink={← sink.get}"
  return series

private def runTrialsOnTheFly (label : String) (useCache : Bool) (images : ImageData) (labels : LabelData)
    (batchSize numBatches hidden : Nat) (lr : Float) (trials : Nat) : IO BreakdownSeries := do
  if numBatches == 0 then
    IO.println s!"{label}: empty batches"
    return { total := #[], prep := #[], compile := #[], eval := #[] }

  let modelInit ← initModelHidden hidden
  let modelRef ← IO.mkRef modelInit
  let idxRef ← IO.mkRef 0
  let sink ← IO.mkRef (0.0 : Float)

  let reset : IO Unit := do
    idxRef.set 0
    modelRef.set modelInit
    sink.set 0.0

  let act : IO (Nat × Nat × Nat) := do
    let idx ← idxRef.get
    if idx == 0 then
      modelRef.set modelInit
    let startIdx := idx * batchSize
    let t0 ← IO.monoNanosNow
    let xData := getBatch images startIdx batchSize
    let yLabels := getBatchLabels labels startIdx batchSize
    let yOneHot := toOneHot yLabels
    let t1 ← IO.monoNanosNow
    let out ← trainStep useCache (← modelRef.get) xData yOneHot batchSize hidden lr
    modelRef.set out.model
    idxRef.set ((idx + 1) % numBatches)
    sink.modify (· + out.loss)
    pure (t1 - t0, out.compileNs, out.evalNs)

  let t0 ← IO.monoNanosNow
  let (p0, c0, e0) ← act
  let t1 ← IO.monoNanosNow
  IO.println s!"{label}: cold total_ms={nsToMs (t1 - t0)} prep_ms={nsToMs p0} compile_ms={nsToMs c0} eval_ms={nsToMs e0}"

  let series ← timeTrialsBreakdown label trials numBatches reset act
  IO.println s!"{label}: sink={← sink.get}"
  return series

def runWith (batchSize : Nat := 32) (numBatches : Nat := 20) (trials : Nat := 5) (hidden : Nat := 128)
    (lr : Float := 0.01) (precompute : Bool := true) (tag : String := "") : IO Unit := do
  let tagSuffix := if tag.isEmpty then "" else s!" ({tag})"
  IO.println s!"=== MNISTTrainCacheBench{tagSuffix} batch={batchSize} batches={numBatches} hidden={hidden} trials={trials} ==="
  IO.println s!"precompute={precompute}"
  if batchSize == 0 then
    IO.println "batchSize=0, nothing to do"
    return ()
  if hidden == 0 then
    IO.println "hidden=0, nothing to do"
    return ()

  let maxImages := batchSize * numBatches
  let dataDir ← resolveDataDir
  IO.println s!"dataDir: {dataDir}"
  let (images, labels) ← loadTrain dataDir (maxImages? := some maxImages)
  let availBatches := images.numImages / batchSize
  let numBatches' := min numBatches availBatches
  if numBatches' == 0 then
    IO.println "no full batches to run"
    return ()

  let labelSuffix := if tag.isEmpty then "" else s!" [{tag}]"
  Interpreter.clearScheduleCache
  Interpreter.clearScheduleCacheStats
  let uncachedSeries ←
    if precompute then
      let batches := buildBatches images labels batchSize numBatches'
      runTrialsPrecomputed s!"uncached (evalMany){labelSuffix}" false batches batchSize hidden lr trials
    else
      runTrialsOnTheFly s!"uncached (evalMany){labelSuffix}" false images labels batchSize numBatches' hidden lr trials
  reportCacheStats s!"uncached{labelSuffix}"

  Interpreter.clearScheduleCache
  Interpreter.clearScheduleCacheStats
  let cachedSeries ←
    if precompute then
      let batches := buildBatches images labels batchSize numBatches'
      runTrialsPrecomputed s!"cached (evalManyCached){labelSuffix}" true batches batchSize hidden lr trials
    else
      runTrialsOnTheFly s!"cached (evalManyCached){labelSuffix}" true images labels batchSize numBatches' hidden lr trials
  reportCacheStats s!"cached{labelSuffix}"

  let uncachedMed := TinyGrad4Bench.median uncachedSeries.total
  let cachedMed := TinyGrad4Bench.median cachedSeries.total
  let ratio : Float := if cachedMed == 0.0 then 0.0 else uncachedMed / cachedMed
  IO.println s!"median speedup: {ratio}x (uncached/cached){labelSuffix}"

  let uncachedCompile := TinyGrad4Bench.median uncachedSeries.compile
  let cachedCompile := TinyGrad4Bench.median cachedSeries.compile
  let compileRatio : Float := if cachedCompile == 0.0 then 0.0 else uncachedCompile / cachedCompile
  IO.println s!"median compile speedup: {compileRatio}x (uncached/cached){labelSuffix}"

  let uncachedEval := TinyGrad4Bench.median uncachedSeries.eval
  let cachedEval := TinyGrad4Bench.median cachedSeries.eval
  let evalRatio : Float := if cachedEval == 0.0 then 0.0 else uncachedEval / cachedEval
  IO.println s!"median eval speedup: {evalRatio}x (uncached/cached){labelSuffix}"

  -- TODO(leanplot): integrate github.com/alok/leanplot to plot total/compile/eval series (use LSP image loop).

def run : IO Unit := do
  match (← IO.getEnv "TINYGRAD4_BENCH_QUICK") with
  | some _ =>
      runWith 16 10 2 64 0.01 true "quick"
  | none =>
      runWith 32 50 7 128 0.01 true "precompute-more"
      runWith 64 20 3 256 0.01 true "big"
      runWith 32 50 5 128 0.01 false "with-prep"

end TinyGrad4Bench.MNISTTrainCacheBench

#eval! TinyGrad4Bench.MNISTTrainCacheBench.run
