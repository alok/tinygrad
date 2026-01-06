import TinyGrad4
import TinyGrad4.Data.MNISTRaw
import TinyGrad4.Data.Prefetch
import TinyGrad4.Data.GPULoader
import TinyGrad4.Backend.Accelerate
import TinyGrad4.Backend.DeviceBuffer
import TinyGrad4.Backend.Metal
import TinyGrad4.Backend.Cuda

/-!
# Compiled MNIST Training Loop

Runs a small MNIST MLP training loop using `Interpreter.compileManyCached` once and
reuses the compiled program for all batches (RawBuffer hot path).
-/

namespace TinyGrad4.Test.MNISTCompiledTrain

open TinyGrad4
open TinyGrad4.Data
open TinyGrad4.Data.MNISTRaw
open TinyGrad4.Data.GPULoader
open TinyGrad4.Backend.DeviceBuffer
open StaticTensor
open Interpreter
open Backend
open Std

private def getEnvNat (key : String) (default : Nat) : IO Nat := do
  match (← IO.getEnv key) with
  | some v =>
    match v.toNat? with
    | some n => pure n
    | none => pure default
  | none => pure default

private def getEnvString (key : String) (default : String) : IO String := do
  match (← IO.getEnv key) with
  | some v => pure v
  | none => pure default

private def getEnvBool (key : String) (default : Bool) : IO Bool := do
  match (← IO.getEnv key) with
  | some v =>
    let v' := v.trimAscii.toString.toLower
    pure (v' == "1" || v' == "true" || v' == "yes")
  | none => pure default

private def parseFloat32? (raw : String) : Option Float32 := do
  let s := raw.trimAscii.toString
  if s.isEmpty then
    none
  else
    let parts := s.splitOn "."
    match parts with
    | [whole] =>
        whole.toNat? |>.map (fun n => (Float.ofNat n).toFloat32)
    | [whole, frac] =>
        let wholeNat? := if whole.isEmpty then some 0 else whole.toNat?
        let fracNat? := if frac.isEmpty then some 0 else frac.toNat?
        match wholeNat?, fracNat? with
        | some w, some f =>
            let denom := (10 : Nat) ^ frac.length
            let val := Float.ofNat w + (Float.ofNat f / Float.ofNat denom)
            some val.toFloat32
        | _, _ => none
    | _ => none

private def getEnvFloat32 (key : String) (default : Float32) : IO Float32 := do
  match (← IO.getEnv key) with
  | some v =>
    match parseFloat32? v with
    | some f => pure f
    | none => pure default
  | none => pure default

private structure BatchData where
  xBytes : ByteArray
  yBuf : RawBuffer

private def normalizeBatchF32 (items : Array MNISTRaw.Sample) : IO ByteArray := do
  let batchSize := items.size
  let pixelsPerImage := MNISTRaw.ImageBuffer.pixelsPerImage
  let totalBytesU8 := batchSize * pixelsPerImage
  let mut u8 := ByteArray.emptyWithCapacity totalBytesU8
  let mut offset := 0
  for item in items do
    let slice := item.pixels
    u8 := ByteArray.copySlice slice.parent slice.offset u8 offset slice.length false
    offset := offset + slice.length
  if TinyGrad4.Backend.Accel.isAvailable then
    pure (TinyGrad4.Backend.Accel.normalizeU8ToF32 u8 0 u8.size)
  else
    let mut floats := Array.mkEmpty totalBytesU8
    for i in [:u8.size] do
      let px := u8[i]!.toNat.toFloat / 255.0
      floats := floats.push px
    pure (RawBuffer.ofF32 ⟨floats⟩).data

private def labelsToOneHot (items : Array MNISTRaw.Sample) : RawBuffer := Id.run do
  let mut out := Array.mkEmpty (items.size * 10)
  for item in items do
    let label := item.label.toUInt64.toNat
    for c in [:10] do
      out := out.push (if c == label then 1.0 else 0.0)
  RawBuffer.ofF32 ⟨out⟩

private def collateBatch (items : Array MNISTRaw.Sample) : IO BatchData := do
  let xBytes ← normalizeBatchF32 items
  let yBuf := labelsToOneHot items
  pure { xBytes, yBuf }

private def makePrefetcher (mnist : MNISTRaw) (batchSize : Nat) (prefetch : Nat) : IO (BatchPrefetcher BatchData) := do
  let cfg : IteratorConfig MNISTRaw := { base := mnist }
  BatchPrefetcher.createFromIteratorCfg cfg batchSize collateBatch true prefetch

private def emptyBuf : RawBuffer := RawBuffer.zeros .float32 0

instance : Inhabited (RawBuffer × RawBuffer) where
  default := (emptyBuf, emptyBuf)

private structure Program where
  w1Id : UOpId
  w2Id : UOpId
  xId : UOpId
  yId : UOpId
  loss : UOp
  newW1 : UOp
  newW2 : UOp
  compiled : Interpreter.Compiled

private def buildProgram (batchSize hidden : Nat) (lr : Float32) : IO Program := do
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

  let roots : List UOp := [lossUop, newW1Uop, newW2Uop]
  let compiled ← Interpreter.compileManyCached roots
  pure { w1Id, w2Id, xId, yId, loss := lossUop, newW1 := newW1Uop, newW2 := newW2Uop, compiled }

private def initWeights (hidden : Nat) : RawBuffer × RawBuffer :=
  let w1 := { dtype := .float32, data := Native.fullF32Bits (784 * hidden) ((0.01 : Float32).toBits) }
  let w2 := { dtype := .float32, data := Native.fullF32Bits (hidden * 10) ((0.01 : Float32).toBits) }
  (w1, w2)

private def fileExists (path : String) : IO Bool := do
  try
    IO.FS.withFile path .read (fun _ => pure ())
    pure true
  catch _ =>
    pure false

private def registerExternalMetal (buf : TinyGrad4.Backend.Metal.MetalBuffer) (byteSize : Nat) (dtype : DType) :
    IO (DeviceBuffer × GPUBufferId) := do
  let id ← DeviceBuffer.registerExternal buf byteSize dtype
  pure (DeviceBuffer.fromGPU id dtype byteSize, id)

private def tagOf (c : Interpreter.Compiled) (u : UOp) : String :=
  match c.implMap[u.uid]? with
  | some impl => impl.tag
  | none => "node"

def run (dataDir : String := "data") (epochs : Nat := 1) (batchSize : Nat := 32)
    (hidden : Nat := 128) (numBatches : Nat := 50) (lr : Float32 := 0.01)
    (useGPU : Bool := false) (useGPULoader : Bool := false) (prefetch : Nat := 8) : IO Unit := do
  if batchSize == 0 || hidden == 0 || numBatches == 0 || epochs == 0 then
    throw (IO.userError "epochs/batchSize/hidden/numBatches must be > 0")

  let trainImagesPath := s!"{dataDir}/train-images-idx3-ubyte"
  let trainLabelsPath := s!"{dataDir}/train-labels-idx1-ubyte"
  if !(← fileExists trainImagesPath) || !(← fileExists trainLabelsPath) then
    throw (IO.userError s!"missing MNIST files in `{dataDir}`.\nexpected:\n  {trainImagesPath}\n  {trainLabelsPath}")

  let maxImages := batchSize * numBatches
  let mnist ← MNISTRaw.loadTrain dataDir (some maxImages)

  let p ← buildProgram batchSize hidden lr
  match p.compiled.implMap[p.newW1.uid]? with
  | some (.fusedSGD _) => pure ()
  | some impl => IO.println s!"warning: expected fusedSGD for w1 update, got {impl.tag}"
  | none => IO.println "warning: expected fusedSGD for w1 update, got none"
  match p.compiled.implMap[p.newW2.uid]? with
  | some (.fusedSGD _) => pure ()
  | some impl => IO.println s!"warning: expected fusedSGD for w2 update, got {impl.tag}"
  | none => IO.println "warning: expected fusedSGD for w2 update, got none"

  IO.println s!"=== MNIST Compiled Train epochs={epochs} batch={batchSize} hidden={hidden} batches={numBatches} ==="
  IO.println s!"root tags: loss={tagOf p.compiled p.loss}, newW1={tagOf p.compiled p.newW1}, newW2={tagOf p.compiled p.newW2}"
  if useGPU then
    let metalAvail ← Metal.isAvailable
    let cudaAvail ← Cuda.isAvailable
    IO.println s!"Metal available: {metalAvail}"
    if cudaAvail then
      let count ← Cuda.deviceCount
      IO.println s!"CUDA available: true (count={count})"
      if count > 0 then
        let info ← Cuda.deviceInfoFor 0
        IO.println s!"CUDA device[0]: {info}"
    else
      IO.println "CUDA available: false"

  let (w1Init, w2Init) := initWeights hidden
  let w1Ref ← IO.mkRef w1Init
  let w2Ref ← IO.mkRef w2Init

  for epoch in [:epochs] do
    let prefetcher ← makePrefetcher mnist batchSize prefetch
    let totalBatches := prefetcher.totalBatches
    let runBatches := min numBatches totalBatches
    if runBatches == 0 then
      prefetcher.cancel
      throw (IO.userError "MNISTCompiledTrain: no batches available (check batchSize/maxImages)")
    let mut totalLoss : Float := 0.0
    let mut dataWaitNs : Nat := 0
    let mut uploadNs : Nat := 0
    let mut computeNs : Nat := 0
    for bi in [:runBatches] do
      let (batch?, waitNs) ← prefetcher.nextWithWait
      dataWaitNs := dataWaitNs + waitNs
      let some batch := batch? | break
      let xBuf : RawBuffer := { dtype := .float32, data := batch.xBytes }
      let env : Env := (∅ : Env)
        |>.insert p.xId xBuf
        |>.insert p.yId batch.yBuf
        |>.insert p.w1Id (← w1Ref.get)
        |>.insert p.w2Id (← w2Ref.get)
      let cache ←
        if useGPU then
          if useGPULoader then
            let metalAvail ← Metal.isAvailable
            if !metalAvail then
              let tComputeStart ← IO.monoNanosNow
              let cache ← Interpreter.evalCompiledRawIO p.compiled env
              let tComputeStop ← IO.monoNanosNow
              computeNs := computeNs + (tComputeStop - tComputeStart)
              pure cache
            else
              let batchShape : Shape := [batchSize, 784]
              let tUploadStart ← IO.monoNanosNow
              let gpuBuf ← ByteArray.toGPUBuffer batch.xBytes .metal batchShape .float32
              let tUploadStop ← IO.monoNanosNow
              uploadNs := uploadNs + (tUploadStop - tUploadStart)
              let (xDb, xId) ←
                match gpuBuf.allocation.handle with
                | .metal buf => registerExternalMetal buf gpuBuf.bytes .float32
                | .cuda _ => throw (IO.userError "GPU loader: CUDA inputs not supported in evalCompiledRawIOGPU")
              let gpuInputs : HashMap UOpId DeviceBuffer := (∅ : HashMap UOpId DeviceBuffer) |>.insert p.xId xDb
              let tComputeStart ← IO.monoNanosNow
              let cache ← Interpreter.evalCompiledRawIOGPUWithInputs p.compiled env gpuInputs
              let tComputeStop ← IO.monoNanosNow
              computeNs := computeNs + (tComputeStop - tComputeStart)
              DeviceBuffer.freeGPU xId
              gpuBuf.free
              pure cache
          else
            let tComputeStart ← IO.monoNanosNow
            let cache ← Interpreter.evalCompiledRawIOGPU p.compiled env
            let tComputeStop ← IO.monoNanosNow
            computeNs := computeNs + (tComputeStop - tComputeStart)
            pure cache
        else
          let tComputeStart ← IO.monoNanosNow
          let cache := Interpreter.evalCompiledRaw p.compiled env
          let tComputeStop ← IO.monoNanosNow
          computeNs := computeNs + (tComputeStop - tComputeStart)
          pure cache
      let w1' := cache.getD p.newW1.uid (RawBuffer.zeros p.newW1.dtype (listProd p.newW1.shape))
      let w2' := cache.getD p.newW2.uid (RawBuffer.zeros p.newW2.dtype (listProd p.newW2.shape))
      w1Ref.set w1'
      w2Ref.set w2'
      let lossBuf := cache.getD p.loss.uid (RawBuffer.zeros p.loss.dtype (listProd p.loss.shape))
      let lossVal := RawBuffer.decodeScalarF32 lossBuf
      totalLoss := totalLoss + lossVal
      if bi % 10 == 0 then
        IO.print s!"\r  epoch {epoch + 1}/{epochs} batch {bi + 1}/{runBatches} loss={lossVal}"
        (← IO.getStdout).flush
    prefetcher.cancel
    IO.println ""
    let avgLoss := totalLoss / (Float.ofNat runBatches)
    IO.println s!"  avg loss: {avgLoss}"
    let dataWaitMs := dataWaitNs.toFloat / 1e6
    let uploadMs := uploadNs.toFloat / 1e6
    let computeMs := computeNs.toFloat / 1e6
    IO.println s!"  data wait: {dataWaitMs} ms, upload: {uploadMs} ms, compute: {computeMs} ms"

def main : IO Unit := do
  let dataDir ← getEnvString "TG4_DATA_DIR" "data"
  let epochs ← getEnvNat "TG4_MNIST_EPOCHS" 1
  let batchSize ← getEnvNat "TG4_MNIST_BATCH" 32
  let hidden ← getEnvNat "TG4_MNIST_HIDDEN" 128
  let numBatches ← getEnvNat "TG4_MNIST_BATCHES" 50
  let lr ← getEnvFloat32 "TG4_MNIST_LR" 0.01
  let useGPU ← getEnvBool "TG4_MNIST_USE_GPU" false
  let useGPULoader ← getEnvBool "TG4_MNIST_USE_GPU_LOADER" false
  let prefetch ← getEnvNat "TG4_MNIST_PREFETCH" 8
  run
    (dataDir := dataDir)
    (epochs := epochs)
    (batchSize := batchSize)
    (hidden := hidden)
    (numBatches := numBatches)
    (lr := lr)
    (useGPU := useGPU)
    (useGPULoader := useGPULoader)
    (prefetch := prefetch)

end TinyGrad4.Test.MNISTCompiledTrain

def main : IO Unit :=
  TinyGrad4.Test.MNISTCompiledTrain.main
