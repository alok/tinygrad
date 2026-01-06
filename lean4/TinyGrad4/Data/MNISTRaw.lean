import TinyGrad4.Data.Slice
import TinyGrad4.Data.Dataset
import TinyGrad4.Data.Transform
import TinyGrad4.Backend.Accelerate
import TinyGrad4.Backend.Metal
import TinyGrad4.Backend.Cuda
import TinyGrad4.Backend.Interpreter
import TinyGrad4.Tensor.Math
import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Typed

-- Disable IO.monoNanosNow linter: benchmark timing uses raw monotonic clocks.
set_option linter.monoNanosNow false

/-!
# MNIST Raw Data Loading (tinygrad-style)

Loads MNIST using zero-copy views, matching tinygrad Python's pattern:
```python
def mnist():
  def _mnist(file): return Tensor.from_url(base_url+file, gunzip=True)
  return _mnist("train-images-idx3-ubyte.gz")[0x10:].reshape(-1,1,28,28), ...
```

Key principles:
1. Load raw bytes into ByteArray (once)
2. Slice operations are zero-copy (just adjust offset)
3. Data stays as uint8 until needed (no Float conversion overhead)
4. Tensor creation is lazy (reshape is just a UOp)

## Usage
```lean
let mnist ← MNISTRaw.load "data"
let batch := mnist.images.slice (batchIdx * 64) 64  -- zero-copy
let tensor ← mnist.toTensor batch [64, 1, 28, 28]
```
-/

namespace TinyGrad4.Data.MNISTRaw

open TinyGrad4.Data
open Interpreter

/-- MNIST image buffer: raw bytes, uint8, shape [N, 784] or [N, 1, 28, 28]
    Uses ByteSlice for zero-copy - keeps reference to original file data. -/
structure ImageBuffer where
  /-- Raw uint8 pixel data as slice into original file (zero-copy) -/
  data : ByteSlice
  /-- Number of images -/
  numImages : Nat
  deriving Repr

/-- MNIST label buffer: raw bytes, uint8, shape [N]
    Uses ByteSlice for zero-copy - keeps reference to original file data. -/
structure LabelBuffer where
  /-- Raw uint8 label data as slice into original file (zero-copy) -/
  data : ByteSlice
  /-- Number of labels -/
  numLabels : Nat
  deriving Repr

/-- MNIST dataset: image and label buffers (zero-copy ready) -/
structure MNISTRaw where
  images : ImageBuffer
  labels : LabelBuffer
  deriving Repr

namespace ImageBuffer

/-- Number of pixels per image (28 * 28 = 784) -/
def pixelsPerImage : Nat := 784

/-- Get a slice of images (zero-copy) -/
def slice (ib : ImageBuffer) (startIdx numImages : Nat) : ByteSlice :=
  let startByte := startIdx * pixelsPerImage
  let numBytes := numImages * pixelsPerImage
  ib.data.slice startByte numBytes

/-- Get a single image as a slice (zero-copy) -/
def getImage (ib : ImageBuffer) (idx : Nat) : ByteSlice :=
  ib.slice idx 1

/-- Total byte size -/
def byteSize (ib : ImageBuffer) : Nat := ib.data.length

/-- Get pixel at (imageIdx, pixelIdx) -/
@[inline] def getPixel (ib : ImageBuffer) (imageIdx pixelIdx : Nat) : UInt8 :=
  ib.data.get! (imageIdx * pixelsPerImage + pixelIdx)

end ImageBuffer

namespace LabelBuffer

/-- Get a slice of labels (zero-copy) -/
def slice (lb : LabelBuffer) (startIdx numLabels : Nat) : ByteSlice :=
  lb.data.slice startIdx numLabels

/-- Get a single label -/
@[inline] def getLabel (lb : LabelBuffer) (idx : Nat) : UInt8 :=
  lb.data.get! idx

/-- Total byte size -/
def byteSize (lb : LabelBuffer) : Nat := lb.data.length

end LabelBuffer

/-! ## Single Sample Access (Dataset Integration) -/

/-- A single MNIST sample: image pixels (zero-copy slice) + label.
    This is the item type returned by Dataset instance. -/
structure Sample where
  /-- 784 bytes of pixel data (zero-copy view into original file) -/
  pixels : ByteSlice
  /-- Label value 0-9 -/
  label : UInt8
  deriving Repr

instance : Inhabited Sample := ⟨{ pixels := ByteSlice.mk' (ByteArray.emptyWithCapacity 0) 0 0, label := 0 }⟩

/-- Get a single sample from MNIST (zero-copy for image) -/
def MNISTRaw.getSample (mnist : MNISTRaw) (idx : Nat) : Sample :=
  { pixels := mnist.images.getImage idx
    label := mnist.labels.getLabel idx }

/-- Dataset instance for MNISTRaw: random-access over individual samples -/
instance : Dataset MNISTRaw Sample where
  len mnist := mnist.images.numImages
  getItem mnist idx _ := pure (mnist.getSample idx)

/-- Parse IDX3 image file header, returning a ByteSlice view into pixel data (zero-copy).
    Data format: [N * 784] uint8 pixels -/
def parseImages (bytes : ByteArray) (maxImages? : Option Nat := none) : IO ImageBuffer := do
  if bytes.size < 16 then
    throw (IO.userError "Image file too small")

  -- Parse header (big-endian)
  let header := bytes.toSlice 0 16
  let magic := header.getU32BE 0
  if magic != 2051 then
    throw (IO.userError s!"Invalid magic: {magic}, expected 2051")

  let numImagesTotal := header.getU32BE 4
  let rows := header.getU32BE 8
  let cols := header.getU32BE 12

  if rows != 28 || cols != 28 then
    throw (IO.userError s!"Expected 28x28 images, got {rows}x{cols}")

  let numImages := match maxImages? with
    | some n => min n numImagesTotal.toNat
    | none => numImagesTotal.toNat

  -- Create zero-copy slice: skip 16-byte header, take pixel data
  -- This is like Python's tensor[0x10:]
  let pixelBytes := numImages * 784
  let dataSlice := bytes.toSlice 16 (16 + pixelBytes)

  pure { data := dataSlice, numImages := numImages }

/-- Parse IDX1 label file header, returning a ByteSlice view into label data (zero-copy).
    Data format: [N] uint8 labels -/
def parseLabels (bytes : ByteArray) (maxLabels? : Option Nat := none) : IO LabelBuffer := do
  if bytes.size < 8 then
    throw (IO.userError "Label file too small")

  let header := bytes.toSlice 0 8
  let magic := header.getU32BE 0
  if magic != 2049 then
    throw (IO.userError s!"Invalid magic: {magic}, expected 2049")

  let numLabelsTotal := header.getU32BE 4
  let numLabels := match maxLabels? with
    | some n => min n numLabelsTotal.toNat
    | none => numLabelsTotal.toNat

  -- Create zero-copy slice: skip 8-byte header, take label data
  let dataSlice := bytes.toSlice 8 (8 + numLabels)

  pure { data := dataSlice, numLabels := numLabels }

/-- Load MNIST training data -/
def loadTrain (dataDir : String := "data") (maxImages? : Option Nat := none) : IO MNISTRaw := do
  let imagesPath := s!"{dataDir}/train-images-idx3-ubyte"
  let labelsPath := s!"{dataDir}/train-labels-idx1-ubyte"

  let imageBytes ← IO.FS.readBinFile imagesPath
  let labelBytes ← IO.FS.readBinFile labelsPath

  let images ← parseImages imageBytes maxImages?
  let labels ← parseLabels labelBytes maxImages?

  pure { images, labels }

/-- Load MNIST test data -/
def loadTest (dataDir : String := "data") (maxImages? : Option Nat := none) : IO MNISTRaw := do
  let imagesPath := s!"{dataDir}/t10k-images-idx3-ubyte"
  let labelsPath := s!"{dataDir}/t10k-labels-idx1-ubyte"

  let imageBytes ← IO.FS.readBinFile imagesPath
  let labelBytes ← IO.FS.readBinFile labelsPath

  let images ← parseImages imageBytes maxImages?
  let labels ← parseLabels labelBytes maxImages?

  pure { images, labels }

/-! ## Tensor Creation from Views -/

/-- Create tensor UOp from a ByteSlice.
    The tensor has dtype uint8 initially - cast to float32 in the UOp graph. -/
def toTensorU8 (slice : ByteSlice) (shape : Shape) : UOpM UOp := do
  -- Copy slice data to RawBuffer for now
  -- Future: could use offset-based BUFFER reference for true zero-copy in UOp graph
  let buf : RawBuffer := { dtype := .uint8, data := slice.toByteArray }
  let uop ← TUOp.vconstRaw buf shape
  pure uop.raw

/-- Create float32 tensor from uint8 image slice.
    Normalizes pixel values to [0, 1] via the UOp graph. -/
def imagesToTensorF32 (ib : ImageBuffer) (startIdx numImages : Nat) (shape : Shape) : UOpM UOp := do
  let slice := ib.slice startIdx numImages
  -- Create uint8 tensor
  let u8Tensor ← toTensorU8 slice shape
  -- Cast to float32 and normalize: x / 255.0
  let u8TU := TUOp.ofRaw u8Tensor
  let f32Tensor ← TUOp.cast u8TU .float32
  let scale ← TUOp.const .float32 (1.0 / 255.0 : Float32)
  let scaleBroadcast ← TUOp.expand scale shape
  let out ← TUOp.binaryOp .MUL f32Tensor scaleBroadcast
  pure out.raw

/-- Create one-hot encoded tensor from uint8 label slice.
    Output shape: [batchSize, numClasses] -/
def labelsToOneHot (lb : LabelBuffer) (startIdx numLabels : Nat) (numClasses : Nat := 10) : UOpM UOp := do
  let slice := lb.slice startIdx numLabels
  let vals : Array Float32 := Id.run do
    let total := numLabels * numClasses
    let mut out := Array.replicate total (0.0 : Float32)
    for i in [:numLabels] do
      let cls := (slice.get! i).toNat
      if cls < numClasses then
        let idx := i * numClasses + cls
        out := out.set! idx 1.0
    out
  let flat ← UOp.vconstF32 vals
  UOp.reshape flat [numLabels, numClasses]

/-! ## Batch Iteration (Zero-Copy) -/

/-- A batch of MNIST data as zero-copy ByteSlices -/
structure Batch where
  /-- Image slice: [batchSize * 784] uint8 bytes -/
  images : ByteSlice
  /-- Label slice: [batchSize] uint8 bytes -/
  labels : ByteSlice
  /-- Batch size -/
  size : Nat
  deriving Repr

/-- Get a batch from MNIST data (zero-copy) -/
def MNISTRaw.getBatch (mnist : MNISTRaw) (batchIdx batchSize : Nat) : Batch :=
  let startIdx := batchIdx * batchSize
  let actualSize := min batchSize (mnist.images.numImages - startIdx)
  { images := mnist.images.slice startIdx actualSize
    labels := mnist.labels.slice startIdx actualSize
    size := actualSize }

/-- Number of batches (drop last incomplete) -/
def MNISTRaw.numBatches (mnist : MNISTRaw) (batchSize : Nat) : Nat :=
  if batchSize == 0 then 0 else mnist.images.numImages / batchSize

/-! ## Benchmark Comparison -/

/-- Sum all bytes in a ByteSlice -/
def ByteSlice.sum (s : ByteSlice) : UInt64 := Id.run do
  let mut total : UInt64 := 0
  for i in [:s.length] do
    total := total + (s.get! i).toUInt64
  total

private def bytesFromUInt32 (v : UInt32) : Array UInt8 :=
  let b0 : UInt8 := UInt8.ofNat (UInt32.toNat (v &&& 0xFF))
  let b1 : UInt8 := UInt8.ofNat (UInt32.toNat ((v >>> 8) &&& 0xFF))
  let b2 : UInt8 := UInt8.ofNat (UInt32.toNat ((v >>> 16) &&& 0xFF))
  let b3 : UInt8 := UInt8.ofNat (UInt32.toNat ((v >>> 24) &&& 0xFF))
  #[b0, b1, b2, b3]

private def pushBytes (out : ByteArray) (bytes : Array UInt8) : ByteArray := Id.run do
  let mut acc := out
  for b in bytes do
    acc := acc.push b
  return acc

private def packI32 (vals : Array Nat) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity (vals.size * 4)
  for v in vals do
    out := pushBytes out (bytesFromUInt32 (UInt32.ofNat v))
  return out

set_option linter.useRawBuffer false in
/-- Comprehensive benchmark matching Python bench_data_loading.py -/
def benchmarkComprehensive (dataDir : String := "data") (batchSize : Nat := 64)
    (maxImages : Nat := 10000) (iterations : Nat := 10) : IO Unit := do
  IO.println "============================================================"
  IO.println "DATA LOADING BENCHMARK: Lean"
  IO.println "============================================================"
  IO.println ""

  let mnist ← loadTrain dataDir (some maxImages)
  let numBatches := mnist.numBatches batchSize

  IO.println s!"Config: {mnist.images.numImages} images, batch_size={batchSize}"
  IO.println s!"Loaded {mnist.images.numImages} images ({mnist.images.byteSize} bytes)"
  IO.println ""
  IO.println s!"Running benchmarks ({iterations} iterations each, median reported)..."
  IO.println ""

  -- 1. Full checksum (sum ALL bytes in each batch) - matches Python numpy vectorized
  let mut checksumTimes : Array Nat := #[]
  let mut finalChecksum : UInt64 := 0
  for _ in [:iterations] do
    let start ← IO.monoNanosNow
    let mut checksum : UInt64 := 0
    for i in [:numBatches] do
      let batch := mnist.getBatch i batchSize
      checksum := checksum + ByteSlice.sum batch.images
    let stop ← IO.monoNanosNow
    checksumTimes := checksumTimes.push (stop - start)
    finalChecksum := checksum

  let checksumMedian := (checksumTimes[iterations / 2]!).toFloat / 1e6
  IO.println s!"  Full checksum (sum all bytes): {checksumMedian} ms ({numBatches.toFloat * 1000.0 / checksumMedian} batch/s)"
  IO.println s!"    checksum = {finalChecksum}"

  -- 2. Normalize to float32 and sum - matches Python numpy + normalize
  let mut normalizeTimes : Array Nat := #[]
  let mut normalizeTotal : Float := 0.0
  for _ in [:iterations] do
    let start ← IO.monoNanosNow
    let mut total : Float := 0.0
    for i in [:numBatches] do
      let batch := mnist.getBatch i batchSize
      -- Normalize each byte to [0,1] and sum
      for j in [:batch.images.length] do
        let px := (batch.images.get! j).toFloat / 255.0
        total := total + px
    let stop ← IO.monoNanosNow
    normalizeTimes := normalizeTimes.push (stop - start)
    normalizeTotal := total

  let normalizeMedian := (normalizeTimes[iterations / 2]!).toFloat / 1e6
  IO.println s!"  Normalize to f32 + sum: {normalizeMedian} ms ({numBatches.toFloat * 1000.0 / normalizeMedian} batch/s)"
  IO.println s!"    total = {normalizeTotal}"

  -- 3. Shuffle + batch (sum all bytes) - using RandKey from Dataset module
  let mut shuffleTimes : Array Nat := #[]
  let mut shuffleChecksum : UInt64 := 0
  for iter in [:iterations] do
    let start ← IO.monoNanosNow
    -- Generate shuffled indices
    let key := RandKey.new (42 + iter)
    let (indices, _) := key.shuffleIndices mnist.images.numImages
    let mut checksum : UInt64 := 0
    for i in [:numBatches] do
      -- Access shuffled samples
      for j in [:batchSize] do
        let idx := indices[i * batchSize + j]!
        -- Sum all pixels for this image (matches numpy batch.sum)
        for p in [:784] do
          let px := mnist.images.getPixel idx p
          checksum := checksum + px.toUInt64
    let stop ← IO.monoNanosNow
    shuffleTimes := shuffleTimes.push (stop - start)
    shuffleChecksum := checksum

  let shuffleMedian := (shuffleTimes[iterations / 2]!).toFloat / 1e6
  IO.println s!"  Shuffle + batch (sum all bytes): {shuffleMedian} ms ({numBatches.toFloat * 1000.0 / shuffleMedian} batch/s)"
  IO.println s!"    checksum = {shuffleChecksum}"

  -- 3b. Shuffle + batch via indexSelect (tinygrad one-hot gather)
  let (sumT, imgBuf, idxBuf) := runTensorM do
    let img ← Tensor.buffer [mnist.images.numImages, 784] .uint8
    let idx ← Tensor.buffer [batchSize] .int32
    let gathered ← StaticTensor.indexSelect img 0 idx
    let gatheredF ← StaticTensor.cast gathered .float32
    let total ← StaticTensor.sum gatheredF
    pure (total, img.uop, idx.uop)

  let imgRaw : RawBuffer := { dtype := .uint8, data := mnist.images.data.toByteArray }
  let baseEnv := setBuffer (∅ : Env) imgBuf imgRaw

  let mut shuffleTGTimes : Array Nat := #[]
  let mut shuffleTGTotal : Float := 0.0
  for iter in [:iterations] do
    let start ← IO.monoNanosNow
    let key := RandKey.new (42 + iter)
    let (indices, _) := key.shuffleIndices mnist.images.numImages
    let mut total : Float := 0.0
    for i in [:numBatches] do
      let startIdx := i * batchSize
      let endIdx := startIdx + batchSize
      let idxSlice := indices.extract startIdx endIdx
      let idxBytes := packI32 idxSlice
      let idxRaw : RawBuffer := { dtype := .int32, data := idxBytes }
      let env := setBuffer baseEnv idxBuf idxRaw
      let out ← evalTensorCached sumT env
      let vals := RawBuffer.toFloatArray out
      let v := if vals.size == 0 then 0.0 else vals[0]!
      total := total + v
    let stop ← IO.monoNanosNow
    shuffleTGTimes := shuffleTGTimes.push (stop - start)
    shuffleTGTotal := total

  let shuffleTGMedian := (shuffleTGTimes[iterations / 2]!).toFloat / 1e6
  IO.println s!"  Shuffle + batch (indexSelect): {shuffleTGMedian} ms ({numBatches.toFloat * 1000.0 / shuffleTGMedian} batch/s)"
  IO.println s!"    total = {shuffleTGTotal}"

  -- 4. Accelerate SIMD checksum
  IO.println ""
  IO.println s!"  Using Accelerate: {TinyGrad4.Backend.Accel.isAvailable}"

  let mut accelChecksumTimes : Array Nat := #[]
  let mut accelChecksum : UInt64 := 0
  for _ in [:iterations] do
    let start ← IO.monoNanosNow
    let mut checksum : UInt64 := 0
    for i in [:numBatches] do
      let batch := mnist.getBatch i batchSize
      -- Use Accelerate SIMD sum
      checksum := checksum + TinyGrad4.Backend.Accel.sumSlice batch.images.parent batch.images.offset batch.images.length
    let stop ← IO.monoNanosNow
    accelChecksumTimes := accelChecksumTimes.push (stop - start)
    accelChecksum := checksum

  let accelChecksumMedian := (accelChecksumTimes[iterations / 2]!).toFloat / 1e6
  IO.println s!"  Accel checksum (SIMD):    {accelChecksumMedian} ms ({numBatches.toFloat * 1000.0 / accelChecksumMedian} batch/s)"
  IO.println s!"    checksum = {accelChecksum}"

  -- 5. Accelerate SIMD normalize + sum (fused)
  let mut accelNormTimes : Array Nat := #[]
  let mut accelNormTotal : Float := 0.0
  for _ in [:iterations] do
    let start ← IO.monoNanosNow
    let mut total : Float := 0.0
    for i in [:numBatches] do
      let batch := mnist.getBatch i batchSize
      -- Use Accelerate fused normalize + sum
      total := total + TinyGrad4.Backend.Accel.normalizeSumSlice batch.images.parent batch.images.offset batch.images.length
    let stop ← IO.monoNanosNow
    accelNormTimes := accelNormTimes.push (stop - start)
    accelNormTotal := total

  let accelNormMedian := (accelNormTimes[iterations / 2]!).toFloat / 1e6
  IO.println s!"  Accel normalize+sum (SIMD): {accelNormMedian} ms ({numBatches.toFloat * 1000.0 / accelNormMedian} batch/s)"
  IO.println s!"    total = {accelNormTotal}"

  -- 6. Metal GPU: normalize + matmul (simulated forward pass like MLX)
  IO.println ""
  IO.println s!"  Metal GPU available: {← TinyGrad4.Backend.Metal.isAvailable}"

  -- Pre-allocate weight matrix W [784, 128] for forward pass
  let M := batchSize  -- batch size
  let K := 784        -- input features
  let N := 128        -- output features

  -- Create weight matrix as ByteArray (float32 zeros for simplicity)
  -- In real usage, weights would come from model parameters
  let mut wBytes := ByteArray.emptyWithCapacity (K * N * 4)
  for _ in [:K * N * 4] do
    wBytes := wBytes.push 0

  let mut metalTimes : Array Nat := #[]
  let mut metalOutputSize : Nat := 0
  for _ in [:iterations] do
    let start ← IO.monoNanosNow

    for i in [:numBatches] do
      let batch := mnist.getBatch i batchSize
      -- 1. Normalize batch data to float32 using Accelerate (CPU, already fast)
      let normalized := TinyGrad4.Backend.Accel.normalizeU8ToF32
        batch.images.parent batch.images.offset batch.images.length

      -- 2. Run matmul on GPU: [M, 784] @ [784, 128] = [M, 128]
      let result := TinyGrad4.Backend.Metal.metalMatmulSync normalized wBytes M K N
      metalOutputSize := result.size

    let stop ← IO.monoNanosNow
    metalTimes := metalTimes.push (stop - start)

  let metalMedian := (metalTimes[iterations / 2]!).toFloat / 1e6
  IO.println s!"  Metal GPU normalize+matmul: {metalMedian} ms ({numBatches.toFloat * 1000.0 / metalMedian} batch/s)"
  IO.println s!"    output size = {metalOutputSize} bytes (expected {M * N * 4})"

  -- 6b. CUDA GPU: normalize + matmul (if available)
  IO.println ""
  let cudaAvailable ← TinyGrad4.Backend.Cuda.isAvailable
  IO.println s!"  CUDA GPU available: {cudaAvailable}"
  let cudaMedian? ←
    if !cudaAvailable then
      pure none
    else
      let _ ← TinyGrad4.Backend.Cuda.setDevice 0
      let mut cudaTimes : Array Nat := #[]
      let mut cudaOutputSize : Nat := 0
      for _ in [:iterations] do
        let start ← IO.monoNanosNow
        for i in [:numBatches] do
          let batch := mnist.getBatch i batchSize
          let normalized := TinyGrad4.Backend.Accel.normalizeU8ToF32
            batch.images.parent batch.images.offset batch.images.length
          let result := TinyGrad4.Backend.Cuda.cudaMatmulSync normalized wBytes M K N
          cudaOutputSize := result.size
        let stop ← IO.monoNanosNow
        cudaTimes := cudaTimes.push (stop - start)
      let cudaMedian := (cudaTimes[iterations / 2]!).toFloat / 1e6
      IO.println s!"  CUDA GPU normalize+matmul: {cudaMedian} ms ({numBatches.toFloat * 1000.0 / cudaMedian} batch/s)"
      IO.println s!"    output size = {cudaOutputSize} bytes (expected {M * N * 4})"
      pure (some cudaMedian)

  IO.println ""
  IO.println "============================================================"
  IO.println "RESULTS SUMMARY"
  IO.println "============================================================"
  IO.println ""
  IO.println "Method                              Time (ms)    Batch/sec"
  IO.println "------------------------------------------------------------"
  IO.println s!"Lean loop checksum                  {checksumMedian}    {numBatches.toFloat * 1000.0 / checksumMedian}"
  IO.println s!"Lean loop normalize                 {normalizeMedian}    {numBatches.toFloat * 1000.0 / normalizeMedian}"
  IO.println s!"Lean shuffle + batch                {shuffleMedian}    {numBatches.toFloat * 1000.0 / shuffleMedian}"
  IO.println s!"Accel SIMD checksum                 {accelChecksumMedian}    {numBatches.toFloat * 1000.0 / accelChecksumMedian}"
  IO.println s!"Accel SIMD normalize+sum            {accelNormMedian}    {numBatches.toFloat * 1000.0 / accelNormMedian}"
  IO.println s!"Metal GPU normalize+matmul          {metalMedian}    {numBatches.toFloat * 1000.0 / metalMedian}"
  match cudaMedian? with
  | some cudaMedian =>
      IO.println s!"CUDA GPU normalize+matmul           {cudaMedian}    {numBatches.toFloat * 1000.0 / cudaMedian}"
  | none =>
      IO.println s!"CUDA GPU normalize+matmul           skipped"
  IO.println ""
  IO.println s!"Speedup (Accel vs Lean loop):"
  IO.println s!"  Checksum:  {checksumMedian / accelChecksumMedian}x"
  IO.println s!"  Normalize: {normalizeMedian / accelNormMedian}x"
  IO.println ""
  IO.println s!"GPU vs CPU comparison:"
  IO.println s!"  Metal normalize+matmul: {metalMedian} ms"
  IO.println s!"  Accel normalize-only:   {accelNormMedian} ms"
  match cudaMedian? with
  | some cudaMedian =>
      IO.println s!"  CUDA normalize+matmul:  {cudaMedian} ms"
  | none =>
      pure ()

  -- 7/8. Zero-copy + shared memory benchmarks (Metal only)
  let metalAvailable ← TinyGrad4.Backend.Metal.isAvailable
  let (zeroCopyMedian?, shmThroughputWrite?, shmThroughputRead?) ←
    if !metalAvailable then
      IO.println ""
      IO.println "  Zero-copy benchmarks: skipped (Metal unavailable)"
      IO.println "  Shared memory benchmarks: skipped (Metal unavailable)"
      pure (none, none, none)
    else
    -- Zero-copy Metal buffer benchmark
    IO.println ""
    IO.println "  Zero-copy benchmarks:"

    -- Check alignment
    let isAligned ← TinyGrad4.Backend.Metal.metalIsAligned mnist.images.data.parent 0
    IO.println s!"    Data page-aligned: {isAligned}"

    let mut zeroCopyTimes : Array Nat := #[]
    for _ in [:iterations] do
      let start ← IO.monoNanosNow
      for i in [:numBatches] do
        let batch := mnist.getBatch i batchSize
        -- Create zero-copy Metal buffer from ByteSlice
        let _buf ← TinyGrad4.Backend.Metal.metalFromByteSlice
          batch.images.parent batch.images.offset batch.images.length
        pure ()
      let stop ← IO.monoNanosNow
      zeroCopyTimes := zeroCopyTimes.push (stop - start)

    let zeroCopyMedian := (zeroCopyTimes[iterations / 2]!).toFloat / 1e6
    IO.println s!"  Zero-copy buffer create: {zeroCopyMedian} ms ({numBatches.toFloat * 1000.0 / zeroCopyMedian} batch/s)"

    -- Shared memory benchmark
    IO.println ""
    IO.println "  Shared memory benchmarks:"

    -- Create shared memory region
    let shmName := "/tg4_bench_shm"
    let shmSize := mnist.images.byteSize

    let mut shmWriteTimes : Array Nat := #[]
    let mut shmReadTimes : Array Nat := #[]

    for _ in [:iterations] do
      -- Write benchmark
      let shm ← TinyGrad4.Backend.Metal.SharedMemory.create shmName shmSize
      let writeStart ← IO.monoNanosNow
      shm.write mnist.images.data.parent
      let writeStop ← IO.monoNanosNow
      shmWriteTimes := shmWriteTimes.push (writeStop - writeStart)

      -- Read benchmark
      let readStart ← IO.monoNanosNow
      let _data ← shm.read 0 shmSize
      let readStop ← IO.monoNanosNow
      shmReadTimes := shmReadTimes.push (readStop - readStart)

      shm.close
      shm.unlink

    let shmWriteMedian := (shmWriteTimes[iterations / 2]!).toFloat / 1e6
    let shmReadMedian := (shmReadTimes[iterations / 2]!).toFloat / 1e6
    let shmThroughputWrite := shmSize.toFloat / shmWriteMedian / 1e6  -- MB/s
    let shmThroughputRead := shmSize.toFloat / shmReadMedian / 1e6   -- MB/s

    IO.println s!"  SharedMem write: {shmWriteMedian} ms ({shmThroughputWrite} GB/s)"
    IO.println s!"  SharedMem read:  {shmReadMedian} ms ({shmThroughputRead} GB/s)"
    pure (some zeroCopyMedian, some shmThroughputWrite, some shmThroughputRead)

  IO.println ""
  IO.println "============================================================"
  IO.println "RESULTS SUMMARY"
  IO.println "============================================================"
  IO.println ""
  IO.println "Method                              Time (ms)    Batch/sec"
  IO.println "------------------------------------------------------------"
  IO.println s!"Lean loop checksum                  {checksumMedian}    {numBatches.toFloat * 1000.0 / checksumMedian}"
  IO.println s!"Lean loop normalize                 {normalizeMedian}    {numBatches.toFloat * 1000.0 / normalizeMedian}"
  IO.println s!"Lean shuffle + batch                {shuffleMedian}    {numBatches.toFloat * 1000.0 / shuffleMedian}"
  IO.println s!"Accel SIMD checksum                 {accelChecksumMedian}    {numBatches.toFloat * 1000.0 / accelChecksumMedian}"
  IO.println s!"Accel SIMD normalize+sum            {accelNormMedian}    {numBatches.toFloat * 1000.0 / accelNormMedian}"
  IO.println s!"Metal GPU normalize+matmul          {metalMedian}    {numBatches.toFloat * 1000.0 / metalMedian}"
  match zeroCopyMedian? with
  | some zeroCopyMedian =>
      IO.println s!"Zero-copy buffer create             {zeroCopyMedian}    {numBatches.toFloat * 1000.0 / zeroCopyMedian}"
  | none =>
      IO.println "Zero-copy buffer create             skipped"
  IO.println ""
  match shmThroughputWrite?, shmThroughputRead? with
  | some shmThroughputWrite, some shmThroughputRead =>
      IO.println s!"Shared memory throughput:"
      IO.println s!"  Write: {shmThroughputWrite} GB/s"
      IO.println s!"  Read:  {shmThroughputRead} GB/s"
  | _, _ =>
      IO.println "Shared memory throughput: skipped"
  IO.println ""
  IO.println "============================================================"
  IO.println "Run Python: uv run lean4/scripts/bench_data_loading.py"
  IO.println "============================================================"

/-- Legacy benchmark for backward compat -/
def benchmarkZeroCopy (dataDir : String := "data") (batchSize : Nat := 64)
    (maxImages : Nat := 10000) : IO Unit :=
  benchmarkComprehensive dataDir batchSize maxImages 10

end TinyGrad4.Data.MNISTRaw
