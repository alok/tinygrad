import TinyGrad4

-- Disable RawBuffer linter: MNIST data loader uses Array Float for parsed image/label data
set_option linter.useRawBuffer false

/-!
# MNIST Data Loader

Parses IDX format files used by MNIST dataset.
-/

namespace TinyGrad4.Data.MNIST

/-- Read 4 bytes as big-endian UInt32 -/
def readUInt32BE (bytes : ByteArray) (offset : Nat) : UInt32 :=
  let b0 := bytes.get! offset
  let b1 := bytes.get! (offset + 1)
  let b2 := bytes.get! (offset + 2)
  let b3 := bytes.get! (offset + 3)
  (b0.toUInt32 <<< 24) ||| (b1.toUInt32 <<< 16) ||| (b2.toUInt32 <<< 8) ||| b3.toUInt32

/-- MNIST image data: [numImages, 784] flattened, normalized to [0, 1] -/
structure ImageData where
  numImages : Nat
  rows : Nat
  cols : Nat
  pixels : Array Float  -- flattened [numImages * rows * cols]

/-- MNIST label data: [numLabels] as Float for compatibility -/
structure LabelData where
  numLabels : Nat
  labels : Array Float  -- one-hot encoding would be [numLabels, 10]

/-- Parse IDX3 image file. Optionally limit to first `maxImages?` images. -/
def parseImages (bytes : ByteArray) (maxImages? : Option Nat := none) : IO ImageData := do
  let magic := readUInt32BE bytes 0
  if magic != 2051 then
    throw (IO.userError s!"Invalid magic number for images: {magic}, expected 2051")

  let numImagesTotal := (readUInt32BE bytes 4).toNat
  let rows := (readUInt32BE bytes 8).toNat
  let cols := (readUInt32BE bytes 12).toNat

  let numImages := match maxImages? with
    | some n => min n numImagesTotal
    | none => numImagesTotal

  -- Read pixel data starting at offset 16
  let pixelCount := numImages * rows * cols
  let mut pixels := Array.mkEmpty pixelCount

  for i in [:pixelCount] do
    let byte := bytes.get! (16 + i)
    -- Normalize to [0, 1]
    let pixel := byte.toNat.toFloat / 255.0
    pixels := pixels.push pixel

  pure { numImages, rows, cols, pixels }

/-- Parse IDX1 label file. Optionally limit to first `maxLabels?` labels. -/
def parseLabels (bytes : ByteArray) (maxLabels? : Option Nat := none) : IO LabelData := do
  let magic := readUInt32BE bytes 0
  if magic != 2049 then
    throw (IO.userError s!"Invalid magic number for labels: {magic}, expected 2049")

  let numLabelsTotal := (readUInt32BE bytes 4).toNat
  let numLabels := match maxLabels? with
    | some n => min n numLabelsTotal
    | none => numLabelsTotal

  -- Read label data starting at offset 8
  let mut labels := Array.mkEmpty numLabels
  for i in [:numLabels] do
    let label := bytes.get! (8 + i)
    labels := labels.push label.toNat.toFloat

  pure { numLabels, labels }

private def readExact (h : IO.FS.Handle) (n : Nat) : IO ByteArray := do
  let bytes ← h.read n.toUSize
  if bytes.size != n then
    throw (IO.userError s!"Short read: wanted {n} bytes, got {bytes.size}")
  pure bytes

private def readImageFilePrefix (path : String) (maxImages? : Option Nat := none) : IO ByteArray := do
  let h ← IO.FS.Handle.mk path .read
  let header ← readExact h 16
  let numImagesTotal := (readUInt32BE header 4).toNat
  let rows := (readUInt32BE header 8).toNat
  let cols := (readUInt32BE header 12).toNat
  let numImages := match maxImages? with
    | some n => min n numImagesTotal
    | none => numImagesTotal
  let pixelCount := numImages * rows * cols
  let pixels ← readExact h pixelCount
  pure (header ++ pixels)

private def readLabelFilePrefix (path : String) (maxLabels? : Option Nat := none) : IO ByteArray := do
  let h ← IO.FS.Handle.mk path .read
  let header ← readExact h 8
  let numLabelsTotal := (readUInt32BE header 4).toNat
  let numLabels := match maxLabels? with
    | some n => min n numLabelsTotal
    | none => numLabelsTotal
  let labels ← readExact h numLabels
  pure (header ++ labels)

/-- Load MNIST training data -/
def loadTrain (dataDir : String := "data") (maxImages? : Option Nat := none) : IO (ImageData × LabelData) := do
  let imagesPath := s!"{dataDir}/train-images-idx3-ubyte"
  let labelsPath := s!"{dataDir}/train-labels-idx1-ubyte"

  let imageBytes ← readImageFilePrefix imagesPath maxImages?
  let labelBytes ← readLabelFilePrefix labelsPath maxImages?

  let images ← parseImages imageBytes maxImages?
  let labels ← parseLabels labelBytes maxImages?

  pure (images, labels)

/-- Load MNIST test data -/
def loadTest (dataDir : String := "data") (maxImages? : Option Nat := none) : IO (ImageData × LabelData) := do
  let imagesPath := s!"{dataDir}/t10k-images-idx3-ubyte"
  let labelsPath := s!"{dataDir}/t10k-labels-idx1-ubyte"

  let imageBytes ← readImageFilePrefix imagesPath maxImages?
  let labelBytes ← readLabelFilePrefix labelsPath maxImages?

  let images ← parseImages imageBytes maxImages?
  let labels ← parseLabels labelBytes maxImages?

  pure (images, labels)

/-- Get a batch of images as flat array [batchSize * 784] -/
def getBatch (images : ImageData) (startIdx batchSize : Nat) : Array Float :=
  let pixelsPerImage := images.rows * images.cols
  let startPixel := startIdx * pixelsPerImage
  let endPixel := min ((startIdx + batchSize) * pixelsPerImage) images.pixels.size
  images.pixels.extract startPixel endPixel

/-- Get a batch of labels -/
def getBatchLabels (labels : LabelData) (startIdx batchSize : Nat) : Array Float :=
  labels.labels.extract startIdx (min (startIdx + batchSize) labels.labels.size)

/-- Convert labels to one-hot encoding [batchSize, 10] -/
def toOneHot (labels : Array Float) (numClasses : Nat := 10) : Array Float := Id.run do
  let batchSize := labels.size
  let mut oneHot := Array.mkEmpty (batchSize * numClasses)
  for i in [:batchSize] do
    let label := labels[i]!.toUInt64.toNat
    for c in [:numClasses] do
      oneHot := oneHot.push (if c == label then 1.0 else 0.0)
  oneHot

/-! ## Typed batch loader -/

/-- MNIST batch loader producing float32 `DataArrayN` batches. -/
structure Loader (batch : Nat) where
  images : ImageData
  labels : LabelData
  dropLast : Bool := true

private def numFullBatches (images : ImageData) (batch : Nat) : Nat :=
  if batch == 0 then 0 else images.numImages / batch

/-- Number of batches for this loader. -/
def Loader.numBatches (l : Loader batch) : Nat :=
  if l.dropLast then numFullBatches l.images batch
  else if batch == 0 then 0 else (l.images.numImages + batch - 1) / batch

private def padTo (arr : Array Float) (target : Nat) : Array Float :=
  if arr.size >= target then arr else arr ++ Array.replicate (target - arr.size) 0.0

/-- Fetch a batch and pack it as float32 bytes. -/
def Loader.getBatch (l : Loader batch) (batchIdx : Nat) : IO (Batch batch [784] [10] .float32 .float32) := do
  let maxBatches := numFullBatches l.images batch
  if l.dropLast && batchIdx >= maxBatches then
    throw (IO.userError s!"MNIST loader: batch {batchIdx} out of range (max {maxBatches})")
  let startIdx := batchIdx * batch
  let xRaw := TinyGrad4.Data.MNIST.getBatch l.images startIdx batch
  let yLabels := TinyGrad4.Data.MNIST.getBatchLabels l.labels startIdx batch
  let yRaw := TinyGrad4.Data.MNIST.toOneHot yLabels
  let x := if l.dropLast then xRaw else padTo xRaw (batch * 784)
  let y := if l.dropLast then yRaw else padTo yRaw (batch * 10)
  let xData : DataArrayN (batch :: [784]) .float32 := DataArrayN.ofArrayF32 (batch :: [784]) x
  let yData : DataArrayN (batch :: [10]) .float32 := DataArrayN.ofArrayF32 (batch :: [10]) y
  pure { x := xData, y := yData }

instance : DataLoader (Loader batch) batch [784] [10] .float32 .float32 where
  numBatches := Loader.numBatches
  getBatch := Loader.getBatch

end TinyGrad4.Data.MNIST
