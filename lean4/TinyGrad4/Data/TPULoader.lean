import TinyGrad4.Shape
import TinyGrad4.Data.Dataset
import TinyGrad4.Data.Prefetch
import TinyGrad4.Data.GPULoader

/-!
# TPU Loader (Host-Backed)

Minimal TPU-aware data loader that stages batches on the host and tags them
with a TPU device ID. Actual TPU transfer is left to downstream backends.
-/

namespace TinyGrad4.Data.TPULoader

open TinyGrad4.Data
open TinyGrad4.Data.GPULoader

/-- Host-backed TPU buffer with static shape/dtype. -/
structure TPUBuffer (shape : Shape) (dtype : DType) where
  data : ByteArray
  device : DeviceId
  deriving Repr

namespace TPUBuffer

/-- Total number of elements. -/
def numel (_b : TPUBuffer shape dtype) : Nat :=
  Shape.numel shape

/-- Byte size of the buffer view. -/
def bytes (_b : TPUBuffer shape dtype) : Nat :=
  Shape.numel shape * dtype.itemsize

/-- Ensure a buffer is on the expected device. -/
def assertDevice (b : TPUBuffer shape dtype) (device : DeviceId) : IO Unit := do
  if b.device != device then
    throw (IO.userError s!"TPUBuffer on {b.device}, expected {device}")

/-- Ensure data size matches shape/dtype. -/
def assertSize (b : TPUBuffer shape dtype) : IO Unit := do
  let expected := b.bytes
  if b.data.size != expected then
    throw (IO.userError s!"TPUBuffer size mismatch: expected {expected} bytes, got {b.data.size}")

end TPUBuffer

/-- Queue item with checkpoint state after this batch. -/
structure QueuedTPUBuffer (shape : Shape) (dtype : DType) where
  buffer : TPUBuffer shape dtype
  state : IteratorState

/-- Background TPU data loader (host staging). -/
structure TPUDataLoader (batch : Nat) (itemShape : Shape) (dtype : DType) where
  queue : IOQueue (QueuedTPUBuffer (batch :: itemShape) dtype)
  worker : Task (Except IO.Error Unit)
  device : DeviceId
  numBatches : Nat
  lastState : IO.Ref IteratorState
  prefetcher? : Option (IteratorPrefetcher ByteArray)

namespace TPUDataLoader

/-! ## ByteArray helpers -/

/-- Concatenate chunks into one ByteArray with a single allocation. -/
private def concatByteArrays (chunks : Array ByteArray) : ByteArray := Id.run do
  let total := chunks.foldl (fun acc b => acc + b.size) 0
  let mut out := ByteArray.emptyWithCapacity total
  let mut offset := 0
  for chunk in chunks do
    out := ByteArray.copySlice chunk 0 out offset chunk.size false
    offset := offset + chunk.size
  out

/-- Build one batch from an iterator, returning the state after the batch. -/
private def nextBatch (iter : DataIterator ByteArray) (batchSize : Nat) (itemBytes : Nat) :
    IO (Option (ByteArray × IteratorState)) := do
  let mut chunks := Array.mkEmpty batchSize
  for _ in [:batchSize] do
    match ← iter.next with
    | some item =>
        if item.size != itemBytes then
          throw (IO.userError s!"TPUDataLoader.nextBatch: expected {itemBytes} bytes, got {item.size}")
        chunks := chunks.push item
    | none => return none
  let state ← iter.checkpoint
  let batchData := concatByteArrays chunks
  pure (some (batchData, state))

/-- Build one batch from a prefetcher, returning the state after the batch. -/
private def nextBatchFromPrefetcher (prefetcher : IteratorPrefetcher ByteArray) (batchSize : Nat)
    (itemBytes : Nat) :
    IO (Option (ByteArray × IteratorState)) := do
  let mut chunks := Array.mkEmpty batchSize
  for _ in [:batchSize] do
    match ← prefetcher.next with
    | some item =>
        if item.size != itemBytes then
          throw (IO.userError s!"TPUDataLoader.prefetch: expected {itemBytes} bytes, got {item.size}")
        chunks := chunks.push item
    | none => return none
  let state ← prefetcher.checkpoint
  let batchData := concatByteArrays chunks
  pure (some (batchData, state))

/-- Create loader from an iterator config (supports checkpoint/resume). -/
def createFromIteratorCfg [Dataset D ByteArray] (cfg : IteratorConfig D) (device : DeviceId)
    (batchSize : Nat) (itemShape : Shape) (bufferSize : Nat := 4) (dtype : DType := .float32)
    : IO (TPUDataLoader batchSize itemShape dtype) := do
  let iter ← Dataset.toIteratorCfg cfg
  let initState ← iter.checkpoint
  let queue ← IOQueue.new bufferSize
  let lastState ← IO.mkRef initState
  let n := Dataset.len cfg.base
  let remaining := remainingItems n cfg.epochs initState
  let numBatches := remaining / batchSize
  let itemBytes := Shape.numel itemShape * dtype.itemsize

  let worker ← IO.asTask (prio := .dedicated) do
    for _ in [:numBatches] do
      if ← IO.checkCanceled then break
      match ← nextBatch iter batchSize itemBytes with
      | some (batchData, state) =>
          let buf : TPUBuffer (batchSize :: itemShape) dtype := { data := batchData, device }
          buf.assertDevice device
          buf.assertSize
          let ok ← queue.push { buffer := buf, state }
          if !ok then break
      | none => break

    queue.finish

  pure { queue, worker, device, numBatches, lastState, prefetcher? := none }

/-- Create loader from an iterator config with a CPU-side prefetcher. -/
def createFromIteratorCfgPrefetch [Dataset D ByteArray] (cfg : IteratorConfig D) (device : DeviceId)
    (batchSize : Nat) (itemShape : Shape) (prefetchSize : Nat := 8)
    (bufferSize : Nat := 4) (dtype : DType := .float32) :
    IO (TPUDataLoader batchSize itemShape dtype) := do
  let prefetcher ← IteratorPrefetcher.createFromIteratorCfg cfg prefetchSize
  let initState ← prefetcher.checkpoint
  let queue ← IOQueue.new bufferSize
  let lastState ← IO.mkRef initState
  let n := Dataset.len cfg.base
  let remaining := remainingItems n cfg.epochs initState
  let numBatches := remaining / batchSize
  let itemBytes := Shape.numel itemShape * dtype.itemsize

  let worker ← IO.asTask (prio := .dedicated) do
    for _ in [:numBatches] do
      if ← IO.checkCanceled then break
      match ← nextBatchFromPrefetcher prefetcher batchSize itemBytes with
      | some (batchData, state) =>
          let buf : TPUBuffer (batchSize :: itemShape) dtype := { data := batchData, device }
          buf.assertDevice device
          buf.assertSize
          let ok ← queue.push { buffer := buf, state }
          if !ok then break
      | none => break

    queue.finish

  pure { queue, worker, device, numBatches, lastState, prefetcher? := some prefetcher }

/-- Create loader from a dataset of ByteArrays. -/
def create [Dataset D ByteArray] (ds : D) (device : DeviceId)
    (batchSize : Nat) (itemShape : Shape) (bufferSize : Nat := 4) (dtype : DType := .float32)
    : IO (TPUDataLoader batchSize itemShape dtype) := do
  match device with
  | .tpu _ => pure ()
  | _ => throw (IO.userError "TPUDataLoader: device must be .tpu")

  let cfg : IteratorConfig D := { base := ds }
  createFromIteratorCfg cfg device batchSize itemShape bufferSize dtype

/-- Get next TPU buffer (blocks until available). -/
def next (loader : TPUDataLoader batch itemShape dtype) : IO (Option (TPUBuffer (batch :: itemShape) dtype)) := do
  match ← loader.queue.pop with
  | some item =>
      loader.lastState.set item.state
      pure (some item.buffer)
  | none => pure none

/-- Get next buffer and wait time spent in queue (ns). -/
def nextWithWait (loader : TPUDataLoader batch itemShape dtype) :
    IO (Option (TPUBuffer (batch :: itemShape) dtype) × Nat) := do
  let (item?, waitNs) ← loader.queue.popWithWait
  match item? with
  | some item =>
      loader.lastState.set item.state
      pure (some item.buffer, waitNs)
  | none => pure (none, waitNs)

/-- Stop the loader. -/
def stop (loader : TPUDataLoader batch itemShape dtype) : IO Unit := do
  loader.queue.finish
  IO.cancel loader.worker
  match loader.prefetcher? with
  | some p => p.cancel
  | none => pure ()

/-- Wait for loader to complete. -/
def wait (loader : TPUDataLoader batch itemShape dtype) : IO Unit := do
  let _ ← IO.wait loader.worker
  match loader.prefetcher? with
  | some p => p.wait
  | none => pure ()

/-- Get checkpoint state for deterministic resume. -/
def checkpoint (loader : TPUDataLoader batch itemShape dtype) : IO IteratorState :=
  loader.lastState.get

/-- Drain queued buffers. -/
def drain (loader : TPUDataLoader batch itemShape dtype) : IO Unit := do
  repeat do
    match ← loader.queue.pop with
    | some _ => pure ()
    | none => break
  match loader.prefetcher? with
  | some p => p.drain
  | none => pure ()

end TPUDataLoader

end TinyGrad4.Data.TPULoader

/-! ## ForIn Instances -/

instance : ForIn IO (TinyGrad4.Data.TPULoader.TPUDataLoader batch itemShape dtype)
    (TinyGrad4.Data.TPULoader.TPUBuffer (batch :: itemShape) dtype) where
  forIn loader init f := do
    let mut acc := init
    repeat do
      match ← TinyGrad4.Data.TPULoader.TPUDataLoader.next loader with
      | none => break
      | some buf =>
        match ← f buf acc with
        | .done a => return a
        | .yield a => acc := a
    pure acc
