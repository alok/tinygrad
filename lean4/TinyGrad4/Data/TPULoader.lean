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

/-- Host-backed TPU buffer. -/
structure TPUBuffer where
  data : ByteArray
  dtype : DType
  device : DeviceId
  deriving Repr

namespace TPUBuffer

/-- Ensure a buffer is on the expected device. -/
def assertDevice (b : TPUBuffer) (device : DeviceId) : IO Unit := do
  if b.device != device then
    throw (IO.userError s!"TPUBuffer on {b.device}, expected {device}")

end TPUBuffer

/-- Queue item with checkpoint state after this batch. -/
structure QueuedTPUBuffer where
  buffer : TPUBuffer
  state : IteratorState

/-- Background TPU data loader (host staging). -/
structure TPUDataLoader where
  queue : IOQueue QueuedTPUBuffer
  worker : Task (Except IO.Error Unit)
  device : DeviceId
  numBatches : Nat
  lastState : IO.Ref IteratorState

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
private def nextBatch (iter : DataIterator ByteArray) (batchSize : Nat) :
    IO (Option (ByteArray × IteratorState)) := do
  let mut chunks := Array.mkEmpty batchSize
  for _ in [:batchSize] do
    match ← iter.next with
    | some item => chunks := chunks.push item
    | none => return none
  let state ← iter.checkpoint
  let batchData := concatByteArrays chunks
  pure (some (batchData, state))

/-- Create loader from an iterator config (supports checkpoint/resume). -/
def createFromIteratorCfg [Dataset D ByteArray] (cfg : IteratorConfig D) (device : DeviceId)
    (batchSize : Nat) (bufferSize : Nat := 4) (dtype : DType := .float32)
    : IO TPUDataLoader := do
  let iter ← Dataset.toIteratorCfg cfg
  let initState ← iter.checkpoint
  let queue ← IOQueue.new bufferSize
  let lastState ← IO.mkRef initState
  let n := Dataset.len cfg.base
  let remaining := remainingItems n cfg.epochs initState
  let numBatches := remaining / batchSize

  let worker ← IO.asTask (prio := .dedicated) do
    for _ in [:numBatches] do
      if ← IO.checkCanceled then break
      match ← nextBatch iter batchSize with
      | some (batchData, state) =>
          let buf : TPUBuffer := { data := batchData, dtype, device }
          buf.assertDevice device
          let ok ← queue.push { buffer := buf, state }
          if !ok then break
      | none => break

    queue.finish

  pure { queue, worker, device, numBatches, lastState }

/-- Create loader from a dataset of ByteArrays. -/
def create [Dataset D ByteArray] (ds : D) (device : DeviceId)
    (batchSize : Nat) (bufferSize : Nat := 4) (dtype : DType := .float32)
    : IO TPUDataLoader := do
  match device with
  | .tpu _ => pure ()
  | _ => throw (IO.userError "TPUDataLoader: device must be .tpu")

  let cfg : IteratorConfig D := { base := ds }
  createFromIteratorCfg cfg device batchSize bufferSize dtype

/-- Get next TPU buffer (blocks until available). -/
def next (loader : TPUDataLoader) : IO (Option TPUBuffer) := do
  match ← loader.queue.pop with
  | some item =>
      loader.lastState.set item.state
      pure (some item.buffer)
  | none => pure none

/-- Get next buffer and wait time spent in queue (ns). -/
def nextWithWait (loader : TPUDataLoader) : IO (Option TPUBuffer × Nat) := do
  let (item?, waitNs) ← loader.queue.popWithWait
  match item? with
  | some item =>
      loader.lastState.set item.state
      pure (some item.buffer, waitNs)
  | none => pure (none, waitNs)

/-- Stop the loader. -/
def stop (loader : TPUDataLoader) : IO Unit := do
  loader.queue.finish
  IO.cancel loader.worker

/-- Wait for loader to complete. -/
def wait (loader : TPUDataLoader) : IO Unit := do
  let _ ← IO.wait loader.worker

/-- Get checkpoint state for deterministic resume. -/
def checkpoint (loader : TPUDataLoader) : IO IteratorState :=
  loader.lastState.get

/-- Drain queued buffers. -/
def drain (loader : TPUDataLoader) : IO Unit := do
  repeat do
    match ← loader.queue.pop with
    | some _ => pure ()
    | none => break

end TPUDataLoader

end TinyGrad4.Data.TPULoader

/-! ## ForIn Instances -/

instance : ForIn IO TinyGrad4.Data.TPULoader.TPUDataLoader TinyGrad4.Data.TPULoader.TPUBuffer where
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
