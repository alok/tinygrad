import TinyGrad4.Data.Dataset
import TinyGrad4.Data.Shuffle
import TinyGrad4.Data.GPULoader
import TinyGrad4.Data.TPULoader

/-!
# Device Loader Resume Smoke Test

Stress test deterministic resume for GPU/TPU loaders by interrupting
mid-epoch and verifying prefix/remainder equality.
-/

namespace TinyGrad4.Test.DeviceLoaderResumeSmoke

open TinyGrad4.Data
open TinyGrad4.Data.GPULoader
open TinyGrad4.Data.TPULoader

/-- Assert condition with message. -/
def assert (cond : Bool) (msg : String) : IO Unit := do
  if !cond then
    throw (IO.userError s!"Assertion failed: {msg}")

def firstMismatch (a b : Array UInt64) : Option (Nat × UInt64 × UInt64) := Id.run do
  let n := Nat.min a.size b.size
  for i in [:n] do
    if a[i]! != b[i]! then
      return some (i, a[i]!, b[i]!)
  if a.size != b.size then
    return some (n, a.getD n 0, b.getD n 0)
  none

def dumpMismatch (label : String) (a b : Array UInt64) : IO Unit := do
  match firstMismatch a b with
  | none => pure ()
  | some (i, va, vb) =>
      IO.eprintln s!"{label}: first mismatch at {i}: expected={va} got={vb} (sizes {a.size}/{b.size})"

/-- Encode a Nat into 4 bytes (little-endian). -/
def encodeNat (n : Nat) : ByteArray :=
  let v : UInt32 := n.toUInt32
  let b0 : UInt8 := (v &&& 0xFF).toUInt8
  let b1 : UInt8 := ((v >>> 8) &&& 0xFF).toUInt8
  let b2 : UInt8 := ((v >>> 16) &&& 0xFF).toUInt8
  let b3 : UInt8 := ((v >>> 24) &&& 0xFF).toUInt8
  ByteArray.mk #[b0, b1, b2, b3]

/-- Simple checksum (sum of bytes). -/
def checksum (ba : ByteArray) : UInt64 := Id.run do
  let mut acc : UInt64 := 0
  for i in [:ba.size] do
    acc := acc + ba[i]!.toUInt64
  acc

/-- Sleep helper (ms). -/
def sleepMs (ms : Nat) : IO Unit :=
  IO.sleep (UInt32.ofNat ms)

/-- Drop first n elements from an array. -/
def arrayDrop (arr : Array T) (n : Nat) : Array T := Id.run do
  let start := min n arr.size
  let mut out := Array.mkEmpty (arr.size - start)
  for i in [start:arr.size] do
    if h : i < arr.size then
      out := out.push (arr[i]'(h))
  out

/-- Take first n elements from an array. -/
def arrayTake (arr : Array T) (n : Nat) : Array T := Id.run do
  let stop := min n arr.size
  let mut out := Array.mkEmpty stop
  for i in [:stop] do
    if h : i < arr.size then
      out := out.push (arr[i]'(h))
  out

/-- Verify seen prefix and resumed remainder against baseline. -/
def verifyResume (label : String) (baseline seen remainder : Array UInt64) (split : Nat) : IO Unit := do
  let expectedPrefix := arrayTake baseline seen.size
  let expectedRemainder := arrayDrop baseline seen.size
  if seen != expectedPrefix then
    dumpMismatch s!"{label} prefix" expectedPrefix seen
  if remainder != expectedRemainder then
    dumpMismatch s!"{label} remainder" expectedRemainder remainder
  assert (seen == expectedPrefix) s!"{label} prefix mismatch at split={split}"
  assert (remainder == expectedRemainder) s!"{label} remainder mismatch at split={split}"

/-- Build iterator config with cached shuffle per epoch. -/
def buildCfg (n epochs : Nat) (key : RandKey) :
    IteratorConfig (CachedShuffledDataset (ArrayDataset ByteArray) ByteArray) := by
  let baseRaw := ofArray ((Array.range n).map encodeNat)
  let base := shuffleDsCachedAtEpoch key 0 baseRaw
  exact {
    base := base,
    epochs := epochs,
    key := key,
    datasetAtEpoch := fun ds k epoch => shuffleDsCachedAtEpoch k epoch ds.inner
  }

/-- Collect batch checksums from a GPU loader. -/
def collectGPU (cfg : IteratorConfig D) (device : DeviceId) (batchSize : Nat)
    (prefetchSize? : Option Nat := none) [Dataset D ByteArray] : IO (Array UInt64) := do
  let loader ← match prefetchSize? with
    | some p => GPUDataLoader.createFromIteratorCfgPrefetch cfg device batchSize p (bufferSize := 2) (dtype := .uint8)
    | none => GPUDataLoader.createFromIteratorCfg cfg device batchSize (bufferSize := 2) (dtype := .uint8)
  let mut out := #[]
  for buf in loader do
    let bytes ← buf.copyOut
    out := out.push (checksum bytes)
    buf.free
  loader.stop
  loader.drain
  pure out

/-- Collect batch checksums from a MultiGPULoader. -/
def collectMultiGPU (cfg : IteratorConfig D) (devices : Array DeviceId) (batchSize : Nat)
    (world? : Option WorldConfig := none)
    (states? : Option (Array (DeviceId × IteratorState)) := none)
    (prefetchSize? : Option Nat := none)
    [Dataset D ByteArray] : IO (Array UInt64) := do
  let pool ← match prefetchSize? with
    | some p =>
        MultiGPULoader.createFromIteratorCfgPrefetch cfg devices batchSize p (bufferSize := 2) (dtype := .uint8)
          (world? := world?) (states? := states?)
    | none =>
        MultiGPULoader.createFromIteratorCfg cfg devices batchSize (bufferSize := 2) (dtype := .uint8)
          (world? := world?) (states? := states?)
  let mut out := #[]
  repeat do
    let batches ← pool.nextAll
    if batches.all (fun (_, b) => b.isNone) then
      break
    for (_, buf?) in batches do
      match buf? with
      | some buf =>
          let bytes ← buf.copyOut
          out := out.push (checksum bytes)
          buf.free
      | none => pure ()
  pool.stop
  pool.drain
  pure out

/-- Collect batch checksums from a TPU loader. -/
def collectTPU (cfg : IteratorConfig D) (device : DeviceId) (batchSize : Nat)
    (prefetchSize? : Option Nat := none) [Dataset D ByteArray] : IO (Array UInt64) := do
  let loader ← match prefetchSize? with
    | some p => TPUDataLoader.createFromIteratorCfgPrefetch cfg device batchSize p (bufferSize := 2) (dtype := .uint8)
    | none => TPUDataLoader.createFromIteratorCfg cfg device batchSize (bufferSize := 2) (dtype := .uint8)
  let mut out := #[]
  for buf in loader do
    out := out.push (checksum buf.data)
  loader.stop
  loader.drain
  pure out

/-- Resume test for GPU loader at a split point. -/
def resumeGPUAtSplit (cfg : IteratorConfig D) (device : DeviceId) (batchSize : Nat)
    (baseline : Array UInt64) (split : Nat) (prefetchSize? : Option Nat := none)
    [Dataset D ByteArray] : IO Unit := do
  let loader ← match prefetchSize? with
    | some p => GPUDataLoader.createFromIteratorCfgPrefetch cfg device batchSize p (bufferSize := 2) (dtype := .uint8)
    | none => GPUDataLoader.createFromIteratorCfg cfg device batchSize (bufferSize := 2) (dtype := .uint8)
  let mut seen := (#[] : Array UInt64)
  let mut count := 0
  while count < split do
    match ← loader.next with
    | some buf =>
        let bytes ← buf.copyOut
        seen := seen.push (checksum bytes)
        buf.free
        count := count + 1
    | none => break

  let state ← loader.checkpoint
  loader.stop
  loader.drain

  let cfg' := {
    cfg with
    startPos := state.position,
    startEpoch := state.epoch,
    key := state.key
  }

  let remainder ← collectGPU cfg' device batchSize (prefetchSize? := prefetchSize?)
  let label := match prefetchSize? with
    | some _ => "GPU prefetch"
    | none => "GPU"
  verifyResume label baseline seen remainder split

/-- Interrupt test for GPU loader at a split point (adversarial cancellation). -/
def interruptGPUAtSplit (cfg : IteratorConfig D) (device : DeviceId) (batchSize : Nat)
    (baseline : Array UInt64) (split : Nat) (interruptDelayMs : Nat := 5)
    (prefetchSize? : Option Nat := none) [Dataset D ByteArray] : IO Unit := do
  let loader ← match prefetchSize? with
    | some p => GPUDataLoader.createFromIteratorCfgPrefetch cfg device batchSize p (bufferSize := 2) (dtype := .uint8)
    | none => GPUDataLoader.createFromIteratorCfg cfg device batchSize (bufferSize := 2) (dtype := .uint8)
  let mut seen := (#[] : Array UInt64)
  let mut count := 0
  while count < split do
    match ← loader.next with
    | some buf =>
        let bytes ← buf.copyOut
        seen := seen.push (checksum bytes)
        buf.free
        count := count + 1
    | none => break

  -- Let producer get ahead, then interrupt abruptly.
  sleepMs interruptDelayMs
  let state ← loader.checkpoint
  IO.cancel loader.worker
  loader.queue.finish
  loader.drain

  let cfg' := {
    cfg with
    startPos := state.position,
    startEpoch := state.epoch,
    key := state.key
  }

  let remainder ← collectGPU cfg' device batchSize (prefetchSize? := prefetchSize?)
  let label := match prefetchSize? with
    | some _ => "GPU interrupt prefetch"
    | none => "GPU interrupt"
  verifyResume label baseline seen remainder split

/-- Resume test for TPU loader at a split point. -/
def resumeTPUAtSplit (cfg : IteratorConfig D) (device : DeviceId) (batchSize : Nat)
    (baseline : Array UInt64) (split : Nat) (prefetchSize? : Option Nat := none)
    [Dataset D ByteArray] : IO Unit := do
  let loader ← match prefetchSize? with
    | some p => TPUDataLoader.createFromIteratorCfgPrefetch cfg device batchSize p (bufferSize := 2) (dtype := .uint8)
    | none => TPUDataLoader.createFromIteratorCfg cfg device batchSize (bufferSize := 2) (dtype := .uint8)
  let mut seen := (#[] : Array UInt64)
  let mut count := 0
  while count < split do
    match ← loader.next with
    | some buf =>
        seen := seen.push (checksum buf.data)
        count := count + 1
    | none => break

  let state ← loader.checkpoint
  loader.stop
  loader.drain

  let cfg' := {
    cfg with
    startPos := state.position,
    startEpoch := state.epoch,
    key := state.key
  }

  let remainder ← collectTPU cfg' device batchSize (prefetchSize? := prefetchSize?)
  let label := match prefetchSize? with
    | some _ => "TPU prefetch"
    | none => "TPU"
  verifyResume label baseline seen remainder split

/-- Interrupt test for TPU loader at a split point (adversarial cancellation). -/
def interruptTPUAtSplit (cfg : IteratorConfig D) (device : DeviceId) (batchSize : Nat)
    (baseline : Array UInt64) (split : Nat) (interruptDelayMs : Nat := 5)
    (prefetchSize? : Option Nat := none) [Dataset D ByteArray] : IO Unit := do
  let loader ← match prefetchSize? with
    | some p => TPUDataLoader.createFromIteratorCfgPrefetch cfg device batchSize p (bufferSize := 2) (dtype := .uint8)
    | none => TPUDataLoader.createFromIteratorCfg cfg device batchSize (bufferSize := 2) (dtype := .uint8)
  let mut seen := (#[] : Array UInt64)
  let mut count := 0
  while count < split do
    match ← loader.next with
    | some buf =>
        seen := seen.push (checksum buf.data)
        count := count + 1
    | none => break

  sleepMs interruptDelayMs
  let state ← loader.checkpoint
  IO.cancel loader.worker
  loader.queue.finish
  loader.drain

  let cfg' := {
    cfg with
    startPos := state.position,
    startEpoch := state.epoch,
    key := state.key
  }

  let remainder ← collectTPU cfg' device batchSize (prefetchSize? := prefetchSize?)
  let label := match prefetchSize? with
    | some _ => "TPU interrupt prefetch"
    | none => "TPU interrupt"
  verifyResume label baseline seen remainder split

/-- Resume test for MultiGPULoader at a split point (rounds). -/
def resumeMultiGPUAtSplit (cfg : IteratorConfig D) (devices : Array DeviceId) (batchSize : Nat)
    (baseline : Array UInt64) (split : Nat) (world? : Option WorldConfig := none)
    (prefetchSize? : Option Nat := none) [Dataset D ByteArray] : IO Unit := do
  let pool ← match prefetchSize? with
    | some p =>
        MultiGPULoader.createFromIteratorCfgPrefetch cfg devices batchSize p (bufferSize := 2) (dtype := .uint8)
          (world? := world?)
    | none =>
        MultiGPULoader.createFromIteratorCfg cfg devices batchSize (bufferSize := 2) (dtype := .uint8)
          (world? := world?)
  let mut seen := (#[] : Array UInt64)
  let mut rounds := 0
  while rounds < split do
    let batches ← pool.nextAll
    if batches.all (fun (_, b) => b.isNone) then
      break
    for (_, buf?) in batches do
      match buf? with
      | some buf =>
          let bytes ← buf.copyOut
          seen := seen.push (checksum bytes)
          buf.free
      | none => pure ()
    rounds := rounds + 1

  let states ← pool.checkpoint
  pool.stop
  pool.drain

  let remainder ← collectMultiGPU cfg devices batchSize (world? := world?) (states? := some states)
    (prefetchSize? := prefetchSize?)
  let label := match prefetchSize? with
    | some _ => "MultiGPU prefetch"
    | none => "MultiGPU"
  verifyResume label baseline seen remainder split

/-- Interrupt test for MultiGPULoader at a split point (adversarial cancellation). -/
def interruptMultiGPUAtSplit (cfg : IteratorConfig D) (devices : Array DeviceId) (batchSize : Nat)
    (baseline : Array UInt64) (split : Nat) (world? : Option WorldConfig := none)
    (interruptDelayMs : Nat := 5) (prefetchSize? : Option Nat := none)
    [Dataset D ByteArray] : IO Unit := do
  let pool ← match prefetchSize? with
    | some p =>
        MultiGPULoader.createFromIteratorCfgPrefetch cfg devices batchSize p (bufferSize := 2) (dtype := .uint8)
          (world? := world?)
    | none =>
        MultiGPULoader.createFromIteratorCfg cfg devices batchSize (bufferSize := 2) (dtype := .uint8)
          (world? := world?)
  let mut seen := (#[] : Array UInt64)
  let mut rounds := 0
  while rounds < split do
    let batches ← pool.nextAll
    if batches.all (fun (_, b) => b.isNone) then
      break
    for (_, buf?) in batches do
      match buf? with
      | some buf =>
          let bytes ← buf.copyOut
          seen := seen.push (checksum bytes)
          buf.free
      | none => pure ()
    rounds := rounds + 1

  sleepMs interruptDelayMs
  let states ← pool.checkpoint
  for w in pool.workers do
    IO.cancel w.worker
  for w in pool.workers do
    w.queue.finish
  pool.drain

  let remainder ← collectMultiGPU cfg devices batchSize (world? := world?) (states? := some states)
    (prefetchSize? := prefetchSize?)
  let label := match prefetchSize? with
    | some _ => "MultiGPU interrupt prefetch"
    | none => "MultiGPU interrupt"
  verifyResume label baseline seen remainder split

/-- Run all checks. -/
def runAll : IO Unit := do
  IO.println "Device loader resume smoke test..."
  let key := RandKey.new 123
  let cfg := buildCfg 128 3 key
  let batchSize := 8
  let world ← WorldConfig.fromEnv
  let world? := if world.worldSize > 1 then some world else none

  -- GPU (if available)
  let devices ← discoverDevices
  let gpuDevices := devices.filter fun d => d != .cpu && (match d with | .tpu _ => false | _ => true)
  if gpuDevices.isEmpty then
    IO.println "No GPU devices - skipping GPU resume test"
  else
    let dev := gpuDevices[0]!
    let baseline ← collectGPU cfg dev batchSize
    let splits := #[0, 1, 3, 5, baseline.size / 2, baseline.size]
    for split in splits do
      resumeGPUAtSplit cfg dev batchSize baseline split
    interruptGPUAtSplit cfg dev batchSize baseline 3
    let prefetchSize? : Option Nat := some 4
    resumeGPUAtSplit cfg dev batchSize baseline 3 (prefetchSize? := prefetchSize?)
    interruptGPUAtSplit cfg dev batchSize baseline 2 (prefetchSize? := prefetchSize?)
    IO.println "✓ GPU resume test passed"

  -- Multi-GPU (if available)
  if gpuDevices.size >= 2 then
    let baselineMulti ← collectMultiGPU cfg gpuDevices batchSize (world? := world?)
    let splitsMulti := #[0, 1, 3, baselineMulti.size / 4, baselineMulti.size / 2, baselineMulti.size]
    for split in splitsMulti do
      resumeMultiGPUAtSplit cfg gpuDevices batchSize baselineMulti split (world? := world?)
    interruptMultiGPUAtSplit cfg gpuDevices batchSize baselineMulti 2 (world? := world?)
    let prefetchSize? : Option Nat := some 4
    resumeMultiGPUAtSplit cfg gpuDevices batchSize baselineMulti 2 (world? := world?) (prefetchSize? := prefetchSize?)
    interruptMultiGPUAtSplit cfg gpuDevices batchSize baselineMulti 1 (world? := world?) (prefetchSize? := prefetchSize?)
    IO.println "✓ MultiGPU resume test passed"

  -- TPU (host-backed)
  let tpuDev : DeviceId := .tpu 0
  let baselineTpu ← collectTPU cfg tpuDev batchSize
  let splitsTpu := #[0, 1, 3, 5, baselineTpu.size / 2, baselineTpu.size]
  for split in splitsTpu do
    resumeTPUAtSplit cfg tpuDev batchSize baselineTpu split
  interruptTPUAtSplit cfg tpuDev batchSize baselineTpu 3
  let prefetchSize? : Option Nat := some 4
  resumeTPUAtSplit cfg tpuDev batchSize baselineTpu 3 (prefetchSize? := prefetchSize?)
  interruptTPUAtSplit cfg tpuDev batchSize baselineTpu 2 (prefetchSize? := prefetchSize?)
  IO.println "✓ TPU resume test passed"

end TinyGrad4.Test.DeviceLoaderResumeSmoke

/-- Entry point -/
def main : IO Unit := TinyGrad4.Test.DeviceLoaderResumeSmoke.runAll
