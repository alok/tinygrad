import TinyGrad4.Data.Dataset
import TinyGrad4.Data.Prefetch

/-!
# Checkpoint - Training State Serialization

Enables saving and restoring training state for fault-tolerant distributed training.

## What Gets Checkpointed
- Iterator position (index into dataset)
- Epoch number
- Random key state (for reproducible shuffling)

## Usage
```lean
-- Save checkpoint
let state ← iterator.checkpoint
saveCheckpoint "checkpoint.bin" state

-- Restore from checkpoint
let state ← loadCheckpoint "checkpoint.bin"
iterator.restore state
```
-/

namespace TinyGrad4.Data

/-! ## Checkpoint State Serialization -/

/-- Serialize UInt64 to bytes (little-endian) -/
def serializeUInt64 (v : UInt64) : ByteArray :=
  let b0 := (v &&& 0xFF).toUInt8
  let b1 := ((v >>> 8) &&& 0xFF).toUInt8
  let b2 := ((v >>> 16) &&& 0xFF).toUInt8
  let b3 := ((v >>> 24) &&& 0xFF).toUInt8
  let b4 := ((v >>> 32) &&& 0xFF).toUInt8
  let b5 := ((v >>> 40) &&& 0xFF).toUInt8
  let b6 := ((v >>> 48) &&& 0xFF).toUInt8
  let b7 := ((v >>> 56) &&& 0xFF).toUInt8
  ByteArray.mk #[b0, b1, b2, b3, b4, b5, b6, b7]

/-- Deserialize UInt64 from bytes (little-endian) -/
def deserializeUInt64 (bytes : ByteArray) (offset : Nat := 0) : Option UInt64 :=
  if offset + 8 > bytes.size then none
  else
    let b0 := bytes[offset]!.toUInt64
    let b1 := bytes[offset + 1]!.toUInt64
    let b2 := bytes[offset + 2]!.toUInt64
    let b3 := bytes[offset + 3]!.toUInt64
    let b4 := bytes[offset + 4]!.toUInt64
    let b5 := bytes[offset + 5]!.toUInt64
    let b6 := bytes[offset + 6]!.toUInt64
    let b7 := bytes[offset + 7]!.toUInt64
    some (b0 ||| (b1 <<< 8) ||| (b2 <<< 16) ||| (b3 <<< 24) |||
          (b4 <<< 32) ||| (b5 <<< 40) ||| (b6 <<< 48) ||| (b7 <<< 56))

/-- Serialize Nat to bytes (as UInt64, capped) -/
def serializeNat (n : Nat) : ByteArray :=
  serializeUInt64 n.toUInt64

/-- Deserialize Nat from bytes -/
def deserializeNat (bytes : ByteArray) (offset : Nat := 0) : Option Nat :=
  (deserializeUInt64 bytes offset).map UInt64.toNat

/-! ## IteratorState Serialization -/

/-- Magic number for checkpoint format validation -/
def checkpointMagic : UInt64 := 0x544734434B505400  -- "TG4CKPT\0"

/-- Checkpoint format version -/
def checkpointVersion : UInt64 := 1

/-- Serialize IteratorState to bytes -/
def serializeIteratorState (state : IteratorState) : ByteArray :=
  serializeUInt64 checkpointMagic ++
  serializeUInt64 checkpointVersion ++
  serializeNat state.position ++
  serializeNat state.epoch ++
  serializeUInt64 state.key.state

/-- Deserialize IteratorState from bytes -/
def deserializeIteratorState (bytes : ByteArray) : Option IteratorState := do
  -- Validate magic
  let magic ← deserializeUInt64 bytes 0
  guard (magic == checkpointMagic)

  -- Validate version
  let version ← deserializeUInt64 bytes 8
  guard (version == checkpointVersion)

  -- Read state
  let position ← deserializeNat bytes 16
  let epoch ← deserializeNat bytes 24
  let keyState ← deserializeUInt64 bytes 32

  pure {
    position := position
    epoch := epoch
    key := { state := keyState }
  }

/-! ## MultiIteratorState Serialization -/

/-- Magic number for multi-iterator checkpoints. -/
def multiCheckpointMagic : UInt64 := 0x5447344D4B505400  -- "TG4MKPT\0"

/-- Multi-iterator checkpoint format version. -/
def multiCheckpointVersion : UInt64 := 2

/-- Serialize MultiIteratorState to bytes. -/
def serializeMultiIteratorState (state : MultiIteratorState) : ByteArray := Id.run do
  let mut out := serializeUInt64 multiCheckpointMagic ++ serializeUInt64 multiCheckpointVersion
  out := out ++ serializeNat state.workerStates.size
  out := out ++ serializeNat state.nextWorker
  for st in state.workerStates do
    out := out ++ serializeNat st.position
    out := out ++ serializeNat st.epoch
    out := out ++ serializeUInt64 st.key.state
  out := out ++ serializeNat state.produced.size
  for count in state.produced do
    out := out ++ serializeNat count
  out

private def readStates (bytes : ByteArray) : Nat → Nat → Array IteratorState → Option (Array IteratorState)
  | 0, _, acc => some acc
  | n + 1, offset, acc =>
      match deserializeNat bytes offset, deserializeNat bytes (offset + 8), deserializeUInt64 bytes (offset + 16) with
      | some position, some epoch, some keyState =>
          let st : IteratorState := { position, epoch, key := { state := keyState } }
          readStates bytes n (offset + 24) (acc.push st)
      | _, _, _ => none

private def readProduced (bytes : ByteArray) : Nat → Nat → Array Nat → Option (Array Nat)
  | 0, _, acc => some acc
  | n + 1, offset, acc =>
      match deserializeNat bytes offset with
      | some count => readProduced bytes n (offset + 8) (acc.push count)
      | none => none

/-- Deserialize MultiIteratorState from bytes. -/
def deserializeMultiIteratorState (bytes : ByteArray) : Option MultiIteratorState := do
  let magic ← deserializeUInt64 bytes 0
  guard (magic == multiCheckpointMagic)
  let version ← deserializeUInt64 bytes 8
  let numWorkers ← deserializeNat bytes 16
  let nextWorker ← deserializeNat bytes 24
  let states ← readStates bytes numWorkers 32 (#[])
  if version == (1 : UInt64) then
    pure { nextWorker, workerStates := states }
  else if version == multiCheckpointVersion then
    let producedOffset := 32 + numWorkers * 24
    let producedCount ← deserializeNat bytes producedOffset
    let produced ← readProduced bytes producedCount (producedOffset + 8) (#[])
    pure { nextWorker, workerStates := states, produced := produced }
  else
    none

/-! ## File-Based Checkpointing -/

/-- Save checkpoint to file -/
def saveCheckpoint (path : System.FilePath) (state : IteratorState) : IO Unit := do
  let bytes := serializeIteratorState state
  IO.FS.writeBinFile path bytes

/-- Load checkpoint from file -/
def loadCheckpoint (path : System.FilePath) : IO IteratorState := do
  let bytes ← IO.FS.readBinFile path
  match deserializeIteratorState bytes with
  | some state => pure state
  | none => throw (IO.userError s!"Invalid checkpoint file: {path}")

/-- Save multi-iterator checkpoint to file. -/
def saveMultiCheckpoint (path : System.FilePath) (state : MultiIteratorState) : IO Unit := do
  let bytes := serializeMultiIteratorState state
  IO.FS.writeBinFile path bytes

/-- Load multi-iterator checkpoint from file. -/
def loadMultiCheckpoint (path : System.FilePath) : IO MultiIteratorState := do
  let bytes ← IO.FS.readBinFile path
  match deserializeMultiIteratorState bytes with
  | some state => pure state
  | none => throw (IO.userError s!"Invalid multi checkpoint file: {path}")

/-- Check if checkpoint exists -/
def checkpointExists (path : System.FilePath) : IO Bool :=
  path.pathExists

/-! ## Checkpoint Manager -/

/-- Manages periodic checkpointing during training -/
structure CheckpointManager where
  /-- Base path for checkpoints -/
  basePath : System.FilePath
  /-- How often to checkpoint (in iterations) -/
  interval : Nat
  /-- Keep last N checkpoints -/
  keepLast : Nat := 3
  /-- Current iteration counter -/
  iteration : IO.Ref Nat

namespace CheckpointManager

/-- Create a new checkpoint manager -/
def new (basePath : System.FilePath) (interval : Nat := 1000) (keepLast : Nat := 3) : IO CheckpointManager := do
  let iteration ← IO.mkRef 0
  pure { basePath, interval, keepLast, iteration }

/-- Generate checkpoint path for iteration -/
def checkpointPath (m : CheckpointManager) (iter : Nat) : System.FilePath :=
  m.basePath / s!"checkpoint_{iter}.bin"

/-- Get latest checkpoint path -/
def latestPath (m : CheckpointManager) : System.FilePath :=
  m.basePath / "checkpoint_latest.bin"

/-- Should checkpoint at current iteration? -/
def shouldCheckpoint (m : CheckpointManager) : IO Bool := do
  let iter ← m.iteration.get
  pure (iter > 0 && iter % m.interval == 0)

/-- Increment iteration and optionally checkpoint -/
def step (m : CheckpointManager) (state : IteratorState) : IO Unit := do
  m.iteration.modify (· + 1)
  if ← m.shouldCheckpoint then
    let iter ← m.iteration.get
    -- Save numbered checkpoint
    saveCheckpoint (m.checkpointPath iter) state
    -- Update latest symlink/copy
    saveCheckpoint m.latestPath state
    -- TODO: Clean old checkpoints (keep only keepLast)

/-- Load latest checkpoint if exists -/
def loadLatest (m : CheckpointManager) : IO (Option IteratorState) := do
  if ← checkpointExists m.latestPath then
    some <$> loadCheckpoint m.latestPath
  else
    pure none

end CheckpointManager

/-! ## Training Loop with Checkpointing -/

/-- Configuration for checkpointed training -/
structure CheckpointConfig where
  /-- Checkpoint directory -/
  checkpointDir : System.FilePath := "checkpoints"
  /-- Save every N iterations -/
  saveInterval : Nat := 1000
  /-- Resume from checkpoint if exists -/
  resumeIfExists : Bool := true

/-- Run training with automatic checkpointing -/
def withCheckpointingCfg [Dataset D T] (config : CheckpointConfig) (iterCfg : IteratorConfig D)
    (trainStep : T → Nat → IO Unit) : IO Unit := do
  -- Create checkpoint directory
  IO.FS.createDirAll config.checkpointDir

  let manager ← CheckpointManager.new config.checkpointDir config.saveInterval

  -- Try to resume from checkpoint
  let mut cfg := iterCfg

  if config.resumeIfExists then
    if let some state ← manager.loadLatest then
      IO.println s!"Resuming from checkpoint: pos={state.position}, epoch={state.epoch}"
      cfg := {
        cfg with
        startPos := state.position
        startEpoch := state.epoch
        key := state.key
      }

  -- Training loop
  let iter ← Dataset.toIteratorCfg cfg
  let mut iteration := 0
  repeat do
    match ← iter.next with
    | none => break
    | some item =>
        trainStep item iteration
        manager.step (← iter.checkpoint)
        iteration := iteration + 1

/-- Run training with automatic checkpointing (default iterator config). -/
def withCheckpointing [Dataset D T] (config : CheckpointConfig) (ds : D)
    (trainStep : T → Nat → IO Unit) : IO Unit := do
  withCheckpointingCfg config {
    base := ds,
    epochs := 10
  } trainStep

end TinyGrad4.Data
