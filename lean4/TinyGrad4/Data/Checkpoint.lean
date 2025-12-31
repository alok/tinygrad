import TinyGrad4.Data.Dataset

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
def withCheckpointing [Dataset D T] (config : CheckpointConfig) (ds : D)
    (trainStep : T → Nat → IO Unit) : IO Unit := do
  -- Create checkpoint directory
  IO.FS.createDirAll config.checkpointDir

  let manager ← CheckpointManager.new config.checkpointDir config.saveInterval

  -- Try to resume from checkpoint
  let mut startPos := 0
  let mut startEpoch := 0
  let mut key := RandKey.new 42

  if config.resumeIfExists then
    if let some state ← manager.loadLatest then
      IO.println s!"Resuming from checkpoint: pos={state.position}, epoch={state.epoch}"
      startPos := state.position
      startEpoch := state.epoch
      key := state.key

  -- Training loop
  let n := Dataset.len ds
  let mut iteration := 0

  for epoch in [startEpoch:startEpoch + 10] do  -- Example: 10 epochs
    for i in [startPos:n] do
      if h : i < n then
        let item ← Dataset.getItem ds i h
        trainStep item iteration

        -- Checkpoint
        let state : IteratorState := {
          position := i + 1
          epoch := epoch
          key := key
        }
        manager.step state

        iteration := iteration + 1

    -- Reset position for next epoch
    startPos := 0
    key := key.fold epoch.toUInt64  -- New key for each epoch

end TinyGrad4.Data
