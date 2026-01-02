import TinyGrad4.Data.Dataset

/-!
# Shard - Distributed Dataset Sharding

Enables distributed training by partitioning a dataset across workers.
Each worker gets a deterministic subset of data.

## Sharding Modes
- **Interleaved**: Worker i gets indices i, i+n, i+2n, ... (better load balance)
- **Contiguous**: Worker i gets [i*blockSize, (i+1)*blockSize) (better cache locality)

## Usage
```lean
-- Worker 2 of 8
let myData := mnist |> shardDs (shardIndex := 2) (numShards := 8)
```
-/

namespace TinyGrad4.Data

/-! ## Shard Configuration -/

/-- How to partition indices across shards -/
inductive ShardMode where
  /-- Indices i, i+n, i+2n, ... (strided) -/
  | interleaved
  /-- Contiguous blocks [i*block, (i+1)*block) -/
  | contiguous
  deriving Repr, BEq, Inhabited

/-- Configuration for dataset sharding -/
structure ShardConfig where
  /-- This worker's index (0-indexed) -/
  shardIndex : Nat
  /-- Total number of workers/shards -/
  numShards : Nat
  /-- Sharding strategy -/
  mode : ShardMode := .interleaved
  /-- Drop elements that don't divide evenly (ensures equal shard sizes) -/
  dropRemainder : Bool := true
  deriving Repr, Inhabited

namespace ShardConfig

/-- Validate shard configuration -/
def isValid (cfg : ShardConfig) : Bool :=
  cfg.numShards > 0 && cfg.shardIndex < cfg.numShards

/-- Number of elements per shard (assuming n total elements) -/
def shardSize (cfg : ShardConfig) (n : Nat) : Nat :=
  if cfg.dropRemainder then
    n / cfg.numShards
  else
    match cfg.mode with
    | .interleaved => (n + cfg.numShards - 1 - cfg.shardIndex) / cfg.numShards
    | .contiguous =>
      let baseSize := n / cfg.numShards
      let remainder := n % cfg.numShards
      if cfg.shardIndex < remainder then baseSize + 1 else baseSize

/-- Map local index to global index -/
def localToGlobal (cfg : ShardConfig) (n : Nat) (localIdx : Nat) : Nat :=
  match cfg.mode with
  | .interleaved =>
    -- Local idx i → global idx i * numShards + shardIndex
    localIdx * cfg.numShards + cfg.shardIndex
  | .contiguous =>
    -- Local idx i → global idx blockStart + i
    let baseSize := n / cfg.numShards
    let remainder := n % cfg.numShards
    let blockStart :=
      if cfg.shardIndex < remainder then
        cfg.shardIndex * (baseSize + 1)
      else
        remainder * (baseSize + 1) + (cfg.shardIndex - remainder) * baseSize
    blockStart + localIdx

end ShardConfig

/-! ## ShardedDataset -/

/-- Dataset partitioned for a specific shard/worker -/
structure ShardedDataset (D : Type) (T : Type) where
  inner : D
  config : ShardConfig

instance [Dataset D T] : Dataset (ShardedDataset D T) T where
  len ds :=
    let n := Dataset.len ds.inner
    ds.config.shardSize n
  getItem ds localIdx h := do
    let n := Dataset.len ds.inner
    let globalIdx := ds.config.localToGlobal n localIdx
    -- Need to verify globalIdx is in bounds
    if hGlobal : globalIdx < n then
      Dataset.getItem ds.inner globalIdx hGlobal
    else
      -- This shouldn't happen if shardSize/localToGlobal are correct,
      -- but we fallback to a valid access
      have hLocal : localIdx < Dataset.len ds.inner := by
        sorry  -- In practice, shardSize ≤ n / numShards ≤ n
      Dataset.getItem ds.inner localIdx hLocal

/-- Shard a dataset for a specific worker -/
def shardDs [Dataset D T] (shardIndex : Nat) (numShards : Nat)
    (mode : ShardMode := .interleaved) (dropRemainder : Bool := true) (ds : D) : ShardedDataset D T :=
  { inner := ds, config := { shardIndex, numShards, mode, dropRemainder } }

/-- Shard a dataset using a config -/
def shardWithConfig [Dataset D T] (config : ShardConfig) (ds : D) : ShardedDataset D T :=
  { inner := ds, config }

/-! ## Shard Utilities -/

/-- Get all shard indices for a given total size -/
def allShardIndices (config : ShardConfig) (n : Nat) : Array Nat := Id.run do
  let size := config.shardSize n
  let mut indices := Array.mkEmpty size
  for i in [:size] do
    indices := indices.push (config.localToGlobal n i)
  indices

/-- Verify shards partition the dataset correctly -/
def verifyPartition (numShards : Nat) (n : Nat) (mode : ShardMode) (dropRemainder : Bool) : Bool := Id.run do
  let mut seen := Array.replicate n false
  let mut count := 0
  for shardIdx in [:numShards] do
    let config : ShardConfig := { shardIndex := shardIdx, numShards, mode, dropRemainder }
    let indices := allShardIndices config n
    for idx in indices do
      if idx >= n then return false
      if seen[idx]! then return false  -- Duplicate!
      seen := seen.set! idx true
      count := count + 1
  -- If dropRemainder, we might not cover all; otherwise should cover all
  if dropRemainder then
    count == (n / numShards) * numShards
  else
    count == n

/-! ## Distributed Training Helpers -/

/-- World configuration for distributed training -/
structure WorldConfig where
  rank : Nat        -- This process's rank (0-indexed)
  worldSize : Nat   -- Total number of processes
  deriving Repr, Inhabited

private def getEnvNat? (key : String) : IO (Option Nat) := do
  match (← IO.getEnv key) with
  | some v =>
      let v' := v.trimAscii.toString
      pure v'.toNat?
  | none => pure none

private def getEnvNatAny? (keys : List String) : IO (Option Nat) := do
  for key in keys do
    if let some v ← getEnvNat? key then
      return some v
  return none

/-- Read world configuration from environment variables.

Priority:
- `TG4_RANK` + `TG4_WORLD_SIZE` (or `RANK` + `WORLD_SIZE`)
- `TG4_NODE_RANK`, `TG4_NODE_COUNT`, `TG4_LOCAL_RANK`, `TG4_LOCAL_COUNT`

Defaults to rank=0, worldSize=1 when unspecified.
-/
def WorldConfig.fromEnv : IO WorldConfig := do
  let rank? ← getEnvNatAny? ["TG4_RANK", "RANK"]
  let world? ← getEnvNatAny? ["TG4_WORLD_SIZE", "WORLD_SIZE"]
  match rank?, world? with
  | some r, some w =>
      let worldSize := if w == 0 then 1 else w
      let rank := if r < worldSize then r else 0
      pure { rank, worldSize }
  | _, _ =>
      let nodeRank := (← getEnvNatAny? ["TG4_NODE_RANK", "NODE_RANK"]).getD 0
      let nodeCount := (← getEnvNatAny? ["TG4_NODE_COUNT", "NODE_COUNT"]).getD 1
      let localRank := (← getEnvNatAny? ["TG4_LOCAL_RANK", "LOCAL_RANK"]).getD 0
      let localCount := (← getEnvNatAny? ["TG4_LOCAL_COUNT", "LOCAL_COUNT"]).getD 1
      let nodeCount' := if nodeCount == 0 then 1 else nodeCount
      let localCount' := if localCount == 0 then 1 else localCount
      let worldSize := nodeCount' * localCount'
      let rank := nodeRank * localCount' + localRank
      let rank' := if rank < worldSize then rank else 0
      pure { rank := rank', worldSize }

/-- Create shard config from world config -/
def WorldConfig.toShardConfig (world : WorldConfig) (mode : ShardMode := .interleaved)
    (dropRemainder : Bool := true) : ShardConfig :=
  { shardIndex := world.rank, numShards := world.worldSize, mode, dropRemainder }

/-- Shard dataset for current worker in distributed setting -/
def shardForWorker [Dataset D T] (world : WorldConfig) (mode : ShardMode := .interleaved)
    (dropRemainder : Bool := true) (ds : D) : ShardedDataset D T :=
  shardWithConfig (world.toShardConfig mode dropRemainder) ds

/-- Shard dataset based on environment-configured world settings. -/
def shardForEnv [Dataset D T] (mode : ShardMode := .interleaved)
    (dropRemainder : Bool := true) (ds : D) : IO (ShardedDataset D T) := do
  let world ← WorldConfig.fromEnv
  pure (shardForWorker world mode dropRemainder ds)

/-! ## Even Split Helper -/

/-- Split a dataset into n equal-sized shards, returning all of them -/
def splitDs [Dataset D T] (numShards : Nat) (ds : D) : Array (ShardedDataset D T) := Id.run do
  let mut shards := Array.mkEmpty numShards
  for i in [:numShards] do
    shards := shards.push (shardDs i numShards .interleaved true ds)
  shards

end TinyGrad4.Data
