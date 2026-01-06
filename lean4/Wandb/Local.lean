import Std
import Std.Time
import Wandb.Env
import Wandb.Filenames
import Wandb.Json
import Wandb.RunId

/-!
Local/offline run writer compatible with {lit}`wandb sync`.
-/

namespace Wandb.Local

open Wandb.Json

/-- Paths for a local W&B run directory. -/
structure RunPaths where
  rootDir : System.FilePath
  wandbDir : System.FilePath
  syncDir : System.FilePath
  filesDir : System.FilePath
  runId : String
  runMode : String
  timeSpec : String

/-- Use {lit}`.wandb` if it exists, otherwise {lit}`wandb`. -/
def selectWandbDir (rootDir : System.FilePath) : IO System.FilePath := do
  let dot := rootDir / ".wandb"
  let useDot ←
    match (← dot.metadata.toBaseIO) with
    | .ok _ => pure true
    | .error _ => pure false
  pure <| if useDot then dot else rootDir / "wandb"

/-- Map {lit}`WANDB_MODE` to the on-disk run mode name. -/
def runModeFromEnv (mode : Option String) : String :=
  match mode with
  | some "offline" => "offline-run"
  | _ => "run"

/-- Current time spec in {lit}`YYYYMMDD_HHMMSS` format. -/
def timeSpecNow : IO String := do
  let zdt ← Std.Time.ZonedDateTime.now
  pure <| Std.Time.ZonedDateTime.format zdt "uuuuMMdd_HHmmss"

/-- Build run paths from environment variables. -/
def RunPaths.fromEnv (runId? : Option String := none) : IO RunPaths := do
  let vars ← Wandb.Env.read
  let rootDir := vars.dir.map System.FilePath.mk |>.getD (System.FilePath.mk ".")
  let runId ←
    match runId? with
    | some v => pure v
    | none =>
      match vars.runId with
      | some v => pure v
      | none => Wandb.RunId.generate
  let timeSpec ← timeSpecNow
  let runMode := runModeFromEnv vars.mode
  let wandbDir ← selectWandbDir rootDir
  let syncDir :=
    match vars.runDir with
    | some d => System.FilePath.mk d
    | none => wandbDir / s!"{runMode}-{timeSpec}-{runId}"
  let filesDir := syncDir / "files"
  pure {
    rootDir := rootDir
    wandbDir := wandbDir
    syncDir := syncDir
    filesDir := filesDir
    runId := runId
    runMode := runMode
    timeSpec := timeSpec
  }

/-- Ensure the run directories exist. -/
def ensureDirs (paths : RunPaths) : IO Unit := do
  IO.FS.createDirAll paths.filesDir

/-- Render a config YAML string in W&B's on-disk format. -/
def renderConfigYaml (entries : List (String × J)) : String := by
  let header := "wandb_version: 1"
  let body :=
    entries.map fun (k, v) =>
      s!"{k}:\n  value: {Json.render v}\n  desc: null"
  let content :=
    match body with
    | [] => header ++ "\n"
    | _ => header ++ "\n\n" ++ String.intercalate "\n" body ++ "\n"
  exact content

/-- Write {lit}`config.yaml` into the run files directory. -/
def writeConfig (paths : RunPaths) (entries : List (String × J)) : IO Unit := do
  let path := paths.filesDir / Wandb.Filenames.config
  IO.FS.writeFile path (renderConfigYaml entries)

/-- Write {lit}`wandb-summary.json` into the run files directory. -/
def writeSummary (paths : RunPaths) (summary : J) : IO Unit := do
  let path := paths.filesDir / Wandb.Filenames.summary
  IO.FS.writeFile path (Json.render summary)

/-- Write {lit}`wandb-metadata.json` into the run files directory. -/
def writeMetadata (paths : RunPaths) (metadata : J) : IO Unit := do
  let path := paths.filesDir / Wandb.Filenames.metadata
  IO.FS.writeFile path (Json.render metadata)

/-- Append a single JSON line to {lit}`wandb-history.jsonl`. -/
def appendHistory (paths : RunPaths) (row : J) : IO Unit := do
  let path := paths.filesDir / Wandb.Filenames.history
  IO.FS.withFile path .append fun h => do
    h.putStrLn (Json.render row)

/-- Build a history row from fields, ensuring {lit}`_step` is present. -/
def historyRow (fields : List (String × J)) (step : Nat) : J :=
  let hasStep := fields.any (fun (k, _) => k == "_step")
  let fields := if hasStep then fields else ("_step", nat step) :: fields
  obj fields

/-- Local run state for incremental history logging. -/
structure LocalRun where
  paths : RunPaths
  step : Nat := 0

/-- Initialize a local run from env defaults. -/
def init (paths? : Option RunPaths := none) : IO LocalRun := do
  let paths ←
    match paths? with
    | some p => pure p
    | none => RunPaths.fromEnv
  ensureDirs paths
  pure { paths := paths, step := 0 }

/-- Log a history row from field pairs. -/
def log
    (run : LocalRun)
    (fields : List (String × J))
    (step? : Option Nat := none)
    (commit? : Option Bool := none) : IO LocalRun := do
  let baseStep := step?.getD run.step
  let commit :=
    match step? with
    | some _ => commit?.getD false
    | none => commit?.getD true
  let row := historyRow fields baseStep
  appendHistory run.paths row
  let step' := if baseStep > run.step then baseStep else run.step
  let nextStep := if commit then step' + 1 else step'
  pure { run with step := nextStep }

/-- Log a JSON object as a history row. -/
def logJson
    (run : LocalRun)
    (row : J)
    (step? : Option Nat := none)
    (commit? : Option Bool := none) : IO LocalRun := do
  match row with
  | .obj kv =>
      let fields := kv.toList
      log run fields step? commit?
  | _ =>
      throw <| IO.userError "history row must be a JSON object"

end Wandb.Local
