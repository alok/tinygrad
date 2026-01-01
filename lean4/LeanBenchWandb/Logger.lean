import Lean
import Std.Time
import LeanBenchNew
import Wandb.Local
import Wandb.Json
import Wandb.Client
import Wandb.Api
import Wandb.Run
import Wandb.RunId

/-!
W&B (local/offline) logging integration for LeanBenchNew.
-/

namespace LeanBenchWandb

open LeanBenchNew
open Wandb.Json
open Std

private def envBool? (name : String) : IO (Option Bool) := do
  match (← IO.getEnv name) with
  | none => pure none
  | some raw =>
      let v := raw.trimAscii.toString.toLower
      if v == "1" || v == "true" || v == "yes" || v == "on" then
        pure (some true)
      else if v == "0" || v == "false" || v == "no" || v == "off" then
        pure (some false)
      else
        pure none

private def computeEnabled : IO Bool := do
  match (← envBool? "LEANBENCH_WANDB") with
  | some v => pure v
  | none =>
      if (← IO.getEnv "LAKE") |>.isSome then
        return false
      match (← envBool? "WANDB_DISABLED") with
      | some true => pure false
      | _ =>
          match (← IO.getEnv "WANDB_MODE").map String.toLower with
          | some "disabled" | some "dryrun" => pure false
          | _ => pure true

private def platformName : String :=
  if System.Platform.isOSX then
    "macos"
  else if System.Platform.isWindows then
    "windows"
  else
    "linux"

private def optField (key : String) (value? : Option String) : List (String × J) :=
  match value? with
  | some v => [(key, str v)]
  | none => []

initialize enabledRef : IO.Ref (Option Bool) ← IO.mkRef none
initialize runRef : IO.Ref (Option Wandb.Local.LocalRun) ← IO.mkRef none
initialize remoteRef : IO.Ref (Option (Wandb.Config × Wandb.Api.RunInfo × HashMap String LogValue × HashMap String LogValue × Nat)) ← IO.mkRef none
initialize remoteFailedRef : IO.Ref Bool ← IO.mkRef false

private def initRun : IO Wandb.Local.LocalRun := do
  let run ← Wandb.Local.init
  let program ← IO.appPath
  let leanVersion := Lean.versionString
  let leanGithash := Lean.githash
  let toolchain ← IO.getEnv "LEAN_TOOLCHAIN"
  let sysroot ← IO.getEnv "LEAN_SYSROOT"
  let leanPath ← IO.getEnv "LEAN_PATH"
  let lake ← IO.getEnv "LAKE"
  let lakeEnv ← IO.getEnv "LAKE_ENV"
  let baseFields : List (String × J) := [
    ("program", str program.toString),
    ("os", str platformName),
    ("leanbench_wandb", bool true),
    ("lean_version", str leanVersion),
    ("lean_githash", str leanGithash)
  ]
  let metadataFields :=
    baseFields
    ++ optField "lean_toolchain" toolchain
    ++ optField "lean_sysroot" sysroot
    ++ optField "lean_path" leanPath
    ++ optField "lake" lake
    ++ optField "lake_env" lakeEnv
  let metadata := obj metadataFields
  Wandb.Local.writeMetadata run.paths metadata
  Wandb.Local.writeConfig run.paths [("leanbench_wandb", bool true)]
  Wandb.Local.writeSummary run.paths (obj [])
  pure run

private def isEnabled : IO Bool := do
  match (← enabledRef.get) with
  | some v => pure v
  | none =>
      let v ← computeEnabled
      enabledRef.set (some v)
      pure v

private def remoteEnabled : IO Bool := do
  if !(← isEnabled) then
    return false
  match (← envBool? "LEANBENCH_WANDB_REMOTE") with
  | some v => return v
  | none =>
      match (← IO.getEnv "WANDB_MODE").map String.toLower with
      | some "offline" | some "disabled" | some "dryrun" => return false
      | _ =>
          match (← IO.getEnv "WANDB_API_KEY") with
          | none => return false
          | some _ => return true

private def runNameFromEnv : IO String := do
  match (← IO.getEnv "WANDB_RUN_ID") with
  | some v => return v
  | none =>
      match (← IO.getEnv "WANDB_NAME") with
      | some v => return v
      | none =>
          let id ← Wandb.RunId.generate
          return s!"leanbench-{id}"

private def getRemote : IO (Option (Wandb.Config × Wandb.Api.RunInfo × HashMap String LogValue × HashMap String LogValue × Nat)) := do
  if !(← remoteEnabled) then
    return none
  if (← remoteFailedRef.get) then
    return none
  match (← remoteRef.get) with
  | some r => return some r
  | none =>
      try
        let cfg ← Wandb.Config.fromEnv
        let runName ← runNameFromEnv
        let upsert ← Wandb.Api.createRun cfg runName {}
        IO.println s!"wandb: https://wandb.ai/{upsert.run.entity}/{upsert.run.project}/runs/{upsert.run.name}"
        let startMs ← IO.monoMsNow
        let state := (cfg, upsert.run, ({} : HashMap String LogValue), ({} : HashMap String LogValue), startMs)
        remoteRef.set (some state)
        return some state
      catch _ =>
        remoteFailedRef.set true
        return none

private def getRun : IO (Option Wandb.Local.LocalRun) := do
  if !(← isEnabled) then
    return none
  match (← runRef.get) with
  | some run => pure (some run)
  | none =>
      let run ← initRun
      runRef.set (some run)
      pure (some run)

private def logValueToJson : LogValue → J
  | .str v => str v
  | .nat v => nat v
  | .int v => int v
  | .float v => float v
  | .bool v => bool v

private def currentTimestamp : IO Float := do
  let ts ← Std.Time.Timestamp.now
  let secs : Int := (Std.Time.Timestamp.toSecondsSinceUnixEpoch ts).val
  pure (Float.ofInt secs)

private def runtimeSeconds (startMs : Nat) : IO Float := do
  let nowMs ← IO.monoMsNow
  let delta := nowMs - startMs
  pure (delta.toFloat / 1000.0)

private def summaryWithMeta (summary : HashMap String LogValue) (timestamp runtime : Float) : J :=
  let base := summary.toList.map (fun (k, v) => (k, logValueToJson v))
  let metaFields := [
    ("_timestamp", float timestamp),
    ("_runtime", float runtime),
    ("_step", nat 0),
    ("_wandb", obj [("runtime", float runtime)])
  ]
  obj (base ++ metaFields)

private def historyWithMeta (summary : HashMap String LogValue) (timestamp runtime : Float) : J :=
  let base := summary.toList.map (fun (k, v) => (k, logValueToJson v))
  let metaFields := [
    ("_timestamp", float timestamp),
    ("_runtime", float runtime),
    ("_step", nat 0)
  ]
  obj (base ++ metaFields)

private def addMetaToSummaryJson (summary : J) (timestamp runtime : Float) : J :=
  let metaFields := [
    ("_timestamp", float timestamp),
    ("_runtime", float runtime),
    ("_step", nat 0),
    ("_wandb", obj [("runtime", float runtime)])
  ]
  match summary with
  | .obj kv => obj (kv.toList ++ metaFields)
  | _ => obj metaFields

private def addMetaToHistoryJson (summary : J) (timestamp runtime : Float) : J :=
  let metaFields := [
    ("_timestamp", float timestamp),
    ("_runtime", float runtime),
    ("_step", nat 0)
  ]
  match summary with
  | .obj kv => obj (kv.toList ++ metaFields)
  | _ => obj metaFields

/-- Update remote summary key/value pair. -/
private def logSummaryKV (key : String) (value : LogValue) : IO Unit := do
  try
    let some (cfg, runInfo, summary, config, startMs) ← getRemote | return ()
    let summary' := summary.insert key value
    let summaryJson := obj (summary'.toList.map (fun (k, v) => (k, logValueToJson v)))
    let _ ← Wandb.Api.setRunSummary cfg runInfo.id summaryJson
    remoteRef.set (some (cfg, runInfo, summary', config, startMs))
  catch _ =>
    pure ()

private def logConfigKV (key : String) (value : LogValue) : IO Unit := do
  try
    let some (cfg, runInfo, summary, config, startMs) ← getRemote | return ()
    let config' := config.insert key value
    let configJson := obj (config'.toList.map (fun (k, v) => (k, logValueToJson v)))
    let _ ← Wandb.Api.setRunConfig cfg runInfo.id configJson
    remoteRef.set (some (cfg, runInfo, summary, config', startMs))
  catch _ =>
    pure ()

/-- Log raw fields to the local W&B run. -/
def logFields (fields : List (String × LogValue)) : IO Unit := do
  try
    let some run ← getRun | return ()
    let jsonFields := fields.map (fun (k, v) => (k, logValueToJson v))
    let run ← Wandb.Local.log run jsonFields
    runRef.set (some run)
  catch _ =>
    pure ()
  try
    let some (cfg, runInfo, summary, config, startMs) ← getRemote | return ()
    let mut summary' := summary
    for (k, v) in fields do
      summary' := summary'.insert k v
    let summaryJson := obj (summary'.toList.map (fun (k, v) => (k, logValueToJson v)))
    let _ ← Wandb.Api.setRunSummary cfg runInfo.id summaryJson
    remoteRef.set (some (cfg, runInfo, summary', config, startMs))
  catch _ =>
    pure ()

/-- Log a benchmark result. -/
def logResult (r : Result) (config : Config) : IO Unit := do
  let fields : List (String × LogValue) := [
    ("benchmark/name", .str r.name),
    ("benchmark/avg_time_ns", .nat r.avgTimeNs),
    ("benchmark/min_time_ns", .nat r.minTimeNs),
    ("benchmark/max_time_ns", .nat r.maxTimeNs),
    ("benchmark/avg_time_s", .float (r.avgTimeNs.toFloat / 1_000_000_000.0)),
    ("benchmark/min_time_s", .float (r.minTimeNs.toFloat / 1_000_000_000.0)),
    ("benchmark/max_time_s", .float (r.maxTimeNs.toFloat / 1_000_000_000.0)),
    ("benchmark/iterations", .nat r.iterations),
    ("benchmark/warmup_iterations", .nat config.warmupIterations),
    ("benchmark/timed_iterations", .nat config.timedIterations)
  ]
  logFields fields

/-- Logger implementation for LeanBenchNew. -/
def logger : Logger :=
  { logFields := logFields, logResult := logResult }

/-- Log a scalar metric. -/
def logScalar (name : String) (value : Float) (unit? : Option String := none) : IO Unit := do
  let base := [
    ("benchmark/name", LogValue.str name),
    ("benchmark/value", LogValue.float value)
  ]
  let fields :=
    match unit? with
    | some unit => base ++ [("benchmark/unit", LogValue.str unit)]
    | none => base
  logFields fields
  logSummaryKV s!"scalar/{name}" (LogValue.float value)

/-- Log a structured metric with parameters. -/
def logMetric
    (group : String)
    (metric : String)
    (value : Float)
    (unit? : Option String := none)
    (params : List (String × LogValue) := []) : IO Unit := do
  let base := [
    ("benchmark/group", LogValue.str group),
    ("benchmark/metric", LogValue.str metric),
    ("benchmark/value", LogValue.float value)
  ]
  let fields :=
    match unit? with
    | some unit => base ++ [("benchmark/unit", LogValue.str unit)]
    | none => base
  let paramFields := params.map (fun (k, v) => (s!"benchmark/param/{k}", v))
  logFields (fields ++ paramFields)
  logSummaryKV s!"{group}/{metric}" (LogValue.float value)
  for (k, v) in params do
    logSummaryKV s!"{group}/param/{k}" v
    logConfigKV k v

/-- Build a string parameter for {name}`logMetric`. -/
def paramStr (key : String) (value : String) : String × LogValue :=
  (key, LogValue.str value)

/-- Build a natural number parameter for {name}`logMetric`. -/
def paramNat (key : String) (value : Nat) : String × LogValue :=
  (key, LogValue.nat value)

/-- Build a float parameter for {name}`logMetric`. -/
def paramFloat (key : String) (value : Float) : String × LogValue :=
  (key, LogValue.float value)

/-- Fetch the current run summary (remote) if available. -/
def fetchSummaryCurrent : IO (Option J) := do
  match (← remoteRef.get) with
  | none => return none
  | some (cfg, runInfo, _, _, _) =>
      let ref : Wandb.RunRef := { entity := runInfo.entity, project := runInfo.project, id := runInfo.name }
      try
        let run ← Wandb.Api.getRun cfg ref
        match run.summaryMetrics with
        | none => return none
        | some raw =>
            match Lean.Json.parse raw with
            | .ok j => return some j
            | .error _ => return none
      catch _ =>
        return none

/-- Fetch summary metrics for an explicit run reference. -/
def fetchSummary (entity project runName : String) : IO (Option J) := do
  try
    let cfg ← Wandb.Config.fromEnv
    let ref : Wandb.RunRef := { entity := entity, project := project, id := runName }
    let run ← Wandb.Api.getRun cfg ref
    match run.summaryMetrics with
    | none => return none
    | some raw =>
        match Lean.Json.parse raw with
        | .ok j => return some j
        | .error _ => return none
  catch _ =>
    return none

/-- Mark the current remote run as finished (no-op when remote logging is disabled). -/
def finishCurrent : IO Unit := do
  try
    let some (cfg, runInfo, summary, config, startMs) ← getRemote | return ()
    let timestamp ← currentTimestamp
    let runtime ← runtimeSeconds startMs
    let summaryJson := summaryWithMeta summary timestamp runtime
    let historyJson := historyWithMeta summary timestamp runtime
    Wandb.Api.finishRun cfg runInfo summaryJson (some historyJson) 0
    remoteRef.set (some (cfg, runInfo, summary, config, startMs))
  catch _ =>
    pure ()

/-- Mark a named run as finished by emitting a file stream completion payload. -/
def finishRunByName (entity project runName : String) : IO Unit := do
  try
    let cfg ← Wandb.Config.fromEnv
    let ref : Wandb.RunRef := { entity := entity, project := project, id := runName }
    let run ← Wandb.Api.getRun cfg ref
    let summaryJson :=
      match run.summaryMetrics with
      | none => obj []
      | some raw =>
          match Lean.Json.parse raw with
          | .ok j => j
          | .error _ => obj []
    let timestamp ← currentTimestamp
    let runtime : Float := 0.0
    let summaryWithMeta := addMetaToSummaryJson summaryJson timestamp runtime
    let historyWithMeta := addMetaToHistoryJson summaryJson timestamp runtime
    Wandb.Api.finishRun cfg run summaryWithMeta (some historyWithMeta) 0
  catch _ =>
    pure ()

end LeanBenchWandb
