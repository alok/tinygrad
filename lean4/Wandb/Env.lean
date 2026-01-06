import Std

/-! Environment helpers for W&B configuration. -/

namespace Wandb.Env

/-- Environment snapshot for W&B configuration. -/
structure Vars where
  apiKey : Option String
  entity : Option String
  project : Option String
  name : Option String
  notes : Option String
  tags : Option String
  runGroup : Option String
  jobType : Option String
  runId : Option String
  runDir : Option String
  baseUrl : Option String
  mode : Option String
  dir : Option String

/-- Read common W&B environment variables. -/
def read : IO Vars := do
  let apiKey ← IO.getEnv "WANDB_API_KEY"
  let entity ← IO.getEnv "WANDB_ENTITY"
  let project ← IO.getEnv "WANDB_PROJECT"
  let name ← IO.getEnv "WANDB_NAME"
  let notes ← IO.getEnv "WANDB_NOTES"
  let tags ← IO.getEnv "WANDB_TAGS"
  let runGroup ← IO.getEnv "WANDB_RUN_GROUP"
  let jobType ← IO.getEnv "WANDB_JOB_TYPE"
  let runId ← IO.getEnv "WANDB_RUN_ID"
  let runDir ← IO.getEnv "WANDB_RUN_DIR"
  let baseUrl ← IO.getEnv "WANDB_BASE_URL"
  let mode ← IO.getEnv "WANDB_MODE"
  let dir ← IO.getEnv "WANDB_DIR"
  pure { apiKey := apiKey, entity := entity, project := project,
         name := name, notes := notes, tags := tags, runGroup := runGroup, jobType := jobType,
         runId := runId, runDir := runDir, baseUrl := baseUrl, mode := mode, dir := dir }

/-- Require an env var or throw. -/
def require (name : String) : IO String := do
  let some val ← IO.getEnv name
    | throw <| IO.userError s!"{name} not set"
  pure val

/-- Require {lit}`WANDB_API_KEY`. -/
def requireApiKey : IO String :=
  require "WANDB_API_KEY"

/-- Require {lit}`WANDB_ENTITY`. -/
def requireEntity : IO String :=
  require "WANDB_ENTITY"

/-- Require {lit}`WANDB_PROJECT`. -/
def requireProject : IO String :=
  require "WANDB_PROJECT"

/-- Require {lit}`WANDB_RUN_ID`. -/
def requireRunId : IO String :=
  require "WANDB_RUN_ID"

end Wandb.Env
