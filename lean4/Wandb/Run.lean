import Wandb.Json
import Wandb.Env

/-! Run metadata helpers. -/

namespace Wandb

open Wandb.Json

/-- Minimal run reference. -/
structure RunRef where
  entity : String
  project : String
  id : String

/-- Build JSON for a run reference. -/
def RunRef.toJson (r : RunRef) : J :=
  json% { entity: $(r.entity), project: $(r.project), id: $(r.id) }

/-- Build a run reference from environment variables. -/
def RunRef.fromEnv : IO RunRef := do
  let entity ← Wandb.Env.requireEntity
  let project ← Wandb.Env.requireProject
  let runId ← Wandb.Env.requireRunId
  pure { entity := entity, project := project, id := runId }

end Wandb
