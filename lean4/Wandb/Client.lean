import Std
import Wandb.Http
import Wandb.Json
import Wandb.Env

/-! Tiny W&B client helpers built on top of the raw HTTP layer. -/

namespace Wandb

open Wandb.Json

/-- W&B client configuration. -/
structure Config where
  baseUrl : String := "https://api.wandb.ai"
  apiKey : String
  entity : Option String := none
  project : Option String := none
  timeoutMs : Option Nat := none
  userAgent : Option String := some "scilean-wandb"

/-- Read {lit}`WANDB_API_KEY` from the environment. -/
def apiKeyFromEnv : IO (Option String) :=
  IO.getEnv "WANDB_API_KEY"

/-- Build a config from environment variables. -/
def Config.fromEnv : IO Config := do
  let vars ← Wandb.Env.read
  let some key := vars.apiKey
    | throw <| IO.userError "WANDB_API_KEY not set"
  let baseUrl := vars.baseUrl.getD "https://api.wandb.ai"
  pure { apiKey := key, baseUrl := baseUrl, entity := vars.entity, project := vars.project }

/-- Default auth header for W&B. -/
def authHeader (apiKey : String) : (String × String) :=
  ("Authorization", "Bearer " ++ apiKey)

/-- Default headers for W&B requests. -/
def defaultHeaders (cfg : Config) : List (String × String) :=
  let auth := authHeader cfg.apiKey
  match cfg.userAgent with
  | none => [auth]
  | some ua => ("User-Agent", ua) :: [auth]

/-- Apply auth headers and default timeout. -/
def applyDefaults (cfg : Config) (req : Http.Request) : Http.Request :=
  let timeout :=
    match req.timeoutMs with
    | some t => some t
    | none => cfg.timeoutMs
  { req with headers := defaultHeaders cfg ++ req.headers, timeoutMs := timeout }

/-- Execute a request with auth headers. -/
def request (cfg : Config) (req : Http.Request) : IO Http.Response :=
  Http.run (applyDefaults cfg req)

/-- Execute a request against a path on the W&B API host. -/
def requestPath (cfg : Config) (req : Http.Request) (path : String) : IO Http.Response :=
  let url := cfg.baseUrl ++ path
  request cfg { req with url := url }

/-- HTTP GET against a W&B API path. -/
def getPath (cfg : Config) (path : String) (headers : List (String × String) := []) : IO Http.Response :=
  requestPath cfg { method := "GET", url := "", headers := headers } path

/-- POST JSON to a W&B API path. -/
def postJsonPath (cfg : Config) (path : String) (body : J) : IO Http.Response :=
  requestPath cfg
    { method := "POST"
      url := ""
      headers := [("Content-Type", "application/json")]
      body := some (Json.render body)
    }
    path

/-- PUT JSON to a W&B API path. -/
def putJsonPath (cfg : Config) (path : String) (body : J) : IO Http.Response :=
  requestPath cfg
    { method := "PUT"
      url := ""
      headers := [("Content-Type", "application/json")]
      body := some (Json.render body)
    }
    path

/-- Build a JSON object with {lit}`query` and optional {lit}`variables`. -/
def graphqlBody (query : String) (variables : Option J) : J :=
  match variables with
  | none => json% { query: $(query) }
  | some vars => json% { query: $(query), variables: $(vars) }

/-- Post a GraphQL query to W&B's endpoint. -/
def postGraphQL (cfg : Config) (query : String) (variables : Option J := none) : IO Http.Response := do
  postJsonPath cfg "/graphql" (graphqlBody query variables)

/-- Require entity/project configuration or throw. -/
def Config.requireProject (cfg : Config) : IO (String × String) := do
  let some entity := cfg.entity
    | throw <| IO.userError "WANDB_ENTITY not set"
  let some project := cfg.project
    | throw <| IO.userError "WANDB_PROJECT not set"
  pure (entity, project)

end Wandb
