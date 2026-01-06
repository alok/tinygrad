import Lean.Data.Json.Parser
import Wandb.Client

/-! GraphQL helpers for W&B. -/

namespace Wandb.GraphQL

open Wandb
open Wandb.Json

/-- GraphQL request payload. -/
structure Request where
  query : String
  variables : Option J := none

/-- Render a GraphQL request as JSON using {lit}`json%`. -/
def Request.toJson (req : Request) : J :=
  match req.variables with
  | none => json% { query: $(req.query) }
  | some vars => json% { query: $(req.query), variables: $(vars) }

/-- Post a GraphQL request using {lit}`Wandb.Client`. -/
def post (cfg : Wandb.Config) (req : Request) : IO Wandb.Http.Response := do
  Wandb.postJsonPath cfg "/graphql" (Request.toJson req)

/-- Parse a GraphQL response body into JSON. -/
def parse (resp : Wandb.Http.Response) : Except String J :=
  Lean.Json.parse resp.stdout

/-- Extract the {lit}`data` field from a GraphQL response. -/
def dataField (resp : J) : Except String J := do
  let obj ← resp.getObj?
  match obj.get? "data" with
  | some v => pure v
  | none => throw "data field not found"

/-- Extract the {lit}`errors` field from a GraphQL response. -/
def errorsField (resp : J) : Except String J := do
  let obj ← resp.getObj?
  match obj.get? "errors" with
  | some v => pure v
  | none => throw "errors field not found"

end Wandb.GraphQL
