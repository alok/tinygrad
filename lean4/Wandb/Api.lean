import Std
import Lean.Data.Json.Parser
import Wandb.Client
import Wandb.GraphQL
import Wandb.Json
import Wandb.Run

/-!
High-level W&B API helpers for runs, history, and files.
-/

namespace Wandb.Api

open Wandb
open Wandb.Json

/-! ### Helpers -/

/-- Convert {name}`Except` into {name}`IO`, throwing {name}`IO.userError` on errors. -/
def exceptToIO (e : Except String α) : IO α := do
  match e with
  | .ok v => pure v
  | .error err => throw <| IO.userError err

/-- Parse JSON from a response body or throw. -/
def parseJson (resp : Wandb.Http.Response) : IO J := do
  exceptToIO (Lean.Json.parse resp.stdout)

/-- Parse GraphQL data or throw on errors. -/
def parseGraphQLData (resp : Wandb.Http.Response) : IO J := do
  let j ← parseJson resp
  match Wandb.GraphQL.errorsField j with
  | .ok errs =>
      throw <| IO.userError s!"GraphQL errors: {Json.render errs}"
  | .error _ => pure ()
  exceptToIO (Wandb.GraphQL.dataField j)

/-- Convert a JSON value into its JSON-string encoding. -/
def jsonString (j : J) : String :=
  Json.render j

/-- Map an optional JSON value into a JSONString payload. -/
def jsonString? (j : Option J) : Option String :=
  j.map jsonString

/-- Append an optional field to a JSON object. -/
def addField (k : String) (v : Option J) (fields : List (String × J)) : List (String × J) :=
  match v with
  | none => fields
  | some val => (k, val) :: fields

/-- Append an optional string field to a JSON object. -/
def addStrField (k : String) (v : Option String) (fields : List (String × J)) : List (String × J) :=
  addField k (v.map Json.str) fields

/-- Append an optional bool field to a JSON object. -/
def addBoolField (k : String) (v : Option Bool) (fields : List (String × J)) : List (String × J) :=
  addField k (v.map Json.bool) fields

/-- Append a string array field to a JSON object. -/
def addStrArrayField (k : String) (v : List String) (fields : List (String × J)) : List (String × J) :=
  if v.isEmpty then
    fields
  else
    addField k (some <| arr (v.map str)) fields

/-! ### Run upsert -/

/-- Inputs for {lit}`upsertRun`. -/
structure RunUpsert where
  id : Option String := none
  name : Option String := none
  project : Option String := none
  entity : Option String := none
  groupName : Option String := none
  description : Option String := none
  displayName : Option String := none
  notes : Option String := none
  tags : List String := []
  config : Option J := none
  summaryMetrics : Option J := none
  jobType : Option String := none
  state : Option String := none
  sweep : Option String := none
  commit : Option String := none
  program : Option String := none
  repo : Option String := none

/-- Result metadata for a run. -/
structure RunInfo where
  id : String
  name : String
  displayName : Option String := none
  description : Option String := none
  notes : Option String := none
  tags : Array String := #[]
  state : Option String := none
  project : String
  entity : String
  config : Option String := none
  summaryMetrics : Option String := none

/-- Parse a run record from GraphQL JSON. -/
def RunInfo.fromJson (j : J) : Except String RunInfo := do
  let id ← j.getObjValAs? String "id"
  let name ← j.getObjValAs? String "name"
  let displayName := j.getObjValAs? String "displayName" |>.toOption
  let description := j.getObjValAs? String "description" |>.toOption
  let notes := j.getObjValAs? String "notes" |>.toOption
  let tags := (j.getObjValAs? (Array String) "tags" |>.toOption).getD #[]
  let state := j.getObjValAs? String "state" |>.toOption
  let config := j.getObjValAs? String "config" |>.toOption
  let summaryMetrics := j.getObjValAs? String "summaryMetrics" |>.toOption
  let projectObj ← j.getObjVal? "project"
  let project ← projectObj.getObjValAs? String "name"
  let entityObj ← projectObj.getObjVal? "entity"
  let entity ← entityObj.getObjValAs? String "name"
  pure {
    id := id
    name := name
    displayName := displayName
    description := description
    notes := notes
    tags := tags
    state := state
    project := project
    entity := entity
    config := config
    summaryMetrics := summaryMetrics
  }

/-- Upsert results include run metadata and an inserted flag. -/
structure UpsertResult where
  run : RunInfo
  inserted : Bool

/-!
GraphQL for run upsert. Uses the {lit}`upsertBucket` mutation and optional fields.
-/
def upsertRunQuery : String :=
"
mutation UpsertBucket(
  $id: String,
  $name: String,
  $project: String,
  $entity: String,
  $groupName: String,
  $description: String,
  $displayName: String,
  $notes: String,
  $commit: String,
  $config: JSONString,
  $program: String,
  $repo: String,
  $jobType: String,
  $state: String,
  $sweep: String,
  $tags: [String!],
  $summaryMetrics: JSONString
) {
  upsertBucket(input: {
    id: $id,
    name: $name,
    modelName: $project,
    entityName: $entity,
    groupName: $groupName,
    description: $description,
    displayName: $displayName,
    notes: $notes,
    config: $config,
    commit: $commit,
    jobProgram: $program,
    jobRepo: $repo,
    jobType: $jobType,
    state: $state,
    sweep: $sweep,
    tags: $tags,
    summaryMetrics: $summaryMetrics
  }) {
    bucket {
      id
      name
      displayName
      description
      notes
      tags
      state
      config
      summaryMetrics
      project { name entity { name } }
    }
    inserted
  }
}
"

/-- Build GraphQL variables for {lit}`upsertRun`. -/
def RunUpsert.toVariables (cfg : Wandb.Config) (u : RunUpsert) : J :=
  let project := match u.project with | some v => some v | none => cfg.project
  let entity := match u.entity with | some v => some v | none => cfg.entity
  let fields : List (String × J) := []
  let fields := addStrField "id" u.id fields
  let fields := addStrField "name" u.name fields
  let fields := addStrField "project" project fields
  let fields := addStrField "entity" entity fields
  let fields := addStrField "groupName" u.groupName fields
  let fields := addStrField "description" u.description fields
  let fields := addStrField "displayName" u.displayName fields
  let fields := addStrField "notes" u.notes fields
  let fields := addStrField "commit" u.commit fields
  let fields := addStrField "program" u.program fields
  let fields := addStrField "repo" u.repo fields
  let fields := addStrField "jobType" u.jobType fields
  let fields := addStrField "state" u.state fields
  let fields := addStrField "sweep" u.sweep fields
  let fields := addStrArrayField "tags" u.tags fields
  let fields := addStrField "config" (jsonString? u.config) fields
  let fields := addStrField "summaryMetrics" (jsonString? u.summaryMetrics) fields
  obj fields.reverse

/-- Create or update a run via GraphQL {lit}`upsertBucket`. -/
def upsertRun (cfg : Wandb.Config) (u : RunUpsert) : IO UpsertResult := do
  let resp ← Wandb.postGraphQL cfg upsertRunQuery (some <| u.toVariables cfg)
  let data ← parseGraphQLData resp
  let upsert ← exceptToIO (data.getObjVal? "upsertBucket")
  let bucket ← exceptToIO (upsert.getObjVal? "bucket")
  let inserted := (upsert.getObjValAs? Bool "inserted" |>.toOption).getD false
  let run ← exceptToIO (RunInfo.fromJson bucket)
  pure { run := run, inserted := inserted }

/-- Create a run by name, defaulting project to {lit}`uncategorized` if unset. -/
def createRun (cfg : Wandb.Config) (name : String) (u : RunUpsert := {}) : IO UpsertResult := do
  let project :=
    match u.project with
    | some v => v
    | none => cfg.project.getD "uncategorized"
  let u := { u with name := some name, project := some project }
  upsertRun cfg u

/-- Update a run by id. -/
def updateRun (cfg : Wandb.Config) (id : String) (u : RunUpsert := {}) : IO UpsertResult := do
  let u := { u with id := some id }
  upsertRun cfg u

/-! ### Run queries -/

/-- GraphQL query for fetching run metadata. -/
def runQuery : String :=
"
query Run($project: String!, $entity: String!, $name: String!) {
  project(name: $project, entityName: $entity) {
    run(name: $name) {
      id
      name
      displayName
      description
      notes
      tags
      state
      config
      summaryMetrics
      project { name entity { name } }
    }
  }
}
"

/-- Fetch run metadata. -/
def getRun (cfg : Wandb.Config) (ref : Wandb.RunRef) : IO RunInfo := do
  let vars := json% { project: $(ref.project), entity: $(ref.entity), name: $(ref.id) }
  let resp ← Wandb.postGraphQL cfg runQuery (some vars)
  let data ← parseGraphQLData resp
  let project ← exceptToIO (data.getObjVal? "project")
  let run ← exceptToIO (project.getObjVal? "run")
  let isNull := match run with | .null => true | _ => false
  if isNull then
    throw <| IO.userError s!"run not found: {ref.entity}/{ref.project}/{ref.id}"
  exceptToIO (RunInfo.fromJson run)

/-- GraphQL mutation for deleting a run by id. -/
def deleteRunQuery : String :=
"
mutation DeleteRun($id: ID!, $deleteArtifacts: Boolean) {
  deleteRun(input: { id: $id, deleteArtifacts: $deleteArtifacts }) {
    clientMutationId
  }
}
"

/-- Delete a run by its internal id. -/
def deleteRunById (cfg : Wandb.Config) (id : String) (deleteArtifacts : Bool := false) : IO Unit := do
  let vars := json% { id: $(id), deleteArtifacts: $(deleteArtifacts) }
  let resp ← Wandb.postGraphQL cfg deleteRunQuery (some vars)
  let _ ← parseGraphQLData resp
  pure ()

/-! ### History queries -/

/-- GraphQL query for sampled history. -/
def sampledHistoryQuery : String :=
"
query RunSampledHistory($project: String!, $entity: String!, $name: String!, $specs: [JSONString!]!) {
  project(name: $project, entityName: $entity) {
    run(name: $name) { sampledHistory(specs: $specs) }
  }
}
"

/-- GraphQL query for full history. -/
def fullHistoryQuery (stream : String) : String :=
"
query RunFullHistory($project: String!, $entity: String!, $name: String!, $samples: Int) {
  project(name: $project, entityName: $entity) {
    run(name: $name) { " ++ stream ++ "(samples: $samples) }
  }
}
"

/-- Fetch sampled history records as JSON objects. -/
def sampledHistory
    (cfg : Wandb.Config)
    (ref : Wandb.RunRef)
    (keys : List String)
    (xAxis : String := "_step")
    (samples : Nat := 500) : IO (Array J) := do
  let keyList := xAxis :: keys
  let spec := json% { keys: $(arr (keyList.map str)), samples: $(samples) }
  let specs := arr [str (Json.render spec)]
  let vars := json% { project: $(ref.project), entity: $(ref.entity), name: $(ref.id), specs: $(specs) }
  let resp ← Wandb.postGraphQL cfg sampledHistoryQuery (some vars)
  let data ← parseGraphQLData resp
  let project ← exceptToIO (data.getObjVal? "project")
  let run ← exceptToIO (project.getObjVal? "run")
  let samplesArr ← exceptToIO (run.getObjValAs? (Array (Array J)) "sampledHistory")
  match samplesArr[0]? with
  | none => pure #[]
  | some values => pure values

/-- Fetch full history as JSON objects. -/
def fullHistory
    (cfg : Wandb.Config)
    (ref : Wandb.RunRef)
    (samples : Nat := 500)
    (stream : String := "history") : IO (Array J) := do
  let vars := json% { project: $(ref.project), entity: $(ref.entity), name: $(ref.id), samples: $(samples) }
  let resp ← Wandb.postGraphQL cfg (fullHistoryQuery stream) (some vars)
  let data ← parseGraphQLData resp
  let project ← exceptToIO (data.getObjVal? "project")
  let run ← exceptToIO (project.getObjVal? "run")
  let lines ← exceptToIO (run.getObjValAs? (Array String) stream)
  let mut out : Array J := #[]
  for line in lines do
    let item ← exceptToIO (Lean.Json.parse line)
    out := out.push item
  pure out

/-! ### File queries -/

/-- File query parameters for a run. -/
structure FileQuery where
  names : List String := []
  pattern : Option String := none
  after : Option String := none
  limit : Nat := 50
  upload : Bool := false

/-- File metadata returned by the API. -/
structure FileInfo where
  id : String
  name : String
  url : Option String := none
  directUrl : Option String := none
  sizeBytes : Option Nat := none
  mimetype : Option String := none
  updatedAt : Option String := none
  md5 : Option String := none

/-- Parse file metadata from JSON. -/
def FileInfo.fromJson (j : J) : Except String FileInfo := do
  let id ← j.getObjValAs? String "id"
  let name ← j.getObjValAs? String "name"
  let url := j.getObjValAs? String "url" |>.toOption
  let directUrl := j.getObjValAs? String "directUrl" |>.toOption
  let sizeBytes := j.getObjValAs? Nat "sizeBytes" |>.toOption
  let mimetype := j.getObjValAs? String "mimetype" |>.toOption
  let updatedAt := j.getObjValAs? String "updatedAt" |>.toOption
  let md5 := j.getObjValAs? String "md5" |>.toOption
  pure {
    id := id
    name := name
    url := url
    directUrl := directUrl
    sizeBytes := sizeBytes
    mimetype := mimetype
    updatedAt := updatedAt
    md5 := md5
  }

/-- A single page of files. -/
structure FilePage where
  files : Array FileInfo
  endCursor : Option String := none
  hasNextPage : Bool := false
  fileCount : Option Nat := none

/-- GraphQL query for listing files for a run. -/
def runFilesQuery : String :=
"
query RunFiles(
  $project: String!,
  $entity: String!,
  $name: String!,
  $fileCursor: String,
  $fileLimit: Int = 50,
  $fileNames: [String] = [],
  $upload: Boolean = false,
  $pattern: String
) {
  project(name: $project, entityName: $entity) {
    run(name: $name) {
      fileCount
      files(names: $fileNames, after: $fileCursor, first: $fileLimit, pattern: $pattern) {
        edges {
          node {
            id
            name
            url(upload: $upload)
            directUrl
            sizeBytes
            mimetype
            updatedAt
            md5
          }
          cursor
        }
        pageInfo {
          endCursor
          hasNextPage
        }
      }
    }
  }
}
"

/-- List files for a run, returning a single page. -/
def listRunFiles (cfg : Wandb.Config) (ref : Wandb.RunRef) (q : FileQuery := {}) : IO FilePage := do
  if (not q.names.isEmpty) && q.pattern.isSome then
    throw <| IO.userError "listRunFiles: names and pattern cannot both be set"
  let mut fields : List (String × J) := []
  fields := addStrField "project" (some ref.project) fields
  fields := addStrField "entity" (some ref.entity) fields
  fields := addStrField "name" (some ref.id) fields
  fields := addStrField "fileCursor" q.after fields
  fields := addField "fileLimit" (some <| nat q.limit) fields
  fields := addField "fileNames" (some <| arr (q.names.map str)) fields
  fields := addBoolField "upload" (some q.upload) fields
  fields := addStrField "pattern" q.pattern fields
  let vars := obj fields.reverse
  let resp ← Wandb.postGraphQL cfg runFilesQuery (some vars)
  let data ← parseGraphQLData resp
  let project ← exceptToIO (data.getObjVal? "project")
  let run ← exceptToIO (project.getObjVal? "run")
  let fileCount := run.getObjValAs? Nat "fileCount" |>.toOption
  let filesObj ← exceptToIO (run.getObjVal? "files")
  let edges ← exceptToIO (filesObj.getObjValAs? (Array J) "edges")
  let mut files : Array FileInfo := #[]
  for edge in edges do
    let node ← exceptToIO (edge.getObjVal? "node")
    let info ← exceptToIO (FileInfo.fromJson node)
    files := files.push info
  let pageInfo ← exceptToIO (filesObj.getObjVal? "pageInfo")
  let endCursor := pageInfo.getObjValAs? String "endCursor" |>.toOption
  let hasNextPage := (pageInfo.getObjValAs? Bool "hasNextPage" |>.toOption).getD false
  pure { files := files, endCursor := endCursor, hasNextPage := hasNextPage, fileCount := fileCount }

/-! ### Runs listing -/

/-- Run list query parameters. -/
structure RunListQuery where
  entity : Option String := none
  project : Option String := none
  cursor : Option String := none
  perPage : Nat := 50
  order : Option String := none
  filters : Option J := none

/-- Lightweight run summary used for lists. -/
structure RunListItem where
  id : String
  name : String
  displayName : Option String := none
  state : Option String := none
  createdAt : Option String := none
  heartbeatAt : Option String := none
  description : Option String := none
  notes : Option String := none
  tags : Array String := #[]
  project : String
  entity : String

/-- Parse a run list item from JSON. -/
def RunListItem.fromJson (j : J) : Except String RunListItem := do
  let id ← j.getObjValAs? String "id"
  let name ← j.getObjValAs? String "name"
  let displayName := j.getObjValAs? String "displayName" |>.toOption
  let state := j.getObjValAs? String "state" |>.toOption
  let createdAt := j.getObjValAs? String "createdAt" |>.toOption
  let heartbeatAt := j.getObjValAs? String "heartbeatAt" |>.toOption
  let description := j.getObjValAs? String "description" |>.toOption
  let notes := j.getObjValAs? String "notes" |>.toOption
  let tags := (j.getObjValAs? (Array String) "tags" |>.toOption).getD #[]
  let projectObj ← j.getObjVal? "project"
  let project ← projectObj.getObjValAs? String "name"
  let entityObj ← projectObj.getObjVal? "entity"
  let entity ← entityObj.getObjValAs? String "name"
  pure {
    id := id
    name := name
    displayName := displayName
    state := state
    createdAt := createdAt
    heartbeatAt := heartbeatAt
    description := description
    notes := notes
    tags := tags
    project := project
    entity := entity
  }

/-- A single page of run list results. -/
structure RunListPage where
  runs : Array RunListItem
  endCursor : Option String := none
  hasNextPage : Bool := false
  runCount : Option Nat := none
  readOnly : Option Bool := none

/-- GraphQL query for listing runs. -/
def runsQuery : String :=
"
query Runs($project: String!, $entity: String!, $cursor: String, $perPage: Int = 50, $order: String, $filters: JSONString) {
  project(name: $project, entityName: $entity) {
    runCount(filters: $filters)
    readOnly
    runs(filters: $filters, after: $cursor, first: $perPage, order: $order) {
      edges {
        node {
          id
          name
          displayName
          state
          createdAt
          heartbeatAt
          description
          notes
          tags
          project { name entity { name } }
        }
        cursor
      }
      pageInfo { endCursor hasNextPage }
    }
  }
}
"

/-- List runs for a project, returning a single page. -/
def listRuns (cfg : Wandb.Config) (q : RunListQuery := {}) : IO RunListPage := do
  let entity ←
    match q.entity with
    | some v => pure v
    | none =>
      match cfg.entity with
      | some v => pure v
      | none => throw <| IO.userError "WANDB_ENTITY not set"
  let project ←
    match q.project with
    | some v => pure v
    | none =>
      match cfg.project with
      | some v => pure v
      | none => throw <| IO.userError "WANDB_PROJECT not set"
  let mut fields : List (String × J) := []
  fields := addStrField "project" (some project) fields
  fields := addStrField "entity" (some entity) fields
  fields := addStrField "cursor" q.cursor fields
  fields := addField "perPage" (some <| nat q.perPage) fields
  fields := addStrField "order" q.order fields
  fields := addStrField "filters" (jsonString? q.filters) fields
  let vars := obj fields.reverse
  let resp ← Wandb.postGraphQL cfg runsQuery (some vars)
  let data ← parseGraphQLData resp
  let projectObj ← exceptToIO (data.getObjVal? "project")
  let runCount := projectObj.getObjValAs? Nat "runCount" |>.toOption
  let readOnly := projectObj.getObjValAs? Bool "readOnly" |>.toOption
  let runsObj ← exceptToIO (projectObj.getObjVal? "runs")
  let edges ← exceptToIO (runsObj.getObjValAs? (Array J) "edges")
  let mut runs : Array RunListItem := #[]
  for edge in edges do
    let node ← exceptToIO (edge.getObjVal? "node")
    let item ← exceptToIO (RunListItem.fromJson node)
    runs := runs.push item
  let pageInfo ← exceptToIO (runsObj.getObjVal? "pageInfo")
  let endCursor := pageInfo.getObjValAs? String "endCursor" |>.toOption
  let hasNextPage := (pageInfo.getObjValAs? Bool "hasNextPage" |>.toOption).getD false
  pure {
    runs := runs
    endCursor := endCursor
    hasNextPage := hasNextPage
    runCount := runCount
    readOnly := readOnly
  }

/-! ### Convenience helpers -/

/-- Update a run's config by internal id. -/
def setRunConfig (cfg : Wandb.Config) (id : String) (config : J) : IO UpsertResult :=
  updateRun cfg id { config := some config }

/-- Update a run's summary metrics by internal id. -/
def setRunSummary (cfg : Wandb.Config) (id : String) (summary : J) : IO UpsertResult :=
  updateRun cfg id { summaryMetrics := some summary }

/-- Update description/notes/tags for a run by internal id. -/
def updateRunMeta
    (cfg : Wandb.Config)
    (id : String)
    (description : Option String := none)
    (notes : Option String := none)
    (tags : List String := []) : IO UpsertResult :=
  updateRun cfg id { description := description, notes := notes, tags := tags }

/-- Mark a run as finished by updating its summary metrics.
    This is a simplified version that sets summary and optionally history. -/
def finishRun (cfg : Wandb.Config) (run : RunInfo) (summary : J) (_history : Option J) (_exitCode : Nat) : IO Unit := do
  let _ ← updateRun cfg run.id { summaryMetrics := some summary }
  pure ()

end Wandb.Api
