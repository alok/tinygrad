import Std
import Lean.Data.Json.Parser
import Wandb.Api
import Wandb.Client
import Wandb.Json

/-!
Run queue helpers for W&B Launch.
-/

namespace Wandb.RunQueue

open Wandb
open Wandb.Json

/-- Run queue metadata. -/
structure RunQueueInfo where
  id : String
  name : String
  createdBy : Option String := none
  access : Option String := none

/-- Parse run queue metadata from JSON. -/
def RunQueueInfo.fromJson (j : J) : Except String RunQueueInfo := do
  let id ← j.getObjValAs? String "id"
  let name ← j.getObjValAs? String "name"
  let createdBy := j.getObjValAs? String "createdBy" |>.toOption
  let access := j.getObjValAs? String "access" |>.toOption
  pure { id := id, name := name, createdBy := createdBy, access := access }

/-- Query run queues for a project. -/
def projectRunQueuesQuery : String :=
"
query ProjectRunQueues($entity: String!, $projectName: String!) {
  project(entityName: $entity, name: $projectName) {
    runQueues { id name createdBy access }
  }
}
"

/-- List run queues for a project. -/
def listProjectRunQueues (cfg : Wandb.Config) (entity project : String) : IO (Array RunQueueInfo) := do
  let vars := json% { entity: $(entity), projectName: $(project) }
  let resp ← Wandb.postGraphQL cfg projectRunQueuesQuery (some vars)
  let data ← Wandb.Api.parseGraphQLData resp
  let projectObj ← Wandb.Api.exceptToIO (data.getObjVal? "project")
  let queues ← Wandb.Api.exceptToIO (projectObj.getObjValAs? (Array J) "runQueues")
  let mut out : Array RunQueueInfo := #[]
  for q in queues do
    let info ← Wandb.Api.exceptToIO (RunQueueInfo.fromJson q)
    out := out.push info
  pure out

/-! ### Create and upsert -/

/-- Inputs for creating a run queue. -/
structure RunQueueCreate where
  entity : String
  project : String
  name : String
  access : String := "PROJECT"
  prioritizationMode : Option String := none
  defaultResourceConfigID : Option String := none

/-- Result for createRunQueue. -/
structure RunQueueCreateResult where
  success : Bool
  queueId : Option String := none

def createRunQueueQueryBasic : String :=
"
mutation createRunQueue(
  $entity: String!,
  $project: String!,
  $queueName: String!,
  $access: RunQueueAccessType!
) {
  createRunQueue(
    input: {
      entityName: $entity,
      projectName: $project,
      queueName: $queueName,
      access: $access
    }
  ) { success queueID }
}
"

def createRunQueueQueryExtended : String :=
"
mutation createRunQueue(
  $entity: String!,
  $project: String!,
  $queueName: String!,
  $access: RunQueueAccessType!,
  $prioritizationMode: RunQueuePrioritizationMode,
  $defaultResourceConfigID: ID
) {
  createRunQueue(
    input: {
      entityName: $entity,
      projectName: $project,
      queueName: $queueName,
      access: $access,
      prioritizationMode: $prioritizationMode,
      defaultResourceConfigID: $defaultResourceConfigID
    }
  ) { success queueID }
}
"

/-- Create a run queue. -/
def createRunQueue (cfg : Wandb.Config) (q : RunQueueCreate) : IO RunQueueCreateResult := do
  let useExtended := q.prioritizationMode.isSome || q.defaultResourceConfigID.isSome
  let vars :=
    if useExtended then
      json% {
        entity: $(q.entity),
        project: $(q.project),
        queueName: $(q.name),
        access: $(q.access),
        prioritizationMode: $(q.prioritizationMode),
        defaultResourceConfigID: $(q.defaultResourceConfigID)
      }
    else
      json% {
        entity: $(q.entity),
        project: $(q.project),
        queueName: $(q.name),
        access: $(q.access)
      }
  let query := if useExtended then createRunQueueQueryExtended else createRunQueueQueryBasic
  let resp ← Wandb.postGraphQL cfg query (some vars)
  let data ← Wandb.Api.parseGraphQLData resp
  let result ← Wandb.Api.exceptToIO (data.getObjVal? "createRunQueue")
  let success := (result.getObjValAs? Bool "success" |>.toOption).getD false
  let queueId := result.getObjValAs? String "queueID" |>.toOption
  pure { success := success, queueId := queueId }

/-- Inputs for upserting a run queue. -/
structure RunQueueUpsert where
  entity : String
  project : String
  name : String
  resourceType : String
  resourceConfig : J
  templateVariables : Option J := none
  prioritizationMode : Option String := none
  externalLinks : Option J := none

def upsertRunQueueQuery : String :=
"
mutation upsertRunQueue(
  $entityName: String!,
  $projectName: String!,
  $queueName: String!,
  $resourceType: String!,
  $resourceConfig: JSONString!,
  $templateVariables: JSONString,
  $prioritizationMode: RunQueuePrioritizationMode,
  $externalLinks: JSONString,
  $clientMutationId: String
) {
  upsertRunQueue(
    input: {
      entityName: $entityName
      projectName: $projectName
      queueName: $queueName
      resourceType: $resourceType
      resourceConfig: $resourceConfig
      templateVariables: $templateVariables
      prioritizationMode: $prioritizationMode
      externalLinks: $externalLinks
      clientMutationId: $clientMutationId
    }
  ) { success configSchemaValidationErrors }
}
"

/-- Upsert a run queue. -/
def upsertRunQueue (cfg : Wandb.Config) (q : RunQueueUpsert) : IO J := do
  let vars := json% {
    entityName: $(q.entity),
    projectName: $(q.project),
    queueName: $(q.name),
    resourceType: $(q.resourceType),
    resourceConfig: $(Wandb.Api.jsonString q.resourceConfig),
    templateVariables: $(Wandb.Api.jsonString? q.templateVariables),
    prioritizationMode: $(q.prioritizationMode),
    externalLinks: $(Wandb.Api.jsonString? q.externalLinks)
  }
  let resp ← Wandb.postGraphQL cfg upsertRunQueueQuery (some vars)
  let data ← Wandb.Api.parseGraphQLData resp
  Wandb.Api.exceptToIO (data.getObjVal? "upsertRunQueue")

/-! ### Push/pop/ack -/

/-- Push request to a run queue. -/
structure RunQueuePush where
  entity : String
  project : String
  name : String
  runSpec : J
  templateVariables : Option J := none
  priority : Option Nat := none

/-- Result from a push/pop request. -/
structure RunQueueItem where
  runQueueItemId : String
  runSpec : Option J := none

def pushToRunQueueByNameQuery (hasPriority hasTemplate : Bool) : String :=
  let params :=
    if hasPriority && hasTemplate then
      "$entityName: String!, $projectName: String!, $queueName: String!, $runSpec: JSONString!, $priority: Int, $templateVariableValues: JSONString"
    else if hasPriority then
      "$entityName: String!, $projectName: String!, $queueName: String!, $runSpec: JSONString!, $priority: Int"
    else if hasTemplate then
      "$entityName: String!, $projectName: String!, $queueName: String!, $runSpec: JSONString!, $templateVariableValues: JSONString"
    else
      "$entityName: String!, $projectName: String!, $queueName: String!, $runSpec: JSONString!"
  let input :=
    if hasPriority && hasTemplate then
      "entityName: $entityName, projectName: $projectName, queueName: $queueName, runSpec: $runSpec, priority: $priority, templateVariableValues: $templateVariableValues"
    else if hasPriority then
      "entityName: $entityName, projectName: $projectName, queueName: $queueName, runSpec: $runSpec, priority: $priority"
    else if hasTemplate then
      "entityName: $entityName, projectName: $projectName, queueName: $queueName, runSpec: $runSpec, templateVariableValues: $templateVariableValues"
    else
      "entityName: $entityName, projectName: $projectName, queueName: $queueName, runSpec: $runSpec"
  "
mutation pushToRunQueueByName(" ++ params ++ ") {
  pushToRunQueueByName(
    input: { " ++ input ++ " }
  ) { runQueueItemId runSpec }
}
"

/-- Push a run spec to a queue by name. -/
def pushToRunQueueByName (cfg : Wandb.Config) (q : RunQueuePush) : IO RunQueueItem := do
  let hasPriority := q.priority.isSome
  let hasTemplate := q.templateVariables.isSome
  let mut fields : List (String × J) := []
  fields := Wandb.Api.addField "entityName" (some <| str q.entity) fields
  fields := Wandb.Api.addField "projectName" (some <| str q.project) fields
  fields := Wandb.Api.addField "queueName" (some <| str q.name) fields
  fields := Wandb.Api.addField "runSpec" (some <| str (Wandb.Api.jsonString q.runSpec)) fields
  fields := Wandb.Api.addField "priority" (q.priority.map (fun n => nat n)) fields
  fields := Wandb.Api.addField "templateVariableValues" (Wandb.Api.jsonString? q.templateVariables |>.map str) fields
  let vars := obj fields.reverse
  let query := pushToRunQueueByNameQuery hasPriority hasTemplate
  let resp ← Wandb.postGraphQL cfg query (some vars)
  let data ← Wandb.Api.parseGraphQLData resp
  let result ← Wandb.Api.exceptToIO (data.getObjVal? "pushToRunQueueByName")
  let runQueueItemId ← Wandb.Api.exceptToIO (result.getObjValAs? String "runQueueItemId")
  let runSpec :=
    match result.getObjValAs? String "runSpec" with
    | .ok s =>
        match Lean.Json.parse s with
        | .ok j => some j
        | .error _ => none
    | .error _ => none
  pure { runQueueItemId := runQueueItemId, runSpec := runSpec }

/-- Pop a run from a queue. -/
def popFromRunQueue
    (cfg : Wandb.Config)
    (entity project queueName : String)
    (launchAgentId : Option String := none) : IO RunQueueItem := do
  let query : String :=
    "
mutation popFromRunQueue($entity: String!, $project: String!, $queueName: String!, $launchAgentId: ID) {
  popFromRunQueue(input: {
    entityName: $entity,
    projectName: $project,
    queueName: $queueName,
    launchAgentId: $launchAgentId
  }) { runQueueItemId runSpec }
}
"
  let vars := json% {
    entity: $(entity),
    project: $(project),
    queueName: $(queueName),
    launchAgentId: $(launchAgentId)
  }
  let resp ← Wandb.postGraphQL cfg query (some vars)
  let data ← Wandb.Api.parseGraphQLData resp
  let result ← Wandb.Api.exceptToIO (data.getObjVal? "popFromRunQueue")
  let runQueueItemId ← Wandb.Api.exceptToIO (result.getObjValAs? String "runQueueItemId")
  let runSpec :=
    match result.getObjValAs? String "runSpec" with
    | .ok s =>
        match Lean.Json.parse s with
        | .ok j => some j
        | .error _ => none
    | .error _ => none
  pure { runQueueItemId := runQueueItemId, runSpec := runSpec }

/-- Ack a run queue item as started by a run id. -/
def ackRunQueueItem (cfg : Wandb.Config) (itemId runId : String) : IO Bool := do
  let query : String :=
    "
mutation ackRunQueueItem($itemId: ID!, $runId: String!) {
  ackRunQueueItem(input: { runQueueItemId: $itemId, runName: $runId }) {
    success
  }
}
"
  let vars := json% { itemId: $(itemId), runId: $(runId) }
  let resp ← Wandb.postGraphQL cfg query (some vars)
  let data ← Wandb.Api.parseGraphQLData resp
  let result ← Wandb.Api.exceptToIO (data.getObjVal? "ackRunQueueItem")
  pure <| (result.getObjValAs? Bool "success" |>.toOption).getD false

/-! ### Fail and warn -/

def failRunQueueItemQueryBasic : String :=
"
mutation failRunQueueItem($runQueueItemId: ID!) {
  failRunQueueItem(input: { runQueueItemId: $runQueueItemId }) { success }
}
"

def failRunQueueItemQueryWithFields : String :=
"
mutation failRunQueueItem($runQueueItemId: ID!, $message: String!, $stage: String!, $filePaths: [String!]) {
  failRunQueueItem(input: {
    runQueueItemId: $runQueueItemId,
    message: $message,
    stage: $stage,
    filePaths: $filePaths
  }) { success }
}
"

/-- Mark a run queue item as failed. -/
def failRunQueueItem
    (cfg : Wandb.Config)
    (itemId : String)
    (message : Option String := none)
    (stage : Option String := none)
    (filePaths : List String := []) : IO Bool := do
  let useFields := message.isSome || stage.isSome || (not filePaths.isEmpty)
  let vars :=
    if useFields then
      json% {
        runQueueItemId: $(itemId),
        message: $(message.getD ""),
        stage: $(stage.getD "run"),
        filePaths: $(arr (filePaths.map str))
      }
    else
      json% { runQueueItemId: $(itemId) }
  let query := if useFields then failRunQueueItemQueryWithFields else failRunQueueItemQueryBasic
  let resp ← Wandb.postGraphQL cfg query (some vars)
  let data ← Wandb.Api.parseGraphQLData resp
  let result ← Wandb.Api.exceptToIO (data.getObjVal? "failRunQueueItem")
  pure <| (result.getObjValAs? Bool "success" |>.toOption).getD false

def updateRunQueueItemWarningQuery : String :=
"
mutation updateRunQueueItemWarning($runQueueItemId: ID!, $message: String!, $stage: String!, $filePaths: [String!]) {
  updateRunQueueItemWarning(input: {
    runQueueItemId: $runQueueItemId,
    message: $message,
    stage: $stage,
    filePaths: $filePaths
  }) { success }
}
"

/-- Add a warning to a run queue item. -/
def updateRunQueueItemWarning
    (cfg : Wandb.Config)
    (itemId : String)
    (message : String)
    (stage : String := "run")
    (filePaths : List String := []) : IO Bool := do
  let vars := json% {
    runQueueItemId: $(itemId),
    message: $(message),
    stage: $(stage),
    filePaths: $(arr (filePaths.map str))
  }
  let resp ← Wandb.postGraphQL cfg updateRunQueueItemWarningQuery (some vars)
  let data ← Wandb.Api.parseGraphQLData resp
  let result ← Wandb.Api.exceptToIO (data.getObjVal? "updateRunQueueItemWarning")
  pure <| (result.getObjValAs? Bool "success" |>.toOption).getD false

end Wandb.RunQueue
