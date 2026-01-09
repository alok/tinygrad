import Std
import Lean.Data.Json.Parser
import Wandb.Api
import Wandb.Client
import Wandb.Files
import Wandb.Json
import Wandb.Run

/-!
Artifact helpers for listing artifacts and files.
-/

namespace Wandb.Artifacts

open Wandb
open Wandb.Json

/-! ### Types -/

/-- Artifact metadata from list queries. -/
structure ArtifactInfo where
  id : String
  collection : String
  versionIndex : Option Nat := none
  version : Option String := none
  typeName : String
  entity : String
  project : String
  description : Option String := none
  state : Option String := none
  size : Option Nat := none
  digest : Option String := none
  fileCount : Option Nat := none
  createdAt : Option String := none
  updatedAt : Option String := none
  aliases : Array String := #[]
  metadata : Option J := none

/-- Run artifact mode. -/
inductive RunArtifactMode where
  | input
  | output

/-- Pagination results for artifact lists. -/
structure ArtifactPage where
  artifacts : Array ArtifactInfo
  endCursor : Option String := none
  hasNextPage : Bool := false
  totalCount : Option Nat := none

/-- Artifact file metadata. -/
structure ArtifactFileInfo where
  id : String
  name : String
  url : Option String := none
  directUrl : Option String := none
  sizeBytes : Option Nat := none
  mimetype : Option String := none
  updatedAt : Option String := none
  digest : Option String := none
  md5 : Option String := none

/-- Page of artifact files. -/
structure ArtifactFilePage where
  files : Array ArtifactFileInfo
  endCursor : Option String := none
  hasNextPage : Bool := false
  totalCount : Option Nat := none

/-! ### Upload helpers -/

/-- Split a {lit}`Header: value` string into a header pair. -/
def parseHeader (s : String) : Option (String × String) :=
  match s.splitOn ":" with
  | [] => none
  | k :: rest =>
      let v := String.intercalate ":" rest
      some (k.trimAscii.toString, v.trimAscii.toString)

/-- Convert header strings into header pairs. -/
def headersFromStrings (xs : Array String) : List (String × String) :=
  xs.foldl (init := []) fun acc s =>
    match parseHeader s with
    | none => acc
    | some h => h :: acc

/-! ### Parsing helpers -/

/-- Parse aliases from an artifact node. -/
def parseAliases (j : J) : Array String :=
  match j.getObjValAs? (Array J) "aliases" with
  | .ok arr =>
      arr.foldl (init := #[]) fun acc a =>
        match a.getObjValAs? String "alias" with
        | .ok s => acc.push s
        | .error _ => acc
  | .error _ => #[]

/-- Parse an artifact info record from JSON. -/
def ArtifactInfo.fromJson (j : J) : Except String ArtifactInfo := do
  let id ← j.getObjValAs? String "id"
  let seqObj ← j.getObjVal? "artifactSequence"
  let collection ← seqObj.getObjValAs? String "name"
  let projectObj ← seqObj.getObjVal? "project"
  let project ← projectObj.getObjValAs? String "name"
  let entityObj ← projectObj.getObjVal? "entity"
  let entity ← entityObj.getObjValAs? String "name"
  let typeObj ← j.getObjVal? "artifactType"
  let typeName ← typeObj.getObjValAs? String "name"
  let versionIndex := j.getObjValAs? Nat "versionIndex" |>.toOption
  let description := j.getObjValAs? String "description" |>.toOption
  let state := j.getObjValAs? String "state" |>.toOption
  let size := j.getObjValAs? Nat "size" |>.toOption
  let digest := j.getObjValAs? String "digest" |>.toOption
  let fileCount := j.getObjValAs? Nat "fileCount" |>.toOption
  let createdAt := j.getObjValAs? String "createdAt" |>.toOption
  let updatedAt := j.getObjValAs? String "updatedAt" |>.toOption
  let metadata := j.getObjVal? "metadata" |>.toOption
  let aliases := parseAliases j
  pure {
    id := id
    collection := collection
    versionIndex := versionIndex
    typeName := typeName
    entity := entity
    project := project
    description := description
    state := state
    size := size
    digest := digest
    fileCount := fileCount
    createdAt := createdAt
    updatedAt := updatedAt
    aliases := aliases
    metadata := metadata
  }

/-- Parse artifact file info from JSON. -/
def ArtifactFileInfo.fromJson (j : J) : Except String ArtifactFileInfo := do
  let id ← j.getObjValAs? String "id"
  let name := (j.getObjValAs? String "name" |>.toOption).getD ""
  let url := j.getObjValAs? String "url" |>.toOption
  let directUrl := j.getObjValAs? String "directUrl" |>.toOption
  let sizeBytes := j.getObjValAs? Nat "sizeBytes" |>.toOption
  let mimetype := j.getObjValAs? String "mimetype" |>.toOption
  let updatedAt := j.getObjValAs? String "updatedAt" |>.toOption
  let digest := j.getObjValAs? String "digest" |>.toOption
  let md5 := j.getObjValAs? String "md5" |>.toOption
  pure {
    id := id
    name := name
    url := url
    directUrl := directUrl
    sizeBytes := sizeBytes
    mimetype := mimetype
    updatedAt := updatedAt
    digest := digest
    md5 := md5
  }

/-! ### Run artifacts -/

def runOutputArtifactsQuery : String :=
"
query RunOutputArtifacts($entity: String!, $project: String!, $run: String!, $cursor: String, $perPage: Int = 50) {
  project(entityName: $entity, name: $project) {
    run(name: $run) {
      artifacts: outputArtifacts(after: $cursor, first: $perPage) {
        totalCount
        pageInfo { endCursor hasNextPage }
        edges {
          node {
            id
            artifactSequence { name project { name entity { name } } }
            versionIndex
            artifactType { name }
            description
            metadata
            state
            size
            digest
            fileCount
            createdAt
            updatedAt
            aliases { alias }
          }
        }
      }
    }
  }
}
"

def runInputArtifactsQuery : String :=
"
query RunInputArtifacts($entity: String!, $project: String!, $run: String!, $cursor: String, $perPage: Int = 50) {
  project(entityName: $entity, name: $project) {
    run(name: $run) {
      artifacts: inputArtifacts(after: $cursor, first: $perPage) {
        totalCount
        pageInfo { endCursor hasNextPage }
        edges {
          node {
            id
            artifactSequence { name project { name entity { name } } }
            versionIndex
            artifactType { name }
            description
            metadata
            state
            size
            digest
            fileCount
            createdAt
            updatedAt
            aliases { alias }
          }
        }
      }
    }
  }
}
"

/-- List input or output artifacts for a run. -/
def listRunArtifacts
    (cfg : Wandb.Config)
    (ref : Wandb.RunRef)
    (mode : RunArtifactMode := .output)
    (cursor : Option String := none)
    (perPage : Nat := 50) : IO ArtifactPage := do
  let vars := json% {
    entity: $(ref.entity),
    project: $(ref.project),
    run: $(ref.id),
    cursor: $(cursor),
    perPage: $(perPage)
  }
  let query := match mode with | .input => runInputArtifactsQuery | .output => runOutputArtifactsQuery
  let resp ← Wandb.postGraphQL cfg query (some vars)
  let data ← Wandb.Api.parseGraphQLData resp
  let projectObj ← Wandb.Api.exceptToIO (data.getObjVal? "project")
  let runObj ← Wandb.Api.exceptToIO (projectObj.getObjVal? "run")
  let artifactsObj ← Wandb.Api.exceptToIO (runObj.getObjVal? "artifacts")
  let totalCount := artifactsObj.getObjValAs? Nat "totalCount" |>.toOption
  let edges ← Wandb.Api.exceptToIO (artifactsObj.getObjValAs? (Array J) "edges")
  let mut artifacts : Array ArtifactInfo := #[]
  for edge in edges do
    let node ← Wandb.Api.exceptToIO (edge.getObjVal? "node")
    let info ← Wandb.Api.exceptToIO (ArtifactInfo.fromJson node)
    artifacts := artifacts.push info
  let pageInfo ← Wandb.Api.exceptToIO (artifactsObj.getObjVal? "pageInfo")
  let endCursor := pageInfo.getObjValAs? String "endCursor" |>.toOption
  let hasNextPage := (pageInfo.getObjValAs? Bool "hasNextPage" |>.toOption).getD false
  pure { artifacts := artifacts, endCursor := endCursor, hasNextPage := hasNextPage, totalCount := totalCount }

/-! ### Project artifacts -/

/-- Parameters for listing artifacts in a collection. -/
structure ProjectArtifactsQuery where
  entity : String
  project : String
  artifactType : String
  collection : String
  cursor : Option String := none
  perPage : Nat := 50
  order : Option String := none
  filters : Option J := none

def projectArtifactsQuery : String :=
"
query ProjectArtifacts($entity: String!, $project: String!, $type: String!, $collection: String!, $cursor: String, $perPage: Int = 50, $order: String, $filters: JSONString) {
  project(entityName: $entity, name: $project) {
    artifactType(name: $type) {
      artifactCollection(name: $collection) {
        artifacts(after: $cursor, first: $perPage, order: $order, filters: $filters) {
          totalCount
          pageInfo { endCursor hasNextPage }
          edges {
            version
            node {
              id
              artifactSequence { name project { name entity { name } } }
              versionIndex
              artifactType { name }
              description
              metadata
              state
              size
              digest
              fileCount
              createdAt
              updatedAt
              aliases { alias }
            }
          }
        }
      }
    }
  }
}
"

/-- List artifacts in a collection. -/
def listProjectArtifacts (cfg : Wandb.Config) (q : ProjectArtifactsQuery) : IO ArtifactPage := do
  let vars := json% {
    entity: $(q.entity),
    project: $(q.project),
    type: $(q.artifactType),
    collection: $(q.collection),
    cursor: $(q.cursor),
    perPage: $(q.perPage),
    order: $(q.order),
    filters: $(Wandb.Api.jsonString? q.filters)
  }
  let resp ← Wandb.postGraphQL cfg projectArtifactsQuery (some vars)
  let data ← Wandb.Api.parseGraphQLData resp
  let projectObj ← Wandb.Api.exceptToIO (data.getObjVal? "project")
  let typeObj ← Wandb.Api.exceptToIO (projectObj.getObjVal? "artifactType")
  let collectionObj ← Wandb.Api.exceptToIO (typeObj.getObjVal? "artifactCollection")
  let artifactsObj ← Wandb.Api.exceptToIO (collectionObj.getObjVal? "artifacts")
  let totalCount := artifactsObj.getObjValAs? Nat "totalCount" |>.toOption
  let edges ← Wandb.Api.exceptToIO (artifactsObj.getObjValAs? (Array J) "edges")
  let mut artifacts : Array ArtifactInfo := #[]
  for edge in edges do
    let node ← Wandb.Api.exceptToIO (edge.getObjVal? "node")
    let info ← Wandb.Api.exceptToIO (ArtifactInfo.fromJson node)
    let version := edge.getObjValAs? String "version" |>.toOption
    artifacts := artifacts.push { info with version := version }
  let pageInfo ← Wandb.Api.exceptToIO (artifactsObj.getObjVal? "pageInfo")
  let endCursor := pageInfo.getObjValAs? String "endCursor" |>.toOption
  let hasNextPage := (pageInfo.getObjValAs? Bool "hasNextPage" |>.toOption).getD false
  pure { artifacts := artifacts, endCursor := endCursor, hasNextPage := hasNextPage, totalCount := totalCount }

/-! ### Artifact files -/

def artifactFilesQuery : String :=
"
query GetArtifactFiles($entity: String!, $project: String!, $type: String!, $name: String!, $fileNames: [String!], $cursor: String, $perPage: Int = 50) {
  project(name: $project, entityName: $entity) {
    artifactType(name: $type) {
      artifact(name: $name) {
        files(names: $fileNames, after: $cursor, first: $perPage) {
          totalCount
          pageInfo { endCursor hasNextPage }
          edges { node { id name: displayName url sizeBytes mimetype updatedAt digest md5 directUrl } }
        }
      }
    }
  }
}
"

def artifactMembershipFilesQuery : String :=
"
query GetArtifactMembershipFiles($entity: String!, $project: String!, $collection: String!, $alias: String!, $fileNames: [String!], $cursor: String, $perPage: Int = 50) {
  project(name: $project, entityName: $entity) {
    artifactCollection(name: $collection) {
      artifactMembership(aliasName: $alias) {
        files(names: $fileNames, after: $cursor, first: $perPage) {
          totalCount
          pageInfo { endCursor hasNextPage }
          edges { node { id name: displayName url sizeBytes mimetype updatedAt digest md5 directUrl } }
        }
      }
    }
  }
}
"

/-- List files for an artifact by type and name. -/
def listArtifactFiles
    (cfg : Wandb.Config)
    (entity project typeName artifactName : String)
    (fileNames : List String := [])
    (cursor : Option String := none)
    (perPage : Nat := 50) : IO ArtifactFilePage := do
  let vars := json% {
    entity: $(entity),
    project: $(project),
    type: $(typeName),
    name: $(artifactName),
    fileNames: $(arr (fileNames.map str)),
    cursor: $(cursor),
    perPage: $(perPage)
  }
  let resp ← Wandb.postGraphQL cfg artifactFilesQuery (some vars)
  let data ← Wandb.Api.parseGraphQLData resp
  let projectObj ← Wandb.Api.exceptToIO (data.getObjVal? "project")
  let typeObj ← Wandb.Api.exceptToIO (projectObj.getObjVal? "artifactType")
  let artifactObj ← Wandb.Api.exceptToIO (typeObj.getObjVal? "artifact")
  let filesObj ← Wandb.Api.exceptToIO (artifactObj.getObjVal? "files")
  let totalCount := filesObj.getObjValAs? Nat "totalCount" |>.toOption
  let edges ← Wandb.Api.exceptToIO (filesObj.getObjValAs? (Array J) "edges")
  let mut files : Array ArtifactFileInfo := #[]
  for edge in edges do
    let node ← Wandb.Api.exceptToIO (edge.getObjVal? "node")
    let info ← Wandb.Api.exceptToIO (ArtifactFileInfo.fromJson node)
    files := files.push info
  let pageInfo ← Wandb.Api.exceptToIO (filesObj.getObjVal? "pageInfo")
  let endCursor := pageInfo.getObjValAs? String "endCursor" |>.toOption
  let hasNextPage := (pageInfo.getObjValAs? Bool "hasNextPage" |>.toOption).getD false
  pure { files := files, endCursor := endCursor, hasNextPage := hasNextPage, totalCount := totalCount }

/-- List files for an artifact via collection and alias. -/
def listArtifactFilesByAlias
    (cfg : Wandb.Config)
    (entity project collection alias : String)
    (fileNames : List String := [])
    (cursor : Option String := none)
    (perPage : Nat := 50) : IO ArtifactFilePage := do
  let vars := json% {
    entity: $(entity),
    project: $(project),
    collection: $(collection),
    alias: $(alias),
    fileNames: $(arr (fileNames.map str)),
    cursor: $(cursor),
    perPage: $(perPage)
  }
  let resp ← Wandb.postGraphQL cfg artifactMembershipFilesQuery (some vars)
  let data ← Wandb.Api.parseGraphQLData resp
  let projectObj ← Wandb.Api.exceptToIO (data.getObjVal? "project")
  let collectionObj ← Wandb.Api.exceptToIO (projectObj.getObjVal? "artifactCollection")
  let membershipObj ← Wandb.Api.exceptToIO (collectionObj.getObjVal? "artifactMembership")
  let filesObj ← Wandb.Api.exceptToIO (membershipObj.getObjVal? "files")
  let totalCount := filesObj.getObjValAs? Nat "totalCount" |>.toOption
  let edges ← Wandb.Api.exceptToIO (filesObj.getObjValAs? (Array J) "edges")
  let mut files : Array ArtifactFileInfo := #[]
  for edge in edges do
    let node ← Wandb.Api.exceptToIO (edge.getObjVal? "node")
    let info ← Wandb.Api.exceptToIO (ArtifactFileInfo.fromJson node)
    files := files.push info
  let pageInfo ← Wandb.Api.exceptToIO (filesObj.getObjVal? "pageInfo")
  let endCursor := pageInfo.getObjValAs? String "endCursor" |>.toOption
  let hasNextPage := (pageInfo.getObjValAs? Bool "hasNextPage" |>.toOption).getD false
  pure { files := files, endCursor := endCursor, hasNextPage := hasNextPage, totalCount := totalCount }

/-! ### Artifact create/upload -/

/-- Inputs for creating an artifact record. -/
structure ArtifactCreate where
  entity : String
  project : String
  runName : Option String := none
  typeName : String
  collection : String
  digest : String
  description : Option String := none
  metadata : Option J := none
  aliases : List String := []

structure ArtifactCreateResult where
  artifactId : String
  state : Option String := none
  sequenceId : Option String := none
  latestId : Option String := none
  latestVersionIndex : Option Nat := none

def createArtifactQuery : String :=
"
mutation CreateArtifact(
  $artifactTypeName: String!,
  $artifactCollectionNames: [String!],
  $entityName: String!,
  $projectName: String!,
  $runName: String,
  $description: String,
  $digest: String!,
  $aliases: [ArtifactAliasInput!],
  $metadata: JSONString
) {
  createArtifact(input: {
    artifactTypeName: $artifactTypeName,
    artifactCollectionNames: $artifactCollectionNames,
    entityName: $entityName,
    projectName: $projectName,
    runName: $runName,
    description: $description,
    digest: $digest,
    digestAlgorithm: MANIFEST_MD5,
    aliases: $aliases,
    metadata: $metadata
  }) {
    artifact {
      id
      state
      artifactSequence {
        id
        latestArtifact { id versionIndex }
      }
    }
  }
}
"

def aliasInputs (collection : String) (aliases : List String) : J :=
  arr <| aliases.map fun a =>
    json% { alias: $(a), artifactCollectionName: $(collection) }

/-- Create an artifact record. -/
def createArtifact (cfg : Wandb.Config) (c : ArtifactCreate) : IO ArtifactCreateResult := do
  let mut fields : List (String × J) := []
  fields := Wandb.Api.addField "artifactTypeName" (some <| str c.typeName) fields
  fields := Wandb.Api.addField "artifactCollectionNames" (some <| arr [str c.collection]) fields
  fields := Wandb.Api.addField "entityName" (some <| str c.entity) fields
  fields := Wandb.Api.addField "projectName" (some <| str c.project) fields
  fields := Wandb.Api.addField "runName" (c.runName.map str) fields
  fields := Wandb.Api.addField "description" (c.description.map str) fields
  fields := Wandb.Api.addField "digest" (some <| str c.digest) fields
  if not c.aliases.isEmpty then
    fields := Wandb.Api.addField "aliases" (some <| aliasInputs c.collection c.aliases) fields
  fields := Wandb.Api.addField "metadata" (Wandb.Api.jsonString? c.metadata |>.map str) fields
  let vars := obj fields.reverse
  let resp ← Wandb.postGraphQL cfg createArtifactQuery (some vars)
  let data ← Wandb.Api.parseGraphQLData resp
  let result ← Wandb.Api.exceptToIO (data.getObjVal? "createArtifact")
  let artifact ← Wandb.Api.exceptToIO (result.getObjVal? "artifact")
  let artifactId ← Wandb.Api.exceptToIO (artifact.getObjValAs? String "id")
  let state := artifact.getObjValAs? String "state" |>.toOption
  let seqObj ← Wandb.Api.exceptToIO (artifact.getObjVal? "artifactSequence")
  let seqId := seqObj.getObjValAs? String "id" |>.toOption
  let latest := seqObj.getObjVal? "latestArtifact" |>.toOption
  let latestId := latest.bind (fun j => (j.getObjValAs? String "id").toOption)
  let latestVersionIndex := latest.bind (fun j => (j.getObjValAs? Nat "versionIndex").toOption)
  pure {
    artifactId := artifactId
    state := state
    sequenceId := seqId
    latestId := latestId
    latestVersionIndex := latestVersionIndex
  }

/-- File metadata for a manifest upload. -/
structure ArtifactManifestFile where
  id : String
  name : String
  displayName : String
  uploadUrl : Option String := none
  uploadHeaders : Array String := #[]

structure ArtifactManifestResult where
  manifestId : String
  file : ArtifactManifestFile

def createArtifactManifestQuery : String :=
"
mutation CreateArtifactManifest(
  $name: String!,
  $digest: String!,
  $artifactID: ID!,
  $entityName: String!,
  $projectName: String!,
  $runName: String!,
  $includeUpload: Boolean!
) {
  createArtifactManifest(input: {
    name: $name,
    digest: $digest,
    artifactID: $artifactID,
    entityName: $entityName,
    projectName: $projectName,
    runName: $runName
  }) {
    artifactManifest {
      id
      file {
        id
        name
        displayName
        uploadUrl @include(if: $includeUpload)
        uploadHeaders @include(if: $includeUpload)
      }
    }
  }
}
"

/-- Create an artifact manifest record. -/
def createArtifactManifest
    (cfg : Wandb.Config)
    (artifactId : String)
    (name : String)
    (digest : String)
    (entity project runName : String)
    (includeUpload : Bool := true) : IO ArtifactManifestResult := do
  let vars := json% {
    name: $(name),
    digest: $(digest),
    artifactID: $(artifactId),
    entityName: $(entity),
    projectName: $(project),
    runName: $(runName),
    includeUpload: $(includeUpload)
  }
  let resp ← Wandb.postGraphQL cfg createArtifactManifestQuery (some vars)
  let data ← Wandb.Api.parseGraphQLData resp
  let result ← Wandb.Api.exceptToIO (data.getObjVal? "createArtifactManifest")
  let manifest ← Wandb.Api.exceptToIO (result.getObjVal? "artifactManifest")
  let manifestId ← Wandb.Api.exceptToIO (manifest.getObjValAs? String "id")
  let fileObj ← Wandb.Api.exceptToIO (manifest.getObjVal? "file")
  let fileId ← Wandb.Api.exceptToIO (fileObj.getObjValAs? String "id")
  let fileName ← Wandb.Api.exceptToIO (fileObj.getObjValAs? String "name")
  let displayName ← Wandb.Api.exceptToIO (fileObj.getObjValAs? String "displayName")
  let uploadUrl := fileObj.getObjValAs? String "uploadUrl" |>.toOption
  let uploadHeaders := (fileObj.getObjValAs? (Array String) "uploadHeaders" |>.toOption).getD #[]
  pure {
    manifestId := manifestId
    file := {
      id := fileId
      name := fileName
      displayName := displayName
      uploadUrl := uploadUrl
      uploadHeaders := uploadHeaders
    }
  }

/-- File spec used to request upload URLs. -/
structure ArtifactFileSpec where
  artifactId : String
  name : String
  md5 : String
  mimetype : Option String := none
  manifestId : Option String := none

/-- Upload metadata returned from {lit}`createArtifactFiles`. -/
structure ArtifactFileUpload where
  id : String
  name : String
  displayName : String
  uploadUrl : Option String := none
  uploadHeaders : Array String := #[]
  artifactId : String

def createArtifactFilesQuery : String :=
"
mutation CreateArtifactFiles(
  $storageLayout: ArtifactStorageLayout!,
  $artifactFiles: [CreateArtifactFileSpecInput!]!
) {
  createArtifactFiles(input: {
    artifactFiles: $artifactFiles,
    storageLayout: $storageLayout
  }) {
    files {
      edges {
        node {
          id
          name
          displayName
          uploadUrl
          uploadHeaders
          artifact { id }
        }
      }
    }
  }
}
"

def fileSpecToJson (f : ArtifactFileSpec) : J :=
  let fields : List (String × J) := []
  let fields := Wandb.Api.addField "artifactID" (some <| str f.artifactId) fields
  let fields := Wandb.Api.addField "name" (some <| str f.name) fields
  let fields := Wandb.Api.addField "md5" (some <| str f.md5) fields
  let fields := Wandb.Api.addField "mimetype" (f.mimetype.map str) fields
  let fields := Wandb.Api.addField "artifactManifestID" (f.manifestId.map str) fields
  obj fields.reverse

/-- Request upload URLs for artifact files. -/
def createArtifactFiles
    (cfg : Wandb.Config)
    (files : List ArtifactFileSpec)
    (storageLayout : String := "V2") : IO (Array ArtifactFileUpload) := do
  let vars := obj [
    ("storageLayout", str storageLayout),
    ("artifactFiles", arr (files.map fileSpecToJson))
  ]
  let resp ← Wandb.postGraphQL cfg createArtifactFilesQuery (some vars)
  let data ← Wandb.Api.parseGraphQLData resp
  let result ← Wandb.Api.exceptToIO (data.getObjVal? "createArtifactFiles")
  let filesObj ← Wandb.Api.exceptToIO (result.getObjVal? "files")
  let edges ← Wandb.Api.exceptToIO (filesObj.getObjValAs? (Array J) "edges")
  let mut out : Array ArtifactFileUpload := #[]
  for edge in edges do
    let node ← Wandb.Api.exceptToIO (edge.getObjVal? "node")
    let id ← Wandb.Api.exceptToIO (node.getObjValAs? String "id")
    let name ← Wandb.Api.exceptToIO (node.getObjValAs? String "name")
    let displayName ← Wandb.Api.exceptToIO (node.getObjValAs? String "displayName")
    let uploadUrl := node.getObjValAs? String "uploadUrl" |>.toOption
    let uploadHeaders := (node.getObjValAs? (Array String) "uploadHeaders" |>.toOption).getD #[]
    let artifactObj ← Wandb.Api.exceptToIO (node.getObjVal? "artifact")
    let artifactId ← Wandb.Api.exceptToIO (artifactObj.getObjValAs? String "id")
    out := out.push {
      id := id
      name := name
      displayName := displayName
      uploadUrl := uploadUrl
      uploadHeaders := uploadHeaders
      artifactId := artifactId
    }
  pure out

/-- Upload a manifest file using the provided upload URL. -/
def uploadArtifactManifest
    (file : ArtifactManifestFile)
    (path : System.FilePath) : IO Http.Response := do
  let some url := file.uploadUrl
    | throw <| IO.userError "manifest upload URL not provided"
  let headers := headersFromStrings file.uploadHeaders
  Wandb.uploadFile url path headers

/-- Upload a file using its artifact upload metadata. -/
def uploadArtifactFile
    (file : ArtifactFileUpload)
    (path : System.FilePath) : IO Http.Response := do
  let some url := file.uploadUrl
    | throw <| IO.userError "artifact file upload URL not provided"
  let headers := headersFromStrings file.uploadHeaders
  Wandb.uploadFile url path headers

def commitArtifactQuery : String :=
"
mutation CommitArtifact($artifactID: ID!) {
  commitArtifact(input: { artifactID: $artifactID }) {
    artifact { id digest }
  }
}
"

/-- Commit an artifact by id. -/
def commitArtifact (cfg : Wandb.Config) (artifactId : String) : IO J := do
  let vars := json% { artifactID: $(artifactId) }
  let resp ← Wandb.postGraphQL cfg commitArtifactQuery (some vars)
  let data ← Wandb.Api.parseGraphQLData resp
  let result ← Wandb.Api.exceptToIO (data.getObjVal? "commitArtifact")
  Wandb.Api.exceptToIO (result.getObjVal? "artifact")

/-! ### Download helpers -/

/-- Download a file from an artifact file info, preferring {lit}`directUrl`. -/
def downloadArtifactFile (info : ArtifactFileInfo) (path : System.FilePath) : IO Http.Response := do
  let url := info.directUrl.getD (info.url.getD "")
  if url.isEmpty then
    throw <| IO.userError "artifact file does not include a URL"
  Wandb.downloadFile url path

end Wandb.Artifacts
