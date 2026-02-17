import Std
import LSpec
import TinyGrad4.Test.Registry

namespace TinyGrad4.Test

open Std

structure DriverOptions where
  profile : TestProfile := .fast
  group? : Option String := none
  filter? : Option String := none
  tagFilters : List String := []
  onlyFilters : List String := []
  seed : Nat := 1337
  listOnly : Bool := false
  json : Bool := false
  showHelp : Bool := false

private def usage : String :=
  String.intercalate "\n"
    [ "TinyGrad4 Lean test driver"
    , ""
    , "Usage:"
    , "  lake test -- [--profile fast|medium|slow] [--group NAME] [--filter TEXT] [--tag TAG] [--only CASE] [--seed NAT] [--list] [--json]"
    , ""
    , "Examples:"
    , "  lake test"
    , "  lake test -- --profile medium"
    , "  lake test -- --profile slow --group indexing --filter ellipsis"
    , "  lake test -- --tag cuda"
    , "  lake test -- --only tensor.rand.seed_determinism"
    , "  lake test -- --list"
    , "  lake test -- --json --profile medium"
    ]

private def selectedTags (opts : DriverOptions) : List String :=
  if opts.tagFilters.isEmpty then ["cpu"] else opts.tagFilters

private def parseArgs : DriverOptions → List String → Except String DriverOptions
  | opts, [] =>
    pure opts
  | opts, "-h" :: rest =>
    parseArgs { opts with showHelp := true } rest
  | opts, "--help" :: rest =>
    parseArgs { opts with showHelp := true } rest
  | opts, "--list" :: rest =>
    parseArgs { opts with listOnly := true } rest
  | opts, "--profile" :: prof :: rest =>
    match TestProfile.parse? prof with
    | some p => parseArgs { opts with profile := p } rest
    | none => throw s!"invalid profile '{prof}' (expected fast|medium|slow)"
  | _, "--profile" :: [] =>
    throw "missing value for --profile"
  | opts, "--group" :: g :: rest =>
    parseArgs { opts with group? := some g } rest
  | _, "--group" :: [] =>
    throw "missing value for --group"
  | opts, "--filter" :: f :: rest =>
    parseArgs { opts with filter? := some f } rest
  | _, "--filter" :: [] =>
    throw "missing value for --filter"
  | opts, "--tag" :: tag :: rest =>
    parseArgs { opts with tagFilters := opts.tagFilters ++ [tag] } rest
  | _, "--tag" :: [] =>
    throw "missing value for --tag"
  | opts, "--only" :: caseName :: rest =>
    parseArgs { opts with onlyFilters := opts.onlyFilters ++ [caseName] } rest
  | _, "--only" :: [] =>
    throw "missing value for --only"
  | opts, "--json" :: rest =>
    parseArgs { opts with json := true } rest
  | opts, "--seed" :: s :: rest =>
    match s.toNat? with
    | some n => parseArgs { opts with seed := n } rest
    | none => throw s!"invalid seed '{s}' (expected a natural number)"
  | _, "--seed" :: [] =>
    throw "missing value for --seed"
  | _, arg :: _ =>
    throw s!"unknown argument '{arg}'"

private def printSelection (cfg : RunConfig) (opts : DriverOptions) (cases : List TestCase) : IO Unit := do
  let tags := String.intercalate "," (selectedTags opts)
  let only := if opts.onlyFilters.isEmpty then "*" else String.intercalate "," opts.onlyFilters
  IO.println s!"profile={repr cfg.profile} seed={cfg.seed} group={opts.group?.getD "*"} filter={opts.filter?.getD "*"} tags={tags} only={only}"
  IO.println s!"selected {cases.length} test case(s)"

private def shellQuote (s : String) : String :=
  if s.contains ' ' || s.contains '"' || s.contains '\'' then
    "\"" ++ s.replace "\"" "\\\"" ++ "\""
  else
    s

private def rerunArgs (opts : DriverOptions) : List String :=
  let profileArgs : List String := ["--profile", opts.profile.toArg, "--seed", toString opts.seed]
  let groupArgs : List String := match opts.group? with | some g => ["--group", g] | none => []
  let filterArgs : List String := match opts.filter? with | some f => ["--filter", f] | none => []
  let tagArgs := (selectedTags opts).foldr (fun t acc => ["--tag", t] ++ acc) []
  let onlyArgs := opts.onlyFilters.foldr (fun t acc => ["--only", t] ++ acc) []
  let jsonArgs := if opts.json then ["--json"] else []
  profileArgs ++ groupArgs ++ filterArgs ++ tagArgs ++ onlyArgs ++ jsonArgs

private def rerunCmd (opts : DriverOptions) : String :=
  let args := String.intercalate " " ((rerunArgs opts).map shellQuote)
  s!"lake test -- {args}"

private def applyOnlyFilters (opts : DriverOptions) (cases : List TestCase) : List TestCase :=
  if opts.onlyFilters.isEmpty then
    cases
  else
    cases.filter fun tc => opts.onlyFilters.contains tc.name

private def missingOnlyFilters (opts : DriverOptions) (cases : List TestCase) : List String :=
  if opts.onlyFilters.isEmpty then
    []
  else
    opts.onlyFilters.filter fun requested =>
      !(cases.any fun tc => tc.name == requested)

private def jsonOptString (s : Option String) : Lean.Json :=
  match s with
  | some v => Lean.Json.str v
  | none => Lean.Json.null

private def testProfileJson (p : TestProfile) : Lean.Json :=
  Lean.Json.str p.toArg

private def caseToJson (tc : TestCase) : Lean.Json :=
  Lean.Json.mkObj
    [ ("name", Lean.Json.str tc.name)
    , ("group", Lean.Json.str tc.group)
    , ("min_profile", testProfileJson tc.minProfile)
    , ("tags", Lean.Json.arr <| (tc.tags.map Lean.Json.str).toArray)
    , ("python_refs", Lean.Json.arr <| (tc.pythonRefs.map Lean.Json.str).toArray)
    ]

private def selectionToJson (cfg : RunConfig) (opts : DriverOptions) (cases : List TestCase) : Lean.Json :=
  Lean.Json.mkObj
    [ ("profile", testProfileJson cfg.profile)
    , ("seed", Lean.toJson cfg.seed)
    , ("group", jsonOptString opts.group?)
    , ("filter", jsonOptString opts.filter?)
    , ("tags", Lean.Json.arr <| (selectedTags opts).map Lean.Json.str |> List.toArray)
    , ("only", Lean.Json.arr <| (opts.onlyFilters.map Lean.Json.str).toArray)
    , ("selected_count", Lean.toJson cases.length)
    , ("selected", Lean.Json.arr <| (cases.map caseToJson).toArray)
    ]

private def runSelectedJson (cfg : RunConfig) (opts : DriverOptions) (cases : List TestCase) : IO UInt32 := do
  let mut allPassed := true
  let mut rows : Array Lean.Json := #[]
  for tc in cases do
    let startMs ← IO.monoMsNow
    let (success, output) ← (tc.suite cfg).runIO
    let elapsedMs := (← IO.monoMsNow) - startMs
    if !success then
      allPassed := false
    let row := Lean.Json.mkObj
      [ ("name", Lean.Json.str tc.name)
      , ("group", Lean.Json.str tc.group)
      , ("min_profile", testProfileJson tc.minProfile)
      , ("tags", Lean.Json.arr <| (tc.tags.map Lean.Json.str).toArray)
      , ("python_refs", Lean.Json.arr <| (tc.pythonRefs.map Lean.Json.str).toArray)
      , ("success", Lean.toJson success)
      , ("elapsed_ms", Lean.toJson elapsedMs)
      , ("output", Lean.Json.str output)
      ]
    rows := rows.push row
  let payload := Lean.Json.mkObj
    [ ("run", selectionToJson cfg opts cases)
    , ("results", Lean.Json.arr rows)
    , ("success", Lean.toJson allPassed)
    , ("rerun_cmd", Lean.Json.str (rerunCmd opts))
    ]
  IO.println (toString payload)
  return if allPassed then 0 else 1

private def printJsonError (code : String) (message : String) (extra : List (String × Lean.Json) := []) : IO Unit := do
  let payload := Lean.Json.mkObj <|
    [ ("success", Lean.toJson false)
    , ("error", Lean.Json.str code)
    , ("message", Lean.Json.str message)
    ] ++ extra
  IO.println (toString payload)

private def runSelected (cfg : RunConfig) (opts : DriverOptions) (cases : List TestCase) : IO UInt32 := do
  if cases.isEmpty then
    if opts.json then
      printJsonError "no_tests_matched" "No tests matched the requested filters."
        [("run", selectionToJson cfg opts cases), ("rerun_cmd", Lean.Json.str (rerunCmd opts))]
    else
      IO.eprintln "No tests matched the requested filters."
    return 1

  if opts.json then
    if opts.listOnly then
      IO.println (toString <| selectionToJson cfg opts cases)
      return 0
    return ← runSelectedJson cfg opts cases

  printSelection cfg opts cases

  if opts.listOnly then
    for tc in cases do
      IO.println s!"- {renderCase tc}"
    return 0

  let suites := toSuiteMap cfg cases
  let status ← LSpec.lspecIO suites []
  if status != 0 then
    IO.eprintln s!"Test run failed. Re-run command: {rerunCmd opts}"
  return status

def runMain (args : List String) : IO UInt32 := do
  match parseArgs {} args with
  | .error err =>
    if args.contains "--json" then
      printJsonError "invalid_args" err [("usage", Lean.Json.str usage)]
    else
      IO.eprintln err
      IO.eprintln ""
      IO.eprintln usage
    return 2
  | .ok opts =>
    if opts.showHelp then
      IO.println usage
      return 0
    let cfg : RunConfig := { profile := opts.profile, seed := opts.seed }
    let candidateCases := selectCases cfg opts.group? opts.filter? (selectedTags opts)
    let missingOnly := missingOnlyFilters opts candidateCases
    if !missingOnly.isEmpty then
      if opts.json then
        printJsonError "unknown_only_case"
          s!"unknown --only case id(s): {String.intercalate ", " missingOnly}"
          [ ("unknown_only", Lean.Json.arr <| (missingOnly.map Lean.Json.str).toArray)
          , ("run", selectionToJson cfg opts candidateCases)
          , ("hint", Lean.Json.str "Run `lake test -- --list` to see available case identifiers.")
          ]
      else
        IO.eprintln s!"unknown --only case id(s): {String.intercalate ", " missingOnly}"
        IO.eprintln "Run `lake test -- --list` to see available case identifiers."
      return 2
    let cases := applyOnlyFilters opts candidateCases
    runSelected cfg opts cases

end TinyGrad4.Test

def main (args : List String) : IO UInt32 :=
  TinyGrad4.Test.runMain args
