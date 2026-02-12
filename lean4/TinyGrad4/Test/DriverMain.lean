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
  seed : Nat := 1337
  listOnly : Bool := false
  showHelp : Bool := false

private def usage : String :=
  String.intercalate "\n"
    [ "TinyGrad4 Lean test driver"
    , ""
    , "Usage:"
    , "  lake test -- [--profile fast|medium|slow] [--group NAME] [--filter TEXT] [--tag TAG] [--seed NAT] [--list]"
    , ""
    , "Examples:"
    , "  lake test"
    , "  lake test -- --profile medium"
    , "  lake test -- --profile slow --group indexing --filter ellipsis"
    , "  lake test -- --tag cuda"
    , "  lake test -- --list"
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
  IO.println s!"profile={repr cfg.profile} seed={cfg.seed} group={opts.group?.getD "*"} filter={opts.filter?.getD "*"} tags={tags}"
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
  profileArgs ++ groupArgs ++ filterArgs ++ tagArgs

private def rerunCmd (opts : DriverOptions) : String :=
  let args := String.intercalate " " ((rerunArgs opts).map shellQuote)
  s!"lake test -- {args}"

private def runSelected (cfg : RunConfig) (opts : DriverOptions) (cases : List TestCase) : IO UInt32 := do
  if cases.isEmpty then
    IO.eprintln "No tests matched the requested filters."
    return 1

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
    IO.eprintln err
    IO.eprintln ""
    IO.eprintln usage
    return 2
  | .ok opts =>
    if opts.showHelp then
      IO.println usage
      return 0
    let cfg : RunConfig := { profile := opts.profile, seed := opts.seed }
    let cases := selectCases cfg opts.group? opts.filter? (selectedTags opts)
    runSelected cfg opts cases

end TinyGrad4.Test

def main (args : List String) : IO UInt32 :=
  TinyGrad4.Test.runMain args
