import Std
import Cli

open Cli System

namespace TinyGrad4.AutoResearch

def version : String := "0.1.0"

def completeMarker : String := "<promise>COMPLETE</promise>"

structure LoopConfig where
  runner : String := "codex"
  model? : Option String := none
  reasoningEffort : String := "xhigh"
  promptFlag : String := "-p"
  iterations : Nat := 1
  sleepSecs : UInt32 := 5
  prdPath : FilePath := "ralph/PRD.md"
  progressPath : FilePath := "ralph/progress.txt"
  promptTemplate? : Option FilePath := none
  extra? : Option String := none
  resumeLast : Bool := false
  resumeId? : Option String := none
  fullAuto : Bool := false
  yolo : Bool := true
  includeLinear : Bool := true
  promptOnly : Bool := false
  runnerArgs : Array String := #[]

private def flagString? (p : Parsed) (name : String) : Option String :=
  p.flag? name |>.map (·.as! String)

private def flagNat? (p : Parsed) (name : String) : Option Nat :=
  p.flag? name |>.map (·.as! Nat)

private def flagStrings (p : Parsed) (name : String) : Array String :=
  (p.flag? name |>.map (·.as! (Array String))).getD #[]

private def configFromParsed (p : Parsed) : LoopConfig :=
  let iterations := if p.hasFlag "once" then 1 else (flagNat? p "iterations").getD 1
  let sleepSecs := UInt32.ofNat ((flagNat? p "sleep").getD 5)
  {
    runner := (flagString? p "runner").getD "codex"
    model? := flagString? p "model"
    reasoningEffort := (flagString? p "reasoning-effort").getD "xhigh"
    promptFlag := (flagString? p "prompt-flag").getD "-p"
    iterations := iterations
    sleepSecs := sleepSecs
    prdPath := FilePath.mk ((flagString? p "prd").getD "ralph/PRD.md")
    progressPath := FilePath.mk ((flagString? p "progress").getD "ralph/progress.txt")
    promptTemplate? := (flagString? p "prompt-template").map FilePath.mk
    extra? := flagString? p "extra"
    resumeLast := p.hasFlag "resume"
    resumeId? := flagString? p "resume-id"
    fullAuto := p.hasFlag "full-auto"
    yolo := !p.hasFlag "no-yolo"
    includeLinear := !p.hasFlag "no-linear"
    promptOnly := p.hasFlag "prompt-only"
    runnerArgs := flagStrings p "runner-arg"
  }

private def ensurePathExists (label : String) (path : FilePath) : IO Unit := do
  if !(← path.pathExists) then
    throw <| IO.userError s!"missing {label}: {path}"

private def validateConfig (cfg : LoopConfig) : IO Unit := do
  ensurePathExists "PRD" cfg.prdPath
  ensurePathExists "progress log" cfg.progressPath
  if let some templatePath := cfg.promptTemplate? then
    ensurePathExists "prompt template" templatePath

private def promptRef (path : FilePath) : String :=
  s!"@{path}"

private def defaultPrompt (cfg : LoopConfig) : String :=
  let prdRef := promptRef cfg.prdPath
  let progressRef := promptRef cfg.progressPath
  let sourceOfTruth :=
    if cfg.includeLinear then
      String.intercalate ""
        [ "Use Linear as the source of truth. The PRD should live in Linear as a "
        , "Project Document (for example, \"<Project> PRD\"). If it is missing, "
        , "create it from the local PRD and keep the local file synced when the "
        , "Linear doc changes.\n\n"
        ]
    else
      "Use the PRD and repo docs as the source of truth. Do not use Linear.\n\n"
  let logTail :=
    if cfg.includeLinear then
      "- Any Linear updates (issue IDs, status, cycle or initiative changes)\n"
    else
      "- Any doc updates\n"
  String.intercalate ""
    [ s!"You are Ralph Wiggum. Read {prdRef} and {progressRef}.\n\n"
    , "Work on exactly one task from the PRD. "
    , sourceOfTruth
    , "Follow repo-local instructions (AGENTS.md, CLAUDE.md, etc.) and use the "
    , "repository's preferred build and test commands.\n"
    , "You have explicit approval to create commits as you go, but only on a "
    , "branch you control. Do not commit to main, and do not amend or rewrite "
    , "history.\n\n"
    , "At the end, append to the progress log:\n"
    , "- What you changed\n"
    , "- Commands and tests you ran\n"
    , "- Next steps or blockers\n"
    , logTail
    , s!"If all tasks are done, output `{completeMarker}`.\n"
    ]

private def applyExtra (extra? : Option String) (prompt : String) : String :=
  match extra? with
  | some extra => s!"{extra}\n\n{prompt}"
  | none => prompt

private def readPrompt (cfg : LoopConfig) : IO String := do
  let prompt ←
    match cfg.promptTemplate? with
    | some templatePath =>
      let template ← IO.FS.readFile templatePath
      pure <| (template.replace "{{PRD}}" (promptRef cfg.prdPath)).replace "{{PROGRESS}}" (promptRef cfg.progressPath)
    | none =>
      pure (defaultPrompt cfg)
  pure (applyExtra cfg.extra? prompt)

private def pushArg (args : Array String) (value : String) : Array String :=
  args.push value

private def pushFlagValue (args : Array String) (flag : String) (value : String) : Array String :=
  args.push flag |>.push value

private def buildCodexArgs (cfg : LoopConfig) : Array String :=
  let args :=
    match cfg.model? with
    | some model => pushFlagValue #[] "--model" model
    | none => #[]
  let args :=
    if cfg.reasoningEffort.isEmpty then
      args
    else
      pushFlagValue args "-c" s!"model_reasoning_effort={cfg.reasoningEffort}"
  let args :=
    if cfg.yolo then
      pushArg args "--dangerously-bypass-approvals-and-sandbox"
    else if cfg.fullAuto then
      pushArg args "--full-auto"
    else
      args
  let args := pushArg args "exec"
  let args :=
    if cfg.resumeLast || cfg.resumeId?.isSome then
      let args := pushArg args "resume"
      match cfg.resumeId? with
      | some resumeId => pushArg args resumeId
      | none => pushArg args "--last"
    else
      args
  let args := args ++ cfg.runnerArgs
  pushArg args "-"

private def buildGenericArgs (cfg : LoopConfig) (prompt : String) : Array String :=
  let args :=
    match cfg.model? with
    | some model => pushFlagValue #[] "--model" model
    | none => #[]
  let runnerArgs :=
    if cfg.yolo && cfg.runner == "claude" && !cfg.runnerArgs.contains "--dangerously-skip-permissions" then
      cfg.runnerArgs.push "--dangerously-skip-permissions"
    else
      cfg.runnerArgs
  let args := args ++ runnerArgs
  let args :=
    if cfg.promptFlag.isEmpty then
      args
    else
      pushArg args cfg.promptFlag
  pushArg args prompt

private def printOutput (out : IO.Process.Output) : IO Unit := do
  if !out.stdout.isEmpty then
    IO.print out.stdout
    if !out.stdout.endsWith "\n" then
      IO.println ""
  if !out.stderr.isEmpty then
    IO.eprint out.stderr
    if !out.stderr.endsWith "\n" then
      IO.eprintln ""

private def runRunner (cfg : LoopConfig) (cwd : FilePath) (prompt : String) : IO IO.Process.Output := do
  if cfg.runner == "codex" then
    IO.Process.output { cmd := "codex", args := buildCodexArgs cfg, cwd := some cwd } (some prompt)
  else
    IO.Process.output { cmd := cfg.runner, args := buildGenericArgs cfg prompt, cwd := some cwd }

private def loopForever (cfg : LoopConfig) (cwd : FilePath) (prompt : String) : Nat → Nat → IO UInt32
  | 0, _ =>
    pure 0
  | remaining + 1, iteration => do
    IO.println s!"[tg4_autoresearch] iteration {iteration}/{cfg.iterations}"
    let out ← runRunner cfg cwd prompt
    printOutput out
    let combined := out.stdout ++ "\n" ++ out.stderr
    if out.exitCode != 0 then
      IO.eprintln s!"[tg4_autoresearch] runner exited with code {out.exitCode}"
      return out.exitCode
    if combined.contains completeMarker then
      IO.println "[tg4_autoresearch] completion detected, stopping."
      return 0
    if remaining > 0 then
      IO.println s!"[tg4_autoresearch] sleeping {cfg.sleepSecs}s"
      IO.sleep (cfg.sleepSecs * 1000)
    loopForever cfg cwd prompt remaining (iteration + 1)

def runLoop (p : Parsed) : IO UInt32 := do
  let cfg := configFromParsed p
  validateConfig cfg
  let prompt ← readPrompt cfg
  if cfg.promptOnly then
    IO.println prompt
    return 0
  let cwd ← IO.currentDir
  loopForever cfg cwd prompt cfg.iterations 1

def mainCmd : Cmd := `[Cli|
  tg4_autoresearch VIA runLoop; [version]
  "Lean-native Ralph/autoresearch loop for TinyGrad4."

  FLAGS:
    runner : String;                  "Runner command (default: codex)."
    model : String;                   "Model name passed to the runner."
    "reasoning-effort" : String;      "Reasoning effort for codex (default: xhigh)."
    "runner-arg" : Array String;      "Extra runner arg (repeatable)."
    "prompt-flag" : String;           "Prompt flag for non-codex runners (default: -p, empty to omit)."
    n, iterations : Nat;              "Number of iterations to run (default: 1)."
    sleep : Nat;                      "Sleep between iterations in seconds (default: 5)."
    prd : String;                     "PRD path (default: ralph/PRD.md)."
    progress : String;                "Progress path (default: ralph/progress.txt)."
    "prompt-template" : String;       "Template file with {{PRD}} and {{PROGRESS}} placeholders."
    extra : String;                   "Extra instructions to prepend to the prompt."
    resume;                           "Resume the most recent codex exec session."
    "resume-id" : String;             "Resume a specific codex exec session ID."
    "full-auto";                      "Use codex --full-auto when yolo is disabled."
    "no-yolo";                        "Disable codex yolo / claude skip-permissions."
    "no-linear";                      "Use a prompt that does not reference Linear."
    once;                             "Alias for --iterations 1."
    "prompt-only";                    "Print the resolved prompt and exit."
]

def mainImpl (args : List String) : IO UInt32 := do
  mainCmd.validate args

end TinyGrad4.AutoResearch

def main (args : List String) : IO UInt32 :=
  TinyGrad4.AutoResearch.mainImpl args
