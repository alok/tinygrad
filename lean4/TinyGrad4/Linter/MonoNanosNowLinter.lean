import Lean

/-!
# IO.monoNanosNow Linter

Warns when `IO.monoNanosNow` (or `monoNanosNow`) is used directly.
Prefer the timing monad helpers (`TinyGrad4.Data.timeSpan`) or `MonadTimeNS.monoNs`
to keep timing instrumentation consistent and easier to refactor.

## Usage

Enable the linter with:
```
set_option linter.monoNanosNow true
```

Disable it locally with:
```
set_option linter.monoNanosNow false in
def f := IO.monoNanosNow  -- justified here
```

The linter auto-skips benchmark/test files based on file path patterns.
-/

namespace TinyGrad4.Linter

open Lean Elab Command Meta

/-- Option to control the IO.monoNanosNow linter -/
register_option linter.monoNanosNow : Bool := {
  defValue := true
  descr := "warn when IO.monoNanosNow is used directly (prefer timing monad helpers)"
}

/-- Check if the linter is enabled -/
def monoNanosNowLinterEnabled : CommandElabM Bool := do
  return linter.monoNanosNow.get (← getOptions)

private def isBenchOrTestFile (fileName : String) : Bool :=
  fileName.contains "/Test/" ||
  fileName.contains "/Benchmark/" ||
  fileName.contains "/Bench/" ||
  fileName.contains "TinyGrad4Bench" ||
  fileName.contains "LeanBench" ||
  fileName.endsWith "Bench.lean" ||
  fileName.endsWith "Benchmark.lean"

private def isMonoNanosNowIdent (stx : Syntax) : Bool :=
  match stx with
  | .ident _ rawVal _ _ =>
      let rawStr := rawVal.toString
      rawStr == "IO.monoNanosNow" || rawStr == "monoNanosNow"
  | .atom _ val =>
      val == "IO.monoNanosNow" || val == "monoNanosNow"
  | _ => false

/-- Find all explicit occurrences of IO.monoNanosNow. -/
partial def findMonoNanosNowIdents (stx : Syntax) : Array Syntax := Id.run do
  let mut result := #[]
  if isMonoNanosNowIdent stx then
    result := result.push stx
  match stx with
  | .node _ _ args =>
      for arg in args do
        result := result ++ findMonoNanosNowIdents arg
  | _ => pure ()
  return result

/-- Warning message for IO.monoNanosNow usage. -/
def monoNanosNowWarning : MessageData :=
  m!"⚠️ Prefer the timing monad helpers over raw `IO.monoNanosNow`.\n\n" ++
  m!"**Why?** Centralizing timing via `TinyGrad4.Data.timeSpan` / `TinyGrad4.MonadTimeNS.monoNs` " ++
  m!"keeps profiling consistent and makes instrumentation easier to change.\n\n" ++
  m!"**Fix:** Use the timing monad or wrap timing in a helper.\n\n" ++
  m!"  ❌  `let t0 ← IO.monoNanosNow`\n" ++
  m!"  ✅  `TinyGrad4.Data.timeSpan \"stage\" action`\n\n" ++
  m!"To disable this lint: `set_option linter.monoNanosNow false` (add a justification comment)."

/-- The IO.monoNanosNow linter run function. -/
def monoNanosNowLinterRun (stx : Syntax) : CommandElabM Unit := do
  unless ← monoNanosNowLinterEnabled do return
  if isBenchOrTestFile (← getFileName) then
    return
  let idents := findMonoNanosNowIdents stx
  for ident in idents do
    logWarningAt ident monoNanosNowWarning

/-- The IO.monoNanosNow linter implementation. -/
def monoNanosNowLinter : Linter := {
  run := monoNanosNowLinterRun
  name := `TinyGrad4.Linter.monoNanosNow
}

/-- Register the linter. -/
initialize addLinter monoNanosNowLinter

end TinyGrad4.Linter
