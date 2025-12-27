/-
Copyright (c) 2024 TinyGrad4. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: TinyGrad4 Contributors
-/
import Lean

/-!
# Float64 Explicit Linter

This linter warns when `Float` is used instead of `Float64`.

The `Float` type in Lean 4 is actually `Float64` under the hood, but using the
shorter name can lead to confusion, especially when working with mixed-precision
numeric code (Float32, Float16, etc.).

This linter encourages explicit use of `Float64` to make precision intentions clear.

## Usage

Enable the linter with:
```
set_option linter.floatExplicit true
```

Disable it locally with:
```
set_option linter.floatExplicit false in
def myFunc (x : Float) := x  -- no warning
```
-/

namespace TinyGrad4.Linter

open Lean Elab Command Meta

/-- Option to control the Float → Float64 linter -/
register_option linter.floatExplicit : Bool := {
  defValue := true
  descr := "warn when `Float` is used instead of explicit `Float64`"
}

/-- Check if the linter is enabled -/
def floatExplicitLinterEnabled : CommandElabM Bool := do
  return linter.floatExplicit.get (← getOptions)

/-- Find all occurrences of `Float` identifier in syntax -/
partial def findFloatIdents (stx : Syntax) : Array Syntax := Id.run do
  let mut result := #[]
  match stx with
  | .node _ _ args =>
    for arg in args do
      result := result ++ findFloatIdents arg
  | .ident _ rawVal _ _ =>
    -- Check if this is the `Float` identifier (not `Float64`, `Float32`, etc.)
    -- We check the raw string to catch exactly "Float" as typed
    let rawStr := rawVal.toString
    if rawStr == "Float" then
      result := result.push stx
  | .atom _ val =>
    if val == "Float" then
      result := result.push stx
  | _ => pure ()
  return result

/-- Generate a helpful error message for AI agents -/
def floatWarningMessage : MessageData :=
  m!"⚠️ Use `Float64` instead of `Float` for clarity.\n\n" ++
  m!"**Why?** `Float` is an alias for `Float64`, but the implicit naming causes confusion " ++
  m!"in mixed-precision code (Float32/Float16/Float64).\n\n" ++
  m!"**Fix:** Replace `Float` with `Float64` to make the 64-bit precision explicit.\n\n" ++
  m!"**Example:**\n" ++
  m!"  ❌  `def f (x : Float) : Float := x * 2.0`\n" ++
  m!"  ✅  `def f (x : Float64) : Float64 := x * 2.0`\n\n" ++
  m!"To disable this lint: `set_option linter.floatExplicit false`"

/-- The Float linter run function -/
def floatExplicitLinterRun (stx : Syntax) : CommandElabM Unit := do
  unless ← floatExplicitLinterEnabled do return
  let floatIdents := findFloatIdents stx
  for ident in floatIdents do
    logWarningAt ident floatWarningMessage

/-- The Float linter implementation -/
def floatExplicitLinter : Linter := {
  run := floatExplicitLinterRun
  name := `TinyGrad4.Linter.floatExplicit
}

/-- Register the linter -/
initialize addLinter floatExplicitLinter

end TinyGrad4.Linter
