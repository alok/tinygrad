/-
Copyright (c) 2024 TinyGrad4. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: TinyGrad4 Contributors
-/
import Lean

/-!
# RawBuffer Linter (FloatArray/FlatArray deprecation)

This linter warns when `FloatArray` or `FlatArray` is used instead of `RawBuffer`.

## Why?

`FloatArray` stores only `Float` (64-bit) and loses dtype information.
`FlatArray` is just an alias for `FloatArray`.

Using `RawBuffer` instead:
1. Preserves dtype information (float16, float32, float64, bfloat16, fp8, etc.)
2. Stores data as `ByteArray` which works across all numeric types
3. Enables dtype-agnostic codegen (the tinygrad way)

## Usage

Enable the linter with:
```
set_option linter.useRawBuffer true
```

Disable it locally with:
```
set_option linter.useRawBuffer false in
def myFunc (x : FloatArray) := x  -- no warning
```
-/

namespace TinyGrad4.Linter

open Lean Elab Command Meta

/-- Option to control the RawBuffer linter -/
register_option linter.useRawBuffer : Bool := {
  defValue := true
  descr := "warn when `FloatArray` or `FlatArray` is used instead of `RawBuffer`"
}

/-- Check if the linter is enabled -/
def rawBufferLinterEnabled : CommandElabM Bool := do
  return linter.useRawBuffer.get (← getOptions)

/-- Find all occurrences of `FloatArray` or `FlatArray` identifier in syntax -/
partial def findArrayIdents (stx : Syntax) : Array Syntax := Id.run do
  let mut result := #[]
  match stx with
  | .node _ _ args =>
    for arg in args do
      result := result ++ findArrayIdents arg
  | .ident _ rawVal _ _ =>
    let rawStr := rawVal.toString
    if rawStr == "FloatArray" || rawStr == "FlatArray" then
      result := result.push stx
  | .atom _ val =>
    if val == "FloatArray" || val == "FlatArray" then
      result := result.push stx
  | _ => pure ()
  return result

/-- Generate a helpful error message -/
def rawBufferWarningMessage : MessageData :=
  m!"⚠️ Use `RawBuffer` instead of `FloatArray`.\n" ++
  m!"**Why?** `FloatArray` only stores Float64 and loses dtype information. " ++
  m!"This breaks support for float16/float32/bfloat16/fp8.\n" ++
  m!"**Fix:** Use `RawBuffer` which stores `ByteArray` with a `DType` tag.\n" ++
  m!"  ❌  `def f (x : FloatArray) := x`\n" ++
  m!"  ✅  `def f (x : RawBuffer) := x`\n" ++
  m!"To disable this lint: `set_option linter.useRawBuffer false`"

/-- The RawBuffer linter run function -/
def rawBufferLinterRun (stx : Syntax) : CommandElabM Unit := do
  unless ← rawBufferLinterEnabled do return
  let arrayIdents := findArrayIdents stx
  for ident in arrayIdents do
    logWarningAt ident rawBufferWarningMessage

/-- The RawBuffer linter implementation -/
def rawBufferLinter : Linter := {
  run := rawBufferLinterRun
  name := `TinyGrad4.Linter.useRawBuffer
}

/-- Register the linter -/
initialize addLinter rawBufferLinter

end TinyGrad4.Linter
