/-
Copyright (c) 2024 TinyGrad4. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: TinyGrad4 Contributors
-/
import Lean

/-!
# RawBuffer Linter (FloatArray/FlatArray/Array Float deprecation)

This linter warns when `FloatArray`, `FlatArray`, or `Array Float` is used instead of `RawBuffer`.

## Why?

`FloatArray` stores only `Float` (64-bit) and loses dtype information.
`FlatArray` is just an alias for `FloatArray`.
`Array Float` has boxing overhead (each Float is heap-allocated) and also loses dtype info.

Using `RawBuffer` instead:
1. Preserves dtype information (float16, float32, float64, bfloat16, fp8, etc.)
2. Stores data as `ByteArray` which works across all numeric types
3. Enables dtype-agnostic codegen (the tinygrad way)
4. Avoids per-element boxing overhead

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
  descr := "warn when `FloatArray`, `FlatArray`, or `Array Float` is used instead of `RawBuffer`"
}

/-- Check if the linter is enabled -/
def rawBufferLinterEnabled : CommandElabM Bool := do
  return linter.useRawBuffer.get (← getOptions)

/-- Check if a syntax node represents `Array Float` type application -/
def isArrayFloat (stx : Syntax) : Bool :=
  match stx with
  | .node _ kind args =>
    -- Match type application: `Array Float`
    if kind == ``Lean.Parser.Term.app && args.size >= 2 then
      let fn := args[0]!
      let arg := args[1]!
      let fnIsArray := match fn with
        | .ident _ rawVal _ _ => rawVal.toString == "Array"
        | _ => false
      let argIsFloat := match arg with
        | .ident _ rawVal _ _ => rawVal.toString == "Float"
        | .node _ _ innerArgs =>
          -- Handle parenthesized or other wrapped cases
          innerArgs.any fun a => match a with
            | .ident _ rawVal _ _ => rawVal.toString == "Float"
            | _ => false
        | _ => false
      fnIsArray && argIsFloat
    else false
  | _ => false

/-- Find all occurrences of deprecated float array types in syntax -/
partial def findArrayIdents (stx : Syntax) : Array Syntax := Id.run do
  let mut result := #[]
  -- Check if current node is `Array Float`
  if isArrayFloat stx then
    result := result.push stx
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

/-- Generate a helpful error message for FloatArray/FlatArray -/
def rawBufferErrorMessage : MessageData :=
  m!"❌ Use `RawBuffer` instead of `FloatArray`.\n" ++
  m!"**Why?** `FloatArray` only stores Float64 and loses dtype information. " ++
  m!"This breaks support for float16/float32/bfloat16/fp8.\n" ++
  m!"**Fix:** Use `RawBuffer` which stores `ByteArray` with a `DType` tag.\n" ++
  m!"  ❌  `def f (x : FloatArray) := x`\n" ++
  m!"  ✅  `def f (x : RawBuffer) := x`\n" ++
  m!"To disable this lint: `set_option linter.useRawBuffer false`"

/-- Generate a helpful error message for Array Float -/
def arrayFloatErrorMessage : MessageData :=
  m!"❌ Use `RawBuffer` instead of `Array Float`.\n" ++
  m!"**Why?** `Array Float` has boxing overhead (each Float is heap-allocated) " ++
  m!"and loses dtype information. This breaks support for float16/float32/bfloat16/fp8.\n" ++
  m!"**Fix:** Use `RawBuffer` which stores `ByteArray` with a `DType` tag.\n" ++
  m!"  ❌  `def f (x : Array Float) := x`\n" ++
  m!"  ✅  `def f (x : RawBuffer) := x`\n" ++
  m!"For literals in tests, use `#[1.0, 2.0]` with local `packF32` helper.\n" ++
  m!"To disable this lint: `set_option linter.useRawBuffer false`"

/-- The RawBuffer linter run function (throws error, not warning) -/
def rawBufferLinterRun (stx : Syntax) : CommandElabM Unit := do
  unless ← rawBufferLinterEnabled do return
  let arrayIdents := findArrayIdents stx
  for ident in arrayIdents do
    let msg := if isArrayFloat ident then arrayFloatErrorMessage else rawBufferErrorMessage
    throwErrorAt ident msg

/-- The RawBuffer linter implementation -/
def rawBufferLinter : Linter := {
  run := rawBufferLinterRun
  name := `TinyGrad4.Linter.useRawBuffer
}

/-- Register the linter -/
initialize addLinter rawBufferLinter

end TinyGrad4.Linter
