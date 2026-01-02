import Lean

/-!
# ByteArray Capacity Linter

Warns when `ByteArray.empty` is used without a capacity hint.
Prefer `ByteArray.emptyWithCapacity` for predictable allocation behavior.

## Usage

Enable the linter with:
```
set_option linter.byteArrayCapacity true
```

Disable it locally with:
```
set_option linter.byteArrayCapacity false in
def f := ByteArray.empty
```
-/

namespace TinyGrad4.Linter

open Lean Elab Command Meta

/-- Option to control the ByteArray capacity linter -/
register_option linter.byteArrayCapacity : Bool := {
  defValue := true
  descr := "warn when `ByteArray.empty` is used without a capacity hint"
}

/-- Check if the linter is enabled -/
def byteArrayCapacityLinterEnabled : CommandElabM Bool := do
  return linter.byteArrayCapacity.get (← getOptions)

private def isByteArrayEmptyIdent (stx : Syntax) : Bool :=
  match stx with
  | .ident _ rawVal _ _ => rawVal.toString == "ByteArray.empty"
  | .atom _ val => val == "ByteArray.empty"
  | _ => false

/-- Find all explicit occurrences of `ByteArray.empty`. -/
partial def findByteArrayEmptyIdents (stx : Syntax) : Array Syntax := Id.run do
  let mut result := #[]
  if isByteArrayEmptyIdent stx then
    result := result.push stx
  match stx with
  | .node _ _ args =>
    for arg in args do
      result := result ++ findByteArrayEmptyIdents arg
  | _ => pure ()
  return result

/-- Warning message for ByteArray.empty usage. -/
def byteArrayCapacityWarning : MessageData :=
  m!"⚠️ Prefer `ByteArray.emptyWithCapacity` to avoid repeated reallocation.\n\n" ++
  m!"**Why?** `ByteArray.empty` starts with capacity 0; repeated `append` grows the buffer " ++
  m!"and can copy data multiple times.\n\n" ++
  m!"**Fix:** Use `ByteArray.emptyWithCapacity` when you can estimate size.\n\n" ++
  m!"  ❌  `let mut out := ByteArray.empty`\n" ++
  m!"  ✅  `let mut out := ByteArray.emptyWithCapacity n`\n\n" ++
  m!"To disable this lint: `set_option linter.byteArrayCapacity false`"

/-- The ByteArray capacity linter run function. -/
def byteArrayCapacityLinterRun (stx : Syntax) : CommandElabM Unit := do
  unless ← byteArrayCapacityLinterEnabled do return
  let idents := findByteArrayEmptyIdents stx
  for ident in idents do
    logWarningAt ident byteArrayCapacityWarning

/-- The ByteArray capacity linter implementation. -/
def byteArrayCapacityLinter : Linter := {
  run := byteArrayCapacityLinterRun
  name := `TinyGrad4.Linter.byteArrayCapacity
}

/-- Register the linter. -/
initialize addLinter byteArrayCapacityLinter

end TinyGrad4.Linter
