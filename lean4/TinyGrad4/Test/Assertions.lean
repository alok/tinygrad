import Float64
import TinyGrad4
import LSpec
import Plausible
import TinyGrad4.Test.Profiles

namespace TinyGrad4.Test.Assertions

open TinyGrad4
open TinyGrad4.Interpreter
open TinyGrad4.Test
open scoped Plausible.Decorations

private def formatExpected (vals : Array Float64) : String :=
  s!"{vals.toList}"

def assertShape (got expected : Shape) (label : String) : IO Unit := do
  if got != expected then
    throw <| IO.userError s!"{label}: shape {got} != {expected}"

def assertEqNat (got expected : Nat) (label : String) : IO Unit := do
  if got != expected then
    throw <| IO.userError s!"{label}: {got} != {expected}"

def assertClose (got expected : Float64) (tol : Float64) (label : String) : IO Unit := do
  let diff := Float64.abs (got - expected)
  if diff > tol then
    throw <| IO.userError s!"{label}: got={got} expected={expected} diff={diff} > {tol}"

def assertRawAllClose (raw : RawBuffer) (expected : Array Float64) (tol : Float64) (label : String) : IO Unit := do
  if raw.numF32 != expected.size then
    throw <| IO.userError s!"{label}: size {raw.numF32} != {expected.size}"
  for i in [:raw.numF32] do
    let got := raw.getF32 i
    let exp := expected[i]!
    let diff := Float64.abs (got - exp)
    if diff > tol then
      throw <| IO.userError s!"{label}: idx {i} got={got} expected={exp} diff={diff} > {tol}; expected={formatExpected expected}"

def ioTest (name : String) (action : IO Unit) : LSpec.TestSeq :=
  .individualIO name none (do
    try
      action
      pure (true, 1, 1, none)
    catch e =>
      pure (false, 0, 1, some e.toString)
  ) .done

def assertPlausible (cfg : RunConfig) (label : String) (p : Prop)
    (p' : Plausible.Decorations.DecorationsOf p := by mk_decorations)
    [Plausible.Testable p'] : IO Unit := do
  match ← Plausible.Testable.checkIO p' (plausibleConfig cfg) with
  | .success _ => pure ()
  | .gaveUp n =>
    throw <| IO.userError s!"{label}: Plausible gave up after {n} discarded attempts"
  | .failure _ vars shrinks =>
    let renderedVars := String.intercalate ", " vars
    throw <| IO.userError s!"{label}: counter-example after {shrinks} shrink(s): {renderedVars}"

end TinyGrad4.Test.Assertions
