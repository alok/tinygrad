import Lean.Data.Json.Elab
import Lean.Data.Json.Printer

/-!
JSON helpers for W&B. Prefer using {lit}`json%` for literals.
-/

namespace Wandb.Json

abbrev J := Lean.Json
abbrev JNumber := Lean.JsonNumber

/-- Build a JSON object from key-value pairs. -/
def obj (fields : List (String × J)) : J :=
  Lean.Json.mkObj fields

/-- Build a JSON array from list elements. -/
def arr (elems : List J) : J :=
  Lean.Json.arr elems.toArray

/-- JSON string literal. -/
def str (s : String) : J :=
  Lean.Json.str s

/-- JSON boolean literal. -/
def bool (b : Bool) : J :=
  Lean.Json.bool b

/-- JSON number from a natural number. -/
def nat (n : Nat) : J :=
  (n : Lean.Json)

/-- JSON number from an integer. -/
def int (n : Int) : J :=
  (n : Lean.Json)

/-- JSON number from a float, falling back to string for {lit}`NaN`/infinite. -/
def float (x : Float) : J :=
  match Lean.JsonNumber.fromFloat? x with
  | Sum.inr n => Lean.Json.num n
  | Sum.inl s => Lean.Json.str s

/-- Render JSON to a compact string. -/
def render (j : J) : String :=
  ToString.toString j

end Wandb.Json
