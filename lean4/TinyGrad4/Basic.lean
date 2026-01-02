-- Import linters early so they run on all files that import Basic
import TinyGrad4.Linter.FloatLinter
import TinyGrad4.Linter.RawBufferLinter
import TinyGrad4.Linter.ByteArrayCapacityLinter

namespace TinyGrad4

/-!
# Basic utilities and axioms for TinyGrad4

## sorry_proof axiom
Following the pattern from Grassmann4 and SciLean, we use `sorry_proof` to defer
proof obligations for Float operations. This allows:
- Computation to work via `#eval`
- Type-checking to pass
- Proofs to be filled in later (or left as axioms for practical use)

The Float type doesn't satisfy ring axioms exactly (due to floating point semantics),
but for practical deep learning this is acceptable.
-/

/-- Axiom for deferring proof obligations. Use sparingly and document why. -/
axiom sorryProofAxiom {P : Prop} : P

/-- Macro for convenient sorry_proof usage -/
macro "sorry_proof" : term => `(sorryProofAxiom)

/-- Product of a list of natural numbers -/
def listProd : List Nat → Nat
  | [] => 1
  | n :: ns => n * listProd ns

/-- Sum of a list of natural numbers -/
def listSum : List Nat → Nat
  | [] => 0
  | n :: ns => n + listSum ns

/-- Zip two lists with a function -/
def listZipWith (f : α → β → γ) : List α → List β → List γ
  | [], _ => []
  | _, [] => []
  | a :: as, b :: bs => f a b :: listZipWith f as bs

/-- Zip three lists with a function -/
def listZipWith3 (f : α → β → γ → δ) : List α → List β → List γ → List δ
  | [], _, _ => []
  | _, [], _ => []
  | _, _, [] => []
  | a :: as, b :: bs, c :: cs => f a b c :: listZipWith3 f as bs cs

/-- Zip four lists with a function -/
def listZipWith4 (f : α → β → γ → δ → ε) : List α → List β → List γ → List δ → List ε
  | [], _, _, _ => []
  | _, [], _, _ => []
  | _, _, [], _ => []
  | _, _, _, [] => []
  | a :: as, b :: bs, c :: cs, d :: ds => f a b c d :: listZipWith4 f as bs cs ds

/-- Zip five lists with a function -/
def listZipWith5 (f : α → β → γ → δ → ε → ζ) : List α → List β → List γ → List δ → List ε → List ζ
  | [], _, _, _, _ => []
  | _, [], _, _, _ => []
  | _, _, [], _, _ => []
  | _, _, _, [], _ => []
  | _, _, _, _, [] => []
  | a :: as, b :: bs, c :: cs, d :: ds, e :: es => f a b c d e :: listZipWith5 f as bs cs ds es

/-- Check if all elements satisfy a predicate -/
def listAll (p : α → Bool) : List α → Bool
  | [] => true
  | a :: as => p a && listAll p as

/-- Check if any element satisfies a predicate -/
def listAny (p : α → Bool) : List α → Bool
  | [] => false
  | a :: as => p a || listAny p as

/-- Range from 0 to n-1 -/
def listRange : Nat → List Nat
  | 0 => []
  | n + 1 => listRange n ++ [n]

/-- Enumerate list with indices -/
def listEnum (xs : List α) : List (Nat × α) :=
  (listRange xs.length).zip xs

/-- Safe list indexing with default -/
def listGetD (xs : List α) (i : Nat) (default : α) : α :=
  match xs, i with
  | [], _ => default
  | x :: _, 0 => x
  | _ :: xs, n + 1 => listGetD xs n default

/-- Find index of element in list, returns list length if not found -/
def listIndexOf [DecidableEq α] (xs : List α) (a : α) : Nat :=
  match xs with
  | [] => 0
  | x :: xs' => if x == a then 0 else 1 + listIndexOf xs' a

/-- Unique identifier type -/
structure Id where
  val : Nat
  deriving DecidableEq, Repr, Hashable, Ord

instance : ToString Id where
  toString id := s!"#{id.val}"

/-- Counter monad for generating unique IDs -/
abbrev IdGenM := StateM Nat

/-- Generate a fresh ID -/
def freshId : IdGenM Id := do
  let n ← get
  set (n + 1)
  pure ⟨n⟩

/-- Run ID generation starting from 0 -/
def runIdGen (m : IdGenM α) : α :=
  (m.run 0).1

end TinyGrad4
