import Std.Data.HashMap

/-!
# CostExpr (symbolic Nat cost expressions)

TinyGrad4's core cost model (`Backend.Cost`) is numeric: it emits `CostQ` tokens and interprets them to a `Nat`.

For compile-time reasoning and "nonstandard interpretation" experiments, it's useful to have a *symbolic* cost language:
we want to plug in a `CostModel` and get a closed-form expression back (potentially in terms of variables like `m`, `n`, `k`).

This module defines a tiny expression language over `Nat`:
- constants
- variables (named by `String`)
- addition and multiplication

It is intentionally minimal: the goal is to be a stable substrate for future extensions (polynomials, min/max, piecewise costs, etc.).
-/

namespace TinyGrad4.Backend

open Std

inductive CostExpr where
  | const (n : Nat)
  | var (name : String)
  | add (a b : CostExpr)
  | mul (a b : CostExpr)
  | min (a b : CostExpr)
  | max (a b : CostExpr)
  deriving Repr, Inhabited, BEq

namespace CostExpr

def zero : CostExpr := .const 0
def one : CostExpr := .const 1

instance : OfNat CostExpr n where
  ofNat := .const n

instance : HAdd CostExpr CostExpr CostExpr where
  hAdd := .add

instance : HMul CostExpr CostExpr CostExpr where
  hMul := .mul

def eval (env : HashMap String Nat) : CostExpr → Nat
  | .const n => n
  | .var name => env.getD name 0
  | .add a b => eval env a + eval env b
  | .mul a b => eval env a * eval env b
  | .min a b => Nat.min (eval env a) (eval env b)
  | .max a b => Nat.max (eval env a) (eval env b)

private def collectAdd : CostExpr → List CostExpr
  | .add a b => collectAdd a ++ collectAdd b
  | e => [e]

private def collectMul : CostExpr → List CostExpr
  | .mul a b => collectMul a ++ collectMul b
  | e => [e]

private def mkAdd (terms : List CostExpr) : CostExpr :=
  match terms with
  | [] => zero
  | [t] => t
  | t :: ts => ts.foldl (init := t) fun acc x => .add acc x

private def mkMul (factors : List CostExpr) : CostExpr :=
  match factors with
  | [] => one
  | [t] => t
  | t :: ts => ts.foldl (init := t) fun acc x => .mul acc x

def simp : CostExpr → CostExpr
  | .const n => .const n
  | .var name => .var name
  | .add a b =>
    let terms := collectAdd (simp a) ++ collectAdd (simp b)
    let (c, rest) := terms.foldl (init := (0, ([] : List CostExpr))) fun (c, rest) t =>
      match t with
      | .const n => (c + n, rest)
      | _ => (c, t :: rest)
    let rest := rest.reverse
    let rest := if c == 0 then rest else rest ++ [.const c]
    mkAdd rest
  | .mul a b =>
    let factors := collectMul (simp a) ++ collectMul (simp b)
    let (prodC, rest, sawZero) :=
      factors.foldl (init := (1, ([] : List CostExpr), false)) fun (prodC, rest, sawZero) f =>
        if sawZero then
          (prodC, rest, true)
        else
          match f with
          | .const 0 => (0, [], true)
          | .const 1 => (prodC, rest, false)
          | .const n => (prodC * n, rest, false)
          | _ => (prodC, f :: rest, false)
    if sawZero then
      zero
    else
      let rest := rest.reverse
      let rest := if prodC == 1 then rest else (.const prodC) :: rest
      mkMul rest
  | .min a b =>
    let a := simp a
    let b := simp b
    match a, b with
    | .const x, .const y => .const (Nat.min x y)
    | _, _ => if a == b then a else .min a b
  | .max a b =>
    let a := simp a
    let b := simp b
    match a, b with
    | .const x, .const y => .const (Nat.max x y)
    | _, _ => if a == b then a else .max a b

private def paren (need : Bool) (s : String) : String :=
  if need then s!"({s})" else s

partial def toStringPrec : Nat → CostExpr → String
  | _, .const n => toString n
  | _, .var name => name
  | prec, .add a b =>
    let thisPrec := 10
    let sa := toStringPrec thisPrec a
    let sb := toStringPrec thisPrec b
    paren (prec > thisPrec) s!"{sa} + {sb}"
  | prec, .mul a b =>
    let thisPrec := 20
    let sa := toStringPrec thisPrec a
    let sb := toStringPrec thisPrec b
    paren (prec > thisPrec) s!"{sa} * {sb}"
  | prec, .min a b =>
    let thisPrec := 30
    let sa := toStringPrec thisPrec a
    let sb := toStringPrec thisPrec b
    paren (prec > thisPrec) s!"min {sa} {sb}"
  | prec, .max a b =>
    let thisPrec := 30
    let sa := toStringPrec thisPrec a
    let sb := toStringPrec thisPrec b
    paren (prec > thisPrec) s!"max {sa} {sb}"

instance : ToString CostExpr where
  toString e := toStringPrec 0 e

end CostExpr

end TinyGrad4.Backend
