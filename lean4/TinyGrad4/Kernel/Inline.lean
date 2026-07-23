import Lean
import TinyGrad4.Kernel.Codegen

/-!
# `kernel!` — inline codegen elaborator

The pure-Lean answer to tinygrad's "cut the abstraction layers" philosophy:
a tensor kernel is written as an ordinary Lean lambda, and a *custom elaborator*
compiles it — at elaboration time — into

1. its spec (`Kernel.Expr`, the typed language from `Kernel/Spec.lean`),
2. Metal + C source, embedded as string literals in the binary,
3. a native Lean implementation (`fn`),
4. a machine-checked proof that spec and implementation agree (`denote_eq`),
   discharged automatically by `rfl`/`simp`/`grind`.

```lean
def saxpy := kernel! "saxpy" fun a x y => a * x + y
-- saxpy : InlineKernel 3
-- saxpy.metal : String  -- full MSL kernel, generated at compile time
-- saxpy.denote_eq       -- proof, generated at compile time
```

Supported fragment (elementwise f32): `+ - * /`, unary `-`, `sqrt`, `recip`,
`exp2`, `log2`, `sin`, `max`, `if a < b then _ else _`, float/nat literals.
Unsupported constructs give a compile-time error at the offending subterm.
-/

namespace TinyGrad4.Kernel.Inline

open Lean Elab Term Meta

/-- Reified kernel expression: the elaborator's working AST.
    Literals carry both the meta-level `Float32` value (for source generation)
    and the original syntax (so the embedded spec and `fn` reuse the *identical*
    term, keeping `denote_eq` provable by `rfl`). -/
inductive RExpr where
  | var (i : Nat)
  | lit (v : Float32) (stx : Term)
  | neg (a : RExpr) | sqrt (a : RExpr) | recip (a : RExpr)
  | exp2 (a : RExpr) | log2 (a : RExpr) | sin (a : RExpr)
  | add (a b : RExpr) | sub (a b : RExpr) | mul (a b : RExpr)
  | div (a b : RExpr) | max (a b : RExpr)
  | ite (a b x y : RExpr)  -- if a < b then x else y

/-- Project to the spec language (used at elaboration time for codegen). -/
def RExpr.toKernel : RExpr → Expr .f32
  | .var i => .input .f32 i
  | .lit v _ => .constF32 v
  | .neg a => .neg a.toKernel
  | .sqrt a => .sqrt a.toKernel
  | .recip a => .reciprocal a.toKernel
  | .exp2 a => .exp2 a.toKernel
  | .log2 a => .log2 a.toKernel
  | .sin a => .sin a.toKernel
  | .add a b => .add a.toKernel b.toKernel
  | .sub a b => .sub a.toKernel b.toKernel
  | .mul a b => .mul a.toKernel b.toKernel
  | .div a b => .div a.toKernel b.toKernel
  | .max a b => .max a.toKernel b.toKernel
  | .ite a b x y => .where_ (.cmplt a.toKernel b.toKernel) x.toKernel y.toKernel

/-- Render the AST as a `Kernel.Expr` term (the embedded spec). -/
partial def RExpr.toExprTerm : RExpr → TermElabM Term
  | .var i => `(Expr.input .f32 $(quote i))
  | .lit _ stx => `(Expr.constF32 $stx)
  | .neg a => do `(Expr.neg $(← a.toExprTerm))
  | .sqrt a => do `(Expr.sqrt $(← a.toExprTerm))
  | .recip a => do `(Expr.reciprocal $(← a.toExprTerm))
  | .exp2 a => do `(Expr.exp2 $(← a.toExprTerm))
  | .log2 a => do `(Expr.log2 $(← a.toExprTerm))
  | .sin a => do `(Expr.sin $(← a.toExprTerm))
  | .add a b => do `(Expr.add $(← a.toExprTerm) $(← b.toExprTerm))
  | .sub a b => do `(Expr.sub $(← a.toExprTerm) $(← b.toExprTerm))
  | .mul a b => do `(Expr.mul $(← a.toExprTerm) $(← b.toExprTerm))
  | .div a b => do `(Expr.div $(← a.toExprTerm) $(← b.toExprTerm))
  | .max a b => do `(Expr.max $(← a.toExprTerm) $(← b.toExprTerm))
  | .ite a b x y => do
    `(Expr.where_ (Expr.cmplt $(← a.toExprTerm) $(← b.toExprTerm)) $(← x.toExprTerm) $(← y.toExprTerm))

/-- Render the AST as native Float32 code over an environment `env : Fin n → Float32`.
    Shapes are chosen to be definitionally equal to what `denote` unfolds to. -/
partial def RExpr.toFnTerm (env : Ident) : RExpr → TermElabM Term
  | .var i => `($env ⟨$(quote i), by decide⟩)
  | .lit _ stx => pure stx
  | .neg a => do let a' ← a.toFnTerm env; `(-$a')
  | .sqrt a => do let a' ← a.toFnTerm env; `(Float32.sqrt $a')
  | .recip a => do let a' ← a.toFnTerm env; `((1.0 : Float32) / $a')
  | .exp2 a => do let a' ← a.toFnTerm env; `(Float32.exp2 $a')
  | .log2 a => do let a' ← a.toFnTerm env; `(Float32.log2 $a')
  | .sin a => do let a' ← a.toFnTerm env; `(Float32.sin $a')
  | .add a b => do let a' ← a.toFnTerm env; let b' ← b.toFnTerm env; `($a' + $b')
  | .sub a b => do let a' ← a.toFnTerm env; let b' ← b.toFnTerm env; `($a' - $b')
  | .mul a b => do let a' ← a.toFnTerm env; let b' ← b.toFnTerm env; `($a' * $b')
  | .div a b => do let a' ← a.toFnTerm env; let b' ← b.toFnTerm env; `($a' / $b')
  | .max a b => do let a' ← a.toFnTerm env; let b' ← b.toFnTerm env; `(Max.max $a' $b')
  | .ite a b x y => do
    let a' ← a.toFnTerm env; let b' ← b.toFnTerm env
    let x' ← x.toFnTerm env; let y' ← y.toFnTerm env
    `(if $a' < $b' then $x' else $y')

/-- Value of a float literal as elaboration would compute it. -/
private def litValue? (stx : Term) : Option Float32 :=
  match stx.raw.isScientificLit? with
  | some (m, sign, e) => some (Float32.ofScientific m sign e)
  | none => stx.raw.isNatLit?.map fun n => (OfNat.ofNat n : Float32)

/-- Reify a term-syntax kernel body into `RExpr`. -/
partial def reify (vars : Std.HashMap Name Nat) (stx : Term) : TermElabM RExpr := do
  match stx with
  | `(($e)) => reify vars e
  | `($a + $b) => return .add (← reify vars a) (← reify vars b)
  | `($a - $b) => return .sub (← reify vars a) (← reify vars b)
  | `($a * $b) => return .mul (← reify vars a) (← reify vars b)
  | `($a / $b) => return .div (← reify vars a) (← reify vars b)
  | `(-$a) => return .neg (← reify vars a)
  | `(max $a $b) => return .max (← reify vars a) (← reify vars b)
  | `(sqrt $a) => return .sqrt (← reify vars a)
  | `(Float32.sqrt $a) => return .sqrt (← reify vars a)
  | `(recip $a) => return .recip (← reify vars a)
  | `(exp2 $a) => return .exp2 (← reify vars a)
  | `(Float32.exp2 $a) => return .exp2 (← reify vars a)
  | `(log2 $a) => return .log2 (← reify vars a)
  | `(Float32.log2 $a) => return .log2 (← reify vars a)
  | `(sin $a) => return .sin (← reify vars a)
  | `(Float32.sin $a) => return .sin (← reify vars a)
  | `(if $a < $b then $x else $y) =>
    return .ite (← reify vars a) (← reify vars b) (← reify vars x) (← reify vars y)
  | `($x:ident) =>
    match vars.get? x.getId with
    | some i => return .var i
    | none => throwErrorAt stx "kernel!: unknown variable `{x.getId}` (not a kernel binder)"
  | _ =>
    match litValue? stx with
    | some v => return .lit v stx
    | none => throwErrorAt stx
        "kernel!: unsupported construct; supported: + - * / max sqrt recip exp2 log2 sin, \
         `if a < b then _ else _`, literals, and binder variables"

/-- Extract binder names from `fun` binders (plain idents or `(x y : T)` groups). -/
private def binderNames (b : Syntax) : TermElabM (Array Name) := do
  if b.isIdent then
    return #[b.getId]
  else if b.getKind == ``Lean.Parser.Term.explicitBinder then
    -- (x y : T) — idents live in the second child
    return b[1].getArgs.filterMap fun s => if s.isIdent then some s.getId else none
  else
    throwErrorAt b "kernel!: use plain binders, e.g. `fun x y => ...` or `fun (x y : Float32) => ...`"

syntax (name := kernelBang) "kernel!" (str)? term : term

/-- The inline-codegen elaborator: reify, generate device source, embed the
    proof-carrying `InlineKernel`. -/
elab_rules : term
  | `(kernel! $[$nm?:str]? fun $binders* => $body) => do
    let mut vars : Std.HashMap Name Nat := {}
    for b in binders do
      for x in ← binderNames b do
        vars := vars.insert x vars.size
    let n := vars.size
    if n == 0 then throwError "kernel!: at least one binder required"
    let r ← reify vars body
    let name := (nm?.map (·.getString)).getD "tg4_inline"
    let k := r.toKernel
    let metal := metalSource name n k
    let cSrc := cSource name n k
    let env := mkIdent `env
    let stx ← `((({ name := $(Syntax.mkStrLit name)
                    expr := $(← r.toExprTerm)
                    metal := $(Syntax.mkStrLit metal)
                    cSrc := $(Syntax.mkStrLit cSrc)
                    fn := fun $env => $(← r.toFnTerm env)
                    denote_eq := by
                      intro env
                      first
                        | rfl
                        | simp [denote, evalExpr, readEnv, f32Ops]
                        | grind [denote, evalExpr, readEnv, f32Ops] } :
                  InlineKernel $(quote n))))
    elabTerm stx none

end TinyGrad4.Kernel.Inline
