import Lean
import TinyGrad4.Backend.Cost

/-!
# CostExprMeta (nonstandard interpretation tooling)

This file provides a tiny compile-time reifier:

`costExpr! <nat-term>` elaborates `<nat-term> : Nat` and produces a `Backend.CostExpr` AST.

The intended use is "nonstandard interpretation" for cost models:

- write cost computations as ordinary Lean code that produces a `Nat` (possibly depending on local `Nat` variables),
- reify it to a `CostExpr` at compile time,
- later evaluate it under different environments.

This is deliberately best-effort:
- it reifies a small arithmetic fragment (`+`, `*`, `min`, `max`),
- and fails with an error if the `Nat` term cannot be reduced into that fragment.
-/

namespace TinyGrad4.Backend

open Lean Meta Elab Term

namespace CostExprMeta

private def mkConstCE (n : Nat) : Expr :=
  mkApp (mkConst ``TinyGrad4.Backend.CostExpr.const) (mkNatLit n)

private def mkVarCE (name : String) : Expr :=
  mkApp (mkConst ``TinyGrad4.Backend.CostExpr.var) (mkStrLit name)

private def mkAddCE (a b : Expr) : Expr :=
  mkApp2 (mkConst ``TinyGrad4.Backend.CostExpr.add) a b

private def mkMulCE (a b : Expr) : Expr :=
  mkApp2 (mkConst ``TinyGrad4.Backend.CostExpr.mul) a b

private def mkMinCE (a b : Expr) : Expr :=
  mkApp2 (mkConst ``TinyGrad4.Backend.CostExpr.min) a b

private def mkMaxCE (a b : Expr) : Expr :=
  mkApp2 (mkConst ``TinyGrad4.Backend.CostExpr.max) a b

private def prepNat (e : Expr) : MetaM Expr := do
  let e ← instantiateMVars e
  let e ← zetaReduce e
  pure e

private def evalCostModelField (cm : Expr) (field : Name) : MetaM Nat := do
  let cm ← prepNat cm
  let fieldExpr := mkApp (mkConst field) cm
  let fieldExpr ← whnf fieldExpr
  let n? ← (evalNat fieldExpr).run
  match n? with
  | some n => pure n
  | none => throwError "costExpr!: cost model field did not reduce to a Nat:\n{fieldExpr}"

mutual

partial def reifyNatToCostExpr (e : Expr) : MetaM Expr := do
  let e ← prepNat e
  let e := e.consumeMData
  let n? ← (evalNat e).run
  if let some n := n? then
    return mkConstCE n
  match e with
  | .fvar id =>
    let decl ← getFVarLocalDecl (mkFVar id)
    return mkVarCE decl.userName.toString
  | _ =>
    let (fn, args) := e.getAppFnArgs
    if fn == ``TinyGrad4.Backend.CostProg.time && args.size == 2 then
      let some qs ← getArrayLit? args[1]! | throwError "costExpr!: CostProg.time expects an array literal"
      let mut acc := mkConstCE 0
      for q in qs do
        acc := mkAddCE acc (← reifyCostQTime args[0]! q)
      return acc
    if fn == ``Nat.succ && args.size == 1 then
      return mkAddCE (← reifyNatToCostExpr args[0]!) (mkConstCE 1)
    else if fn == ``Nat.add && args.size == 2 then
      return mkAddCE (← reifyNatToCostExpr args[0]!) (← reifyNatToCostExpr args[1]!)
    else if fn == ``HAdd.hAdd && args.size >= 2 then
      let a := args[args.size - 2]!
      let b := args[args.size - 1]!
      return mkAddCE (← reifyNatToCostExpr a) (← reifyNatToCostExpr b)
    else if fn == ``Nat.mul && args.size == 2 then
      return mkMulCE (← reifyNatToCostExpr args[0]!) (← reifyNatToCostExpr args[1]!)
    else if fn == ``HMul.hMul && args.size >= 2 then
      let a := args[args.size - 2]!
      let b := args[args.size - 1]!
      return mkMulCE (← reifyNatToCostExpr a) (← reifyNatToCostExpr b)
    else if fn == ``Nat.min && args.size == 2 then
      return mkMinCE (← reifyNatToCostExpr args[0]!) (← reifyNatToCostExpr args[1]!)
    else if fn == ``Nat.max && args.size == 2 then
      return mkMaxCE (← reifyNatToCostExpr args[0]!) (← reifyNatToCostExpr args[1]!)
    else
      throwError "costExpr!: unsupported Nat term:\n{e}"

partial def reifyCostQTime (cm : Expr) (q : Expr) : MetaM Expr := do
  let q ← prepNat q
  let q := q.consumeMData
  let (fn, args) := q.getAppFnArgs
  if fn == ``TinyGrad4.Backend.CostQ.launch then
    let ko ← evalCostModelField cm ``TinyGrad4.Backend.CostModel.kernelOverhead
    let n := if args.size == 0 then mkNatLit 1 else args[0]!
    let n := (← reifyNatToCostExpr n)
    return mkMulCE (mkConstCE ko) n
  else if fn == ``TinyGrad4.Backend.CostQ.mem && args.size == 2 then
    let rb ← evalCostModelField cm ``TinyGrad4.Backend.CostModel.memReadByte
    let wb ← evalCostModelField cm ``TinyGrad4.Backend.CostModel.memWriteByte
    let r := (← reifyNatToCostExpr args[0]!)
    let w := (← reifyNatToCostExpr args[1]!)
    let rt := mkMulCE (mkConstCE rb) r
    let wt := mkMulCE (mkConstCE wb) w
    return mkAddCE rt wt
  else if fn == ``TinyGrad4.Backend.CostQ.memView && args.size == 2 then
    let rb ← evalCostModelField cm ``TinyGrad4.Backend.CostModel.memReadViewByte
    let wb ← evalCostModelField cm ``TinyGrad4.Backend.CostModel.memWriteViewByte
    let r := (← reifyNatToCostExpr args[0]!)
    let w := (← reifyNatToCostExpr args[1]!)
    let rt := mkMulCE (mkConstCE rb) r
    let wt := mkMulCE (mkConstCE wb) w
    return mkAddCE rt wt
  else if fn == ``TinyGrad4.Backend.CostQ.elemwise then
    let elem ← evalCostModelField cm ``TinyGrad4.Backend.CostModel.elem
    if args.size < 1 then
      throwError "costExpr!: CostQ.elemwise missing args"
    let numel := (← reifyNatToCostExpr args[0]!)
    let ops ← if args.size >= 2 then reifyNatToCostExpr args[1]! else pure (mkConstCE 1)
    return mkMulCE (mkMulCE (mkConstCE elem) numel) ops
  else if fn == ``TinyGrad4.Backend.CostQ.move && args.size == 1 then
    let moveElem ← evalCostModelField cm ``TinyGrad4.Backend.CostModel.moveElem
    let numel := (← reifyNatToCostExpr args[0]!)
    return mkMulCE (mkConstCE moveElem) numel
  else if fn == ``TinyGrad4.Backend.CostQ.reduce && args.size == 1 then
    let reduceElem ← evalCostModelField cm ``TinyGrad4.Backend.CostModel.reduceElem
    let inNumel := (← reifyNatToCostExpr args[0]!)
    return mkMulCE (mkConstCE reduceElem) inNumel
  else if fn == ``TinyGrad4.Backend.CostQ.matmul then
    if args.size < 1 then
      throwError "costExpr!: CostQ.matmul missing args"
    let mulAdds := (← reifyNatToCostExpr args[0]!)
    let viewExpr := if args.size >= 2 then args[1]! else mkConst ``Bool.false
    let view? : Option Bool := match viewExpr.consumeMData with
      | .const ``Bool.true _ => some true
      | .const ``Bool.false _ => some false
      | _ => none
    let view ← match view? with
      | some b => pure b
      | none => throwError "costExpr!: CostQ.matmul view must be a Bool literal"
    let wField := if view then ``TinyGrad4.Backend.CostModel.matmulViewMulAdd else ``TinyGrad4.Backend.CostModel.matmulMulAdd
    let w ← evalCostModelField cm wField
    return mkMulCE (mkConstCE w) mulAdds
  else
    throwError "costExpr!: unsupported CostQ term:\n{q}"

end

end CostExprMeta

/-- Reify a `Nat` expression into a `CostExpr` at elaboration time. -/
syntax:max (name := tg4CostExprBang) "costExpr! " term : term

@[term_elab tg4CostExprBang] def elabCostExprBang : TermElab := fun stx expectedType? => do
  let natTerm := stx[1]
  let natExpr ← elabTerm natTerm (some (mkConst ``Nat))
  synthesizeSyntheticMVarsNoPostponing
  let ceExpr ← CostExprMeta.reifyNatToCostExpr natExpr
  let ceExpr := mkApp (mkConst ``TinyGrad4.Backend.CostExpr.simp) ceExpr
  match expectedType? with
  | some expectedType =>
    if !(← isDefEq expectedType (mkConst ``TinyGrad4.Backend.CostExpr)) then
      throwErrorAt stx "costExpr! expected type CostExpr, got {expectedType}"
    return ceExpr
  | none =>
    return ceExpr

end TinyGrad4.Backend
