import Lean
import TinyGrad4.Backend.LeanPtxEmit

/-!
# Lean PTX Meta Helpers

Provides an elaborator-backed config literal for the Lean PTX emitter:

```
emitPtx! {
  m := 4096, n := 4096, k := 4096,
  blockM := 64, blockN := 128, blockK := 64, numWarps := 4,
  variant := TinyGrad4.Backend.LeanPtxEmit.PtxVariant.smem
}
```

Fields are optional. Missing fields are taken from env defaults via
`LeanPtxEmit.configFromEnvOverride`.
-/

namespace TinyGrad4.Backend.LeanPtxMeta

open Lean Elab Term Meta

declare_syntax_cat tg4PtxField
syntax ident " := " term : tg4PtxField
syntax (name := tg4PtxCfgBang) "ptxCfg! " "{" sepBy(tg4PtxField, ", ") "}" : term
syntax (name := tg4EmitPtxBang) "emitPtx! " "{" sepBy(tg4PtxField, ", ") "}" : term
syntax (name := tg4PtxSourceBang) "ptxSource! " "{" sepBy(tg4PtxField, ", ") "}" : term

private def fieldType (name : Name) : Option Expr :=
  match name.toString with
  | "ptxPath" => some (mkConst ``System.FilePath)
  | "kernelName" => some (mkConst ``String)
  | "m" => some (mkConst ``Nat)
  | "n" => some (mkConst ``Nat)
  | "k" => some (mkConst ``Nat)
  | "strideAm" => some (mkConst ``Nat)
  | "strideAk" => some (mkConst ``Nat)
  | "strideBk" => some (mkConst ``Nat)
  | "strideBn" => some (mkConst ``Nat)
  | "strideCm" => some (mkConst ``Nat)
  | "strideCn" => some (mkConst ``Nat)
  | "aOffset" => some (mkConst ``Nat)
  | "bOffset" => some (mkConst ``Nat)
  | "cOffset" => some (mkConst ``Nat)
  | "biasOffset" => some (mkConst ``Nat)
  | "bias2Offset" => some (mkConst ``Nat)
  | "maskMStart" => some (mkConst ``Nat)
  | "maskMEnd" => some (mkConst ``Nat)
  | "maskNStart" => some (mkConst ``Nat)
  | "maskNEnd" => some (mkConst ``Nat)
  | "maskKStart" => some (mkConst ``Nat)
  | "maskKEnd" => some (mkConst ``Nat)
  | "blockM" => some (mkConst ``Nat)
  | "blockN" => some (mkConst ``Nat)
  | "blockK" => some (mkConst ``Nat)
  | "numWarps" => some (mkConst ``Nat)
  | "ptxVersion" => some (mkConst ``Nat)
  | "sm" => some (mkConst ``Nat)
  | "withBias" => some (mkConst ``Bool)
  | "withBias2" => some (mkConst ``Bool)
  | "scaleBits" => some (mkConst ``UInt32)
  | "relu" => some (mkConst ``Bool)
  | "variant" => some (mkConst ``TinyGrad4.Backend.LeanPtxEmit.PtxVariant)
  | _ => none

private def overrideField (name : Name) : Option Name :=
  match name.toString with
  | "ptxPath" => some `ptxPath?
  | "kernelName" => some `kernelName?
  | "m" => some `m?
  | "n" => some `n?
  | "k" => some `k?
  | "strideAm" => some `strideAm?
  | "strideAk" => some `strideAk?
  | "strideBk" => some `strideBk?
  | "strideBn" => some `strideBn?
  | "strideCm" => some `strideCm?
  | "strideCn" => some `strideCn?
  | "aOffset" => some `aOffset?
  | "bOffset" => some `bOffset?
  | "cOffset" => some `cOffset?
  | "biasOffset" => some `biasOffset?
  | "bias2Offset" => some `bias2Offset?
  | "maskMStart" => some `maskMStart?
  | "maskMEnd" => some `maskMEnd?
  | "maskNStart" => some `maskNStart?
  | "maskNEnd" => some `maskNEnd?
  | "maskKStart" => some `maskKStart?
  | "maskKEnd" => some `maskKEnd?
  | "blockM" => some `blockM?
  | "blockN" => some `blockN?
  | "blockK" => some `blockK?
  | "numWarps" => some `numWarps?
  | "ptxVersion" => some `ptxVersion?
  | "sm" => some `sm?
  | "withBias" => some `withBias?
  | "withBias2" => some `withBias2?
  | "scaleBits" => some `scaleBits?
  | "relu" => some `relu?
  | "variant" => some `variant?
  | _ => none

private def mkNone (ty : Expr) : Expr :=
  mkApp (mkConst ``Option.none) ty

private def mkSome (ty val : Expr) : Expr :=
  mkApp2 (mkConst ``Option.some) ty val

private def buildOverrideExpr (fields : Array Syntax) : TermElabM Expr := do
  let mut values : List (Name × Expr) := []
  for field in fields do
    let id := field[0].getId
    let ty ←
      match fieldType id with
      | some ty => pure ty
      | none => throwErrorAt field s!"ptxCfg!: unknown field '{id}'"
    let overrideName ←
      match overrideField id with
      | some n => pure n
      | none => throwErrorAt field s!"ptxCfg!: unknown field '{id}'"
    if values.any (fun (name, _) => name == overrideName) then
      throwErrorAt field s!"ptxCfg!: duplicate field '{id}'"
    let expr ← elabTerm field[2] (some ty)
    values := (overrideName, mkSome ty expr) :: values
  let defaults : List (Name × Expr) :=
    [ (`ptxPath?, mkNone (mkConst ``System.FilePath))
    , (`kernelName?, mkNone (mkConst ``String))
    , (`m?, mkNone (mkConst ``Nat))
    , (`n?, mkNone (mkConst ``Nat))
    , (`k?, mkNone (mkConst ``Nat))
    , (`strideAm?, mkNone (mkConst ``Nat))
    , (`strideAk?, mkNone (mkConst ``Nat))
    , (`strideBk?, mkNone (mkConst ``Nat))
    , (`strideBn?, mkNone (mkConst ``Nat))
    , (`strideCm?, mkNone (mkConst ``Nat))
    , (`strideCn?, mkNone (mkConst ``Nat))
    , (`aOffset?, mkNone (mkConst ``Nat))
    , (`bOffset?, mkNone (mkConst ``Nat))
    , (`cOffset?, mkNone (mkConst ``Nat))
    , (`biasOffset?, mkNone (mkConst ``Nat))
    , (`bias2Offset?, mkNone (mkConst ``Nat))
    , (`maskMStart?, mkNone (mkConst ``Nat))
    , (`maskMEnd?, mkNone (mkConst ``Nat))
    , (`maskNStart?, mkNone (mkConst ``Nat))
    , (`maskNEnd?, mkNone (mkConst ``Nat))
    , (`maskKStart?, mkNone (mkConst ``Nat))
    , (`maskKEnd?, mkNone (mkConst ``Nat))
    , (`blockM?, mkNone (mkConst ``Nat))
    , (`blockN?, mkNone (mkConst ``Nat))
    , (`blockK?, mkNone (mkConst ``Nat))
    , (`numWarps?, mkNone (mkConst ``Nat))
    , (`ptxVersion?, mkNone (mkConst ``Nat))
    , (`sm?, mkNone (mkConst ``Nat))
    , (`withBias?, mkNone (mkConst ``Bool))
    , (`withBias2?, mkNone (mkConst ``Bool))
    , (`scaleBits?, mkNone (mkConst ``UInt32))
    , (`relu?, mkNone (mkConst ``Bool))
    , (`variant?, mkNone (mkConst ``TinyGrad4.Backend.LeanPtxEmit.PtxVariant)) ]
  let mut args : Array Expr := #[]
  for (name, defaultExpr) in defaults do
    let expr := (values.find? (fun (n, _) => n == name) |>.map Prod.snd).getD defaultExpr
    args := args.push expr
  return mkAppN (mkConst ``TinyGrad4.Backend.LeanPtxEmit.EmitOverride.mk) args

@[term_elab tg4PtxCfgBang] def elabPtxCfgBang : TermElab := fun stx expectedType? => do
  let fields := stx[1].getArgs
  let expr ← buildOverrideExpr fields
  match expectedType? with
  | some expectedType =>
    if !(← isDefEq expectedType (mkConst ``TinyGrad4.Backend.LeanPtxEmit.EmitOverride)) then
      throwErrorAt stx "ptxCfg! expected type EmitOverride"
    return expr
  | none =>
    return expr

@[term_elab tg4EmitPtxBang] def elabEmitPtxBang : TermElab := fun stx expectedType? => do
  let fields := stx[1].getArgs
  let ov ← buildOverrideExpr fields
  let expr := mkApp (mkConst ``TinyGrad4.Backend.LeanPtxEmit.emitFromOverride) ov
  match expectedType? with
  | some expectedType =>
    let expected := mkApp (mkConst ``IO) (mkConst ``UInt32)
    if !(← isDefEq expectedType expected) then
      return expr
    return expr
  | none => return expr

@[term_elab tg4PtxSourceBang] def elabPtxSourceBang : TermElab := fun stx expectedType? => do
  let fields := stx[1].getArgs
  let ov ← buildOverrideExpr fields
  let expr := mkApp (mkConst ``TinyGrad4.Backend.LeanPtxEmit.ptxSourceFromOverride) ov
  match expectedType? with
  | some expectedType =>
    let expected := mkApp (mkConst ``IO) (mkConst ``String)
    if !(← isDefEq expectedType expected) then
      return expr
    return expr
  | none => return expr

end TinyGrad4.Backend.LeanPtxMeta
