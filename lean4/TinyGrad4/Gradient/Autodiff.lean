import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import TinyGrad4.Tensor.Tensor
import TinyGrad4.Gradient.Rules
import Std.Data.HashMap

namespace TinyGrad4

open Std
open Gradient

abbrev GradMap := HashMap UOpId UOp

namespace Autodiff

def computeGradient (root : UOp) (targets : List UOp) : UOpM GradMap := do
  let topo := UOp.toposort root
  let initGrad ← UOp.const root.dtype 1.0
  let mut gradMap : GradMap := (∅ : GradMap).insert root.uid initGrad

  for u in topo.reverse do
    match gradMap[u.uid]? with
    | none => continue
    | some grad =>
      let gradResult ← gradOp u grad
      for (src, maybeGrad) in u.src.zip gradResult.srcGrads do
        match maybeGrad with
        | none => continue
        | some srcGrad =>
          match gradMap[src.uid]? with
          | none => gradMap := gradMap.insert src.uid srcGrad
          | some existingGrad =>
            let accumulated ← UOp.add existingGrad srcGrad
            gradMap := gradMap.insert src.uid accumulated

  let mut result : GradMap := ∅
  for target in targets do
    match gradMap[target.uid]? with
    | some grad => result := result.insert target.uid grad
    | none => pure ()
  return result

def getGrad (gradMap : GradMap) (u : UOp) : Option UOp := gradMap[u.uid]?

end Autodiff

namespace StaticTensor

def backward {d : DType} (loss : Scalar d) (params : List UOp) : UOpM GradMap :=
  Autodiff.computeGradient loss.uop params

def grad {s : List Nat} {d : DType} (loss : Scalar d) (param : StaticTensor s d)
    : UOpM (Option (StaticTensor s d)) := do
  let gradMap ← Autodiff.computeGradient loss.uop [param.uop]
  match gradMap[param.uop.uid]? with
  | some gradUop => pure (some { uop := gradUop, h_shape := sorry_proof })
  | none => pure none

end StaticTensor

end TinyGrad4
