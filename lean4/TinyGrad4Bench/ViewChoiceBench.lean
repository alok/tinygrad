import TinyGrad4

/-!
# ViewChoiceBench

Sanity-check the "materialize-to-fast vs virtualize-as-view" choice infrastructure.

This is not a performance benchmark; it is meant to be a quick way to see how `CostModel` knobs
change Phase C fusion selection.
-/

namespace TinyGrad4Bench.ViewChoiceBench

open TinyGrad4
open StaticTensor
open Backend

private def pickTag (root : UOp) (rep : Backend.Fusion.Report) : String :=
  match rep.chosen.toList.find? (fun c => c.root == root.uid) with
  | some c => c.tag
  | none => "node"

private def buildEwise : UOp := Id.run do
  let (_, y) := runTensorM do
    let x ← Tensor.buffer [64, 64] .float32
    let xp ← StaticTensor.permute x [1, 0]
    let c ← UOp.const .float32 1.0
    let y ← UOp.add xp.uop c
    pure (x.uop, y)
  y

private def buildReduce : UOp := Id.run do
  let (_, out) := runTensorM do
    let x ← Tensor.buffer [64, 64] .float32
    let xp ← StaticTensor.permute x [1, 0]
    let xpad ← StaticTensor.pad xp [(1, 1), (0, 0)]
    let c ← UOp.const .float32 1.0
    let y ← UOp.add xpad.uop c
    let out ← UOp.max_ y [] false
    pure (x.uop, out)
  out

private def buildMixed : UOp := Id.run do
  let (_, out) := runTensorM do
    let x ← Tensor.buffer [64, 64] .float32
    let xp ← StaticTensor.permute x [1, 0]

    let b ← Tensor.buffer [1, 64] .float32
    let be ← StaticTensor.expand b [64, 64]
    let out ← UOp.add xp.uop be.uop
    pure (x.uop, out)
  out

def run : IO Unit := do
  let y := buildEwise
  let out := buildReduce
  let mix := buildMixed

  let cm0 := Backend.defaultCostModel
  let cmViewHate : Backend.CostModel :=
    { cm0 with
      memReadViewByte := cm0.memReadByte * 1000
      memWriteViewByte := cm0.memWriteByte * 1000 }

  let rep0 := Backend.Fusion.report [y] (cm := cm0)
  let rep1 := Backend.Fusion.report [y] (cm := cmViewHate)
  IO.println s!"ewise selection: default={pickTag y rep0}, viewHate={pickTag y rep1}"

  let rep2 := Backend.Fusion.report [out] (cm := cm0)
  let rep3 := Backend.Fusion.report [out] (cm := cmViewHate)
  IO.println s!"reduce selection: default={pickTag out rep2}, viewHate={pickTag out rep3}"

  let rep4 := Backend.Fusion.report [mix] (cm := cm0)
  let rep5 := Backend.Fusion.report [mix] (cm := cmViewHate)
  IO.println s!"mixed selection: default={pickTag mix rep4}, viewHate={pickTag mix rep5}"

end TinyGrad4Bench.ViewChoiceBench

#eval! TinyGrad4Bench.ViewChoiceBench.run
