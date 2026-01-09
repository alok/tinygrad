import TinyGrad4.Backend.CostExpr

/-!
# Cost queries (trusted)

We want cost models that are:
- explicit about *what* is being counted (kernel launches, elementwise ops, moves, matmuls, ...)
- reusable across different backends (CPU-ish vs GPU-ish weights)
- hard to "game" by sprinkling ad-hoc `tick`s inside algorithms

This file defines a tiny "query language" for costs: `CostQ`.
We can emit `CostProg := Array CostQ` from compiler passes and then interpret it with a `CostModel`.

This is a stepping stone toward a freer/free-monad approach (à la CSLib's query models),
but we keep it minimal for now.
-/

namespace TinyGrad4.Backend

/-- Coarse cost model weights (all trusted). -/
structure CostModel where
  /-- Fixed overhead per kernel call. Tune higher for GPU-like models. -/
  kernelOverhead : Nat := 1000
  /-- Cost per element for generic elementwise ops. -/
  elem : Nat := 1
  /-- Cost per element for materialized movement ops (permute/pad/shrink/...). -/
  moveElem : Nat := 2
  /-- Cost per byte read from memory. -/
  memReadByte : Nat := 1
  /-- Cost per byte written to memory. -/
  memWriteByte : Nat := 1
  /-- Cost per byte read via a strided/masked view (models uncoalesced reads / extra address math). -/
  memReadViewByte : Nat := 2
  /-- Cost per byte written via a strided/masked view (rare today; outputs are typically contiguous). -/
  memWriteViewByte : Nat := 1
  /-- Cost per element touched by reductions. -/
  reduceElem : Nat := 1
  /-- Cost per multiply-accumulate in a "fast" matmul kernel. -/
  matmulMulAdd : Nat := 1
  /-- Cost per multiply-accumulate in a view/strided matmul kernel. -/
  matmulViewMulAdd : Nat := 2
  deriving Repr, BEq, Hashable

def defaultCostModel : CostModel := {}

/-- Linear feature vector for CostProg (used for tuning/learning). -/
structure CostFeatures where
  launches : Nat := 0
  memRead : Nat := 0
  memWrite : Nat := 0
  memViewRead : Nat := 0
  memViewWrite : Nat := 0
  elemOps : Nat := 0
  moveElems : Nat := 0
  reduceElems : Nat := 0
  matmulMulAdds : Nat := 0
  matmulViewMulAdds : Nat := 0
  deriving Repr

namespace CostFeatures

def add (a b : CostFeatures) : CostFeatures :=
  { launches := a.launches + b.launches
    memRead := a.memRead + b.memRead
    memWrite := a.memWrite + b.memWrite
    memViewRead := a.memViewRead + b.memViewRead
    memViewWrite := a.memViewWrite + b.memViewWrite
    elemOps := a.elemOps + b.elemOps
    moveElems := a.moveElems + b.moveElems
    reduceElems := a.reduceElems + b.reduceElems
    matmulMulAdds := a.matmulMulAdds + b.matmulMulAdds
    matmulViewMulAdds := a.matmulViewMulAdds + b.matmulViewMulAdds }

def time (cm : CostModel) (f : CostFeatures) : Nat :=
  cm.kernelOverhead * f.launches +
  cm.memReadByte * f.memRead +
  cm.memWriteByte * f.memWrite +
  cm.memReadViewByte * f.memViewRead +
  cm.memWriteViewByte * f.memViewWrite +
  cm.elem * f.elemOps +
  cm.moveElem * f.moveElems +
  cm.reduceElem * f.reduceElems +
  cm.matmulMulAdd * f.matmulMulAdds +
  cm.matmulViewMulAdd * f.matmulViewMulAdds

end CostFeatures

/-- A single cost "query" token. -/
inductive CostQ where
  | launch (n : Nat := 1)
  | mem (readBytes : Nat) (writeBytes : Nat)
  | memView (readBytes : Nat) (writeBytes : Nat)
  | elemwise (numel : Nat) (ops : Nat := 1)
  | move (numel : Nat)
  | reduce (inNumel : Nat)
  | matmul (mulAdds : Nat) (view : Bool := false)
  deriving Repr, Inhabited

abbrev CostProg := Array CostQ

namespace CostQ

def time (cm : CostModel) : CostQ → Nat
  | .launch n => cm.kernelOverhead * n
  | .mem r w => cm.memReadByte * r + cm.memWriteByte * w
  | .memView r w => cm.memReadViewByte * r + cm.memWriteViewByte * w
  | .elemwise numel ops => cm.elem * numel * ops
  | .move numel => cm.moveElem * numel
  | .reduce inNumel => cm.reduceElem * inNumel
  | .matmul mulAdds view =>
    (if view then cm.matmulViewMulAdd else cm.matmulMulAdd) * mulAdds

end CostQ

namespace CostProg

def time (cm : CostModel) (p : CostProg) : Nat :=
  p.foldl (init := 0) fun acc q => acc + q.time cm

def launches (p : CostProg) : Nat :=
  p.foldl (init := 0) fun acc q =>
    match q with
    | .launch n => acc + n
    | _ => acc

def bytesRead (p : CostProg) : Nat :=
  p.foldl (init := 0) fun acc q =>
    match q with
    | .mem r _ | .memView r _ => acc + r
    | _ => acc

def bytesWrite (p : CostProg) : Nat :=
  p.foldl (init := 0) fun acc q =>
    match q with
    | .mem _ w | .memView _ w => acc + w
    | _ => acc

def features (p : CostProg) : CostFeatures :=
  p.foldl (init := ({} : CostFeatures)) fun acc q =>
    match q with
    | .launch n =>
      CostFeatures.add acc { launches := n }
    | .mem r w =>
      CostFeatures.add acc { memRead := r, memWrite := w }
    | .memView r w =>
      CostFeatures.add acc { memViewRead := r, memViewWrite := w }
    | .elemwise numel ops =>
      CostFeatures.add acc { elemOps := numel * ops }
    | .move numel =>
      CostFeatures.add acc { moveElems := numel }
    | .reduce inNumel =>
      CostFeatures.add acc { reduceElems := inNumel }
    | .matmul mulAdds view =>
      if view then
        CostFeatures.add acc { matmulViewMulAdds := mulAdds }
      else
        CostFeatures.add acc { matmulMulAdds := mulAdds }

end CostProg

/-- A symbolic variant of `CostQ`, where payloads are `CostExpr` instead of `Nat`. -/
inductive CostQExpr where
  | launch (n : CostExpr := 1)
  | mem (readBytes : CostExpr) (writeBytes : CostExpr)
  | memView (readBytes : CostExpr) (writeBytes : CostExpr)
  | elemwise (numel : CostExpr) (ops : CostExpr := 1)
  | move (numel : CostExpr)
  | reduce (inNumel : CostExpr)
  | matmul (mulAdds : CostExpr) (view : Bool := false)
  deriving Repr, Inhabited

abbrev CostProgExpr := Array CostQExpr

namespace CostQExpr

private def c (n : Nat) : CostExpr := .const n

def time (cm : CostModel) : CostQExpr → CostExpr
  | .launch n => (c cm.kernelOverhead * n).simp
  | .mem r w => (c cm.memReadByte * r + c cm.memWriteByte * w).simp
  | .memView r w => (c cm.memReadViewByte * r + c cm.memWriteViewByte * w).simp
  | .elemwise numel ops => (c cm.elem * numel * ops).simp
  | .move numel => (c cm.moveElem * numel).simp
  | .reduce inNumel => (c cm.reduceElem * inNumel).simp
  | .matmul mulAdds view =>
    let w := if view then cm.matmulViewMulAdd else cm.matmulMulAdd
    (c w * mulAdds).simp

end CostQExpr

namespace CostProgExpr

def time (cm : CostModel) (p : CostProgExpr) : CostExpr :=
  p.foldl (init := (0 : CostExpr)) fun acc q => (acc + q.time cm).simp

def bytesRead (p : CostProgExpr) : CostExpr :=
  p.foldl (init := (0 : CostExpr)) fun acc q =>
    match q with
    | .mem r _ | .memView r _ => (acc + r).simp
    | _ => acc

def bytesWrite (p : CostProgExpr) : CostExpr :=
  p.foldl (init := (0 : CostExpr)) fun acc q =>
    match q with
    | .mem _ w | .memView _ w => (acc + w).simp
    | _ => acc

end CostProgExpr

namespace CostQ

def toExpr : CostQ → CostQExpr
  | .launch n => .launch (.const n)
  | .mem r w => .mem (.const r) (.const w)
  | .memView r w => .memView (.const r) (.const w)
  | .elemwise numel ops => .elemwise (.const numel) (.const ops)
  | .move numel => .move (.const numel)
  | .reduce inNumel => .reduce (.const inNumel)
  | .matmul mulAdds view => .matmul (.const mulAdds) view

end CostQ

namespace CostProg

def toExpr (p : CostProg) : CostProgExpr :=
  p.map (fun q => q.toExpr)

end CostProg

end TinyGrad4.Backend
