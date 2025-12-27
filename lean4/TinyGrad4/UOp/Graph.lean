import TinyGrad4.UOp.UOp
import Std.Data.HashMap
import Std.Data.HashSet

namespace TinyGrad4

open Std

abbrev UOpIdSet := HashSet UOpId
abbrev UOpIdMap := HashMap UOpId UOp
abbrev ConsumerMap := HashMap UOpId (List UOp)

namespace UOpIdSet

def mkEmpty : UOpIdSet := ∅
def member (s : UOpIdSet) (id : UOpId) : Bool := s.contains id
def add (s : UOpIdSet) (id : UOpId) : UOpIdSet := s.insert id
def map (s : UOpIdSet) (f : UOpId → UOpId) : UOpIdSet :=
  s.fold (init := (∅ : UOpIdSet)) fun acc uid => acc.insert (f uid)

end UOpIdSet

namespace UOp

/-- Topological sort of UOp graph -/
partial def toposort (root : UOp) (fuel : Nat := 100000) : List UOp :=
  let (_, result) := go root UOpIdSet.mkEmpty [] fuel
  result.reverse
where
  go (u : UOp) (visited : UOpIdSet) (acc : List UOp) (fuel : Nat) : UOpIdSet × List UOp :=
    if fuel == 0 then (visited, acc)
    else if UOpIdSet.member visited u.uid then (visited, acc)
    else
      let visited' := UOpIdSet.add visited u.uid
      let (visited'', acc') := u.src.foldl
        (fun (v, a) child => go child v a (fuel - 1))
        (visited', acc)
      (visited'', u :: acc')

def allNodes (root : UOp) : List UOp := toposort root
def nodeCount (root : UOp) : Nat := (allNodes root).length

def buildConsumerMap (root : UOp) : ConsumerMap :=
  let nodes := toposort root
  nodes.foldl (fun m u =>
    u.src.foldl (fun m' s =>
      let existing := m'.getD s.uid []
      m'.insert s.uid (u :: existing)
    ) m
  ) ∅

partial def ancestors (u : UOp) (fuel : Nat := 100000) : UOpIdSet :=
  go u UOpIdSet.mkEmpty fuel
where
  go (u : UOp) (visited : UOpIdSet) (fuel : Nat) : UOpIdSet :=
    if fuel == 0 then visited
    else if UOpIdSet.member visited u.uid then visited
    else
      let visited' := UOpIdSet.add visited u.uid
      u.src.foldl (fun v s => go s v (fuel - 1)) visited'

def dependsOn (u1 u2 : UOp) : Bool := UOpIdSet.member (ancestors u1) u2.uid

def mapNodes (root : UOp) (f : UOp → α) : List α := (toposort root).map f
def filterNodes (root : UOp) (p : UOp → Bool) : List UOp := (toposort root).filter p
def findOp (root : UOp) (op : Ops) : List UOp := filterNodes root (·.op == op)
def findBuffers (root : UOp) : List UOp := filterNodes root (·.isBuffer)
def findConsts (root : UOp) : List UOp := filterNodes root (·.isConst)
def findLeaves (root : UOp) : List UOp := filterNodes root (·.src.isEmpty)

def pretty (u : UOp) : String :=
  let srcIds := u.src.map (·.uid.id) |>.map toString |> String.intercalate ", "
  s!"{u.uid}: {repr u.op} [{srcIds}] -> {u.shape}"

/-- Collect a topological order from multiple roots (deduplicated). -/
partial def toposortMany (roots : List UOp) (fuel : Nat := 100000) : List UOp :=
  let (_, result) := roots.foldl (fun (visited, acc) r => go r visited acc fuel) (UOpIdSet.mkEmpty, [])  -- type: (UOpIdSet × List UOp)
  result.reverse
where
  go (u : UOp) (visited : UOpIdSet) (acc : List UOp) (fuel : Nat) : UOpIdSet × List UOp :=
    if fuel == 0 then (visited, acc)
    else if UOpIdSet.member visited u.uid then (visited, acc)
    else
      let visited' := UOpIdSet.add visited u.uid
      let (visited'', acc') := u.src.foldl
        (fun (v, a) child => go child v a (fuel - 1))
        (visited', acc)
      (visited'', u :: acc')

structure ValidationError where
  uid : UOpId
  op : Ops
  msg : String
  deriving Repr

namespace ValidationError

def render (e : ValidationError) : String :=
  s!"{e.uid}: {repr e.op} - {e.msg}"

end ValidationError

private def pushErr (errs : List ValidationError) (u : UOp) (msg : String) : List ValidationError :=
  { uid := u.uid, op := u.op, msg } :: errs

private def check (errs : List ValidationError) (u : UOp) (ok : Bool) (msg : String) : List ValidationError :=
  if ok then errs else pushErr errs u msg

private def checkArity (errs : List ValidationError) (u : UOp) : List ValidationError :=
  match u.op.arity with
  | .nullary => check errs u (u.src.isEmpty) s!"expected 0 src, got {u.src.length}"
  | .unary => check errs u (u.src.length == 1) s!"expected 1 src, got {u.src.length}"
  | .binary => check errs u (u.src.length == 2) s!"expected 2 src, got {u.src.length}"
  | .ternary => check errs u (u.src.length == 3) s!"expected 3 src, got {u.src.length}"
  | .variadic => errs

  private def validateNode (u : UOp) : List ValidationError := Id.run do
    let mut errs : List ValidationError := []
    errs := checkArity errs u

    match u.op with
    | .CONST | .VCONST =>
      errs := check errs u (u.src.isEmpty) "const must have no src"
    | .BUFFER =>
      errs := check errs u (u.src.isEmpty) "buffer must have no src"
    | .RESHAPE =>
      match u.src, u.arg with
      | [x], .shape s =>
        errs := check errs u (s == u.shape) "reshape arg shape must equal u.shape"
        errs := check errs u (Shape.reshapeValid x.shape u.shape) s!"invalid reshape {x.shape} -> {u.shape}"
      | _, _ =>
        errs := pushErr errs u "reshape expects 1 src and arg=.shape"
    | .EXPAND =>
      match u.src, u.arg with
      | [x], .shape s =>
        errs := check errs u (s == u.shape) "expand arg shape must equal u.shape"
        errs := check errs u ((Shape.broadcast x.shape u.shape) == some u.shape) s!"invalid expand {x.shape} -> {u.shape}"
      | _, _ =>
        errs := pushErr errs u "expand expects 1 src and arg=.shape"
    | .PERMUTE =>
      match u.src, u.arg with
      | [x], .permutation perm =>
        errs := check errs u (Shape.permuteValid x.shape perm) s!"invalid permutation {perm} for shape {x.shape}"
        errs := check errs u (u.shape == Shape.permute x.shape perm) "permute shape mismatch"
      | _, _ =>
        errs := pushErr errs u "permute expects 1 src and arg=.permutation"
    | .PAD =>
      match u.src, u.arg with
      | [x], .padding padding =>
        errs := check errs u (padding.length == x.shape.length) "pad expects padding length == rank"
        errs := check errs u (u.shape == Shape.pad x.shape padding) "pad shape mismatch"
      | _, _ =>
        errs := pushErr errs u "pad expects 1 src and arg=.padding"
    | .SHRINK =>
      match u.src, u.arg with
      | [x], .bounds bounds =>
        errs := check errs u (Shape.shrinkValid x.shape bounds) "invalid shrink bounds"
        errs := check errs u (u.shape == Shape.shrink x.shape bounds) "shrink shape mismatch"
      | _, _ =>
        errs := pushErr errs u "shrink expects 1 src and arg=.bounds"
    | .FLIP =>
      match u.src, u.arg with
      | [x], .axes axes =>
        let okAxes := axes.all (fun ax => ax < x.shape.length)
        errs := check errs u okAxes "flip axes out of bounds"
        errs := check errs u (u.shape == x.shape) "flip must preserve shape"
      | _, _ =>
        errs := pushErr errs u "flip expects 1 src and arg=.axes"
    | .CAT =>
      match u.arg with
      | .axes [axis] =>
        if u.src.isEmpty then
          errs := pushErr errs u "cat expects at least 1 src"
        else
          let shapes := u.src.map (fun s => s.shape)
          let dtype := u.src[0]!.dtype
          errs := check errs u (listAll (fun s => s.dtype == dtype) u.src) "cat dtype mismatch"
          errs := check errs u (u.dtype == dtype) "cat dtype mismatch"
          errs := check errs u (Shape.concatListValid shapes axis) "cat invalid shapes"
          errs := check errs u (u.shape == Shape.concatOutList shapes axis) "cat shape mismatch"
      | _ =>
        errs := pushErr errs u "cat expects arg=.axes [axis]"
    | .REDUCE_AXIS =>
      match u.src, u.arg with
      | [x], .reduceWithAxes op axes =>
        let rank := x.shape.length
        let okAxes := axes.all (fun ax => ax < rank)
        let keepdim := u.shape.length == rank
        errs := check errs u okAxes "reduce axes out of bounds"
        errs := check errs u (u.dtype == x.dtype) "reduce must preserve dtype"
        errs := check errs u (u.shape == Shape.reduce x.shape axes keepdim) "reduce shape mismatch"
        errs := check errs u (op == .ADD || op == .MAX) "reduce op must be ADD or MAX (for now)"
      | _, _ =>
        errs := pushErr errs u "reduce_axis expects 1 src and arg=.reduceWithAxes"
    | .CONTRACT =>
      match u.src with
      | [a, b] =>
        errs := check errs u (u.dtype == DType.promote a.dtype b.dtype) "contract dtype mismatch"
        errs := check errs u ((Shape.matmulShape a.shape b.shape) == some u.shape) "contract shape mismatch"
      | _ =>
        errs := pushErr errs u "contract expects 2 src"
    | .WHERE =>
      match u.src with
      | [c, x, y] =>
        let xyShape := Shape.broadcast x.shape y.shape
        let outShape := xyShape.bind (fun s => Shape.broadcast c.shape s)
        errs := check errs u (outShape == some u.shape) "where shape mismatch"
        errs := check errs u (c.dtype == .bool) "where cond must be bool"
        errs := check errs u (x.dtype == u.dtype) "where x dtype mismatch"
        errs := check errs u (y.dtype == u.dtype) "where y dtype mismatch"
      | _ =>
        errs := pushErr errs u "where expects 3 src"
    | .DETACH | .CONTIGUOUS | .CONTIGUOUS_BACKWARD =>
      match u.src, u.arg with
      | [x], .empty =>
        errs := check errs u (u.shape == x.shape) "detach/contiguous must preserve shape"
        errs := check errs u (u.dtype == x.dtype) "detach/contiguous must preserve dtype"
      | _, _ =>
        errs := pushErr errs u "detach/contiguous expects 1 src and arg=.empty"
    | .CAST | .BITCAST =>
      match u.src, u.arg with
      | [x], .empty =>
        errs := check errs u (u.shape == x.shape) "cast/bitcast must preserve shape"
      | _, _ =>
        errs := pushErr errs u "cast/bitcast expects 1 src and arg=.empty"
    | .RANGE =>
      match u.src, u.arg with
      | [end_], .rangeSpec _ _ _ =>
        errs := check errs u (u.dtype == .index) "range dtype must be index"
        errs := check errs u (u.shape == []) "range shape must be scalar"
        errs := check errs u (end_.dtype == .index) "range end must be index"
      | _, _ =>
        errs := pushErr errs u "range expects 1 src and arg=.rangeSpec"
    | .SPECIAL =>
      match u.src, u.arg with
      | [end_], .specialName _ =>
        errs := check errs u (u.dtype == .index) "special dtype must be index"
        errs := check errs u (u.shape == []) "special shape must be scalar"
        errs := check errs u (end_.dtype == .index) "special end must be index"
      | _, _ =>
        errs := pushErr errs u "special expects 1 src and arg=.specialName"
    | .REWRITE_ERROR =>
      match u.arg with
      | .error _ => pure ()
      | _ => errs := pushErr errs u "rewrite_error expects arg=.error"
    | .PROGRAM =>
      let okShape := u.shape == []
      let okType := u.dtype == .void
      let okSrc :=
        match u.src with
        | [s] => s.op == .SINK
        | [s, l] => s.op == .SINK && l.op == .LINEAR
        | [s, l, src] => s.op == .SINK && l.op == .LINEAR && src.op == .SOURCE
        | _ => false
      errs := check errs u okShape "program must have scalar shape"
      errs := check errs u okType "program dtype must be void"
      errs := check errs u okSrc "program expects src=[SINK] or [SINK, LINEAR] or [SINK, LINEAR, SOURCE]"
    | .LINEAR =>
      errs := check errs u (u.shape == []) "linear must have scalar shape"
      errs := check errs u (u.dtype == .void) "linear dtype must be void"
      errs := check errs u (u.arg == .empty) "linear expects arg=.empty"
    | .SOURCE =>
      errs := check errs u (u.shape == []) "source must have scalar shape"
      errs := check errs u (u.dtype == .void) "source dtype must be void"
      match u.arg with
      | .source _ => pure ()
      | _ => errs := pushErr errs u "source expects arg=.source"
      errs := check errs u (u.src.isEmpty) "source expects no src"
    | .KERNEL =>
      -- Kernel nodes are lowered boundaries; src/arg requirements depend on backend.
      pure ()
    | .CUSTOM_KERNEL =>
      -- Custom kernel boundary; shape/arg are backend-defined.
      pure ()
    | _ =>
      if u.op.isALU then
        if u.op.isUnary then
          match u.src with
          | [x] =>
            errs := check errs u (u.shape == x.shape) "unary op must preserve shape"
            errs := check errs u (u.dtype == x.dtype) "unary op must preserve dtype"
          | _ =>
            errs := pushErr errs u "unary ALU expects 1 src"
        else if u.op.isBinary then
          match u.src with
          | [x, y] =>
            errs := check errs u ((Shape.broadcast x.shape y.shape) == some u.shape) "binary op shape mismatch"
            if u.op.producesBoolean then
              errs := check errs u (u.dtype == .bool) "comparison op must have bool dtype"
              errs := check errs u (x.dtype.scalar == y.dtype.scalar) "comparison operands must share scalar dtype"
            else
              errs := check errs u (u.dtype == DType.promote x.dtype y.dtype) "binary op dtype mismatch"
          | _ =>
            errs := pushErr errs u "binary ALU expects 2 src"
        else if u.op.isTernary then
          -- WHERE and MULACC are validated explicitly above when they are used.
          pure ()
        else
          pure ()
      else
        -- Many ops are not yet modeled by the Lean runtime; accept them structurally for now.
        pure ()

    return errs.reverse

def validateMany (roots : List UOp) : Array ValidationError := Id.run do
  let nodes := toposortMany roots
  let mut out : Array ValidationError := #[]
  for u in nodes do
    for e in validateNode u do
      out := out.push e
  return out

end UOp

end TinyGrad4
