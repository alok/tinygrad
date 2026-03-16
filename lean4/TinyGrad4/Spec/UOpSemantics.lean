import Float64
import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph

namespace TinyGrad4
namespace Spec
namespace UOpSpec

/-!
# Lower-level UOp semantics

This layer mirrors the current `UOp.validateMany` rules, but returns a checked signature with a best-known device.
It is intentionally conservative: when the current Lean encoding does not carry enough information to recover an
exact device or axis fact, we return the strongest fact that is actually justified.
-/

/-- Checked signature for a `UOp`. -/
structure UOpSig where
  dtype : DType
  shape : Shape
  device? : Option Backend.DeviceType := none
  shardAxis? : Option Nat := none
  deriving Repr, BEq, DecidableEq

private def uargDevice? : UArg → Option Backend.DeviceType
  | .device dev => some (Backend.parseDeviceType dev)
  | _ => none

private def firstDevice? (srcs : List UOpSig) : Option Backend.DeviceType :=
  srcs.head?.bind (·.device?)

private def agreeingDevice? (srcs : List UOpSig) : Option Backend.DeviceType :=
  let ds := srcs.filterMap (·.device?)
  match ds with
  | [] => none
  | d :: rest =>
    if rest.all (· == d) then some d else none

private def arityOk (u : UOp) : Bool :=
  match u.op.arity with
  | .nullary => u.src.isEmpty
  | .unary => u.src.length == 1
  | .binary => u.src.length == 2
  | .ternary => u.src.length == 3
  | .variadic => true

private def sigOf (u : UOp) (device? : Option Backend.DeviceType) : UOpSig :=
  { dtype := u.dtype, shape := u.shape, device? := u.device? <|> device?, shardAxis? := u.shardAxis? }

/-- Checked signature for a `UOp`, if it matches the currently modeled lower-level rules. -/
partial def check? (u : UOp) : Option UOpSig := do
  let srcs ← u.src.mapM check?
  if !arityOk u then
    none
  else
    match u.op with
    | .CONST =>
      if u.shape == [] then
        match u.arg with
        | .constInt _ | .constFloat _ | .constF32Bits _ | .constBool _ => some (sigOf u none)
        | _ => none
      else
        none
    | .VCONST =>
      if u.src.isEmpty then
        match u.arg with
        | .constF32Array _ | .constBytesArray _ => some (sigOf u none)
        | _ => none
      else
        none
    | .BUFFER =>
      if u.src.isEmpty then
        match uargDevice? u.arg with
        | some dev => some (sigOf u (some dev))
        | none => none
      else
        none
    | .RESHAPE =>
      match srcs, u.arg with
      | [x], .shape s =>
        if u.dtype == x.dtype && s == u.shape && Shape.reshapeValid x.shape u.shape then
          some (sigOf u x.device?)
        else none
      | _, _ => none
    | .EXPAND =>
      match srcs, u.arg with
      | [x], .shape s =>
        if u.dtype == x.dtype && s == u.shape && (Shape.broadcast x.shape u.shape == some u.shape) then
          some (sigOf u x.device?)
        else none
      | _, _ => none
    | .PERMUTE =>
      match srcs, u.arg with
      | [x], .permutation perm =>
        if u.dtype == x.dtype && Shape.permuteValid x.shape perm && u.shape == Shape.permute x.shape perm then
          some (sigOf u x.device?)
        else none
      | _, _ => none
    | .PAD =>
      match srcs, u.arg with
      | [x], .padding padding =>
        if u.dtype == x.dtype && padding.length == x.shape.length && u.shape == Shape.pad x.shape padding then
          some (sigOf u x.device?)
        else none
      | _, _ => none
    | .SHRINK =>
      match srcs, u.arg with
      | [x], .bounds bounds =>
        if u.dtype == x.dtype && Shape.shrinkValid x.shape bounds && u.shape == Shape.shrink x.shape bounds then
          some (sigOf u x.device?)
        else none
      | _, _ => none
    | .FLIP =>
      match srcs, u.arg with
      | [x], .axes axes =>
        if u.dtype == x.dtype && axes.all (fun ax => ax < x.shape.length) && u.shape == x.shape then
          some (sigOf u x.device?)
        else none
      | _, _ => none
    | .CAT =>
      match u.arg with
      | .axes [axis] =>
        match srcs with
        | [] => none
        | head :: _ =>
          let dtype := head.dtype
          let shapes := srcs.map (·.shape)
          if srcs.all (fun s => s.dtype == dtype) && u.dtype == dtype &&
              Shape.concatListValid shapes axis && u.shape == Shape.concatOutList shapes axis then
            some (sigOf u (agreeingDevice? srcs))
          else none
      | _ => none
    | .REDUCE_AXIS =>
      match srcs, u.arg with
      | [x], .reduceWithAxes op axes =>
        let rank := x.shape.length
        let keepdim := u.shape.length == rank
        if axes.all (fun ax => ax < rank) && u.dtype == x.dtype &&
            u.shape == Shape.reduce x.shape axes keepdim &&
            (op == .ADD || op == .MAX) then
          some (sigOf u x.device?)
        else none
      | _, _ => none
    | .CONTRACT =>
      match srcs with
      | [a, b] =>
        if u.dtype == DType.promote a.dtype b.dtype && Shape.matmulShape a.shape b.shape == some u.shape then
          some (sigOf u (agreeingDevice? [a, b]))
        else none
      | _ => none
    | .WHERE =>
      match srcs with
      | [c, x, y] =>
        let xyShape := Shape.broadcast x.shape y.shape
        let outShape := xyShape.bind (fun s => Shape.broadcast c.shape s)
        if c.dtype == .bool && x.dtype == u.dtype && y.dtype == u.dtype && outShape == some u.shape then
          some (sigOf u (agreeingDevice? [c, x, y]))
        else none
      | _ => none
    | .DETACH | .CONTIGUOUS | .CONTIGUOUS_BACKWARD =>
      match srcs, u.arg with
      | [x], .empty =>
        if u.shape == x.shape && u.dtype == x.dtype then some (sigOf u x.device?) else none
      | _, _ => none
    | .CAST =>
      match srcs, u.arg with
      | [x], .empty =>
        if u.shape == x.shape then some (sigOf u x.device?) else none
      | _, _ => none
    | .BITCAST =>
      match srcs, u.arg with
      | [x], .empty =>
        if u.shape == x.shape && x.dtype.itemsize == u.dtype.itemsize then
          some (sigOf u x.device?)
        else none
      | _, _ => none
    | .RANGE =>
      match srcs, u.arg with
      | [end_], .rangeSpec _ _ _ =>
        if u.dtype == .index && u.shape == [] && end_.dtype == .index then some (sigOf u none) else none
      | _, _ => none
    | .SPECIAL =>
      match srcs, u.arg with
      | [end_], .specialName _ =>
        if u.dtype == .index && u.shape == [] && end_.dtype == .index then some (sigOf u none) else none
      | _, _ => none
    | .PROGRAM =>
      let okSrc :=
        match u.src with
        | [s] => s.op == .SINK
        | [s, l] => s.op == .SINK && l.op == .LINEAR
        | [s, l, src] => s.op == .SINK && l.op == .LINEAR && src.op == .SOURCE
        | _ => false
      if u.shape == [] && u.dtype == .void && okSrc then some (sigOf u none) else none
    | .LINEAR =>
      if u.shape == [] && u.dtype == .void && u.arg == .empty then some (sigOf u none) else none
    | .SOURCE =>
      match u.arg with
      | .source _ =>
        if u.shape == [] && u.dtype == .void && u.src.isEmpty then some (sigOf u none) else none
      | _ => none
    | .LOAD =>
      match srcs with
      | [buf, idx] =>
        if u.dtype == buf.dtype && idx.dtype == .index && (u.shape == idx.shape || idx.shape == []) then
          some (sigOf u buf.device?)
        else none
      | _ => none
    | .STORE =>
      match srcs with
      | [buf, idx, val] =>
        if u.dtype == .void && u.shape == buf.shape && idx.dtype == .index && val.dtype == buf.dtype &&
            (val.shape == idx.shape || idx.shape == []) then
          some (sigOf u buf.device?)
        else none
      | _ => none
    | .AFTER =>
      match srcs with
      | x :: _ =>
        if u.dtype == x.dtype && u.shape == x.shape then some (sigOf u x.device?) else none
      | [] => none
    | .SINK =>
      if u.dtype == .void && u.shape == [] then some (sigOf u none) else none
    | .COPY =>
      match srcs with
      | x :: _ =>
        if u.dtype == x.dtype && u.shape == x.shape then
          some (sigOf u (uargDevice? u.arg <|> firstDevice? srcs))
        else none
      | [] => none
    | .BUFFER_VIEW =>
      match srcs with
      | [x] =>
        if u.dtype == x.dtype then some (sigOf u x.device?) else none
      | _ => none
    | _ =>
      if u.op.isALU then
        if u.op.isUnary then
          match srcs with
          | [x] =>
            if u.shape == x.shape && u.dtype == x.dtype then some (sigOf u x.device?) else none
          | _ => none
        else if u.op.isBinary then
          match srcs with
          | [x, y] =>
            if Shape.broadcast x.shape y.shape == some u.shape then
              if u.op.producesBoolean then
                if u.dtype == .bool && x.dtype.scalar == y.dtype.scalar then
                  some (sigOf u (agreeingDevice? [x, y]))
                else none
              else if u.dtype == DType.promote x.dtype y.dtype then
                some (sigOf u (agreeingDevice? [x, y]))
              else none
            else none
          | _ => none
        else if u.op.isTernary then
          some (sigOf u (agreeingDevice? srcs))
        else
          some (sigOf u (agreeingDevice? srcs))
      else
        -- Conservative fallback for ops not yet modeled more precisely.
        some (sigOf u (uargDevice? u.arg <|> agreeingDevice? srcs <|> firstDevice? srcs))

/-- Boolean checker mirroring `check?`. -/
def valid (u : UOp) : Bool :=
  (check? u).isSome

end UOpSpec
end Spec
end TinyGrad4
