import TinyGrad4.Basic
import TinyGrad4.Shape

namespace TinyGrad4
namespace Spec

/-!
# Broadcast + stride semantics (reference)

Ported from TensorLib's broadcast/stride rules. This is a pure spec layer, not runtime.
- Shapes are `List Nat` (row-major).
- Strides are `List Int`, with 0 meaning a broadcasted axis.
-/

/-- Logical strides for a row-major contiguous buffer. -/
abbrev Strides := List Int

/-- Row-major unit strides for a shape. -/
def unitStrides (shape : Shape) : Strides :=
  Shape.unitStrides shape

/-- Broadcast a list of shapes (left-to-right), returning the output shape if possible. -/
def broadcastList (shapes : List Shape) : Option Shape :=
  let rec go (acc : Shape) (rest : List Shape) : Option Shape :=
    match rest with
    | [] => some acc
    | s :: rest =>
      match Shape.broadcast acc s with
      | none => none
      | some out => go out rest
  match shapes with
  | [] => none
  | s :: rest => go s rest

/--
Compute new strides when broadcasting `(fromShape, fromStrides)` to `toShape`.
Returns `none` when broadcasting is invalid.
-/
def broadcastStrides (fromShape : Shape) (fromStrides : Strides) (toShape : Shape) : Option Strides :=
  Shape.broadcastStrides fromShape fromStrides toShape

end Spec
end TinyGrad4
