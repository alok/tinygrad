import Float64
import TinyGrad4.DType
import TinyGrad4.Ops
import TinyGrad4.Shape
import TinyGrad4.Spec.Indexing

namespace TinyGrad4
namespace Spec

/-!
# Executable tensor spec

`tinyspec` is a good high-level map of tinygrad's tensor semantics. This module keeps the Lean port in an executable,
typed form: shape and dtype rules are values/functions we can test, not just prose tables.
-/

/-- A tensor signature tracked by the pure spec layer. -/
structure TensorDesc where
  shape : Shape
  dtype : DType
  deriving Repr, BEq, DecidableEq

/-- Tensor movement operators with pure shape semantics. -/
inductive MovementOp where
  | reshape (newShape : Shape)
  | expand (newShape : Shape)
  | permute (perm : List Nat)
  | pad (padding : List (Nat × Nat))
  | shrink (bounds : List (Nat × Nat))
  | flip (axes : List Nat)
  deriving Repr, BEq, Inhabited

namespace MovementOp

/-- Output shape of a movement op, if the move is valid. -/
def outShape? (op : MovementOp) (shape : Shape) : Option Shape :=
  match op with
  | .reshape newShape =>
    if Shape.reshapeValid shape newShape then some newShape else none
  | .expand newShape =>
    if Shape.expandValid shape newShape then some newShape else none
  | .permute perm =>
    if Shape.permuteValid shape perm then some (Shape.permute shape perm) else none
  | .pad padding =>
    if Shape.padValid shape padding then some (Shape.pad shape padding) else none
  | .shrink bounds =>
    if Shape.shrinkValid shape bounds then some (Shape.shrink shape bounds) else none
  | .flip axes =>
    if Shape.flipValid shape axes then some shape else none

/-- Apply a movement op to a tensor signature. -/
def apply? (desc : TensorDesc) (op : MovementOp) : Option TensorDesc :=
  (outShape? op desc.shape).map fun shape => { desc with shape }

end MovementOp

/-- Apply basic indexing to a tensor signature. -/
def basicIndex? (desc : TensorDesc) (items : List BasicIndexItem) : Option TensorDesc :=
  (inferBasicIndexShape desc.shape items).map fun shape => { desc with shape }

/-- Reduce spec: reduce op + axes + keepdim flag. -/
structure ReduceSpec where
  op : Ops
  axes : List Nat := []
  keepdim : Bool := true
  deriving Repr, BEq, DecidableEq

namespace ReduceSpec

/-- Reduction ops supported by the pure tensor spec. -/
def supportsOp : Ops → Bool
  | .ADD | .MAX | .MUL | .NOOP => true
  | _ => false

/-- Empty axes mean "reduce over all dimensions", matching tensor APIs. -/
def normalizedAxes (spec : ReduceSpec) (shape : Shape) : List Nat :=
  if spec.axes.isEmpty then listRange shape.length else spec.axes

/-- Output shape of a reduction, if the op + axes are valid. -/
def outShape? (spec : ReduceSpec) (shape : Shape) : Option Shape :=
  let axes := spec.normalizedAxes shape
  if supportsOp spec.op && listAll (fun axis => axis < shape.length) axes then
    some (Shape.reduce shape axes spec.keepdim)
  else
    none

/-- Apply a reduction to a tensor signature. -/
def apply? (desc : TensorDesc) (spec : ReduceSpec) : Option TensorDesc :=
  (outShape? spec desc.shape).map fun shape => { desc with shape }

end ReduceSpec

private def binaryOutDType? (op : Ops) (lhs rhs : DType) : Option DType :=
  match op with
  | .ADD | .MUL | .SUB | .FDIV | .IDIV | .MAX | .MOD | .POW | .SHL | .SHR | .XOR | .OR | .AND =>
    some (DType.promote lhs rhs)
  | .CMPLT | .CMPNE | .CMPEQ =>
    some .bool
  | _ =>
    none

private def broadcast3 (s1 s2 s3 : Shape) : Option Shape := do
  let s12 ← Shape.broadcast s1 s2
  Shape.broadcast s12 s3

/-- Result type of a unary tensor op. `CAST` and `BITCAST` require a target dtype. -/
def unaryResult? (op : Ops) (desc : TensorDesc) (target? : Option DType := none) : Option TensorDesc :=
  match op with
  | .CAST =>
    target?.map fun dtype => { desc with dtype }
  | .BITCAST =>
    match target? with
    | some dtype =>
      if desc.dtype.itemsize == dtype.itemsize then some { desc with dtype } else none
    | none => none
  | .EXP2 | .LOG2 | .SIN | .COS | .TAN | .SQRT | .RECIPROCAL | .NEG | .TRUNC =>
    some desc
  | _ =>
    none

/-- Result type of a binary tensor op with broadcasting + promotion. -/
def binaryResult? (op : Ops) (lhs rhs : TensorDesc) : Option TensorDesc := do
  let shape ← Shape.broadcast lhs.shape rhs.shape
  let dtype ← binaryOutDType? op lhs.dtype rhs.dtype
  pure { shape, dtype }

/-- Result type of a ternary tensor op. -/
def ternaryResult? (op : Ops) (a b c : TensorDesc) : Option TensorDesc :=
  match op with
  | .WHERE =>
    if a.dtype != .bool || b.dtype != c.dtype then
      none
    else
      match broadcast3 a.shape b.shape c.shape with
      | some shape => some { shape, dtype := b.dtype }
      | none => none
  | .MULACC =>
    match broadcast3 a.shape b.shape c.shape with
    | some shape =>
      let dtype := DType.promote (DType.promote a.dtype b.dtype) c.dtype
      some { shape, dtype }
    | none =>
      none
  | _ =>
    none

/-- Concatenation shape + dtype rule from the spec layer. -/
def catResult? (inputs : List TensorDesc) (axis : Nat) : Option TensorDesc :=
  match inputs with
  | [] => none
  | head :: tail =>
    let sameDType := listAll (fun input => input.dtype == head.dtype) tail
    let shapes := inputs.map (·.shape)
    if sameDType && Shape.concatListValid shapes axis then
      some { shape := Shape.concatOutList shapes axis, dtype := head.dtype }
    else
      none

/-- Matmul / contraction shape + dtype rule. -/
def contract2DResult? (lhs rhs : TensorDesc) : Option TensorDesc :=
  (Shape.matmulShape lhs.shape rhs.shape).map fun shape =>
    { shape, dtype := DType.promote lhs.dtype rhs.dtype }

end Spec
end TinyGrad4
