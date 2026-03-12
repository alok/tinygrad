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

/-- Scalar constant signature. -/
def constResult (dtype : DType) : TensorDesc :=
  { shape := [], dtype }

/-- Vector constant signature. -/
def vconstResult (size : Nat) (dtype : DType) : TensorDesc :=
  { shape := [size], dtype }

/-- Full/zeros/ones-style constructors produce exactly the requested shape and dtype. -/
def fullResult (shape : Shape) (dtype : DType) : TensorDesc :=
  { shape, dtype }

/-- Boolean full constructor. -/
def fullBoolResult (shape : Shape) : TensorDesc :=
  { shape, dtype := .bool }

/-- Eye constructor yields a matrix with the requested dtype. -/
def eyeResult (rows : Nat) (cols : Nat := rows) (dtype : DType := .float32) : TensorDesc :=
  { shape := [rows, cols], dtype }

/-- Arange/linspace/rand/randn/randint/randperm all produce a flat vector over their requested extent. -/
def arangeResult (size : Nat) (dtype : DType := .float32) : TensorDesc :=
  { shape := [size], dtype }

/-- Like-style constructors preserve the source tensor's signature. -/
def likeResult (desc : TensorDesc) : TensorDesc := desc

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

private def scalarDesc (dtype : DType) : TensorDesc :=
  constResult dtype

private def gatherShapeOk (shape idxShape : Shape) (dim : Nat) : Bool :=
  shape.length == idxShape.length &&
  dim < shape.length &&
  listAll (fun i => if i == dim then true else listGetD shape i 0 >= listGetD idxShape i 0) (listRange shape.length)

/-- Gather returns the index tensor's shape and preserves the payload dtype. -/
def gatherResult? (src index : TensorDesc) (dim : Nat) : Option TensorDesc :=
  if !index.dtype.isInt then
    none
  else if gatherShapeOk src.shape index.shape dim then
    some { shape := index.shape, dtype := src.dtype }
  else
    none

/-- Flattened `take` keeps the index shape and source dtype. -/
def takeResult? (src index : TensorDesc) : Option TensorDesc :=
  if index.dtype.isInt then
    some { shape := index.shape, dtype := src.dtype }
  else
    none

private def scatterShapeOk (selfShape idxShape srcShape : Shape) (dim : Nat) : Bool :=
  selfShape.length == idxShape.length &&
  selfShape.length == srcShape.length &&
  dim < selfShape.length &&
  listAll (fun i =>
    if i == dim then
      true
    else
      listGetD selfShape i 0 >= listGetD idxShape i 0 &&
      listGetD srcShape i 0 >= listGetD idxShape i 0) (listRange selfShape.length)

/-- Basic scatter preserves the destination signature when the indexing shapes line up. -/
def scatterResult? (self index src : TensorDesc) (dim : Nat) : Option TensorDesc :=
  if !index.dtype.isInt || self.dtype != src.dtype then
    none
  else if scatterShapeOk self.shape index.shape src.shape dim then
    some self
  else
    none

/-- Scalar scatter fills an index-shaped source tensor conceptually, then preserves the destination signature. -/
def scatterScalarResult? (self index : TensorDesc) (dim : Nat) : Option TensorDesc :=
  if !index.dtype.isInt then
    none
  else if scatterShapeOk self.shape index.shape index.shape dim then
    some self
  else
    none

/-- Scatter-reduce modes exposed by the pure spec layer. -/
inductive ScatterReduceMode where
  | sum
  | prod
  | mean
  | amax
  | amin
  deriving Repr, BEq, DecidableEq, Inhabited

/-- Scatter-reduce preserves the destination signature when the indexing shapes line up. -/
def scatterReduceResult? (self index src : TensorDesc) (dim : Nat)
    (_mode : ScatterReduceMode) (_includeSelf : Bool := true) : Option TensorDesc :=
  scatterResult? self index src dim

/-- Scalar scatter-reduce preserves the destination signature when the indexing shapes line up. -/
def scatterReduceScalarResult? (self index : TensorDesc) (dim : Nat)
    (_mode : ScatterReduceMode) (_includeSelf : Bool := true) : Option TensorDesc :=
  scatterScalarResult? self index dim

/-- Vector-to-matrix diagonal constructor. -/
def diagResult? (desc : TensorDesc) : Option TensorDesc :=
  match desc.shape with
  | [n] => some { shape := [n, n], dtype := desc.dtype }
  | _ => none

/-- Main diagonal view for square matrices. -/
def diagonalResult? (desc : TensorDesc) : Option TensorDesc :=
  match desc.shape with
  | [rows, cols] =>
    if rows == cols then some { shape := [rows], dtype := desc.dtype } else none
  | _ =>
    none

/-- Triangular masking preserves shape and dtype on rank-2+ tensors. -/
def triuResult? (desc : TensorDesc) : Option TensorDesc :=
  if desc.shape.length < 2 then none else some desc

/-- Triangular masking preserves shape and dtype on rank-2+ tensors. -/
def trilResult? (desc : TensorDesc) : Option TensorDesc :=
  if desc.shape.length < 2 then none else some desc

/-- Static-lane unfold currently supports only the last axis, matching the runtime implementation. -/
def unfoldResult? (desc : TensorDesc) (dim size step : Nat) : Option TensorDesc :=
  let shape := desc.shape
  if size == 0 || step == 0 || shape.isEmpty then
    none
  else if dim >= shape.length || dim != shape.length - 1 then
    none
  else
    let axisSize := listGetD shape dim 0
    if size > axisSize then
      none
    else
      some { shape := Shape.poolOut shape [size] [step] [1], dtype := desc.dtype }

/-- Packed masked-select bridge: front-packed payload plus a scalar valid-count. -/
def maskedSelectPackedResult? (src mask : TensorDesc) : Option (TensorDesc × TensorDesc) :=
  if src.shape != mask.shape || mask.dtype != .bool then
    none
  else
    some ({ shape := [Shape.numel src.shape], dtype := src.dtype }, scalarDesc .int32)

/-- Copy preserves tensor signature inside this pure shape/dtype layer. -/
def copyResult (desc : TensorDesc) : TensorDesc := desc

/-- Marker ops that are identity at the signature level. -/
def identityLikeUnaryResult? (op : Ops) (desc : TensorDesc) : Option TensorDesc :=
  match op with
  | .DETACH | .CONTIGUOUS | .CONTIGUOUS_BACKWARD => some desc
  | _ => none

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
