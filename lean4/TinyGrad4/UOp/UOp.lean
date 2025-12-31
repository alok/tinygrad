import TinyGrad4.Basic
import TinyGrad4.DType
import TinyGrad4.Shape
import TinyGrad4.Ops
import TinyGrad4.Tags
import TinyGrad4.Backend.Buffer
import Std.Data.HashMap

-- ByteArray needs Repr for UArg deriving
instance : Repr ByteArray where
  reprPrec b _ := s!"ByteArray({b.size} bytes)"

namespace TinyGrad4

open Std

instance : Hashable Float where
  hash f :=
    let bits := f.toBits
    let bits := if bits == (0x8000000000000000 : UInt64) then 0 else bits
    hash bits

/-!
# UOp - Unified Operation Node

The core IR for TinyGrad4.
-/

/-- Arguments for UOp operations -/
inductive UArg where
  | empty  -- renamed from `none` to avoid collision with Option.none
  | constInt (v : Int)
  | constFloat (v : Float)
  | constF32Bits (bits : UInt32)
  | constF32Array (bits : Array UInt32)
  | constBytesArray (bytes : ByteArray)  -- dtype-agnostic raw bytes storage
  | constBool (v : Bool)
  | shape (s : Shape)
  | permutation (perm : List Nat)
  | padding (pad : List (Nat × Nat))
  | bounds (b : List (Nat × Nat))
  | axes (a : List Nat)
  | reduceOp (op : Ops)
  | reduceWithAxes (op : Ops) (axes : List Nat)
  | bufferIdx (idx : Nat)
  | device (dev : String)
  | specialName (name : String)
  | source (code : String)
  | rangeSpec (axisId : Nat) (axisType : AxisType) (args : List Nat)
  | error (msg : String)
  | varSpec (name : String) (min max : Int)
  deriving Repr, Inhabited, BEq, Hashable

namespace UArg

def getShape : UArg → Option Shape
  | .shape s => some s
  | _ => Option.none

def getPermutation : UArg → Option (List Nat)
  | .permutation p => some p
  | _ => Option.none

def getAxes : UArg → Option (List Nat)
  | .axes a => some a
  | _ => Option.none

def getRangeSpec : UArg → Option (Nat × AxisType × List Nat)
  | .rangeSpec axisId axisType args => some (axisId, axisType, args)
  | _ => Option.none

/-- Convert Int to Float -/
private def intToFloat (v : Int) : Float :=
  if v >= 0 then v.toNat.toFloat else -((-v).toNat.toFloat)

def getFloat : UArg → Option Float
  | .constFloat v => some v
  | .constInt v => some (intToFloat v)
  | .constF32Bits bits => some ((Float32.ofBits bits).toFloat)
  | .constBool v => some (if v then 1.0 else 0.0)
  | _ => Option.none

end UArg

/-- Unique identifier for UOp nodes -/
structure UOpId where
  id : Nat
  deriving DecidableEq, Repr, Hashable, Ord

instance : Inhabited UOpId where
  default := ⟨0⟩

instance : ToString UOpId where
  toString uid := s!"u{uid.id}"

/-- Core UOp structure -/
structure UOp where
  uid : UOpId
  op : Ops
  dtype : DType
  src : List UOp
  arg : UArg
  shape : Shape
  deriving Repr

instance : Inhabited UOp where
  default := { uid := ⟨0⟩, op := .CONST, dtype := .float32, src := [], arg := .empty, shape := [] }

namespace UOp

def numSrc (u : UOp) : Nat := u.src.length
def isConst (u : UOp) : Bool := u.op == .CONST || u.op == .VCONST
def isBuffer (u : UOp) : Bool := u.op == .BUFFER
def isALU (u : UOp) : Bool := u.op.isALU
def isMovement (u : UOp) : Bool := u.op.isMovement
def isReduce (u : UOp) : Bool := u.op.isReduce
def numel (u : UOp) : Nat := u.shape.numel
def rank (u : UOp) : Nat := u.shape.rank

end UOp

/-- Hash-consing key for a UOp node (ignores `uid`). -/
structure UOpKey where
  op : Ops
  dtype : DType
  src : List UOpId
  arg : UArg
  shape : Shape
  deriving BEq, Hashable, Repr

/-- State for UOp construction -/
structure UOpState where
  nextId : Nat := 0
  intern : HashMap UOpKey UOp := ∅
  internLimit : Nat := 0

abbrev UOpM := StateM UOpState

def freshUOpId : UOpM UOpId := do
  let st ← get
  set { st with nextId := st.nextId + 1 }
  pure ⟨st.nextId⟩

def runUOpM (m : UOpM α) : α := (m.run {}).1

/-- Set a max size for the hash-consing table (0 disables). When exceeded, the table is cleared. -/
def setInternLimit (limit : Nat) : UOpM Unit := do
  modify fun st => { st with internLimit := limit }

/-- Clear the hash-consing table to allow UOp nodes to be collected. -/
def clearIntern : UOpM Unit := do
  modify fun st => { st with intern := ∅ }

def runUOpMWith (st : UOpState) (m : UOpM α) : α := (m.run st).1

namespace UOp

private def keyOf (op : Ops) (dtype : DType) (src : List UOp) (arg : UArg) (shape : Shape) : UOpKey :=
  { op, dtype, src := src.map (·.uid), arg, shape }

private def shouldIntern : Ops → Bool
  | .BUFFER | .RANGE | .SPECIAL | .UNIQUE | .LUNIQUE => false
  | _ => true

private def mkUOp (op : Ops) (dtype : DType) (src : List UOp) (arg : UArg) (shape : Shape) : UOpM UOp := do
  if !shouldIntern op then
    let uid ← freshUOpId
    pure { uid, op, dtype, src, arg, shape }
  else
    let k := keyOf op dtype src arg shape
    let st ← get
    match st.intern.get? k with
    | some u => pure u
    | none =>
      let uid ← freshUOpId
      let u : UOp := { uid, op, dtype, src, arg, shape }
      let st1 ← get
      let st' := { st1 with intern := st1.intern.insert k u }
      let st'' :=
        if st'.internLimit != 0 && st'.intern.size > st'.internLimit then
          { st' with intern := ∅ }
        else
          st'
      set st''
      pure u

def const (dtype : DType) (value : Float32) : UOpM UOp := do
  match dtype with
  | .float32 =>
    mkUOp .CONST dtype [] (.constF32Bits value.toBits) []
  | _ =>
    mkUOp .CONST dtype [] (.constFloat value.toFloat) []

def constInt (dtype : DType) (value : Int) : UOpM UOp := do
  mkUOp .CONST dtype [] (.constInt value) []

def constNat (dtype : DType) (value : Nat) : UOpM UOp := do
  constInt dtype (Int.ofNat value)

def constBool (value : Bool) : UOpM UOp := do
  mkUOp .CONST .bool [] (.constBool value) []

def vconstF32 (vals : Array Float32) : UOpM UOp := do
  let bits := vals.map (fun v => v.toBits)
  mkUOp .VCONST .float32 [] (.constF32Array bits) [vals.size]

/-- Create a vector constant from raw bytes with a given dtype.
    The number of elements is computed from bytes.size / dtype.itemsize. -/
def vconst (bytes : ByteArray) (dtype : DType) (shape : Shape) : UOpM UOp := do
  mkUOp .VCONST dtype [] (.constBytesArray bytes) shape

/-- Create a vector constant from a RawBuffer.
    Shape is inferred from the buffer size and dtype. -/
def vconstRaw (buf : RawBuffer) (shape : Shape) : UOpM UOp := do
  mkUOp .VCONST buf.dtype [] (.constBytesArray buf.data) shape

def buffer (dtype : DType) (shape : Shape) (dev : String := "CPU") : UOpM UOp := do
  let u ← mkUOp .BUFFER dtype [] (.device dev) shape
  pure (device[dev] u)

def range (end_ : UOp) (axisId : Nat) (axisType : AxisType := .LOOP) (args : List Nat := []) : UOpM UOp := do
  mkUOp .RANGE .index [end_] (.rangeSpec axisId axisType args) []

def special (end_ : UOp) (name : String) : UOpM UOp := do
  mkUOp .SPECIAL .index [end_] (.specialName name) []

def program (sink : UOp) : UOpM UOp := do
  mkUOp .PROGRAM .void [sink] .empty []

def linear (uops : List UOp) : UOpM UOp := do
  mkUOp .LINEAR .void uops .empty []

def source (code : String) : UOpM UOp := do
  mkUOp .SOURCE .void [] (.source code) []

def customKernel (src : List UOp) : UOpM UOp := do
  mkUOp .CUSTOM_KERNEL .void src .empty []

def unaryOp (op : Ops) (x : UOp) : UOpM UOp := do
  mkUOp op x.dtype [x] .empty x.shape

def binaryOp (op : Ops) (x y : UOp) : UOpM UOp := do
  let shape ← match Shape.broadcast x.shape y.shape with
    | some s => pure s
    | Option.none => panic! s!"Cannot broadcast shapes"
  let dtype := if op.producesBoolean then .bool else DType.promote x.dtype y.dtype
  mkUOp op dtype [x, y] .empty shape

def reshape (x : UOp) (newShape : Shape) : UOpM UOp := do
  if !Shape.reshapeValid x.shape newShape then
    panic! s!"Invalid reshape {x.shape} -> {newShape}"
  mkUOp .RESHAPE x.dtype [x] (.shape newShape) newShape

def expand (x : UOp) (newShape : Shape) : UOpM UOp := do
  if (Shape.broadcast x.shape newShape) != some newShape then
    panic! s!"Invalid expand {x.shape} -> {newShape}"
  mkUOp .EXPAND x.dtype [x] (.shape newShape) newShape

def permute (x : UOp) (perm : List Nat) : UOpM UOp := do
  if !Shape.permuteValid x.shape perm then
    panic! s!"Invalid permute {perm} for shape {x.shape}"
  let newShape := Shape.permute x.shape perm
  mkUOp .PERMUTE x.dtype [x] (.permutation perm) newShape

def pad (x : UOp) (padding : List (Nat × Nat)) : UOpM UOp := do
  if padding.length != x.shape.length then
    panic! s!"Invalid pad: padding rank {padding.length} != shape rank {x.shape.length}"
  let newShape := Shape.pad x.shape padding
  mkUOp .PAD x.dtype [x] (.padding padding) newShape

def shrink (x : UOp) (bounds : List (Nat × Nat)) : UOpM UOp := do
  if !Shape.shrinkValid x.shape bounds then
    panic! s!"Invalid shrink {x.shape} with bounds {bounds}"
  let newShape := Shape.shrink x.shape bounds
  mkUOp .SHRINK x.dtype [x] (.bounds bounds) newShape

def flip (x : UOp) (axes : List Nat) : UOpM UOp := do
  let okAxes := axes.all (fun ax => ax < x.shape.length)
  if !okAxes then
    panic! s!"Invalid flip axes {axes} for shape {x.shape}"
  mkUOp .FLIP x.dtype [x] (.axes axes) x.shape

def reduce (x : UOp) (reduceOp : Ops) (axes : List Nat) (keepdim : Bool := true) : UOpM UOp := do
  let okAxes := axes.all (fun ax => ax < x.shape.length)
  if !okAxes then
    panic! s!"Invalid reduce axes {axes} for shape {x.shape}"
  let newShape := Shape.reduce x.shape axes keepdim
  mkUOp .REDUCE_AXIS x.dtype [x] (.reduceWithAxes reduceOp axes) newShape

/-- Tensor contraction (matmul): (..., m, k) @ (..., k, n) -> (..., m, n). -/
def contract2D (a b : UOp) : UOpM UOp := do
  let outShape ← match Shape.matmulShape a.shape b.shape with
    | some s => pure s
    | none => panic! s!"Invalid matmul shapes: {a.shape} @ {b.shape}"
  let dtype := DType.promote a.dtype b.dtype
  mkUOp .CONTRACT dtype [a, b] .empty outShape

def add (x y : UOp) : UOpM UOp := binaryOp .ADD x y
def mul (x y : UOp) : UOpM UOp := binaryOp .MUL x y
def sub (x y : UOp) : UOpM UOp := binaryOp .SUB x y
def div (x y : UOp) : UOpM UOp := binaryOp .FDIV x y
def neg (x : UOp) : UOpM UOp := unaryOp .NEG x
def trunc (x : UOp) : UOpM UOp := unaryOp .TRUNC x
def exp2 (x : UOp) : UOpM UOp := unaryOp .EXP2 x
def log2 (x : UOp) : UOpM UOp := unaryOp .LOG2 x
def sqrt (x : UOp) : UOpM UOp := unaryOp .SQRT x
def sin (x : UOp) : UOpM UOp := unaryOp .SIN x
def cos (x : UOp) : UOpM UOp := unaryOp .COS x
def tan (x : UOp) : UOpM UOp := unaryOp .TAN x
def recip (x : UOp) : UOpM UOp := unaryOp .RECIPROCAL x
def pow (x y : UOp) : UOpM UOp := binaryOp .POW x y

def sum (x : UOp) (axes : List Nat := []) (keepdim : Bool := true) : UOpM UOp :=
  let axes' := if axes.isEmpty then listRange x.rank else axes
  reduce x .ADD axes' keepdim

def max_ (x : UOp) (axes : List Nat := []) (keepdim : Bool := true) : UOpM UOp :=
  let axes' := if axes.isEmpty then listRange x.rank else axes
  reduce x .MAX axes' keepdim

-- Binary max (element-wise)
def maxBinary (x y : UOp) : UOpM UOp := binaryOp .MAX x y

-- Comparison ops
def cmplt (x y : UOp) : UOpM UOp := binaryOp .CMPLT x y
def cmpeq (x y : UOp) : UOpM UOp := binaryOp .CMPEQ x y
def cmpne (x y : UOp) : UOpM UOp := binaryOp .CMPNE x y
def bitand (x y : UOp) : UOpM UOp := binaryOp .AND x y
def bitor (x y : UOp) : UOpM UOp := binaryOp .OR x y
def bitxor (x y : UOp) : UOpM UOp := binaryOp .XOR x y

def cast (x : UOp) (dtype : DType) : UOpM UOp := do
  mkUOp .CAST dtype [x] .empty x.shape

def bitcast (x : UOp) (dtype : DType) : UOpM UOp := do
  if x.dtype.itemsize != dtype.itemsize then
    panic! s!"bitcast: dtype size mismatch {repr x.dtype} -> {repr dtype}"
  mkUOp .BITCAST dtype [x] .empty x.shape

def cat (xs : List UOp) (axis : Nat) : UOpM UOp := do
  if xs.isEmpty then
    panic! "cat: empty list"
  let dtype := xs[0]!.dtype
  if !listAll (fun u => u.dtype == dtype) xs then
    panic! s!"cat: dtype mismatch {repr (xs.map (fun u => u.dtype))}"
  let shapes := xs.map (fun u => u.shape)
  if !Shape.concatListValid shapes axis then
    panic! s!"cat: invalid shapes {shapes} on axis {axis}"
  let outShape := Shape.concatOutList shapes axis
  mkUOp .CAT dtype xs (.axes [axis]) outShape

-- Ternary where: cond ? x : y
def where_ (cond x y : UOp) : UOpM UOp := do
  if cond.dtype != .bool then
    panic! s!"where_ expects cond dtype bool, got {repr cond.dtype}"
  let shapeXY ← match Shape.broadcast x.shape y.shape with
    | some s => pure s
    | Option.none => panic! s!"Cannot broadcast shapes in where"
  let shape ← match Shape.broadcast cond.shape shapeXY with
    | some s => pure s
    | Option.none => panic! s!"Cannot broadcast shapes in where"
  mkUOp .WHERE x.dtype [cond, x, y] .empty shape

end UOp

end TinyGrad4
