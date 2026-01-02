import TinyGrad4.Basic
import TinyGrad4.DType
import TinyGrad4.Shape
import TinyGrad4.Ops
import TinyGrad4.UOp.UOp

namespace TinyGrad4

/-!
Typed UOp wrapper.

This is a shallow, proof-light layer: key metadata is pushed into the type,
but we defer most proofs with `sorry_proof` for now.
-/

/-- Typed UOp node with op, shape, rank, and dtype in the type. -/
structure TUOp (op : Ops) (shape : Shape) (rank : Nat) (dtype : DType) where
  raw : UOp
  h_op : raw.op = op
  h_shape : raw.shape = shape
  h_rank : Shape.rank shape = rank
  h_dtype : raw.dtype = dtype
  deriving Repr

abbrev TUOpM := UOpM

namespace TUOp

abbrev rankOf (s : Shape) : Nat := Shape.rank s

def mk {op : Ops} {s : Shape} {r : Nat} {d : DType} (u : UOp) : TUOp op s r d :=
  { raw := u, h_op := sorry_proof, h_shape := sorry_proof, h_rank := sorry_proof, h_dtype := sorry_proof }

def ofRaw (u : UOp) : TUOp u.op u.shape (rankOf u.shape) u.dtype :=
  { raw := u, h_op := rfl, h_shape := rfl, h_rank := rfl, h_dtype := rfl }

def castShape {op : Ops} {s : Shape} {r : Nat} {d : DType} (t : TUOp op s r d) (s' : Shape) :
    TUOp op s' (rankOf s') d :=
  { raw := t.raw, h_op := sorry_proof, h_shape := sorry_proof, h_rank := sorry_proof, h_dtype := sorry_proof }

def castDType {op : Ops} {s : Shape} {r : Nat} {d : DType} (t : TUOp op s r d) (d' : DType) :
    TUOp op s r d' :=
  { raw := t.raw, h_op := sorry_proof, h_shape := sorry_proof, h_rank := sorry_proof, h_dtype := sorry_proof }

def broadcastShape (s1 s2 : Shape) : Shape :=
  match Shape.broadcast s1 s2 with
  | some s => s
  | none => panic! s!"Cannot broadcast shapes {s1} and {s2}"

def matmulShape (s1 s2 : Shape) : Shape :=
  match Shape.matmulShape s1 s2 with
  | some s => s
  | none => panic! s!"Invalid matmul shapes: {s1} @ {s2}"

def const (dtype : DType) (value : Float32) : TUOpM (TUOp .CONST [] (rankOf []) dtype) := do
  let raw ← UOp.const dtype value
  pure (mk raw)

def constInt (dtype : DType) (value : Int) : TUOpM (TUOp .CONST [] (rankOf []) dtype) := do
  let raw ← UOp.constInt dtype value
  pure (mk raw)

def constBool (value : Bool) : TUOpM (TUOp .CONST [] (rankOf []) .bool) := do
  let raw ← UOp.constBool value
  pure (mk raw)

def buffer (dtype : DType) (shape : Shape) (dev : String := "CPU") :
    TUOpM (TUOp .BUFFER shape (rankOf shape) dtype) := do
  let raw ← UOp.buffer dtype shape dev
  pure (mk raw)

def unaryOp (op : Ops) (x : TUOp _ s r d) : TUOpM (TUOp op s r d) := do
  let raw ← UOp.unaryOp op x.raw
  pure (mk raw)

def binaryOp (op : Ops) (x : TUOp _ sx _ dx) (y : TUOp _ sy _ dy) :
    TUOpM
      (TUOp op (broadcastShape sx sy) (rankOf (broadcastShape sx sy))
        (if op.producesBoolean then .bool else DType.promote dx dy)) := do
  let raw ← UOp.binaryOp op x.raw y.raw
  pure (mk raw)

def cast (x : TUOp _ s r d) (dtype : DType) : TUOpM (TUOp .CAST s r dtype) := do
  let raw ← UOp.cast x.raw dtype
  pure (mk raw)

def bitcast (x : TUOp _ s r d) (dtype : DType) : TUOpM (TUOp .BITCAST s r dtype) := do
  let raw ← UOp.bitcast x.raw dtype
  pure (mk raw)

def reshape (x : TUOp _ s _ d) (newShape : Shape) :
    TUOpM (TUOp .RESHAPE newShape (rankOf newShape) d) := do
  let raw ← UOp.reshape x.raw newShape
  pure (mk raw)

def expand (x : TUOp _ s _ d) (newShape : Shape) :
    TUOpM (TUOp .EXPAND newShape (rankOf newShape) d) := do
  let raw ← UOp.expand x.raw newShape
  pure (mk raw)

def permute (x : TUOp _ s _ d) (perm : List Nat) :
    TUOpM (TUOp .PERMUTE (Shape.permute s perm) (rankOf (Shape.permute s perm)) d) := do
  let raw ← UOp.permute x.raw perm
  pure (mk raw)

def pad (x : TUOp _ s _ d) (padding : List (Nat × Nat)) :
    TUOpM (TUOp .PAD (Shape.pad s padding) (rankOf (Shape.pad s padding)) d) := do
  let raw ← UOp.pad x.raw padding
  pure (mk raw)

def shrink (x : TUOp _ s _ d) (bounds : List (Nat × Nat)) :
    TUOpM (TUOp .SHRINK (Shape.shrink s bounds) (rankOf (Shape.shrink s bounds)) d) := do
  let raw ← UOp.shrink x.raw bounds
  pure (mk raw)

def flip (x : TUOp _ s r d) (axes : List Nat) : TUOpM (TUOp .FLIP s r d) := do
  let raw ← UOp.flip x.raw axes
  pure (mk raw)

def reduce (x : TUOp _ s _ d) (reduceOp : Ops) (axes : List Nat) (keepdim : Bool := true) :
    TUOpM (TUOp .REDUCE_AXIS (Shape.reduce s axes keepdim) (rankOf (Shape.reduce s axes keepdim)) d) := do
  let raw ← UOp.reduce x.raw reduceOp axes keepdim
  pure (mk raw)

def contract2D (a : TUOp _ sa _ da) (b : TUOp _ sb _ db) :
    TUOpM
      (TUOp .CONTRACT (matmulShape sa sb) (rankOf (matmulShape sa sb)) (DType.promote da db)) := do
  let raw ← UOp.contract2D a.raw b.raw
  pure (mk raw)

def where_ (cond : TUOp _ sc _ .bool) (x : TUOp _ sx _ dx) (y : TUOp _ sy _ dx) :
    TUOpM
      (TUOp .WHERE (broadcastShape sc (broadcastShape sx sy))
        (rankOf (broadcastShape sc (broadcastShape sx sy))) dx) := do
  let raw ← UOp.where_ cond.raw x.raw y.raw
  pure (mk raw)

def add (x : TUOp _ sx _ dx) (y : TUOp _ sy _ dy) :=
  binaryOp .ADD x y

def mul (x : TUOp _ sx _ dx) (y : TUOp _ sy _ dy) :=
  binaryOp .MUL x y

def sub (x : TUOp _ sx _ dx) (y : TUOp _ sy _ dy) :=
  binaryOp .SUB x y

def div (x : TUOp _ sx _ dx) (y : TUOp _ sy _ dy) :=
  binaryOp .FDIV x y

def neg (x : TUOp _ s r d) :=
  unaryOp .NEG x

def cmplt (x : TUOp _ sx _ dx) (y : TUOp _ sy _ dy) :=
  binaryOp .CMPLT x y

def cmpeq (x : TUOp _ sx _ dx) (y : TUOp _ sy _ dy) :=
  binaryOp .CMPEQ x y

def cmpne (x : TUOp _ sx _ dx) (y : TUOp _ sy _ dy) :=
  binaryOp .CMPNE x y

def sum (x : TUOp _ s _ d) (axes : List Nat := []) (keepdim : Bool := true) :=
  let axes' := if axes.isEmpty then listRange s.length else axes
  reduce x .ADD axes' keepdim

def max_ (x : TUOp _ s _ d) (axes : List Nat := []) (keepdim : Bool := true) :=
  let axes' := if axes.isEmpty then listRange s.length else axes
  reduce x .MAX axes' keepdim

end TUOp

end TinyGrad4
