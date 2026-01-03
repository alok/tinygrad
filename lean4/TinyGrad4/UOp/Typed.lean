import TinyGrad4.Basic
import TinyGrad4.DType
import TinyGrad4.Shape
import TinyGrad4.Ops
import TinyGrad4.UOp.UOp

namespace TinyGrad4

/-!
Typed UOp wrapper.

Design notes:
- `TUOp` stores op/shape/rank/dtype in the type, so consumers can carry static metadata through
  graph construction without extra runtime checks.
- Proofs are intentionally shallow (`sorry_proof`); the goal is type-directed structure first,
  and deeper proofs later. This keeps the typed path usable while we migrate call sites.
- `mkUnsafe`/`ofRaw`/`castShape`/`castDType` are the escape hatches: they let us bridge from raw UOp,
  align with actual runtime shapes, and defer verification to later phases.
- Typed helpers (e.g., `cat`, `catList`) take shape witnesses so well-formedness can live at the
  type level even before we fully prove it.
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

def mkUnsafe {op : Ops} {s : Shape} {r : Nat} {d : DType} (u : UOp) : TUOp op s r d :=
  { raw := u, h_op := sorry_proof, h_shape := sorry_proof, h_rank := sorry_proof, h_dtype := sorry_proof }

def ofRaw (u : UOp) : TUOp u.op u.shape (rankOf u.shape) u.dtype :=
  { raw := u, h_op := rfl, h_shape := rfl, h_rank := rfl, h_dtype := rfl }

def castShape {op : Ops} {s : Shape} {r : Nat} {d : DType} (t : TUOp op s r d) (s' : Shape) :
    TUOp op s' (rankOf s') d :=
  { raw := t.raw, h_op := sorry_proof, h_shape := sorry_proof, h_rank := sorry_proof, h_dtype := sorry_proof }

inductive TUOpList (d : DType) : List Shape -> Type where
  | nil : TUOpList d []
  | cons {op : Ops} {s : Shape} {r : Nat} {ss : List Shape} :
      TUOp op s r d -> TUOpList d ss -> TUOpList d (s :: ss)

def TUOpList.toUOps {d : DType} {shapes : List Shape} (xs : TUOpList d shapes) : List UOp :=
  match xs with
  | .nil => []
  | .cons x xs => x.raw :: xs.toUOps

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
  pure (mkUnsafe raw)

def constInt (dtype : DType) (value : Int) : TUOpM (TUOp .CONST [] (rankOf []) dtype) := do
  let raw ← UOp.constInt dtype value
  pure (mkUnsafe raw)

def constNat (dtype : DType) (value : Nat) : TUOpM (TUOp .CONST [] (rankOf []) dtype) := do
  let raw ← UOp.constNat dtype value
  pure (mkUnsafe raw)

def constBool (value : Bool) : TUOpM (TUOp .CONST [] (rankOf []) .bool) := do
  let raw ← UOp.constBool value
  pure (mkUnsafe raw)

def vconstF32 (vals : Array Float32) : TUOpM (TUOp .VCONST [vals.size] (rankOf [vals.size]) .float32) := do
  let raw ← UOp.vconstF32 vals
  pure (mkUnsafe raw)

def vconst (bytes : ByteArray) (dtype : DType) (shape : Shape) :
    TUOpM (TUOp .VCONST shape (rankOf shape) dtype) := do
  let raw ← UOp.vconst bytes dtype shape
  pure (mkUnsafe raw)

def vconstRaw (buf : RawBuffer) (shape : Shape) :
    TUOpM (TUOp .VCONST shape (rankOf shape) buf.dtype) := do
  let raw ← UOp.vconstRaw buf shape
  pure (mkUnsafe raw)

def buffer (dtype : DType) (shape : Shape) (dev : String := "CPU") :
    TUOpM (TUOp .BUFFER shape (rankOf shape) dtype) := do
  let raw ← UOp.buffer dtype shape dev
  pure (mkUnsafe raw)

def unaryOp {opx : Ops} {s : Shape} {r : Nat} {d : DType} (op : Ops) (x : TUOp opx s r d) :
    TUOpM (TUOp op s r d) := do
  let raw ← UOp.unaryOp op x.raw
  pure (mkUnsafe raw)

def binaryOp {opx opy : Ops} {sx sy : Shape} {rx ry : Nat} {dx dy : DType}
    (op : Ops) (x : TUOp opx sx rx dx) (y : TUOp opy sy ry dy) :
    TUOpM
      (TUOp op (broadcastShape sx sy) (rankOf (broadcastShape sx sy))
        (if op.producesBoolean then .bool else DType.promote dx dy)) := do
  let raw ← UOp.binaryOp op x.raw y.raw
  pure (mkUnsafe raw)

def binaryOpB {opx opy : Ops} {sx sy out : Shape} {rx ry : Nat} {dx dy : DType}
    [Shape.BroadcastShape sx sy out] (op : Ops) (x : TUOp opx sx rx dx) (y : TUOp opy sy ry dy) :
    TUOpM (TUOp op out (rankOf out) (if op.producesBoolean then .bool else DType.promote dx dy)) := do
  let raw ← UOp.binaryOp op x.raw y.raw
  pure (mkUnsafe raw)

def cast {opx : Ops} {s : Shape} {r : Nat} {d : DType} (x : TUOp opx s r d) (dtype : DType) :
    TUOpM (TUOp .CAST s r dtype) := do
  let raw ← UOp.cast x.raw dtype
  pure (mkUnsafe raw)

def bitcast {opx : Ops} {s : Shape} {r : Nat} {d : DType} (x : TUOp opx s r d) (dtype : DType) :
    TUOpM (TUOp .BITCAST s r dtype) := do
  let raw ← UOp.bitcast x.raw dtype
  pure (mkUnsafe raw)

def cat {opx opy : Ops} {sx sy : Shape} {rx ry : Nat} {d : DType} {out : Shape} {axis : Nat}
    [Shape.ConcatShape sx sy axis out] (x : TUOp opx sx rx d) (y : TUOp opy sy ry d) :
    TUOpM (TUOp .CAT out (rankOf out) d) := do
  let raw ← UOp.cat [x.raw, y.raw] axis
  pure (mkUnsafe raw)

def catList {d : DType} {shapes : List Shape} {axis : Nat} {out : Shape}
    [h : Shape.ConcatListShape shapes axis out] (xs : TUOpList d shapes) :
    TUOpM (TUOp .CAT out (rankOf out) d) := do
  let _ := h
  let raw ← UOp.cat xs.toUOps axis
  pure (mkUnsafe raw)

def reshape {opx : Ops} {s : Shape} {r : Nat} {d : DType} (x : TUOp opx s r d) (newShape : Shape) :
    TUOpM (TUOp .RESHAPE newShape (rankOf newShape) d) := do
  let raw ← UOp.reshape x.raw newShape
  pure (mkUnsafe raw)

def expand {opx : Ops} {s : Shape} {r : Nat} {d : DType} (x : TUOp opx s r d) (newShape : Shape) :
    TUOpM (TUOp .EXPAND newShape (rankOf newShape) d) := do
  let raw ← UOp.expand x.raw newShape
  pure (mkUnsafe raw)

def permute {opx : Ops} {s : Shape} {r : Nat} {d : DType} (x : TUOp opx s r d) (perm : List Nat) :
    TUOpM (TUOp .PERMUTE (Shape.permute s perm) (rankOf (Shape.permute s perm)) d) := do
  let raw ← UOp.permute x.raw perm
  pure (mkUnsafe raw)

def pad {opx : Ops} {s : Shape} {r : Nat} {d : DType} (x : TUOp opx s r d) (padding : List (Nat × Nat)) :
    TUOpM (TUOp .PAD (Shape.pad s padding) (rankOf (Shape.pad s padding)) d) := do
  let raw ← UOp.pad x.raw padding
  pure (mkUnsafe raw)

def shrink {opx : Ops} {s : Shape} {r : Nat} {d : DType} (x : TUOp opx s r d) (bounds : List (Nat × Nat)) :
    TUOpM (TUOp .SHRINK (Shape.shrink s bounds) (rankOf (Shape.shrink s bounds)) d) := do
  let raw ← UOp.shrink x.raw bounds
  pure (mkUnsafe raw)

def flip {opx : Ops} {s : Shape} {r : Nat} {d : DType} (x : TUOp opx s r d) (axes : List Nat) :
    TUOpM (TUOp .FLIP s r d) := do
  let raw ← UOp.flip x.raw axes
  pure (mkUnsafe raw)

def reduce {opx : Ops} {s : Shape} {r : Nat} {d : DType} (x : TUOp opx s r d)
    (reduceOp : Ops) (axes : List Nat) (keepdim : Bool := true) :
    TUOpM (TUOp .REDUCE_AXIS (Shape.reduce s axes keepdim) (rankOf (Shape.reduce s axes keepdim)) d) := do
  let raw ← UOp.reduce x.raw reduceOp axes keepdim
  pure (mkUnsafe raw)

def contract2D {opa opb : Ops} {sa sb : Shape} {ra rb : Nat} {da db : DType}
    (a : TUOp opa sa ra da) (b : TUOp opb sb rb db) :
    TUOpM
      (TUOp .CONTRACT (matmulShape sa sb) (rankOf (matmulShape sa sb)) (DType.promote da db)) := do
  let raw ← UOp.contract2D a.raw b.raw
  pure (mkUnsafe raw)

def contract2DB {opa opb : Ops} {sa sb out : Shape} {ra rb : Nat} {da db : DType}
    [Shape.MatmulShape sa sb out] (a : TUOp opa sa ra da) (b : TUOp opb sb rb db) :
    TUOpM (TUOp .CONTRACT out (rankOf out) (DType.promote da db)) := do
  let raw ← UOp.contract2D a.raw b.raw
  pure (mkUnsafe raw)

def where_ {opc opx opy : Ops} {sc sx sy : Shape} {rc rx ry : Nat} {dx : DType}
    (cond : TUOp opc sc rc .bool) (x : TUOp opx sx rx dx) (y : TUOp opy sy ry dx) :
    TUOpM
      (TUOp .WHERE (broadcastShape sc (broadcastShape sx sy))
        (rankOf (broadcastShape sc (broadcastShape sx sy))) dx) := do
  let raw ← UOp.where_ cond.raw x.raw y.raw
  pure (mkUnsafe raw)

def add {opx opy : Ops} {sx sy : Shape} {rx ry : Nat} {dx dy : DType}
    (x : TUOp opx sx rx dx) (y : TUOp opy sy ry dy) :=
  binaryOp .ADD x y

def mul {opx opy : Ops} {sx sy : Shape} {rx ry : Nat} {dx dy : DType}
    (x : TUOp opx sx rx dx) (y : TUOp opy sy ry dy) :=
  binaryOp .MUL x y

def sub {opx opy : Ops} {sx sy : Shape} {rx ry : Nat} {dx dy : DType}
    (x : TUOp opx sx rx dx) (y : TUOp opy sy ry dy) :=
  binaryOp .SUB x y

def div {opx opy : Ops} {sx sy : Shape} {rx ry : Nat} {dx dy : DType}
    (x : TUOp opx sx rx dx) (y : TUOp opy sy ry dy) :=
  binaryOp .FDIV x y

def neg {opx : Ops} {s : Shape} {r : Nat} {d : DType} (x : TUOp opx s r d) :=
  unaryOp .NEG x

def cmplt {opx opy : Ops} {sx sy : Shape} {rx ry : Nat} {dx dy : DType}
    (x : TUOp opx sx rx dx) (y : TUOp opy sy ry dy) :=
  binaryOp .CMPLT x y

def cmpeq {opx opy : Ops} {sx sy : Shape} {rx ry : Nat} {dx dy : DType}
    (x : TUOp opx sx rx dx) (y : TUOp opy sy ry dy) :=
  binaryOp .CMPEQ x y

def cmpne {opx opy : Ops} {sx sy : Shape} {rx ry : Nat} {dx dy : DType}
    (x : TUOp opx sx rx dx) (y : TUOp opy sy ry dy) :=
  binaryOp .CMPNE x y

def sum {opx : Ops} {s : Shape} {r : Nat} {d : DType} (x : TUOp opx s r d)
    (axes : List Nat := []) (keepdim : Bool := true) :=
  let axes' := if axes.isEmpty then listRange s.length else axes
  reduce x .ADD axes' keepdim

def max_ {opx : Ops} {s : Shape} {r : Nat} {d : DType} (x : TUOp opx s r d)
    (axes : List Nat := []) (keepdim : Bool := true) :=
  let axes' := if axes.isEmpty then listRange s.length else axes
  reduce x .MAX axes' keepdim

end TUOp

end TinyGrad4
