import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph

namespace TinyGrad4

/-- A tensor with shape known at compile time -/
structure StaticTensor (shape : List Nat) (dtype : DType) where
  uop : UOp
  h_shape : uop.shape = shape := by rfl
  requiresGrad : Bool := false
  deriving Repr

instance {s : List Nat} {d : DType} : Inhabited (StaticTensor s d) :=
  ⟨{ uop := default, h_shape := sorry_proof, requiresGrad := false }⟩

inductive TensorList (d : DType) : List Shape → Type where
  | nil : TensorList d []
  | cons {s : Shape} {ss : List Shape} (t : StaticTensor s d) (ts : TensorList d ss) : TensorList d (s :: ss)

namespace TensorList

def anyRequiresGrad {d : DType} {shapes : List Shape} : TensorList d shapes → Bool
  | .nil => false
  | .cons t rest => t.requiresGrad || anyRequiresGrad rest

def toUOps {d : DType} {shapes : List Shape} : TensorList d shapes → List UOp
  | .nil => []
  | .cons t rest => t.uop :: toUOps rest

end TensorList

namespace StaticTensor

def shape {s : List Nat} {d : DType} (_ : StaticTensor s d) : List Nat := s
def dtype {s : List Nat} {d : DType} (_ : StaticTensor s d) : DType := d
def numel {s : List Nat} {d : DType} (_ : StaticTensor s d) : Nat := listProd s
def rank {s : List Nat} {d : DType} (_ : StaticTensor s d) : Nat := s.length
def elementSize {s : List Nat} {d : DType} (_ : StaticTensor s d) : Nat := d.itemsize
def nbytes {s : List Nat} {d : DType} (_ : StaticTensor s d) : Nat := listProd s * d.itemsize

end StaticTensor

abbrev TensorM := UOpM
def runTensorM (m : TensorM α) : α := runUOpM m
def runTensorMWith (st : UOpState) (m : TensorM α) : α := runUOpMWith st m

/-- Run a tensor builder with a capped hash-consing table (0 disables). -/
def runTensorMWithInternLimit (limit : Nat) (m : TensorM α) : α :=
  runUOpMWith { internLimit := limit } m

namespace Tensor

def full (shape : List Nat) (dtype : DType) (value : Float32) : TensorM (StaticTensor shape dtype) := do
  let const ← UOp.const dtype value
  let expanded ← UOp.expand const shape
  pure { uop := expanded, requiresGrad := false, h_shape := sorry_proof }

def fullInt (shape : List Nat) (dtype : DType) (value : Int) : TensorM (StaticTensor shape dtype) := do
  let const ← UOp.constInt dtype value
  let expanded ← UOp.expand const shape
  pure { uop := expanded, requiresGrad := false, h_shape := sorry_proof }

def fullNat (shape : List Nat) (dtype : DType) (value : Nat) : TensorM (StaticTensor shape dtype) := do
  let const ← UOp.constNat dtype value
  let expanded ← UOp.expand const shape
  pure { uop := expanded, requiresGrad := false, h_shape := sorry_proof }

def fullBool (shape : List Nat) (value : Bool) : TensorM (StaticTensor shape .bool) := do
  let const ← UOp.constBool value
  let expanded ← UOp.expand const shape
  pure { uop := expanded, requiresGrad := false, h_shape := sorry_proof }

def zeros (shape : List Nat) (dtype : DType := .float32) : TensorM (StaticTensor shape dtype) :=
  full shape dtype 0.0

def ones (shape : List Nat) (dtype : DType := .float32) : TensorM (StaticTensor shape dtype) :=
  full shape dtype 1.0

private def fromArrayF32 (shape : Shape) (vals : Array Float32) : TensorM (StaticTensor shape .float32) := do
  let u ← UOp.vconstF32 vals
  let base : StaticTensor [vals.size] .float32 := { uop := u, h_shape := sorry_proof }
  let reshaped ← UOp.reshape base.uop shape
  pure { uop := reshaped, requiresGrad := false, h_shape := sorry_proof }

private def intToFloat (v : Int) : Float :=
  if v >= 0 then
    Float.ofNat v.toNat
  else
    -Float.ofNat (-v).toNat

private def twoPow64 : Float :=
  Float.ofNat ((2 : Nat) ^ 64)

private def lcgStep (s : UInt64) : UInt64 :=
  s * 6364136223846793005 + 1

private def nextUniform (s : UInt64) : UInt64 × Float :=
  let s' := lcgStep s
  let f := (Float.ofNat s'.toNat) / twoPow64
  (s', f)

private def randArray (n : Nat) (seed : Nat) : Array Float32 := Id.run do
  let mut out := Array.emptyWithCapacity n
  let mut s := UInt64.ofNat seed
  for _ in [:n] do
    let (s', u) := nextUniform s
    s := s'
    out := out.push u.toFloat32
  return out

private def randnArray (n : Nat) (seed : Nat) : Array Float32 := Id.run do
  let mut out := Array.emptyWithCapacity n
  let mut s := UInt64.ofNat seed
  for _ in [:n] do
    let mut acc : Float := 0.0
    for _ in [:12] do
      let (s', u) := nextUniform s
      s := s'
      acc := acc + u
    out := out.push (acc - 6.0).toFloat32
  return out

private def randintArray (n : Nat) (seed : Nat) (low high : Int) : Array Float32 := Id.run do
  let mut out := Array.emptyWithCapacity n
  let mut s := UInt64.ofNat seed
  let rangeInt := high - low
  let rangeNat := if rangeInt <= 0 then 1 else Int.toNat rangeInt
  for _ in [:n] do
    s := lcgStep s
    let r := s.toNat % rangeNat
    let v := low + Int.ofNat r
    out := out.push (intToFloat v).toFloat32
  return out

def arange (n : Nat) (dtype : DType := .float32) : TensorM (StaticTensor [n] dtype) := do
  let vals : Array Float32 := Id.run do
    let mut out := Array.emptyWithCapacity n
    for i in [:n] do
      out := out.push (Float.ofNat i).toFloat32
    return out
  let base ← fromArrayF32 [n] vals
  if dtype == .float32 then
    pure { uop := base.uop, requiresGrad := false, h_shape := sorry_proof }
  else
    let casted ← UOp.cast base.uop dtype
    pure { uop := casted, requiresGrad := false, h_shape := sorry_proof }

def linspace (start stop : Float32) (steps : Nat) (dtype : DType := .float32)
    : TensorM (StaticTensor [steps] dtype) := do
  let startF := start.toFloat
  let stopF := stop.toFloat
  let step :=
    if steps <= 1 then 0.0
    else (stopF - startF) / (Float.ofNat (steps - 1))
  let vals : Array Float32 := Id.run do
    let mut out := Array.emptyWithCapacity steps
    for i in [:steps] do
      let v := startF + step * (Float.ofNat i)
      out := out.push v.toFloat32
    return out
  let base ← fromArrayF32 [steps] vals
  if dtype == .float32 then
    pure { uop := base.uop, requiresGrad := false, h_shape := sorry_proof }
  else
    let casted ← UOp.cast base.uop dtype
    pure { uop := casted, requiresGrad := false, h_shape := sorry_proof }

def rand (shape : Shape) (dtype : DType := .float32) (seed : Nat := 0)
    : TensorM (StaticTensor shape dtype) := do
  let numel := listProd shape
  let vals := randArray numel seed
  let base ← fromArrayF32 [numel] vals
  let reshaped ← UOp.reshape base.uop shape
  if dtype == .float32 then
    pure { uop := reshaped, requiresGrad := false, h_shape := sorry_proof }
  else
    let casted ← UOp.cast reshaped dtype
    pure { uop := casted, requiresGrad := false, h_shape := sorry_proof }

def randn (shape : Shape) (dtype : DType := .float32) (seed : Nat := 0)
    : TensorM (StaticTensor shape dtype) := do
  let numel := listProd shape
  let vals := randnArray numel seed
  let base ← fromArrayF32 [numel] vals
  let reshaped ← UOp.reshape base.uop shape
  if dtype == .float32 then
    pure { uop := reshaped, requiresGrad := false, h_shape := sorry_proof }
  else
    let casted ← UOp.cast reshaped dtype
    pure { uop := casted, requiresGrad := false, h_shape := sorry_proof }

def randint (shape : Shape) (low high : Int) (dtype : DType := .int32) (seed : Nat := 0)
    : TensorM (StaticTensor shape dtype) := do
  let numel := listProd shape
  let vals := randintArray numel seed low high
  let base ← fromArrayF32 [numel] vals
  let reshaped ← UOp.reshape base.uop shape
  if dtype == .float32 then
    pure { uop := reshaped, requiresGrad := false, h_shape := sorry_proof }
  else
    let casted ← UOp.cast reshaped dtype
    pure { uop := casted, requiresGrad := false, h_shape := sorry_proof }

def buffer (shape : List Nat) (dtype : DType := .float32) (device : String := "CPU")
    : TensorM (StaticTensor shape dtype) := do
  let buf ← UOp.buffer dtype shape device
  pure { uop := buf, requiresGrad := false, h_shape := sorry_proof }

/-- Set a max size for the UOp interning table (0 disables). -/
def setInternLimit (limit : Nat) : TensorM Unit :=
  TinyGrad4.setInternLimit limit

/-- Clear the UOp interning table to allow graphs to be collected. -/
def clearIntern : TensorM Unit :=
  TinyGrad4.clearIntern

def zerosLike {s : List Nat} {d : DType} (_t : StaticTensor s d) : TensorM (StaticTensor s d) :=
  zeros s d

def onesLike {s : List Nat} {d : DType} (_t : StaticTensor s d) : TensorM (StaticTensor s d) :=
  ones s d

def fullLike {s : List Nat} {d : DType} (_t : StaticTensor s d) (value : Float32)
    : TensorM (StaticTensor s d) := do
  let base ← full s .float32 value
  if d == .float32 then
    pure { uop := base.uop, requiresGrad := false, h_shape := sorry_proof }
  else
    let casted ← UOp.cast base.uop d
    pure { uop := casted, requiresGrad := false, h_shape := sorry_proof }

def randLike {s : List Nat} {d : DType} (_t : StaticTensor s d) (seed : Nat := 0)
    : TensorM (StaticTensor s d) :=
  rand s d seed

def randnLike {s : List Nat} {d : DType} (_t : StaticTensor s d) (seed : Nat := 0)
    : TensorM (StaticTensor s d) :=
  randn s d seed

def randintLike {s : List Nat} {d : DType} (_t : StaticTensor s d) (low high : Int) (seed : Nat := 0)
    : TensorM (StaticTensor s d) :=
  randint s low high d seed

end Tensor

abbrev Scalar (dtype : DType) := StaticTensor [] dtype
abbrev Vector (n : Nat) (dtype : DType) := StaticTensor [n] dtype
abbrev Matrix (m n : Nat) (dtype : DType) := StaticTensor [m, n] dtype
abbrev BMatrix (b m n : Nat) (dtype : DType) := StaticTensor [b, m, n] dtype

end TinyGrad4
