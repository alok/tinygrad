import Float64
import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import TinyGrad4.Backend.DeviceType

namespace TinyGrad4

/-- A tensor with shape known at compile time -/
structure StaticTensor (shape : List Nat) (dtype : DType) (device : Backend.DeviceType) where
  uop : UOp
  h_shape : uop.shape = shape := by rfl
  h_dtype : uop.dtype = dtype := by rfl
  requiresGrad : Bool := false
  deriving Repr

instance {s : List Nat} {d : DType} {device : Backend.DeviceType} : Inhabited (StaticTensor s d device) :=
  ⟨{ uop := {(default : UOp) with shape := s, dtype := d}, h_shape := by rfl, h_dtype := by rfl, requiresGrad := false }⟩

inductive TensorList (d : DType) (device : Backend.DeviceType) : List Shape → Type where
  | nil : TensorList d device []
  | cons {s : Shape} {ss : List Shape} (t : StaticTensor s d device) (ts : TensorList d device ss) : TensorList d device (s :: ss)

namespace TensorList

def anyRequiresGrad {d : DType} {device : Backend.DeviceType} {shapes : List Shape} : TensorList d device shapes → Bool
  | .nil => false
  | .cons t rest => t.requiresGrad || anyRequiresGrad rest

def toUOps {d : DType} {device : Backend.DeviceType} {shapes : List Shape} : TensorList d device shapes → List UOp
  | .nil => []
  | .cons t rest => t.uop :: toUOps rest

end TensorList

namespace StaticTensor

def shape {s : List Nat} {d : DType} {device : Backend.DeviceType} (_ : StaticTensor s d device) : List Nat := s
def dtype {s : List Nat} {d : DType} {device : Backend.DeviceType} (_ : StaticTensor s d device) : DType := d
def deviceType {s : List Nat} {d : DType} {device : Backend.DeviceType} (_ : StaticTensor s d device) : Backend.DeviceType := device
def numel {s : List Nat} {d : DType} {device : Backend.DeviceType} (_ : StaticTensor s d device) : Nat := listProd s
def rank {s : List Nat} {d : DType} {device : Backend.DeviceType} (_ : StaticTensor s d device) : Nat := s.length
def elementSize {s : List Nat} {d : DType} {device : Backend.DeviceType} (_ : StaticTensor s d device) : Nat := d.itemsize
def nbytes {s : List Nat} {d : DType} {device : Backend.DeviceType} (_ : StaticTensor s d device) : Nat := listProd s * d.itemsize

/-- Build a StaticTensor from a UOp, checking shape + dtype at runtime. -/
private def uopDeviceType (u : UOp) : Option Backend.DeviceType :=
  match u.op, u.arg with
  | .BUFFER, .device dev => some (Backend.parseDeviceType dev)
  | _, _ => none

def ofUOp {s : List Nat} {d : DType} {device : Backend.DeviceType} (u : UOp)
    (requiresGrad : Bool := false) : StaticTensor s d device :=
  if hShape : u.shape = s then
    if hType : u.dtype = d then
      match uopDeviceType u with
      | some udev =>
        if udev = device then
          { uop := u, h_shape := hShape, h_dtype := hType, requiresGrad }
        else
          panic! s!"StaticTensor device mismatch: expected {repr device}, got {repr udev}"
      | none =>
        { uop := u, h_shape := hShape, h_dtype := hType, requiresGrad }
    else
      panic! s!"StaticTensor dtype mismatch: expected {repr d}, got {repr u.dtype}"
  else
    panic! s!"StaticTensor shape mismatch: expected {repr s}, got {repr u.shape}"

def ofUOpEq {s : List Nat} {d : DType} {device : Backend.DeviceType} (u : UOp)
    (hShape : u.shape = s) (hType : u.dtype = d) (requiresGrad : Bool := false) : StaticTensor s d device :=
  { uop := u, h_shape := hShape, h_dtype := hType, requiresGrad }

end StaticTensor

abbrev TensorM := UOpM
def runTensorM (m : TensorM α) : α := runUOpM m
def runTensorMWith (st : UOpState) (m : TensorM α) : α := runUOpMWith st m

/-- Run a tensor builder with a capped hash-consing table (0 disables). -/
def runTensorMWithInternLimit (limit : Nat) (m : TensorM α) : α :=
  runUOpMWith { internLimit := limit } m

namespace Tensor

def full (shape : List Nat) (dtype : DType) (value : Float32) (device : Backend.DeviceType := .CPU)
    : TensorM (StaticTensor shape dtype device) := do
  let const ← UOp.const dtype value
  let expanded ← UOp.expand const shape
  pure (StaticTensor.ofUOp expanded)

def fullInt (shape : List Nat) (dtype : DType) (value : Int) (device : Backend.DeviceType := .CPU)
    : TensorM (StaticTensor shape dtype device) := do
  let const ← UOp.constInt dtype value
  let expanded ← UOp.expand const shape
  pure (StaticTensor.ofUOp expanded)

def fullNat (shape : List Nat) (dtype : DType) (value : Nat) (device : Backend.DeviceType := .CPU)
    : TensorM (StaticTensor shape dtype device) := do
  let const ← UOp.constNat dtype value
  let expanded ← UOp.expand const shape
  pure (StaticTensor.ofUOp expanded)

def fullBool (shape : List Nat) (value : Bool) (device : Backend.DeviceType := .CPU)
    : TensorM (StaticTensor shape .bool device) := do
  let const ← UOp.constBool value
  let expanded ← UOp.expand const shape
  pure (StaticTensor.ofUOp expanded)

def zeros (shape : List Nat) (dtype : DType := .float32) (device : Backend.DeviceType := .CPU)
    : TensorM (StaticTensor shape dtype device) :=
  full (device := device) shape dtype 0.0

def ones (shape : List Nat) (dtype : DType := .float32) (device : Backend.DeviceType := .CPU)
    : TensorM (StaticTensor shape dtype device) :=
  full (device := device) shape dtype 1.0

private def fromArrayF32 {device : Backend.DeviceType} (shape : Shape) (vals : Array Float32)
    : TensorM (StaticTensor shape .float32 device) := do
  let u ← UOp.vconstF32 vals
  let base : StaticTensor [vals.size] .float32 device := StaticTensor.ofUOp u
  let reshaped ← UOp.reshape base.uop shape
  pure (StaticTensor.ofUOp reshaped)

private def intToFloat (v : Int) : Float64 :=
  if v >= 0 then
    Float64.ofNat v.toNat
  else
    -Float64.ofNat (-v).toNat

private def twoPow64 : Float64 :=
  Float64.ofNat ((2 : Nat) ^ 64)

private def lcgStep (s : UInt64) : UInt64 :=
  s * 6364136223846793005 + 1

private def nextUniform (s : UInt64) : UInt64 × Float64 :=
  let s' := lcgStep s
  let f := (Float64.ofNat s'.toNat) / twoPow64
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
    let mut acc : Float64 := 0.0
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

def arange (n : Nat) (dtype : DType := .float32) (device : Backend.DeviceType := .CPU)
    : TensorM (StaticTensor [n] dtype device) := do
  let vals : Array Float32 := Id.run do
    let mut out := Array.emptyWithCapacity n
    for i in [:n] do
      out := out.push (Float64.ofNat i).toFloat32
    return out
  let base ← fromArrayF32 (device := device) [n] vals
  if dtype == .float32 then
    pure (StaticTensor.ofUOp base.uop)
  else
    let casted ← UOp.cast base.uop dtype
    pure (StaticTensor.ofUOp casted)

def linspace (start stop : Float32) (steps : Nat) (dtype : DType := .float32) (device : Backend.DeviceType := .CPU)
    : TensorM (StaticTensor [steps] dtype device) := do
  let startF := start.toFloat
  let stopF := stop.toFloat
  let step :=
    if steps <= 1 then 0.0
    else (stopF - startF) / (Float64.ofNat (steps - 1))
  let vals : Array Float32 := Id.run do
    let mut out := Array.emptyWithCapacity steps
    for i in [:steps] do
      let v := startF + step * (Float64.ofNat i)
      out := out.push v.toFloat32
    return out
  let base ← fromArrayF32 (device := device) [steps] vals
  if dtype == .float32 then
    pure (StaticTensor.ofUOp base.uop)
  else
    let casted ← UOp.cast base.uop dtype
    pure (StaticTensor.ofUOp casted)

def rand (shape : Shape) (dtype : DType := .float32) (seed : Nat := 0) (device : Backend.DeviceType := .CPU)
    : TensorM (StaticTensor shape dtype device) := do
  let numel := listProd shape
  let vals := randArray numel seed
  let base ← fromArrayF32 (device := device) [numel] vals
  let reshaped ← UOp.reshape base.uop shape
  if dtype == .float32 then
    pure (StaticTensor.ofUOp reshaped)
  else
    let casted ← UOp.cast reshaped dtype
    pure (StaticTensor.ofUOp casted)

def randn (shape : Shape) (dtype : DType := .float32) (seed : Nat := 0) (device : Backend.DeviceType := .CPU)
    : TensorM (StaticTensor shape dtype device) := do
  let numel := listProd shape
  let vals := randnArray numel seed
  let base ← fromArrayF32 (device := device) [numel] vals
  let reshaped ← UOp.reshape base.uop shape
  if dtype == .float32 then
    pure (StaticTensor.ofUOp reshaped)
  else
    let casted ← UOp.cast reshaped dtype
    pure (StaticTensor.ofUOp casted)

def randint (shape : Shape) (low high : Int) (dtype : DType := .int32) (seed : Nat := 0) (device : Backend.DeviceType := .CPU)
    : TensorM (StaticTensor shape dtype device) := do
  let numel := listProd shape
  let vals := randintArray numel seed low high
  let base ← fromArrayF32 (device := device) [numel] vals
  let reshaped ← UOp.reshape base.uop shape
  if dtype == .float32 then
    pure (StaticTensor.ofUOp reshaped)
  else
    let casted ← UOp.cast reshaped dtype
    pure (StaticTensor.ofUOp casted)

def buffer (shape : List Nat) (dtype : DType := .float32) (device : Backend.DeviceType := .CPU)
    : TensorM (StaticTensor shape dtype device) := do
  let buf ← UOp.buffer dtype shape device
  pure (StaticTensor.ofUOp buf)

/-- Set a max size for the UOp interning table (0 disables). -/
def setInternLimit (limit : Nat) : TensorM Unit :=
  TinyGrad4.setInternLimit limit

/-- Clear the UOp interning table to allow graphs to be collected. -/
def clearIntern : TensorM Unit :=
  TinyGrad4.clearIntern

def zerosLike {s : List Nat} {d : DType} {device : Backend.DeviceType} (_t : StaticTensor s d device)
    : TensorM (StaticTensor s d device) :=
  zeros (device := device) s d

def onesLike {s : List Nat} {d : DType} {device : Backend.DeviceType} (_t : StaticTensor s d device)
    : TensorM (StaticTensor s d device) :=
  ones (device := device) s d

def fullLike {s : List Nat} {d : DType} {device : Backend.DeviceType} (_t : StaticTensor s d device) (value : Float32)
    : TensorM (StaticTensor s d device) := do
  let base ← full (device := device) s .float32 value
  if d == .float32 then
    pure (StaticTensor.ofUOp base.uop)
  else
    let casted ← UOp.cast base.uop d
    pure (StaticTensor.ofUOp casted)

def randLike {s : List Nat} {d : DType} {device : Backend.DeviceType} (_t : StaticTensor s d device) (seed : Nat := 0)
    : TensorM (StaticTensor s d device) :=
  rand (device := device) s d seed

def randnLike {s : List Nat} {d : DType} {device : Backend.DeviceType} (_t : StaticTensor s d device) (seed : Nat := 0)
    : TensorM (StaticTensor s d device) :=
  randn (device := device) s d seed

def randintLike {s : List Nat} {d : DType} {device : Backend.DeviceType} (_t : StaticTensor s d device) (low high : Int) (seed : Nat := 0)
    : TensorM (StaticTensor s d device) :=
  randint (device := device) s low high d seed

end Tensor

abbrev Scalar (dtype : DType) (device : Backend.DeviceType := .CPU) := StaticTensor [] dtype device
abbrev Vector (n : Nat) (dtype : DType) (device : Backend.DeviceType := .CPU) := StaticTensor [n] dtype device
abbrev Matrix (m n : Nat) (dtype : DType) (device : Backend.DeviceType := .CPU) := StaticTensor [m, n] dtype device
abbrev BMatrix (b m n : Nat) (dtype : DType) (device : Backend.DeviceType := .CPU) := StaticTensor [b, m, n] dtype device

end TinyGrad4
