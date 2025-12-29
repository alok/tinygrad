import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import TinyGrad4.Tensor.Tensor
import TinyGrad4.Optim.UOpOpt
import TinyGrad4.Backend.Buffer
import TinyGrad4.Backend.Native
import TinyGrad4.Backend.Fusion
import TinyGrad4.Backend.FusedEwise
import TinyGrad4.Backend.FusedReduce
import TinyGrad4.Backend.FusedContract
import TinyGrad4.Backend.FusedSGD
import TinyGrad4.Backend.FusedMatmul
import TinyGrad4.Backend.FusedSoftmax
import TinyGrad4.Backend.Metal
import TinyGrad4.Tags
import Std.Data.HashMap

namespace TinyGrad4

open Std
open TinyGrad4.Backend

abbrev Env := HashMap UOpId RawBuffer

namespace RawBuffer

def zeros (dtype : DType) (numel : Nat) : RawBuffer :=
  match dtype with
  | .float32 =>
    { dtype, data := Native.fullF32Bits numel 0 }
  | _ =>
    { dtype, data := ByteArray.mk (Array.replicate (numel * dtype.itemsize) 0) }

/-- Pack Float values as Float32 into RawBuffer -/
def ofFloats (arr : Array Float) : RawBuffer := Id.run do
  let mut bytes := ByteArray.emptyWithCapacity (arr.size * 4)
  for v in arr do
    let f32 := v.toFloat32
    let bits := f32.toBits
    bytes := bytes.push (bits.toNat.toUInt8)
    bytes := bytes.push ((bits.toNat >>> 8).toUInt8)
    bytes := bytes.push ((bits.toNat >>> 16).toUInt8)
    bytes := bytes.push ((bits.toNat >>> 24).toUInt8)
  { dtype := .float32, data := bytes }

/-- Pack Float32 values directly into RawBuffer -/
def ofFloat32s (arr : Array Float32) : RawBuffer := Id.run do
  let mut bytes := ByteArray.emptyWithCapacity (arr.size * 4)
  for v in arr do
    let bits := v.toBits
    bytes := bytes.push (bits.toNat.toUInt8)
    bytes := bytes.push ((bits.toNat >>> 8).toUInt8)
    bytes := bytes.push ((bits.toNat >>> 16).toUInt8)
    bytes := bytes.push ((bits.toNat >>> 24).toUInt8)
  { dtype := .float32, data := bytes }

/-- Pack Float array to F32 bytes (internal helper) -/
def packFloatsToF32Bytes (arr : Array Float) : ByteArray := Id.run do
  let mut bytes := ByteArray.emptyWithCapacity (arr.size * 4)
  for v in arr do
    let f32 := v.toFloat32
    let bits := f32.toBits
    bytes := bytes.push (bits.toNat.toUInt8)
    bytes := bytes.push ((bits.toNat >>> 8).toUInt8)
    bytes := bytes.push ((bits.toNat >>> 16).toUInt8)
    bytes := bytes.push ((bits.toNat >>> 24).toUInt8)
  bytes

/-- Unpack F32 bytes to Float array (internal helper) -/
def unpackF32Bytes (data : ByteArray) : Array Float := Id.run do
  let numel := data.size / 4
  let mut out := Array.emptyWithCapacity numel
  for i in [:numel] do
    let b0 := data.get! (i * 4)
    let b1 := data.get! (i * 4 + 1)
    let b2 := data.get! (i * 4 + 2)
    let b3 := data.get! (i * 4 + 3)
    let bits : UInt32 :=
      (UInt32.ofNat b0.toNat) |||
      ((UInt32.ofNat b1.toNat) <<< 8) |||
      ((UInt32.ofNat b2.toNat) <<< 16) |||
      ((UInt32.ofNat b3.toNat) <<< 24)
    out := out.push (Float32.ofBits bits).toFloat
  out

/-- Format RawBuffer as string for display (no intermediate array allocation) -/
def reprAsString (b : RawBuffer) : String := Id.run do
  match b.dtype with
  | .float32 =>
    let numel := b.data.size / 4
    if numel == 0 then return "#[]"
    let mut s := "#["
    for i in [:numel] do
      if i > 0 then s := s ++ ", "
      let b0 := b.data.get! (i * 4)
      let b1 := b.data.get! (i * 4 + 1)
      let b2 := b.data.get! (i * 4 + 2)
      let b3 := b.data.get! (i * 4 + 3)
      let bits : UInt32 :=
        (UInt32.ofNat b0.toNat) |||
        ((UInt32.ofNat b1.toNat) <<< 8) |||
        ((UInt32.ofNat b2.toNat) <<< 16) |||
        ((UInt32.ofNat b3.toNat) <<< 24)
      s := s ++ toString (Float32.ofBits bits).toFloat
    s ++ "]"
  | .bool =>
    if b.data.size == 0 then return "#[]"
    let mut s := "#["
    for i in [:b.data.size] do
      if i > 0 then s := s ++ ", "
      s := s ++ if b.data.get! i == 0 then "0" else "1"
    s ++ "]"
  | _ => s!"<RawBuffer {repr b.dtype} {b.data.size} bytes>"

instance : Repr RawBuffer where
  reprPrec b _ := b.reprAsString

instance : ToString RawBuffer where
  toString b := b.reprAsString

def decodeScalarF32? (b : RawBuffer) : Option Float :=
  if b.dtype != .float32 then
    none
  else if b.data.size < 4 then
    none
  else
    let b0 := b.data.get! 0
    let b1 := b.data.get! 1
    let b2 := b.data.get! 2
    let b3 := b.data.get! 3
    let bits : UInt32 :=
      (UInt32.ofNat b0.toNat) |||
      ((UInt32.ofNat b1.toNat) <<< 8) |||
      ((UInt32.ofNat b2.toNat) <<< 16) |||
      ((UInt32.ofNat b3.toNat) <<< 24)
    some ((Float32.ofBits bits).toFloat)

def decodeScalarF32 (b : RawBuffer) : Float :=
  (decodeScalarF32? b).getD 0.0

/-- Get float32 at index i (returns as Float for Lean compatibility) -/
def getF32 (b : RawBuffer) (i : Nat) : Float :=
  if b.dtype != .float32 then 0.0
  else if i * 4 + 3 >= b.data.size then 0.0
  else
    let b0 := b.data.get! (i * 4)
    let b1 := b.data.get! (i * 4 + 1)
    let b2 := b.data.get! (i * 4 + 2)
    let b3 := b.data.get! (i * 4 + 3)
    let bits : UInt32 :=
      (UInt32.ofNat b0.toNat) |||
      ((UInt32.ofNat b1.toNat) <<< 8) |||
      ((UInt32.ofNat b2.toNat) <<< 16) |||
      ((UInt32.ofNat b3.toNat) <<< 24)
    (Float32.ofBits bits).toFloat

/-- Number of float32 elements -/
def numF32 (b : RawBuffer) : Nat :=
  if b.dtype != .float32 then 0 else b.data.size / 4

/-- Size of buffer in elements (dtype-aware) -/
def size (b : RawBuffer) : Nat :=
  b.data.size / b.dtype.itemsize

end RawBuffer

/-- GetElem instance for RawBuffer - enables buf[i] syntax -/
instance : GetElem RawBuffer Nat Float (fun b i => i < b.numF32) where
  getElem b i _ := b.getF32 i

namespace Interpreter

def validateOnCompile : Bool := true
def optimizeOnCompile : Bool := true

def padShapeTo (s : Shape) (len : Nat) : Shape :=
  if s.length >= len then s else List.replicate (len - s.length) 1 ++ s

def unflattenIndex (flatIdx : Nat) (shape : Shape) : List Nat :=
  shape.foldr (fun dim (idx, acc) =>
    let (q, r) := (idx / dim, idx % dim)
    (q, r :: acc)
  ) (flatIdx, []) |>.2

def flattenIndex (indices : List Nat) (shape : Shape) : Nat :=
  (indices.zip shape).foldl (fun acc (idx, dim) => acc * dim + idx) 0

def stridesOf (shape : Shape) : List Nat :=
  shape.foldr (fun dim (strides, prod) => (prod :: strides, prod * dim)) ([], 1) |>.1

def offsetOf (indices strides : List Nat) : Nat :=
  (indices.zip strides).foldl (fun acc (idx, stride) => acc + idx * stride) 0

def broadcastBytes (src : ByteArray) (fromShape toShape : Shape) (elemBytes : Nat) : ByteArray := Id.run do
  let numelTo := listProd toShape
  let mut out := ByteArray.empty

  let rank := toShape.length
  let fromShape' := padShapeTo fromShape rank
  if src.size < listProd fromShape' * elemBytes then
    return ByteArray.empty

  for i in [:numelTo] do
    let toIndices := unflattenIndex i toShape
    let fromIndices := listZipWith (fun idx dim => if dim == 1 then 0 else idx) toIndices fromShape'
    let fromIdx := flattenIndex fromIndices fromShape'
    let start := fromIdx * elemBytes
    out := out ++ src.extract start (start + elemBytes)
  return out

def permuteBytes (src : ByteArray) (fromShape : Shape) (perm : List Nat) (elemBytes : Nat) : ByteArray := Id.run do
  let toShape := perm.map (fun i => listGetD fromShape i 0)
  let numelTo := listProd toShape
  let mut out := ByteArray.empty

  if src.size < listProd fromShape * elemBytes then
    return ByteArray.empty

  for i in [:numelTo] do
    let toIndices := unflattenIndex i toShape
    let fromIndices := (listRange fromShape.length).map fun j =>
      listGetD toIndices (listIndexOf perm j) 0
    let fromIdx := flattenIndex fromIndices fromShape
    let start := fromIdx * elemBytes
    out := out ++ src.extract start (start + elemBytes)
  return out

private def padInIndices? (outIdxs fromShape padLeft : List Nat) : Option (List Nat) :=
  match outIdxs, fromShape, padLeft with
  | [], [], [] => some []
  | o :: os, dim :: ds, l :: ls =>
    if o < l || o >= l + dim then
      none
    else
      match padInIndices? os ds ls with
      | some rest => some ((o - l) :: rest)
      | none => none
  | _, _, _ => none

def padBytes (src : ByteArray) (fromShape padLeft _padRight outShape : Shape) (elemBytes : Nat) : ByteArray := Id.run do
  let outNumel := listProd outShape
  let mut out := ByteArray.emptyWithCapacity (outNumel * elemBytes)
  let zeroElem : Array UInt8 := Array.replicate elemBytes 0

  if src.size < listProd fromShape * elemBytes then
    return ByteArray.empty

  for i in [:outNumel] do
    let outIdxs := unflattenIndex i outShape
    match padInIndices? outIdxs fromShape padLeft with
    | some inIdxs =>
      let inIdx := flattenIndex inIdxs fromShape
      let start := inIdx * elemBytes
      out := out ++ src.extract start (start + elemBytes)
    | none =>
      for b in zeroElem do
        out := out.push b
  return out

def shrinkBytes (src : ByteArray) (fromShape starts _stops outShape : Shape) (elemBytes : Nat) : ByteArray := Id.run do
  let outNumel := listProd outShape
  let mut out := ByteArray.emptyWithCapacity (outNumel * elemBytes)

  if src.size < listProd fromShape * elemBytes then
    return ByteArray.empty

  for i in [:outNumel] do
    let outIdxs := unflattenIndex i outShape
    let inIdxs := listZipWith (fun idx start => idx + start) outIdxs starts
    let inIdx := flattenIndex inIdxs fromShape
    let start := inIdx * elemBytes
    out := out ++ src.extract start (start + elemBytes)
  return out

def whereBytes (cond x y : ByteArray) (cSh xSh ySh outShape : Shape) (elemBytes : Nat) : ByteArray := Id.run do
  let outNumel := listProd outShape
  let mut out := ByteArray.emptyWithCapacity (outNumel * elemBytes)

  let cData :=
    if cSh == outShape then cond else broadcastBytes cond cSh outShape 1
  let xData :=
    if xSh == outShape then x else broadcastBytes x xSh outShape elemBytes
  let yData :=
    if ySh == outShape then y else broadcastBytes y ySh outShape elemBytes

  if cData.isEmpty || xData.isEmpty || yData.isEmpty then
    return ByteArray.empty

  for i in [:outNumel] do
    let pickX := cData.get! i != 0
    let base := i * elemBytes
    let src := if pickX then xData else yData
    out := out ++ src.extract base (base + elemBytes)
  return out

private def intToUInt8Mod (v : Int) : UInt8 :=
  UInt8.ofNat (Int8.ofInt v).toBitVec.toNat

private def intToUInt16Mod (v : Int) : UInt16 :=
  UInt16.ofNat (Int16.ofInt v).toBitVec.toNat

private def intToUInt32Mod (v : Int) : UInt32 :=
  UInt32.ofNat (Int32.ofInt v).toBitVec.toNat

private def intToUInt64Mod (v : Int) : UInt64 :=
  UInt64.ofNat (Int64.ofInt v).toBitVec.toNat

private def bytesFromUInt16 (v : UInt16) : Array UInt8 :=
  let b0 : UInt8 := UInt8.ofNat (UInt16.toNat (v &&& 0xFF))
  let b1 : UInt8 := UInt8.ofNat (UInt16.toNat ((v >>> 8) &&& 0xFF))
  #[b0, b1]

private def bytesFromUInt32 (v : UInt32) : Array UInt8 :=
  let b0 : UInt8 := UInt8.ofNat (UInt32.toNat (v &&& 0xFF))
  let b1 : UInt8 := UInt8.ofNat (UInt32.toNat ((v >>> 8) &&& 0xFF))
  let b2 : UInt8 := UInt8.ofNat (UInt32.toNat ((v >>> 16) &&& 0xFF))
  let b3 : UInt8 := UInt8.ofNat (UInt32.toNat ((v >>> 24) &&& 0xFF))
  #[b0, b1, b2, b3]

private def bytesFromUInt64 (v : UInt64) : Array UInt8 :=
  let b0 : UInt8 := UInt8.ofNat (UInt64.toNat (v &&& 0xFF))
  let b1 : UInt8 := UInt8.ofNat (UInt64.toNat ((v >>> 8) &&& 0xFF))
  let b2 : UInt8 := UInt8.ofNat (UInt64.toNat ((v >>> 16) &&& 0xFF))
  let b3 : UInt8 := UInt8.ofNat (UInt64.toNat ((v >>> 24) &&& 0xFF))
  let b4 : UInt8 := UInt8.ofNat (UInt64.toNat ((v >>> 32) &&& 0xFF))
  let b5 : UInt8 := UInt8.ofNat (UInt64.toNat ((v >>> 40) &&& 0xFF))
  let b6 : UInt8 := UInt8.ofNat (UInt64.toNat ((v >>> 48) &&& 0xFF))
  let b7 : UInt8 := UInt8.ofNat (UInt64.toNat ((v >>> 56) &&& 0xFF))
  #[b0, b1, b2, b3, b4, b5, b6, b7]

private def readU16LE (b : ByteArray) (offset : Nat) : UInt16 :=
  let b0 := b.get! offset
  let b1 := b.get! (offset + 1)
  (UInt16.ofNat b0.toNat) ||| ((UInt16.ofNat b1.toNat) <<< 8)

private def readU32LE (b : ByteArray) (offset : Nat) : UInt32 :=
  let b0 := b.get! offset
  let b1 := b.get! (offset + 1)
  let b2 := b.get! (offset + 2)
  let b3 := b.get! (offset + 3)
  (UInt32.ofNat b0.toNat) |||
    ((UInt32.ofNat b1.toNat) <<< 8) |||
    ((UInt32.ofNat b2.toNat) <<< 16) |||
    ((UInt32.ofNat b3.toNat) <<< 24)

private def readU64LE (b : ByteArray) (offset : Nat) : UInt64 :=
  let b0 := b.get! offset
  let b1 := b.get! (offset + 1)
  let b2 := b.get! (offset + 2)
  let b3 := b.get! (offset + 3)
  let b4 := b.get! (offset + 4)
  let b5 := b.get! (offset + 5)
  let b6 := b.get! (offset + 6)
  let b7 := b.get! (offset + 7)
  (UInt64.ofNat b0.toNat) |||
    ((UInt64.ofNat b1.toNat) <<< 8) |||
    ((UInt64.ofNat b2.toNat) <<< 16) |||
    ((UInt64.ofNat b3.toNat) <<< 24) |||
    ((UInt64.ofNat b4.toNat) <<< 32) |||
    ((UInt64.ofNat b5.toNat) <<< 40) |||
    ((UInt64.ofNat b6.toNat) <<< 48) |||
    ((UInt64.ofNat b7.toNat) <<< 56)

private def readF32At (b : ByteArray) (idx : Nat) : Float :=
  let base := idx * 4
  let bits := readU32LE b base
  (Float32.ofBits bits).toFloat

private def intToFloat (v : Int) : Float :=
  if v >= 0 then
    Float.ofNat v.toNat
  else
    -Float.ofNat (-v).toNat

private def readAsInt (dtype : DType) (b : ByteArray) (idx : Nat) : Int :=
  match dtype with
  | .float32 =>
    let v := readF32At b idx
    (Float.toInt64 v).toInt
  | .bool =>
    if b.get! idx == 0 then 0 else 1
  | .int8 =>
    let v := Int8.ofBitVec (BitVec.ofNat 8 (b.get! idx).toNat)
    v.toInt
  | .uint8 =>
    Int.ofNat (b.get! idx).toNat
  | .int16 =>
    let v := Int16.ofBitVec (BitVec.ofNat 16 (readU16LE b (idx * 2)).toNat)
    v.toInt
  | .uint16 =>
    Int.ofNat (UInt16.toNat (readU16LE b (idx * 2)))
  | .int32 =>
    let v := Int32.ofBitVec (BitVec.ofNat 32 (readU32LE b (idx * 4)).toNat)
    v.toInt
  | .uint32 =>
    Int.ofNat (UInt32.toNat (readU32LE b (idx * 4)))
  | .int64 | .index =>
    let v := Int64.ofBitVec (BitVec.ofNat 64 (readU64LE b (idx * 8)).toNat)
    v.toInt
  | .uint64 =>
    Int.ofNat (UInt64.toNat (readU64LE b (idx * 8)))
  | _ =>
    0

private def readAsFloat (dtype : DType) (b : ByteArray) (idx : Nat) : Float :=
  match dtype with
  | .float32 => readF32At b idx
  | _ => intToFloat (readAsInt dtype b idx)

private def pushBytes (out : ByteArray) (bytes : Array UInt8) : ByteArray := Id.run do
  let mut acc := out
  for b in bytes do
    acc := acc.push b
  return acc

private def repeatBytes (bytes : Array UInt8) (count : Nat) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity (bytes.size * count)
  for _ in [:count] do
    for b in bytes do
      out := out.push b
  return out

private def constIntArg (arg : UArg) : Int :=
  match arg with
  | .constInt v => v
  | .constFloat v => (Float.toInt64 v).toInt
  | .constF32Bits bits => (Float.toInt64 (Float32.ofBits bits).toFloat).toInt
  | .constBool v => if v then 1 else 0
  | _ => 0

private def constBytes (dtype : DType) (arg : UArg) (numel : Nat) : Option ByteArray :=
  let v := constIntArg arg
  let bytes : Option (Array UInt8) :=
    match dtype with
    | .int8 | .uint8 => some #[intToUInt8Mod v]
    | .int16 | .uint16 => some (bytesFromUInt16 (intToUInt16Mod v))
    | .int32 | .uint32 => some (bytesFromUInt32 (intToUInt32Mod v))
    | .int64 | .uint64 | .index => some (bytesFromUInt64 (intToUInt64Mod v))
    | _ => none
  bytes.map (fun b => repeatBytes b numel)

private def insertNatAsc (x : Nat) : List Nat → List Nat
  | [] => [x]
  | y :: ys => if x <= y then x :: y :: ys else y :: insertNatAsc x ys

private def sortNatAsc (xs : List Nat) : List Nat :=
  xs.foldl (fun acc x => insertNatAsc x acc) []

private def sortNatDesc (xs : List Nat) : List Nat :=
  (sortNatAsc xs).reverse

private def dedupNat (xs : List Nat) : List Nat := Id.run do
  let mut out : List Nat := []
  for x in xs do
    if !(out.contains x) then
      out := out ++ [x]
  return out

private partial def toposortMany (roots : List UOp) (fuel : Nat := 100000) : List UOp :=
  let (_, result) := roots.foldl (fun (visited, acc) r => go r visited acc fuel) (UOpIdSet.mkEmpty, [])
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

private def getSrc (cache : HashMap UOpId RawBuffer) (u : UOp) (idx : Nat) : RawBuffer :=
  let s := u.src[idx]!
  cache.getD s.uid (RawBuffer.zeros s.dtype (listProd s.shape))

private def numelOfArrayShape (shape : Array Nat) : Nat :=
  shape.foldl (fun acc d => acc * d) 1

private def evalFusedEwise (u : UOp) (plan : FusedEwise.Plan) (env : Env) (cache : HashMap UOpId RawBuffer) : RawBuffer :=
  if u.dtype != .float32 then
    RawBuffer.zeros u.dtype (listProd u.shape)
  else
    Id.run do
      let n := plan.leafBases.size
      let mut inputs : Array ByteArray := Array.emptyWithCapacity n
      for i in [:n] do
        let dtCode := plan.leafDtypes[i]!
        let dtype : DType := if dtCode == 1 then .bool else .float32
        let uid := plan.leafBases[i]!
        let fallback := env.getD uid (RawBuffer.zeros dtype 0)
        let buf := cache.getD uid fallback
        inputs := (dbgTraceIfShared "evalFusedEwise.inputs shared" inputs).push buf.data

      -- Dispatch based on specialized kernel (pattern detected in Lean, not C)
      let outBytes := match plan.kernel with
        -- Unary contiguous fast paths - direct call to simple kernels
        | .negContiguous => Native.negF32 inputs[0]!
        | .sqrtContiguous => Native.sqrtF32 inputs[0]!
        | .recipContiguous => Native.reciprocalF32 inputs[0]!
        | .exp2Contiguous => Native.exp2F32 inputs[0]!
        | .log2Contiguous => Native.log2F32 inputs[0]!
        | .sinContiguous => Native.sinF32 inputs[0]!
        | .cosContiguous => Native.cosF32 inputs[0]!
        | .tanContiguous => Native.tanF32 inputs[0]!
        -- Binary contiguous fast paths
        | .addContiguous => Native.addF32 inputs[0]! inputs[1]!
        | .subContiguous => Native.subF32 inputs[0]! inputs[1]!
        | .mulContiguous => Native.mulF32 inputs[0]! inputs[1]!
        | .divContiguous => Native.divF32 inputs[0]! inputs[1]!
        | .maxContiguous => Native.maxF32 inputs[0]! inputs[1]!
        | .powContiguous => Native.powF32 inputs[0]! inputs[1]!
        -- Fallback: bytecode interpreter
        | .bytecode =>
          if plan.fast then
            Native.fusedEwiseF32
              inputs
              plan.leafShapes
              plan.leafDtypes
              u.shape.toArray
              plan.prog
          else if plan.needsStack then
            Native.fusedEwiseViewStackF32
              inputs
              plan.leafStackShapes plan.leafStackStrides
              plan.leafStackOffsets
              plan.leafStackMaskStarts plan.leafStackMaskEnds
              plan.leafDtypes
              u.shape.toArray
              plan.prog
          else
            Native.fusedEwiseViewF32
              inputs
              plan.leafStrides plan.leafOffsets
              plan.leafMaskStarts plan.leafMaskEnds
              plan.leafDtypes
              u.shape.toArray
              plan.prog
      if outBytes.isEmpty then
        return RawBuffer.zeros u.dtype (listProd u.shape)
      return { dtype := .float32, data := outBytes }

private def evalFusedReduce (u : UOp) (plan : FusedReduce.Plan) (env : Env) (cache : HashMap UOpId RawBuffer) : RawBuffer :=
  if u.dtype != .float32 then
    RawBuffer.zeros u.dtype (listProd u.shape)
  else
    Id.run do
      let n := plan.ewise.leafBases.size
      let mut inputs : Array ByteArray := Array.emptyWithCapacity n
      for i in [:n] do
        let dtCode := plan.ewise.leafDtypes[i]!
        let dtype : DType := if dtCode == 1 then .bool else .float32
        let uid := plan.ewise.leafBases[i]!
        let fallback := env.getD uid (RawBuffer.zeros dtype 0)
        let buf := cache.getD uid fallback
        inputs := (dbgTraceIfShared "evalFusedReduce.inputs shared" inputs).push buf.data

      let outBytes :=
        let outNumel := listProd u.shape
        let fast := plan.ewise.fast
        if outNumel == 1 then
          match plan.reduceOp with
          | .ADD =>
            if fast then
              Native.fusedReduceSumAllF32
                inputs
                plan.ewise.leafShapes
                plan.ewise.leafDtypes
                plan.fullShape
                plan.ewise.prog
            else if plan.ewise.needsStack then
              Native.fusedReduceSumAllViewStackF32
                inputs
                plan.ewise.leafStackShapes plan.ewise.leafStackStrides
                plan.ewise.leafStackOffsets
                plan.ewise.leafStackMaskStarts plan.ewise.leafStackMaskEnds
                plan.ewise.leafDtypes
                plan.fullShape
                plan.ewise.prog
            else
              Native.fusedReduceSumAllViewF32
                inputs
                plan.ewise.leafStrides plan.ewise.leafOffsets
                plan.ewise.leafMaskStarts plan.ewise.leafMaskEnds
                plan.ewise.leafDtypes
                plan.fullShape
                plan.ewise.prog
          | .MAX =>
            if fast then
              Native.fusedReduceMaxAllF32
                inputs
                plan.ewise.leafShapes
                plan.ewise.leafDtypes
                plan.fullShape
                plan.ewise.prog
            else if plan.ewise.needsStack then
              Native.fusedReduceMaxAllViewStackF32
                inputs
                plan.ewise.leafStackShapes plan.ewise.leafStackStrides
                plan.ewise.leafStackOffsets
                plan.ewise.leafStackMaskStarts plan.ewise.leafStackMaskEnds
                plan.ewise.leafDtypes
                plan.fullShape
                plan.ewise.prog
            else
              Native.fusedReduceMaxAllViewF32
                inputs
                plan.ewise.leafStrides plan.ewise.leafOffsets
                plan.ewise.leafMaskStarts plan.ewise.leafMaskEnds
                plan.ewise.leafDtypes
                plan.fullShape
                plan.ewise.prog
          | _ => ByteArray.empty
        else
          let axes := plan.axes
          if axes.size == 1 then
            let axis := axes[0]!
            let isLast := axis + 1 == plan.fullShape.size
            match plan.reduceOp with
            | .ADD =>
              if fast then
                if isLast then
                  Native.fusedReduceSumLastF32
                    inputs
                    plan.ewise.leafShapes
                    plan.ewise.leafDtypes
                    plan.fullShape
                    plan.ewise.prog
                else
                  Native.fusedReduceSumAxisF32
                    inputs
                    plan.ewise.leafShapes
                    plan.ewise.leafDtypes
                    plan.fullShape
                    axis
                    plan.ewise.prog
              else if plan.ewise.needsStack then
                if isLast then
                  Native.fusedReduceSumLastViewStackF32
                    inputs
                    plan.ewise.leafStackShapes plan.ewise.leafStackStrides
                    plan.ewise.leafStackOffsets
                    plan.ewise.leafStackMaskStarts plan.ewise.leafStackMaskEnds
                    plan.ewise.leafDtypes
                    plan.fullShape
                    plan.ewise.prog
                else
                  Native.fusedReduceSumAxisViewStackF32
                    inputs
                    plan.ewise.leafStackShapes plan.ewise.leafStackStrides
                    plan.ewise.leafStackOffsets
                    plan.ewise.leafStackMaskStarts plan.ewise.leafStackMaskEnds
                    plan.ewise.leafDtypes
                    plan.fullShape
                    axis
                    plan.ewise.prog
              else
                if isLast then
                  Native.fusedReduceSumLastViewF32
                    inputs
                    plan.ewise.leafStrides plan.ewise.leafOffsets
                    plan.ewise.leafMaskStarts plan.ewise.leafMaskEnds
                    plan.ewise.leafDtypes
                    plan.fullShape
                    plan.ewise.prog
                else
                  Native.fusedReduceSumAxisViewF32
                    inputs
                    plan.ewise.leafStrides plan.ewise.leafOffsets
                    plan.ewise.leafMaskStarts plan.ewise.leafMaskEnds
                    plan.ewise.leafDtypes
                    plan.fullShape
                    axis
                    plan.ewise.prog
            | .MAX =>
              if fast then
                if isLast then
                  Native.fusedReduceMaxLastF32
                    inputs
                    plan.ewise.leafShapes
                    plan.ewise.leafDtypes
                    plan.fullShape
                    plan.ewise.prog
                else
                  Native.fusedReduceMaxAxisF32
                    inputs
                    plan.ewise.leafShapes
                    plan.ewise.leafDtypes
                    plan.fullShape
                    axis
                    plan.ewise.prog
              else if plan.ewise.needsStack then
                if isLast then
                  Native.fusedReduceMaxLastViewStackF32
                    inputs
                    plan.ewise.leafStackShapes plan.ewise.leafStackStrides
                    plan.ewise.leafStackOffsets
                    plan.ewise.leafStackMaskStarts plan.ewise.leafStackMaskEnds
                    plan.ewise.leafDtypes
                    plan.fullShape
                    plan.ewise.prog
                else
                  Native.fusedReduceMaxAxisViewStackF32
                    inputs
                    plan.ewise.leafStackShapes plan.ewise.leafStackStrides
                    plan.ewise.leafStackOffsets
                    plan.ewise.leafStackMaskStarts plan.ewise.leafStackMaskEnds
                    plan.ewise.leafDtypes
                    plan.fullShape
                    axis
                    plan.ewise.prog
              else
                if isLast then
                  Native.fusedReduceMaxLastViewF32
                    inputs
                    plan.ewise.leafStrides plan.ewise.leafOffsets
                    plan.ewise.leafMaskStarts plan.ewise.leafMaskEnds
                    plan.ewise.leafDtypes
                    plan.fullShape
                    plan.ewise.prog
                else
                  Native.fusedReduceMaxAxisViewF32
                    inputs
                    plan.ewise.leafStrides plan.ewise.leafOffsets
                    plan.ewise.leafMaskStarts plan.ewise.leafMaskEnds
                    plan.ewise.leafDtypes
                    plan.fullShape
                    axis
                    plan.ewise.prog
            | _ => ByteArray.empty
          else
            let outShape := u.shape.toArray
            match plan.reduceOp with
            | .ADD =>
              if fast then
                Native.fusedReduceSumAxesF32
                  inputs
                  plan.ewise.leafShapes
                  plan.ewise.leafDtypes
                  plan.fullShape
                  outShape
                  axes
                  plan.ewise.prog
              else if plan.ewise.needsStack then
                Native.fusedReduceSumAxesViewStackF32
                  inputs
                  plan.ewise.leafStackShapes plan.ewise.leafStackStrides
                  plan.ewise.leafStackOffsets
                  plan.ewise.leafStackMaskStarts plan.ewise.leafStackMaskEnds
                  plan.ewise.leafDtypes
                  plan.fullShape
                  outShape
                  axes
                  plan.ewise.prog
              else
                Native.fusedReduceSumAxesViewF32
                  inputs
                  plan.ewise.leafStrides plan.ewise.leafOffsets
                  plan.ewise.leafMaskStarts plan.ewise.leafMaskEnds
                  plan.ewise.leafDtypes
                  plan.fullShape
                  outShape
                  axes
                  plan.ewise.prog
            | .MAX =>
              if fast then
                Native.fusedReduceMaxAxesF32
                  inputs
                  plan.ewise.leafShapes
                  plan.ewise.leafDtypes
                  plan.fullShape
                  outShape
                  axes
                  plan.ewise.prog
              else if plan.ewise.needsStack then
                Native.fusedReduceMaxAxesViewStackF32
                  inputs
                  plan.ewise.leafStackShapes plan.ewise.leafStackStrides
                  plan.ewise.leafStackOffsets
                  plan.ewise.leafStackMaskStarts plan.ewise.leafStackMaskEnds
                  plan.ewise.leafDtypes
                  plan.fullShape
                  outShape
                  axes
                  plan.ewise.prog
              else
                Native.fusedReduceMaxAxesViewF32
                  inputs
                  plan.ewise.leafStrides plan.ewise.leafOffsets
                  plan.ewise.leafMaskStarts plan.ewise.leafMaskEnds
                  plan.ewise.leafDtypes
                  plan.fullShape
                  outShape
                  axes
                  plan.ewise.prog
            | _ => ByteArray.empty

      if outBytes.isEmpty then
        return RawBuffer.zeros u.dtype (listProd u.shape)
      return { dtype := .float32, data := outBytes }

  private def evalFusedContract (u : UOp) (plan : FusedContract.Plan) (env : Env) (cache : HashMap UOpId RawBuffer) : RawBuffer :=
    if u.dtype != .float32 then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else
      Id.run do
        let aFallback := env.getD plan.aBase (RawBuffer.zeros .float32 plan.aNumel)
        let bFallback := env.getD plan.bBase (RawBuffer.zeros .float32 plan.bNumel)
        let aBuf := cache.getD plan.aBase aFallback
        let bBuf := cache.getD plan.bBase bFallback

        let outBytes :=
          if plan.needsStack then
            Native.matmulViewStackF32
              aBuf.data bBuf.data
              plan.aStackShapes plan.aStackStrides
              plan.aStackOffsets
              plan.aStackMaskStarts plan.aStackMaskEnds
              plan.bStackShapes plan.bStackStrides
              plan.bStackOffsets
              plan.bStackMaskStarts plan.bStackMaskEnds
              u.shape.toArray
              plan.k
          else
            Native.matmulViewF32
              aBuf.data bBuf.data
              plan.aStrides plan.aOffset plan.aMaskStarts plan.aMaskEnds
              plan.bStrides plan.bOffset plan.bMaskStarts plan.bMaskEnds
              u.shape.toArray
              plan.k
        if outBytes.isEmpty then
          return RawBuffer.zeros u.dtype (listProd u.shape)
        return { dtype := .float32, data := outBytes }

  private def evalFusedSGD (u : UOp) (plan : FusedSGD.Plan) (env : Env) (cache : HashMap UOpId RawBuffer) : RawBuffer :=
    if u.dtype != .float32 then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else
      Id.run do
        let n := listProd u.shape
        let wFallback := env.getD plan.w (RawBuffer.zeros .float32 n)
        let gFallback := env.getD plan.grad (RawBuffer.zeros .float32 n)
        let wBuf := cache.getD plan.w wFallback
        let gBuf := cache.getD plan.grad gFallback

        let outBytes := Native.sgdUpdateF32 wBuf.data gBuf.data plan.lr
        if outBytes.isEmpty then
          return RawBuffer.zeros u.dtype n
        return { dtype := .float32, data := outBytes }

  private def evalFusedMatmulBias (u : UOp) (plan : FusedMatmul.Plan) (env : Env) (cache : HashMap UOpId RawBuffer) : RawBuffer :=
    if u.dtype != .float32 then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else
      Id.run do
        let aFallback := env.getD plan.aBase (RawBuffer.zeros .float32 0)
        let bFallback := env.getD plan.bBase (RawBuffer.zeros .float32 0)
        let biasFallback := env.getD plan.bias (RawBuffer.zeros .float32 plan.biasNumel)
        let aBuf := cache.getD plan.aBase aFallback
        let bBuf := cache.getD plan.bBase bFallback
        let biasBuf := cache.getD plan.bias biasFallback

        if plan.needsStack then
          let outShape := u.shape.toArray
          let outBytes0 :=
            Native.matmulViewStackF32
              aBuf.data bBuf.data
              plan.aStackShapes plan.aStackStrides
              plan.aStackOffsets
              plan.aStackMaskStarts plan.aStackMaskEnds
              plan.bStackShapes plan.bStackStrides
              plan.bStackOffsets
              plan.bStackMaskStarts plan.bStackMaskEnds
              outShape
              plan.k
          if outBytes0.isEmpty then
            return RawBuffer.zeros u.dtype (listProd u.shape)
          let mut outBytes := outBytes0
          match plan.scaleBits with
          | some bits =>
            let scale := Native.fullF32Bits 1 bits
            outBytes := Native.mulBcastF32 outBytes scale outShape #[] outShape
          | none => ()
          outBytes := Native.addBcastF32 outBytes biasBuf.data outShape plan.biasShape outShape
          match plan.bias2 with
          | some bias2 =>
            let bias2Fallback := env.getD bias2 (RawBuffer.zeros .float32 plan.bias2Numel)
            let bias2Buf := cache.getD bias2 bias2Fallback
            outBytes := Native.addBcastF32 outBytes bias2Buf.data outShape plan.bias2Shape outShape
          | none => ()
          if plan.relu then
            let zero := Native.fullF32Bits 1 0
            outBytes := Native.maxBcastF32 outBytes zero outShape #[] outShape
          return { dtype := .float32, data := outBytes }

        let isBatched := !plan.aStarts.isEmpty
        let fast :=
          !plan.needsStack &&
          plan.aFast && plan.bFast && plan.biasFast &&
          match plan.bias2 with
          | some _ => plan.bias2Fast
          | none => true
        let outBytes :=
          if fast then
            match plan.bias2 with
            | none =>
              match plan.scaleBits with
              | some scaleBits =>
                if isBatched then
                  if plan.relu then
                    Native.matmulBatchedBiasScaleReluF32
                      aBuf.data bBuf.data biasBuf.data plan.biasShape2d
                      plan.aStarts plan.bStarts plan.biasStarts
                      plan.m plan.k plan.n
                      scaleBits
                  else
                    Native.matmulBatchedBiasScaleF32
                      aBuf.data bBuf.data biasBuf.data plan.biasShape2d
                      plan.aStarts plan.bStarts plan.biasStarts
                      plan.m plan.k plan.n
                      scaleBits
                else if plan.relu then
                  Native.matmulBiasScaleReluF32 aBuf.data bBuf.data biasBuf.data plan.biasShape2d plan.m plan.k plan.n scaleBits
                else
                  Native.matmulBiasScaleF32 aBuf.data bBuf.data biasBuf.data plan.biasShape2d plan.m plan.k plan.n scaleBits
              | none =>
                if isBatched then
                  if plan.relu then
                    Native.matmulBatchedBiasReluF32
                      aBuf.data bBuf.data biasBuf.data plan.biasShape2d
                      plan.aStarts plan.bStarts plan.biasStarts
                      plan.m plan.k plan.n
                  else
                    Native.matmulBatchedBiasF32
                      aBuf.data bBuf.data biasBuf.data plan.biasShape2d
                      plan.aStarts plan.bStarts plan.biasStarts
                      plan.m plan.k plan.n
                else if plan.relu then
                  Native.matmulBiasReluF32 aBuf.data bBuf.data biasBuf.data plan.biasShape2d plan.m plan.k plan.n
                else
                  Native.matmulBiasF32 aBuf.data bBuf.data biasBuf.data plan.biasShape2d plan.m plan.k plan.n
            | some bias2 =>
              let bias2Fallback := env.getD bias2 (RawBuffer.zeros .float32 plan.bias2Numel)
              let bias2Buf := cache.getD bias2 bias2Fallback
              match plan.scaleBits with
              | some scaleBits =>
                if isBatched then
                  if plan.relu then
                    Native.matmulBatchedBias2ScaleReluF32
                      aBuf.data bBuf.data
                      biasBuf.data plan.biasShape2d
                      bias2Buf.data plan.bias2Shape2d
                      plan.aStarts plan.bStarts plan.biasStarts plan.bias2Starts
                      plan.m plan.k plan.n
                      scaleBits
                  else
                    Native.matmulBatchedBias2ScaleF32
                      aBuf.data bBuf.data
                      biasBuf.data plan.biasShape2d
                      bias2Buf.data plan.bias2Shape2d
                      plan.aStarts plan.bStarts plan.biasStarts plan.bias2Starts
                      plan.m plan.k plan.n
                      scaleBits
                else if plan.relu then
                  Native.matmulBias2ScaleReluF32
                    aBuf.data bBuf.data
                    biasBuf.data plan.biasShape2d
                    bias2Buf.data plan.bias2Shape2d
                    plan.m plan.k plan.n
                    scaleBits
                else
                  Native.matmulBias2ScaleF32
                    aBuf.data bBuf.data
                    biasBuf.data plan.biasShape2d
                    bias2Buf.data plan.bias2Shape2d
                    plan.m plan.k plan.n
                    scaleBits
              | none =>
                if isBatched then
                  if plan.relu then
                    Native.matmulBatchedBias2ReluF32
                      aBuf.data bBuf.data
                      biasBuf.data plan.biasShape2d
                      bias2Buf.data plan.bias2Shape2d
                      plan.aStarts plan.bStarts plan.biasStarts plan.bias2Starts
                      plan.m plan.k plan.n
                  else
                    Native.matmulBatchedBias2F32
                      aBuf.data bBuf.data
                      biasBuf.data plan.biasShape2d
                      bias2Buf.data plan.bias2Shape2d
                      plan.aStarts plan.bStarts plan.biasStarts plan.bias2Starts
                      plan.m plan.k plan.n
                else if plan.relu then
                  Native.matmulBias2ReluF32 aBuf.data bBuf.data biasBuf.data plan.biasShape2d bias2Buf.data plan.bias2Shape2d plan.m plan.k plan.n
                else
                  Native.matmulBias2F32 aBuf.data bBuf.data biasBuf.data plan.biasShape2d bias2Buf.data plan.bias2Shape2d plan.m plan.k plan.n
          else
            match plan.bias2 with
            | none =>
              match plan.scaleBits with
              | some scaleBits =>
                if plan.relu then
                  Native.matmulViewBiasScaleReluF32
                    aBuf.data bBuf.data biasBuf.data
                    plan.aStrides plan.aOffset plan.aMaskStarts plan.aMaskEnds
                    plan.bStrides plan.bOffset plan.bMaskStarts plan.bMaskEnds
                    plan.biasStrides plan.biasOffset plan.biasMaskStarts plan.biasMaskEnds
                    u.shape.toArray
                    plan.k
                    scaleBits
                else
                  Native.matmulViewBiasScaleF32
                    aBuf.data bBuf.data biasBuf.data
                    plan.aStrides plan.aOffset plan.aMaskStarts plan.aMaskEnds
                    plan.bStrides plan.bOffset plan.bMaskStarts plan.bMaskEnds
                    plan.biasStrides plan.biasOffset plan.biasMaskStarts plan.biasMaskEnds
                    u.shape.toArray
                    plan.k
                    scaleBits
              | none =>
                if plan.relu then
                  Native.matmulViewBiasReluF32
                    aBuf.data bBuf.data biasBuf.data
                    plan.aStrides plan.aOffset plan.aMaskStarts plan.aMaskEnds
                    plan.bStrides plan.bOffset plan.bMaskStarts plan.bMaskEnds
                    plan.biasStrides plan.biasOffset plan.biasMaskStarts plan.biasMaskEnds
                    u.shape.toArray
                    plan.k
                else
                  Native.matmulViewBiasF32
                    aBuf.data bBuf.data biasBuf.data
                    plan.aStrides plan.aOffset plan.aMaskStarts plan.aMaskEnds
                    plan.bStrides plan.bOffset plan.bMaskStarts plan.bMaskEnds
                    plan.biasStrides plan.biasOffset plan.biasMaskStarts plan.biasMaskEnds
                    u.shape.toArray
                    plan.k
            | some bias2 =>
              let bias2Fallback := env.getD bias2 (RawBuffer.zeros .float32 plan.bias2Numel)
              let bias2Buf := cache.getD bias2 bias2Fallback
              match plan.scaleBits with
              | some scaleBits =>
                if plan.relu then
                  Native.matmulViewBias2ScaleReluF32
                    aBuf.data bBuf.data biasBuf.data bias2Buf.data
                    plan.aStrides plan.aOffset plan.aMaskStarts plan.aMaskEnds
                    plan.bStrides plan.bOffset plan.bMaskStarts plan.bMaskEnds
                    plan.biasStrides plan.biasOffset plan.biasMaskStarts plan.biasMaskEnds
                    plan.bias2Strides plan.bias2Offset plan.bias2MaskStarts plan.bias2MaskEnds
                    u.shape.toArray
                    plan.k
                    scaleBits
                else
                  Native.matmulViewBias2ScaleF32
                    aBuf.data bBuf.data biasBuf.data bias2Buf.data
                    plan.aStrides plan.aOffset plan.aMaskStarts plan.aMaskEnds
                    plan.bStrides plan.bOffset plan.bMaskStarts plan.bMaskEnds
                    plan.biasStrides plan.biasOffset plan.biasMaskStarts plan.biasMaskEnds
                    plan.bias2Strides plan.bias2Offset plan.bias2MaskStarts plan.bias2MaskEnds
                    u.shape.toArray
                    plan.k
                    scaleBits
              | none =>
                if plan.relu then
                  Native.matmulViewBias2ReluF32
                    aBuf.data bBuf.data biasBuf.data bias2Buf.data
                    plan.aStrides plan.aOffset plan.aMaskStarts plan.aMaskEnds
                    plan.bStrides plan.bOffset plan.bMaskStarts plan.bMaskEnds
                    plan.biasStrides plan.biasOffset plan.biasMaskStarts plan.biasMaskEnds
                    plan.bias2Strides plan.bias2Offset plan.bias2MaskStarts plan.bias2MaskEnds
                    u.shape.toArray
                    plan.k
                else
                  Native.matmulViewBias2F32
                    aBuf.data bBuf.data biasBuf.data bias2Buf.data
                    plan.aStrides plan.aOffset plan.aMaskStarts plan.aMaskEnds
                    plan.bStrides plan.bOffset plan.bMaskStarts plan.bMaskEnds
                    plan.biasStrides plan.biasOffset plan.biasMaskStarts plan.biasMaskEnds
                    plan.bias2Strides plan.bias2Offset plan.bias2MaskStarts plan.bias2MaskEnds
                    u.shape.toArray
                    plan.k
        if outBytes.isEmpty then
          return RawBuffer.zeros u.dtype (listProd u.shape)
        return { dtype := .float32, data := outBytes }

  private def evalFusedSoftmax (u : UOp) (plan : FusedSoftmax.Plan) (env : Env) (cache : HashMap UOpId RawBuffer) : RawBuffer :=
    if u.dtype != .float32 then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else
      Id.run do
        let inNumel := plan.outer * plan.inner
        let aFallback := env.getD plan.input (RawBuffer.zeros .float32 inNumel)
        let aBuf := cache.getD plan.input aFallback
        let outBytes :=
          if plan.log then
            Native.logSoftmaxLastF32 aBuf.data plan.outer plan.inner plan.scaleBits plan.ln2Bits
          else
            Native.softmaxLastF32 aBuf.data plan.outer plan.inner plan.scaleBits
        if outBytes.isEmpty then
          return RawBuffer.zeros u.dtype (listProd u.shape)
        return { dtype := .float32, data := outBytes }

private def isZeroConst (u : UOp) : Bool :=
  u.op == .CONST && u.shape == [] && u.dtype == .float32 &&
    match u.arg with
    | .constF32Bits bits => bits == 0
    | .constFloat v => v == 0.0
    | _ => false

private def packF32BitsArray (bits : Array UInt32) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity (bits.size * 4)
  for v in bits do
    let b0 : UInt8 := UInt8.ofNat (UInt32.toNat (v &&& 0xFF))
    let b1 : UInt8 := UInt8.ofNat (UInt32.toNat ((v >>> 8) &&& 0xFF))
    let b2 : UInt8 := UInt8.ofNat (UInt32.toNat ((v >>> 16) &&& 0xFF))
    let b3 : UInt8 := UInt8.ofNat (UInt32.toNat ((v >>> 24) &&& 0xFF))
    out := out.push b0
    out := out.push b1
    out := out.push b2
    out := out.push b3
  return out

private def cmpF32 (x y : RawBuffer) (xShape yShape outShape : Shape) (pred : Float → Float → Bool) : ByteArray := Id.run do
  let xData :=
    if xShape == outShape then
      x.data
    else
      Native.expandBcastF32 x.data xShape.toArray outShape.toArray
  let yData :=
    if yShape == outShape then
      y.data
    else
      Native.expandBcastF32 y.data yShape.toArray outShape.toArray
  let xs := RawBuffer.unpackF32Bytes xData
  let ys := RawBuffer.unpackF32Bytes yData
  let n := listProd outShape
  let mut out := ByteArray.emptyWithCapacity n
  for i in [:n] do
    let v : UInt8 := if pred xs[i]! ys[i]! then 1 else 0
    out := out.push v
  return out

private def cmpU8 (x y : RawBuffer) (xShape yShape outShape : Shape) (pred : UInt8 → UInt8 → Bool) : ByteArray := Id.run do
  let xData :=
    if xShape == outShape then
      x.data
    else
      Native.expandBcastU8 x.data xShape.toArray outShape.toArray
  let yData :=
    if yShape == outShape then
      y.data
    else
      Native.expandBcastU8 y.data yShape.toArray outShape.toArray
  let n := listProd outShape
  let mut out := ByteArray.emptyWithCapacity n
  for i in [:n] do
    let v : UInt8 := if pred (xData.get! i) (yData.get! i) then 1 else 0
    out := out.push v
  return out

private def bitwiseU8 (x y : RawBuffer) (xShape yShape outShape : Shape) (op : UInt8 → UInt8 → UInt8) : ByteArray := Id.run do
  let xData :=
    if xShape == outShape then
      x.data
    else
      Native.expandBcastU8 x.data xShape.toArray outShape.toArray
  let yData :=
    if yShape == outShape then
      y.data
    else
      Native.expandBcastU8 y.data yShape.toArray outShape.toArray
  let n := listProd outShape
  let mut out := ByteArray.emptyWithCapacity n
  for i in [:n] do
    out := out.push (op (xData.get! i) (yData.get! i))
  return out

private def evalNode (u : UOp) (env : Env) (cache : HashMap UOpId RawBuffer) : RawBuffer :=
  match u.op with
  | .CONST =>
    let numel := listProd u.shape
    match u.dtype with
    | .float32 =>
      match u.arg with
      | .constF32Bits bits =>
        { dtype := .float32, data := Native.fullF32Bits numel bits }
      | .constFloat v =>
        { dtype := .float32, data := Native.fullF32 numel v }
      | _ =>
        RawBuffer.zeros u.dtype numel
    | .bool =>
      match u.arg with
      | .constBool v =>
        { dtype := .bool, data := ByteArray.mk (Array.replicate numel (if v then 1 else 0)) }
      | _ =>
        let v := u.arg.getFloat.getD 0.0
        { dtype := .bool, data := ByteArray.mk (Array.replicate numel (if v == 0.0 then 0 else 1)) }
    | .int8 | .uint8 | .int16 | .uint16 | .int32 | .uint32 | .int64 | .uint64 | .index =>
      match constBytes u.dtype u.arg numel with
      | some data => RawBuffer.mk u.dtype data
      | none => RawBuffer.zeros u.dtype numel
    | _ =>
      RawBuffer.zeros u.dtype numel

  | .VCONST =>
    match u.dtype, u.arg with
    | .float32, .constF32Array bits =>
      { dtype := .float32, data := packF32BitsArray bits }
    | dtype, .constBytesArray bytes =>
      -- Raw bytes with dtype - just use directly
      { dtype := dtype, data := bytes }
    | _, _ =>
      RawBuffer.zeros u.dtype (listProd u.shape)

  | .BUFFER =>
    env.getD u.uid (RawBuffer.zeros u.dtype (listProd u.shape))

  | .NEG =>
    let x := getSrc cache u 0
    match u.dtype with
    | .float32 => { dtype := .float32, data := Native.negF32 x.data }
    | _ => RawBuffer.zeros u.dtype (listProd u.shape)

  | .SQRT =>
    let x := getSrc cache u 0
    match u.dtype with
    | .float32 => { dtype := .float32, data := Native.sqrtF32 x.data }
    | _ => RawBuffer.zeros u.dtype (listProd u.shape)

  | .RECIPROCAL =>
    let x := getSrc cache u 0
    match u.dtype with
    | .float32 => { dtype := .float32, data := Native.reciprocalF32 x.data }
    | _ => RawBuffer.zeros u.dtype (listProd u.shape)

  | .EXP2 =>
    let x := getSrc cache u 0
    match u.dtype with
    | .float32 => { dtype := .float32, data := Native.exp2F32 x.data }
    | _ => RawBuffer.zeros u.dtype (listProd u.shape)

  | .LOG2 =>
    let x := getSrc cache u 0
    match u.dtype with
    | .float32 => { dtype := .float32, data := Native.log2F32 x.data }
    | _ => RawBuffer.zeros u.dtype (listProd u.shape)

  | .SIN =>
    let x := getSrc cache u 0
    match u.dtype with
    | .float32 => { dtype := .float32, data := Native.sinF32 x.data }
    | _ => RawBuffer.zeros u.dtype (listProd u.shape)

  | .COS =>
    let x := getSrc cache u 0
    match u.dtype with
    | .float32 => { dtype := .float32, data := Native.cosF32 x.data }
    | _ => RawBuffer.zeros u.dtype (listProd u.shape)

  | .TAN =>
    let x := getSrc cache u 0
    match u.dtype with
    | .float32 => { dtype := .float32, data := Native.tanF32 x.data }
    | _ => RawBuffer.zeros u.dtype (listProd u.shape)

  | .ADD =>
    let x := getSrc cache u 0
    let y := getSrc cache u 1
    if u.dtype != .float32 || x.dtype != .float32 || y.dtype != .float32 then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else
      let xSh := u.src[0]!.shape
      let ySh := u.src[1]!.shape
      if xSh == u.shape && ySh == u.shape then
        { dtype := .float32, data := Native.addF32 x.data y.data }
      else
        { dtype := .float32, data := Native.addBcastF32 x.data y.data xSh.toArray ySh.toArray u.shape.toArray }
  | .MUL =>
    let x := getSrc cache u 0
    let y := getSrc cache u 1
    if u.dtype != .float32 || x.dtype != .float32 || y.dtype != .float32 then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else
      let xSh := u.src[0]!.shape
      let ySh := u.src[1]!.shape
      if xSh == u.shape && ySh == u.shape then
        { dtype := .float32, data := Native.mulF32 x.data y.data }
      else
        { dtype := .float32, data := Native.mulBcastF32 x.data y.data xSh.toArray ySh.toArray u.shape.toArray }
  | .SUB =>
    let x := getSrc cache u 0
    let y := getSrc cache u 1
    if u.dtype != .float32 || x.dtype != .float32 || y.dtype != .float32 then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else
      let xSh := u.src[0]!.shape
      let ySh := u.src[1]!.shape
      if xSh == u.shape && ySh == u.shape then
        { dtype := .float32, data := Native.subF32 x.data y.data }
      else
        { dtype := .float32, data := Native.subBcastF32 x.data y.data xSh.toArray ySh.toArray u.shape.toArray }
  | .FDIV =>
    let x := getSrc cache u 0
    let y := getSrc cache u 1
    if u.dtype != .float32 || x.dtype != .float32 || y.dtype != .float32 then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else
      let xSh := u.src[0]!.shape
      let ySh := u.src[1]!.shape
      if xSh == u.shape && ySh == u.shape then
        { dtype := .float32, data := Native.divF32 x.data y.data }
      else
        { dtype := .float32, data := Native.divBcastF32 x.data y.data xSh.toArray ySh.toArray u.shape.toArray }

  | .MAX =>
    let x := getSrc cache u 0
    let y := getSrc cache u 1
    if u.dtype != .float32 || x.dtype != .float32 || y.dtype != .float32 then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else
      let xU := u.src[0]!
      let yU := u.src[1]!
      let xSh := xU.shape
      let ySh := yU.shape
      if isZeroConst xU && ySh == u.shape then
        { dtype := .float32, data := Native.reluF32 y.data }
      else if isZeroConst yU && xSh == u.shape then
        { dtype := .float32, data := Native.reluF32 x.data }
      else if xSh == u.shape && ySh == u.shape then
        { dtype := .float32, data := Native.maxF32 x.data y.data }
      else
        { dtype := .float32, data := Native.maxBcastF32 x.data y.data xSh.toArray ySh.toArray u.shape.toArray }

  | .POW =>
    let x := getSrc cache u 0
    let y := getSrc cache u 1
    if u.dtype != .float32 || x.dtype != .float32 || y.dtype != .float32 then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else
      let xSh := u.src[0]!.shape
      let ySh := u.src[1]!.shape
      if xSh == u.shape && ySh == u.shape then
        { dtype := .float32, data := Native.powF32 x.data y.data }
      else
        { dtype := .float32, data := Native.powBcastF32 x.data y.data xSh.toArray ySh.toArray u.shape.toArray }

  | .CMPLT =>
    let x := getSrc cache u 0
    let y := getSrc cache u 1
    if u.dtype != .bool || x.dtype != .float32 || y.dtype != .float32 then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else
      let xSh := u.src[0]!.shape
      let ySh := u.src[1]!.shape
      if xSh == u.shape && ySh == u.shape then
        { dtype := .bool, data := Native.cmpltF32 x.data y.data }
      else
        { dtype := .bool, data := Native.cmpltBcastF32 x.data y.data xSh.toArray ySh.toArray u.shape.toArray }

  | .CMPEQ =>
    let x := getSrc cache u 0
    let y := getSrc cache u 1
    if u.dtype != .bool then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else
      let xSh := u.src[0]!.shape
      let ySh := u.src[1]!.shape
      match x.dtype, y.dtype with
      | .float32, .float32 =>
        { dtype := .bool, data := cmpF32 x y xSh ySh u.shape (fun a b => a == b) }
      | .bool, .bool =>
        { dtype := .bool, data := cmpU8 x y xSh ySh u.shape (fun a b => a == b) }
      | _, _ =>
        RawBuffer.zeros u.dtype (listProd u.shape)

  | .CMPNE =>
    let x := getSrc cache u 0
    let y := getSrc cache u 1
    if u.dtype != .bool then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else
      let xSh := u.src[0]!.shape
      let ySh := u.src[1]!.shape
      match x.dtype, y.dtype with
      | .float32, .float32 =>
        { dtype := .bool, data := cmpF32 x y xSh ySh u.shape (fun a b => a != b) }
      | .bool, .bool =>
        { dtype := .bool, data := cmpU8 x y xSh ySh u.shape (fun a b => a != b) }
      | _, _ =>
        RawBuffer.zeros u.dtype (listProd u.shape)

  | .AND =>
    let x := getSrc cache u 0
    let y := getSrc cache u 1
    if u.dtype != .bool || x.dtype != .bool || y.dtype != .bool then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else
      let xSh := u.src[0]!.shape
      let ySh := u.src[1]!.shape
      let op := fun a b => if (a != 0) && (b != 0) then 1 else 0
      { dtype := .bool, data := bitwiseU8 x y xSh ySh u.shape op }

  | .OR =>
    let x := getSrc cache u 0
    let y := getSrc cache u 1
    if u.dtype != .bool || x.dtype != .bool || y.dtype != .bool then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else
      let xSh := u.src[0]!.shape
      let ySh := u.src[1]!.shape
      let op := fun a b => if (a != 0) || (b != 0) then 1 else 0
      { dtype := .bool, data := bitwiseU8 x y xSh ySh u.shape op }

  | .XOR =>
    let x := getSrc cache u 0
    let y := getSrc cache u 1
    if u.dtype != .bool || x.dtype != .bool || y.dtype != .bool then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else
      let xSh := u.src[0]!.shape
      let ySh := u.src[1]!.shape
      let op := fun a b => if (a != 0) != (b != 0) then 1 else 0
      { dtype := .bool, data := bitwiseU8 x y xSh ySh u.shape op }

  | .WHERE =>
    let cond := getSrc cache u 0
    let x := getSrc cache u 1
    let y := getSrc cache u 2
    if cond.dtype != .bool || x.dtype != u.dtype || y.dtype != u.dtype then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else
      let cSh := u.src[0]!.shape
      let xSh := u.src[1]!.shape
      let ySh := u.src[2]!.shape
      match u.dtype with
      | .float32 =>
        if cSh == u.shape && xSh == u.shape && ySh == u.shape then
          { dtype := .float32, data := Native.whereF32 cond.data x.data y.data (listProd u.shape) }
        else
          { dtype := .float32
            data := Native.whereBcastF32 cond.data x.data y.data cSh.toArray xSh.toArray ySh.toArray u.shape.toArray }
      | _ =>
        let elemSize := u.dtype.itemsize
        if elemSize == 0 then
          RawBuffer.zeros u.dtype (listProd u.shape)
        else
          let out := whereBytes cond.data x.data y.data cSh xSh ySh u.shape elemSize
          if out.isEmpty && listProd u.shape != 0 then
            RawBuffer.zeros u.dtype (listProd u.shape)
          else
            RawBuffer.mk u.dtype out

  | .CONTRACT =>
    let a := getSrc cache u 0
    let b := getSrc cache u 1
    let aSh := u.src[0]!.shape
    let bSh := u.src[1]!.shape
    let rA := aSh.length
    let rB := bSh.length
    if u.dtype != .float32 || a.dtype != .float32 || b.dtype != .float32 then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else if rA < 2 || rB < 2 then
      RawBuffer.zeros u.dtype (listProd u.shape)
    else
      let m := listGetD aSh (rA - 2) 0
      let k := listGetD aSh (rA - 1) 0
      let k2 := listGetD bSh (rB - 2) 0
      let n := listGetD bSh (rB - 1) 0
      if k != k2 then
        RawBuffer.zeros u.dtype (listProd u.shape)
      else if rA == 2 && rB == 2 then
        -- GPU matmul (Metal) - falls back to zeros on GPU error
        { dtype := .float32, data := Metal.metalMatmulSync a.data b.data m k n }
      else
        let batchA := aSh.take (rA - 2)
        let batchB := bSh.take (rB - 2)
        match Shape.broadcast batchA batchB with
        | none => RawBuffer.zeros u.dtype (listProd u.shape)
        | some batchOut =>
          let batchNumel := listProd batchOut
          let lenBatch := batchOut.length
          let batchA' := List.replicate (lenBatch - batchA.length) 1 ++ batchA
          let batchB' := List.replicate (lenBatch - batchB.length) 1 ++ batchB
          let stridesA := stridesOf aSh
          let stridesB := stridesOf bSh
          let batchStridesA := List.replicate (lenBatch - batchA.length) 0 ++ stridesA.take batchA.length
          let batchStridesB := List.replicate (lenBatch - batchB.length) 0 ++ stridesB.take batchB.length
          Id.run do
            let mut aStarts : Array Nat := Array.emptyWithCapacity batchNumel
            let mut bStarts : Array Nat := Array.emptyWithCapacity batchNumel
            for bi in [:batchNumel] do
              let batchIdx := unflattenIndex bi batchOut
              let aBatchIdx := listZipWith (fun idx dim => if dim == 1 then 0 else idx) batchIdx batchA'
              let bBatchIdx := listZipWith (fun idx dim => if dim == 1 then 0 else idx) batchIdx batchB'
              let baseA := offsetOf aBatchIdx batchStridesA
              let baseB := offsetOf bBatchIdx batchStridesB
              let aStart := baseA * 4
              let bStart := baseB * 4
              aStarts := aStarts.push aStart
              bStarts := bStarts.push bStart
            return { dtype := .float32, data := Native.matmulBatchedF32 a.data b.data aStarts bStarts m k n }

  | .RESHAPE => getSrc cache u 0

  | .EXPAND =>
    let src := getSrc cache u 0
    match u.dtype with
    | .float32 =>
      if u.src[0]!.shape == u.shape then
        src
      else
        { dtype := .float32, data := Native.expandBcastF32 src.data u.src[0]!.shape.toArray u.shape.toArray }
    | .bool =>
      if u.src[0]!.shape == u.shape then
        src
      else
        { dtype := .bool, data := Native.expandBcastU8 src.data u.src[0]!.shape.toArray u.shape.toArray }
    | _ =>
      if u.src[0]!.shape == u.shape then
        src
      else
        let outNumel := listProd u.shape
        let out := broadcastBytes src.data u.src[0]!.shape u.shape u.dtype.itemsize
        if out.isEmpty && outNumel != 0 then
          RawBuffer.zeros u.dtype outNumel
        else
          RawBuffer.mk u.dtype out

  | .PERMUTE =>
    let src := getSrc cache u 0
    let perm := u.arg.getPermutation.getD []
    match u.dtype with
    | .float32 =>
      if perm == [1, 0] && u.src[0]!.shape.length == 2 then
        let m := listGetD u.src[0]!.shape 0 0
        let n := listGetD u.src[0]!.shape 1 0
        { dtype := .float32, data := Native.transpose2dF32 src.data m n }
      else
        { dtype := .float32, data := Native.permuteF32 src.data u.src[0]!.shape.toArray perm.toArray }
    | .bool =>
      { dtype := .bool, data := Native.permuteU8 src.data u.src[0]!.shape.toArray perm.toArray }
    | _ =>
      if src.dtype != u.dtype then
        RawBuffer.zeros u.dtype (listProd u.shape)
      else
        let outNumel := listProd u.shape
        let elemSize := u.dtype.itemsize
        if elemSize == 0 then
          RawBuffer.zeros u.dtype outNumel
        else
          let out := permuteBytes src.data u.src[0]!.shape perm elemSize
          if out.isEmpty && outNumel != 0 then
            RawBuffer.zeros u.dtype outNumel
          else
            RawBuffer.mk u.dtype out

  | .PAD =>
    let src := getSrc cache u 0
    match u.arg with
    | .padding padding =>
      let rank := u.src[0]!.shape.length
      if padding.length != rank then
        RawBuffer.zeros u.dtype (listProd u.shape)
      else
        let padL := padding.map (fun (l, _) => l)
        let padR := padding.map (fun (_, r) => r)
        match u.dtype with
        | .float32 =>
          if src.dtype != .float32 then
            RawBuffer.zeros u.dtype (listProd u.shape)
          else
            { dtype := .float32
              data := Native.padF32 src.data u.src[0]!.shape.toArray padL.toArray padR.toArray }
        | .bool =>
          if src.dtype != .bool then
            RawBuffer.zeros u.dtype (listProd u.shape)
          else
            { dtype := .bool
              data := Native.padU8 src.data u.src[0]!.shape.toArray padL.toArray padR.toArray }
        | _ =>
          if src.dtype != u.dtype then
            RawBuffer.zeros u.dtype (listProd u.shape)
          else
            let outNumel := listProd u.shape
            let elemSize := u.dtype.itemsize
            if elemSize == 0 then
              RawBuffer.zeros u.dtype outNumel
            else
              let out := padBytes src.data u.src[0]!.shape padL padR u.shape elemSize
              if out.isEmpty && outNumel != 0 then
                RawBuffer.zeros u.dtype outNumel
              else
                RawBuffer.mk u.dtype out
    | _ =>
      RawBuffer.zeros u.dtype (listProd u.shape)

  | .SHRINK =>
    let src := getSrc cache u 0
    match u.arg with
    | .bounds bounds =>
      let rank := u.src[0]!.shape.length
      if bounds.length != rank then
        RawBuffer.zeros u.dtype (listProd u.shape)
      else
        let starts := bounds.map (fun (s, _) => s)
        let stops := bounds.map (fun (_, e) => e)
        match u.dtype with
        | .float32 =>
          if src.dtype != .float32 then
            RawBuffer.zeros u.dtype (listProd u.shape)
          else
            { dtype := .float32
              data := Native.shrinkF32 src.data u.src[0]!.shape.toArray starts.toArray stops.toArray }
        | .bool =>
          if src.dtype != .bool then
            RawBuffer.zeros u.dtype (listProd u.shape)
          else
            { dtype := .bool
              data := Native.shrinkU8 src.data u.src[0]!.shape.toArray starts.toArray stops.toArray }
        | _ =>
          if src.dtype != u.dtype then
            RawBuffer.zeros u.dtype (listProd u.shape)
          else
            let outNumel := listProd u.shape
            let elemSize := u.dtype.itemsize
            if elemSize == 0 then
              RawBuffer.zeros u.dtype outNumel
            else
              let out := shrinkBytes src.data u.src[0]!.shape starts stops u.shape elemSize
              if out.isEmpty && outNumel != 0 then
                RawBuffer.zeros u.dtype outNumel
              else
                RawBuffer.mk u.dtype out
    | _ =>
      RawBuffer.zeros u.dtype (listProd u.shape)

  | .FLIP =>
    let src := getSrc cache u 0
    let axes := u.arg.getAxes.getD []
    if axes.isEmpty then
      src
    else
      match u.dtype with
      | .float32 =>
        if src.dtype != .float32 then
          RawBuffer.zeros u.dtype (listProd u.shape)
        else
          { dtype := .float32
            data := Native.flipF32 src.data u.src[0]!.shape.toArray axes.toArray }
      | .bool =>
        if src.dtype != .bool then
          RawBuffer.zeros u.dtype (listProd u.shape)
        else
          { dtype := .bool
            data := Native.flipU8 src.data u.src[0]!.shape.toArray axes.toArray }
      | _ =>
        RawBuffer.zeros u.dtype (listProd u.shape)

  | .CAT =>
    Id.run do
      let axes := u.arg.getAxes.getD []
      if axes.length != 1 then
        return RawBuffer.zeros u.dtype (listProd u.shape)
      let ax := axes[0]!
      let rank := u.shape.length
      if ax >= rank || u.src.isEmpty then
        return RawBuffer.zeros u.dtype (listProd u.shape)
      let n := u.src.length
      let mut inputs : Array ByteArray := Array.emptyWithCapacity n
      let mut shapes : Array (Array Nat) := Array.emptyWithCapacity n
      for i in [:n] do
        let srcU := u.src[i]!
        let buf := getSrc cache u i
        if buf.dtype != u.dtype || srcU.shape.length != rank then
          return RawBuffer.zeros u.dtype (listProd u.shape)
        inputs := inputs.push buf.data
        shapes := shapes.push srcU.shape.toArray
      match u.dtype with
      | .float32 =>
        let out := Native.catF32 inputs shapes ax u.shape.toArray
        if out.isEmpty then
          RawBuffer.zeros u.dtype (listProd u.shape)
        else
          RawBuffer.mk .float32 out
      | .bool =>
        let out := Native.catU8 inputs shapes ax u.shape.toArray
        if out.isEmpty then
          RawBuffer.zeros u.dtype (listProd u.shape)
        else
          RawBuffer.mk .bool out
      | _ =>
        let elemSize := u.dtype.itemsize
        if elemSize == 0 then
          RawBuffer.zeros u.dtype (listProd u.shape)
        else
          let out := Native.catBytes inputs shapes ax u.shape.toArray elemSize
          if out.isEmpty then
            RawBuffer.zeros u.dtype (listProd u.shape)
          else
            RawBuffer.mk u.dtype out

  | .REDUCE_AXIS =>
    let src := getSrc cache u 0
    match u.dtype with
    | .float32 =>
      match u.arg with
      | .reduceWithAxes reduceOp axes =>
        Id.run do
          let srcShape := u.src[0]!.shape
          let srcNumel := listProd srcShape
          let outNumel := listProd u.shape
          if outNumel == 1 then
            match reduceOp with
            | .ADD => return { dtype := .float32, data := Native.sumAllF32 src.data }
            | .MAX => return { dtype := .float32, data := Native.reduceMaxLastF32 src.data 1 srcNumel }
            | _ => return RawBuffer.zeros u.dtype outNumel

          if axes.isEmpty then
            return RawBuffer.zeros u.dtype outNumel

          let keepdim := u.shape.length == srcShape.length
          let axes' :=
            let uniq := dedupNat axes
            if keepdim then sortNatAsc uniq else sortNatDesc uniq

          let mut curData := src.data
          let mut curShape := srcShape
          for ax in axes' do
            let curRank := curShape.length
            if ax >= curRank then
              return RawBuffer.zeros u.dtype outNumel

            let reduceDim := listGetD curShape ax 0
            let outer := listProd (curShape.take ax)
            let inner := listProd (curShape.drop (ax + 1))

            curData :=
              match reduceOp with
              | .ADD =>
                if curRank > 0 && ax == curRank - 1 then
                  Native.reduceSumLastF32 curData outer reduceDim
                else
                  Native.reduceSumAxisF32 curData outer reduceDim inner
              | .MAX =>
                if curRank > 0 && ax == curRank - 1 then
                  Native.reduceMaxLastF32 curData outer reduceDim
                else
                  Native.reduceMaxAxisF32 curData outer reduceDim inner
              | _ =>
                ByteArray.empty

            if curData.isEmpty then
              return RawBuffer.zeros u.dtype outNumel

            curShape := Shape.reduce curShape [ax] keepdim

          if curShape != u.shape then
            return RawBuffer.zeros u.dtype outNumel

          return { dtype := .float32, data := curData }
      | _ =>
        RawBuffer.zeros u.dtype (listProd u.shape)
    | _ =>
      RawBuffer.zeros u.dtype (listProd u.shape)

  | .CONTIGUOUS | .DETACH =>
    getSrc cache u 0

  | .BITCAST =>
    let src := getSrc cache u 0
    let numel := listProd u.shape
    if src.dtype.itemsize != u.dtype.itemsize then
      RawBuffer.zeros u.dtype numel
    else
      let expected := numel * u.dtype.itemsize
      if src.data.size != expected then
        RawBuffer.zeros u.dtype numel
      else
        RawBuffer.mk u.dtype src.data

  | .CAST =>
    Id.run do
      let src := getSrc cache u 0
      let numel := listProd u.shape
      match u.dtype with
      | .float32 =>
        let mut out : Array Float := Array.emptyWithCapacity numel
        for i in [:numel] do
          out := out.push (readAsFloat src.dtype src.data i)
        return { dtype := .float32, data := RawBuffer.packFloatsToF32Bytes out }
      | .bool =>
        let mut out := ByteArray.emptyWithCapacity numel
        for i in [:numel] do
          let v :=
            match src.dtype with
            | .float32 => if readF32At src.data i == 0.0 then 0 else 1
            | _ => if readAsInt src.dtype src.data i == 0 then 0 else 1
          out := out.push v
        return RawBuffer.mk .bool out
      | .int8 | .uint8 | .int16 | .uint16 | .int32 | .uint32 | .int64 | .uint64 | .index =>
        let mut out := ByteArray.emptyWithCapacity (numel * u.dtype.itemsize)
        for i in [:numel] do
          let v :=
            match src.dtype with
            | .float32 =>
              (Float.toInt64 (readF32At src.data i)).toInt
            | _ => readAsInt src.dtype src.data i
          match u.dtype with
          | .int8 | .uint8 =>
            out := out.push (intToUInt8Mod v)
          | .int16 | .uint16 =>
            out := pushBytes out (bytesFromUInt16 (intToUInt16Mod v))
          | .int32 | .uint32 =>
            out := pushBytes out (bytesFromUInt32 (intToUInt32Mod v))
          | .int64 | .uint64 | .index =>
            out := pushBytes out (bytesFromUInt64 (intToUInt64Mod v))
          | _ => pure ()
        return RawBuffer.mk u.dtype out
      | _ =>
        return RawBuffer.zeros u.dtype numel

  | _ =>
    RawBuffer.zeros u.dtype (listProd u.shape)

  structure ExecItem where
    ast : UOp
    impl : Option Fusion.Impl
    tag : String
    deriving Repr

  structure Compiled where
    roots : List UOp
    nodes : List UOp
    keepIds : UOpIdSet
    implMap : HashMap UOpId Fusion.Impl
    schedule : List ExecItem
    deriving Repr

  structure NodeKey where
    op : Ops
    dtype : DType
    arg : UArg
    shape : Shape
    src : Array Nat
    deriving BEq, Hashable, Repr

  structure GraphKey where
    nodes : Array NodeKey
    roots : Array Nat
    cm : CostModel
    deriving BEq, Hashable, Repr

  structure ScheduleCacheEntry where
    ids : Array UOpId
    implMap : HashMap UOpId Fusion.Impl
    deriving Repr

  initialize scheduleCacheRef : IO.Ref (HashMap GraphKey ScheduleCacheEntry) ← IO.mkRef ∅
  initialize scheduleCacheLimitRef : IO.Ref Nat ← IO.mkRef 0

  structure ScheduleCacheStats where
    hits : Nat := 0
    misses : Nat := 0
    rebuilds : Nat := 0
    hitNs : Nat := 0
    missNs : Nat := 0
    deriving Repr

  instance : Inhabited ScheduleCacheStats := ⟨{}⟩

  initialize scheduleCacheStatsRef : IO.Ref ScheduleCacheStats ← IO.mkRef {}
  initialize timingSinkRef : IO.Ref Nat ← IO.mkRef 0

  def clearScheduleCacheStats : IO Unit :=
    scheduleCacheStatsRef.set {}

  def getScheduleCacheStats : IO ScheduleCacheStats :=
    scheduleCacheStatsRef.get

  private def addCacheHit (dtNs : Nat) : IO Unit :=
    scheduleCacheStatsRef.modify fun s =>
      { s with
        hits := s.hits + 1
        hitNs := s.hitNs + dtNs }

  private def addCacheMiss (dtNs : Nat) (rebuild : Bool := false) : IO Unit :=
    scheduleCacheStatsRef.modify fun s =>
      { s with
        misses := s.misses + 1
        missNs := s.missNs + dtNs
        rebuilds := s.rebuilds + (if rebuild then 1 else 0) }

  private def touchTiming (n : Nat) : IO Unit :=
    timingSinkRef.modify (· + n)

  private def forceCompiled (c : Compiled) : Nat :=
    let roots := c.roots.length
    let nodes := c.nodes.length
    let schedule := c.schedule.length
    let impls := c.implMap.fold (init := 0) (fun acc _ _ => acc + 1)
    roots + nodes + schedule + impls

  private def forceCache (cache : HashMap UOpId RawBuffer) : Nat :=
    cache.fold (init := 0) fun acc _ buf => acc + buf.data.size

  private def forceCompiledIO (c : Compiled) : IO Unit :=
    touchTiming (forceCompiled c)

  private def forceCacheIO (cache : HashMap UOpId RawBuffer) : IO Unit :=
    touchTiming (forceCache cache)

  def setScheduleCacheLimit (limit : Nat) : IO Unit :=
    scheduleCacheLimitRef.set limit

  def getScheduleCacheLimit : IO Nat :=
    scheduleCacheLimitRef.get

  def clearScheduleCache : IO Unit :=
    scheduleCacheRef.set ∅

  def getScheduleCacheSize : IO Nat := do
    let cache ← scheduleCacheRef.get
    pure cache.size

  /-- Clear schedule cache and stats to free memory when graphs grow large. -/
  def gcScheduleCache : IO Unit := do
    clearScheduleCache
    clearScheduleCacheStats

  private def buildNodeMap (nodes : List UOp) : HashMap UOpId UOp :=
    nodes.foldl (init := (∅ : HashMap UOpId UOp)) fun m u => m.insert u.uid u

  private def graphKeyOf (nodes : List UOp) (roots : List UOp) (cm : CostModel) : GraphKey := Id.run do
    let nodeArr := nodes.toArray
    let mut idToIdx : HashMap UOpId Nat := ∅
    for i in [:nodeArr.size] do
      idToIdx := idToIdx.insert nodeArr[i]!.uid i
    let mut nodeKeys : Array NodeKey := Array.emptyWithCapacity nodeArr.size
    for i in [:nodeArr.size] do
      let u := nodeArr[i]!
      let srcIdx := (u.src.map fun s => idToIdx.getD s.uid 0).toArray
      nodeKeys := nodeKeys.push { op := u.op, dtype := u.dtype, arg := u.arg, shape := u.shape, src := srcIdx }
    let rootIdx := (roots.map fun r => idToIdx.getD r.uid 0).toArray
    return { nodes := nodeKeys, roots := rootIdx, cm }

  private def buildIdMap (oldIds newIds : Array UOpId) : HashMap UOpId UOpId := Id.run do
    let n := Nat.min oldIds.size newIds.size
    let mut out : HashMap UOpId UOpId := ∅
    for i in [:n] do
      out := out.insert oldIds[i]! newIds[i]!
    return out

  private def remapId (idMap : HashMap UOpId UOpId) (uid : UOpId) : UOpId :=
    idMap.getD uid uid

  private def remapImplMap (implMap : HashMap UOpId Fusion.Impl)
      (idMap : HashMap UOpId UOpId) : HashMap UOpId Fusion.Impl :=
    implMap.fold (init := (∅ : HashMap UOpId Fusion.Impl)) fun acc uid impl =>
      let uid' := remapId idMap uid
      let impl' := impl.mapIds (remapId idMap)
      acc.insert uid' impl'

  private def compileManyWithImplMap (roots0 : List UOp) (nodes0 : List UOp) (keepIds : UOpIdSet)
      (implMap : HashMap UOpId Fusion.Impl) : Compiled := Id.run do
    let nodeMap := buildNodeMap nodes0
    let mut kernMap : HashMap UOpId UOp := ∅

    for u in nodes0 do
      let impl := implMap.getD u.uid (.node (u.src.map (fun s => s.uid)))
      let depUids := impl.deps
      let src' : List UOp := depUids.map fun sid =>
        match kernMap[sid]? with
        | some v => v
        | none =>
          match nodeMap[sid]? with
          | some v => v
          | none => default

      let u0 : UOp :=
        match impl with
        | .node _ => { u with src := src' }
        | _ => { u with op := .KERNEL, src := src', arg := .empty }
      let u' : UOp := fusion[impl.tag] (cost[impl.score] u0)
      kernMap := kernMap.insert u.uid u'

    let roots' := roots0.map fun r => kernMap.getD r.uid r
    let nodes := toposortMany roots'

    if validateOnCompile then
      let errs := UOp.validateMany roots'
      if errs.size != 0 then
        let shown := errs.toList.take 10 |>.map (fun e => e.render)
        let msg := String.intercalate "\n" shown
        panic! s!"compileMany: invalid kernelized graph ({errs.size} errors)\n{msg}"

    let schedule := nodes.map fun u =>
      let impl? := if u.op == .KERNEL then implMap[u.uid]? else none
      let tag := match impl? with
        | some impl => impl.tag
        | none => "node"
      { ast := u, impl := impl?, tag }

    return { roots := roots', nodes, keepIds, implMap, schedule }

  def compileMany (roots : List UOp) (cm : CostModel := defaultCostModel) : Compiled := Id.run do
    let roots0 :=
      if optimizeOnCompile then
        TinyGrad4.Optim.optimizeKeepUids roots
      else
        roots

    if validateOnCompile then
      let errs := UOp.validateMany roots0
      if errs.size != 0 then
        let shown := errs.toList.take 10 |>.map (fun e => e.render)
        let msg := String.intercalate "\n" shown
        panic! s!"compileMany: invalid input graph ({errs.size} errors)\n{msg}"

    let nodes0 := toposortMany roots0
    let keepIds := roots0.foldl (fun s r => UOpIdSet.add s r.uid) UOpIdSet.mkEmpty

    let mut refCnt0 : HashMap UOpId Nat := ∅
    for u in nodes0 do
      for s in u.src do
        refCnt0 := refCnt0.insert s.uid (refCnt0.getD s.uid 0 + 1)

    let implMap : HashMap UOpId Fusion.Impl :=
      Fusion.selectPhaseC nodes0 keepIds refCnt0 (cm := cm)

    return compileManyWithImplMap roots0 nodes0 keepIds implMap

  def compileManyCached (roots : List UOp) (cm : CostModel := defaultCostModel) : IO Compiled := do
    let t0 ← IO.monoNanosNow
    let roots0 :=
      if optimizeOnCompile then
        TinyGrad4.Optim.optimizeKeepUids roots
      else
        roots

    if validateOnCompile then
      let errs := UOp.validateMany roots0
      if errs.size != 0 then
        let shown := errs.toList.take 10 |>.map (fun e => e.render)
        let msg := String.intercalate "\n" shown
        panic! s!"compileMany: invalid input graph ({errs.size} errors)\n{msg}"

    let nodes0 := toposortMany roots0
    let nodeIds := (nodes0.map fun u => u.uid).toArray
    let keepIds := roots0.foldl (fun s r => UOpIdSet.add s r.uid) UOpIdSet.mkEmpty
    let key := graphKeyOf nodes0 roots0 cm

    let cache ← scheduleCacheRef.get
    match cache[key]? with
    | some entry =>
      if entry.ids.size != nodeIds.size then
        let mut refCnt0 : HashMap UOpId Nat := ∅
        for u in nodes0 do
          for s in u.src do
            refCnt0 := refCnt0.insert s.uid (refCnt0.getD s.uid 0 + 1)
        let implMap : HashMap UOpId Fusion.Impl :=
          Fusion.selectPhaseC nodes0 keepIds refCnt0 (cm := cm)
        let entry' : ScheduleCacheEntry := { ids := nodeIds, implMap }
        let limit ← scheduleCacheLimitRef.get
        let cache' := cache.insert key entry'
        let cache'' :=
          if limit != 0 && cache'.size > limit then
            (∅ : HashMap GraphKey ScheduleCacheEntry).insert key entry'
          else
            cache'
        scheduleCacheRef.set cache''
        let compiled := compileManyWithImplMap roots0 nodes0 keepIds implMap
        forceCompiledIO compiled
        let t1 ← IO.monoNanosNow
        addCacheMiss (t1 - t0) (rebuild := true)
        pure compiled
      else
        let idMap := buildIdMap entry.ids nodeIds
        let implMap := remapImplMap entry.implMap idMap
        let compiled := compileManyWithImplMap roots0 nodes0 keepIds implMap
        forceCompiledIO compiled
        let t1 ← IO.monoNanosNow
        addCacheHit (t1 - t0)
        pure compiled
    | none =>
      let mut refCnt0 : HashMap UOpId Nat := ∅
      for u in nodes0 do
        for s in u.src do
          refCnt0 := refCnt0.insert s.uid (refCnt0.getD s.uid 0 + 1)
      let implMap : HashMap UOpId Fusion.Impl :=
        Fusion.selectPhaseC nodes0 keepIds refCnt0 (cm := cm)
      let entry : ScheduleCacheEntry := { ids := nodeIds, implMap }
      let limit ← scheduleCacheLimitRef.get
      let cache' := cache.insert key entry
      let cache'' :=
        if limit != 0 && cache'.size > limit then
          (∅ : HashMap GraphKey ScheduleCacheEntry).insert key entry
        else
          cache'
      scheduleCacheRef.set cache''
      let compiled := compileManyWithImplMap roots0 nodes0 keepIds implMap
      forceCompiledIO compiled
      let t1 ← IO.monoNanosNow
      addCacheMiss (t1 - t0)
      pure compiled

  def compile (u : UOp) : Compiled :=
    compileMany [u]

  def compileCached (u : UOp) (cm : CostModel := defaultCostModel) : IO Compiled :=
    compileManyCached [u] (cm := cm)

  def compileManyTimed (roots : List UOp) (cm : CostModel := defaultCostModel) : IO (Compiled × Nat) := do
    let t0 ← IO.monoNanosNow
    let c := compileMany roots (cm := cm)
    forceCompiledIO c
    let t1 ← IO.monoNanosNow
    pure (c, t1 - t0)

  def compileManyCachedTimed (roots : List UOp) (cm : CostModel := defaultCostModel) : IO (Compiled × Nat) := do
    let t0 ← IO.monoNanosNow
    let c ← compileManyCached roots (cm := cm)
    let t1 ← IO.monoNanosNow
    pure (c, t1 - t0)

  def evalCompiledRaw (c : Compiled) (env : Env) : HashMap UOpId RawBuffer := Id.run do
    let deps (u : UOp) : List UOpId := u.src.map (fun s => s.uid)

    let mut refCnt : HashMap UOpId Nat := ∅
    for u in c.nodes do
      for sid in deps u do
        refCnt := refCnt.insert sid (refCnt.getD sid 0 + 1)

    let mut cache : HashMap UOpId RawBuffer := ∅
    for item in c.schedule do
      let u := item.ast
      let v :=
        match u.op with
        | .KERNEL =>
          match item.impl with
          | some impl =>
            match impl with
            | .fusedMatmul plan => evalFusedMatmulBias u plan env cache
            | .fusedSoftmax plan => evalFusedSoftmax u plan env cache
            | .fusedReduce plan => evalFusedReduce u plan env cache
            | .fusedEwise plan => evalFusedEwise u plan env cache
            | .fusedContract plan => evalFusedContract u plan env cache
            | .fusedSGD plan => evalFusedSGD u plan env cache
            | .fusedLayerNorm _ => evalNode u env cache  -- fallback until dedicated eval
            | .fusedGELU _ => evalNode u env cache       -- fallback until dedicated eval
            | .node _ => evalNode u env cache
          | none =>
            evalNode u env cache
        | _ =>
          evalNode u env cache
      cache := cache.insert u.uid v

      for sid in deps u do
        let cnt := refCnt.getD sid 0
        if cnt > 0 then
          let cnt' := cnt - 1
          refCnt := refCnt.insert sid cnt'
          if cnt' == 0 && !UOpIdSet.member c.keepIds sid then
            cache := cache.erase sid

    return cache

  structure KernelTiming where
    uid : UOpId
    tag : String
    shape : Shape
    numel : Nat
    dtype : DType
    ms : Float
    deriving Repr

  def evalCompiledRawTimed (c : Compiled) (env : Env) : IO (HashMap UOpId RawBuffer × Array KernelTiming) := do
    let deps (u : UOp) : List UOpId := u.src.map (fun s => s.uid)

    let mut refCnt : HashMap UOpId Nat := ∅
    for u in c.nodes do
      for sid in deps u do
        refCnt := refCnt.insert sid (refCnt.getD sid 0 + 1)

    let mut cache : HashMap UOpId RawBuffer := ∅
    let mut timings : Array KernelTiming := #[]

    for item in c.schedule do
      let u := item.ast
      let (v, timing?) ←
        if u.op == .KERNEL then
          let start ← IO.monoNanosNow
          let v :=
            match item.impl with
            | some impl =>
              match impl with
              | .fusedMatmul plan => evalFusedMatmulBias u plan env cache
              | .fusedSoftmax plan => evalFusedSoftmax u plan env cache
              | .fusedReduce plan => evalFusedReduce u plan env cache
              | .fusedEwise plan => evalFusedEwise u plan env cache
              | .fusedContract plan => evalFusedContract u plan env cache
              | .fusedSGD plan => evalFusedSGD u plan env cache
              | .fusedLayerNorm _ => evalNode u env cache  -- fallback until dedicated eval
              | .fusedGELU _ => evalNode u env cache       -- fallback until dedicated eval
              | .node _ => evalNode u env cache
            | none =>
              evalNode u env cache
          let stop ← IO.monoNanosNow
          let dtNs : Nat := stop - start
          let ms : Float := (Float.ofNat dtNs) / 1.0e6
          pure (v, some { uid := u.uid, tag := item.tag, shape := u.shape, numel := u.numel, dtype := u.dtype, ms })
        else
          let v := evalNode u env cache
          pure (v, none)

      cache := cache.insert u.uid v
      match timing? with
      | some t => timings := timings.push t
      | none => pure ()

      for sid in deps u do
        let cnt := refCnt.getD sid 0
        if cnt > 0 then
          let cnt' := cnt - 1
          refCnt := refCnt.insert sid cnt'
          if cnt' == 0 && !UOpIdSet.member c.keepIds sid then
            cache := cache.erase sid

    return (cache, timings)

  def evalCompiledRawTimedTotal (c : Compiled) (env : Env) : IO (HashMap UOpId RawBuffer × Nat) := do
    let t0 ← IO.monoNanosNow
    let cache := evalCompiledRaw c env
    forceCacheIO cache
    let t1 ← IO.monoNanosNow
    pure (cache, t1 - t0)

  -- All eval functions return RawBuffer. Use .decode for display.
  def evalCompiled (c : Compiled) (env : Env) : HashMap UOpId RawBuffer :=
    evalCompiledRaw c env

  def evalMany (roots : List UOp) (env : Env) : HashMap UOpId RawBuffer := Id.run do
    evalCompiledRaw (compileMany roots) env

  def evalManyCached (roots : List UOp) (env : Env) : IO (HashMap UOpId RawBuffer) := do
    let compiled ← compileManyCached roots
    pure (evalCompiledRaw compiled env)

  def evalManyTimed (roots : List UOp) (env : Env) : IO (HashMap UOpId RawBuffer × Nat × Nat) := do
    let (compiled, compileNs) ← compileManyTimed roots
    let (cache, evalNs) ← evalCompiledRawTimedTotal compiled env
    pure (cache, compileNs, evalNs)

  def evalManyCachedTimed (roots : List UOp) (env : Env) : IO (HashMap UOpId RawBuffer × Nat × Nat) := do
    let (compiled, compileNs) ← compileManyCachedTimed roots
    let (cache, evalNs) ← evalCompiledRawTimedTotal compiled env
    pure (cache, compileNs, evalNs)

  def evalCached (u : UOp) (env : Env) : IO RawBuffer := do
    let cache ← evalManyCached [u] env
    pure (cache.getD u.uid (RawBuffer.zeros u.dtype (listProd u.shape)))

def eval (u : UOp) (env : Env) : RawBuffer :=
  let cache := evalMany [u] env
  cache.getD u.uid (RawBuffer.zeros u.dtype (listProd u.shape))

def evalTensor {s : List Nat} {d : DType} (t : StaticTensor s d) (env : Env := ∅) : RawBuffer :=
  eval t.uop env

def evalTensorCached {s : List Nat} {d : DType} (t : StaticTensor s d) (env : Env := ∅) : IO RawBuffer :=
  evalCached t.uop env

def setBuffer (env : Env) (u : UOp) (data : RawBuffer) : Env :=
  env.insert u.uid data

end Interpreter

end TinyGrad4
