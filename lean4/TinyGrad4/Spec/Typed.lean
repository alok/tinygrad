import Float64
import TinyGrad4.Spec.Semantics

namespace TinyGrad4
namespace Spec
namespace Typed

/-!
# Typed tensor signatures

This is the low-level proof/constraint layer for the executable spec. Unlike `Spec.TensorDesc`, these signatures
carry shape and dtype in the Lean type, so successful composition can expose more information to the compiler.
-/

/-- A tensor signature with shape and dtype tracked in the Lean type. -/
structure Sig (shape : Shape) (dtype : DType) where
  toDesc : TensorDesc := { shape, dtype }

/-- Forget the indices and recover the ergonomic descriptor. -/
def erase {shape : Shape} {dtype : DType} (sig : Sig shape dtype) : TensorDesc :=
  sig.toDesc

/-- Result dtype for typed binary ops. -/
def binaryDType (op : Ops) (lhs rhs : DType) : DType :=
  if op.producesBoolean then .bool else DType.promote lhs rhs

/-- Result dtype for typed unary ops. -/
def unaryDType (op : Ops) (src target : DType) : DType :=
  match op with
  | .CAST | .BITCAST => target
  | _ => src

def const (dtype : DType) : Sig [] dtype := {}
def vconst (size : Nat) (dtype : DType) : Sig [size] dtype := {}
def full (shape : Shape) (dtype : DType) : Sig shape dtype := {}
def fullBool (shape : Shape) : Sig shape .bool := {}
def eye (rows : Nat) (cols : Nat := rows) (dtype : DType := .float32) : Sig [rows, cols] dtype := {}
def arange (size : Nat) (dtype : DType := .float32) : Sig [size] dtype := {}
def like {shape : Shape} {dtype : DType} (_ : Sig shape dtype) : Sig shape dtype := {}
def copy {shape : Shape} {dtype : DType} (_ : Sig shape dtype) : Sig shape dtype := {}
def detach {shape : Shape} {dtype : DType} (_ : Sig shape dtype) : Sig shape dtype := {}
def contiguous {shape : Shape} {dtype : DType} (_ : Sig shape dtype) : Sig shape dtype := {}
def contiguousBackward {shape : Shape} {dtype : DType} (_ : Sig shape dtype) : Sig shape dtype := {}

def reshape {s newShape : Shape} {d : DType}
    (_ : Sig s d) (_h : Shape.reshapeValid s newShape = true) : Sig newShape d := {}

def expand {s newShape : Shape} {d : DType}
    (_ : Sig s d) (_h : Shape.expandValid s newShape = true) : Sig newShape d := {}

def permute {s : Shape} {d : DType} (_ : Sig s d) (perm : List Nat)
    (_h : Shape.permuteValid s perm = true) : Sig (Shape.permute s perm) d := {}

def pad {s : Shape} {d : DType} (_ : Sig s d) (padding : List (Nat × Nat))
    (_h : Shape.padValid s padding = true) : Sig (Shape.pad s padding) d := {}

def shrink {s : Shape} {d : DType} (_ : Sig s d) (bounds : List (Nat × Nat))
    (_h : Shape.shrinkValid s bounds = true) : Sig (Shape.shrink s bounds) d := {}

def flip {s : Shape} {d : DType} (_ : Sig s d) (_axes : List Nat)
    (_h : Shape.flipValid s _axes = true) : Sig s d := {}

def binary {s1 s2 : Shape} {d1 d2 : DType} (op : Ops)
    (_ : Sig s1 d1) (_ : Sig s2 d2) (_h : Shape.broadcastable s1 s2 = true) :
    Sig (Shape.broadcastOut s1 s2) (binaryDType op d1 d2) := {}

def where_ {s1 s2 s3 : Shape} {d : DType}
    (_ : Sig s1 .bool) (_ : Sig s2 d) (_ : Sig s3 d)
    (_hXY : Shape.broadcastable s2 s3 = true)
    (_hCond : Shape.broadcastable s1 (Shape.broadcastOut s2 s3) = true) :
    Sig (Shape.broadcastOut s1 (Shape.broadcastOut s2 s3)) d := {}

def gather {srcShape idxShape : Shape} {srcD idxD : DType}
    (_ : Sig srcShape srcD) (_ : Sig idxShape idxD) (dim : Nat)
    (_hIdx : idxD.isInt = true) (_hCompat : gatherShapeOk srcShape idxShape dim = true) :
    Sig idxShape srcD := {}

def take {srcShape idxShape : Shape} {srcD idxD : DType}
    (_ : Sig srcShape srcD) (_ : Sig idxShape idxD) (_hIdx : idxD.isInt = true) :
    Sig idxShape srcD := {}

def diag {n : Nat} {d : DType} (_ : Sig [n] d) : Sig [n, n] d := {}
def diagonal {n : Nat} {d : DType} (_ : Sig [n, n] d) : Sig [n] d := {}

def triu {s : Shape} {d : DType} (_ : Sig s d) (_h : 2 <= s.length) : Sig s d := {}
def tril {s : Shape} {d : DType} (_ : Sig s d) (_h : 2 <= s.length) : Sig s d := {}

def unfold {s : Shape} {d : DType} (_ : Sig s d) (dim size step : Nat)
    (_hSize : 0 < size) (_hStep : 0 < step)
    (_hDim : dim = s.length - 1) (_hLe : size <= listGetD s dim 0) :
    Sig (Shape.poolOut s [size] [step] [1]) d := {}

def scatter {selfShape idxShape srcShape : Shape} {d idxD : DType}
    (_ : Sig selfShape d) (_ : Sig idxShape idxD) (_ : Sig srcShape d) (dim : Nat)
    (_hIdx : idxD.isInt = true) (_hCompat : scatterShapeOk selfShape idxShape srcShape dim = true) :
    Sig selfShape d := {}

def scatterScalar {selfShape idxShape : Shape} {d idxD : DType}
    (_ : Sig selfShape d) (_ : Sig idxShape idxD) (dim : Nat)
    (_hIdx : idxD.isInt = true) (_hCompat : scatterShapeOk selfShape idxShape idxShape dim = true) :
    Sig selfShape d := {}

def scatterReduce {selfShape idxShape srcShape : Shape} {d idxD : DType}
    (_ : Sig selfShape d) (_ : Sig idxShape idxD) (_ : Sig srcShape d) (dim : Nat)
    (_mode : ScatterReduceMode) (_hIdx : idxD.isInt = true)
    (_hCompat : scatterShapeOk selfShape idxShape srcShape dim = true) :
    Sig selfShape d := {}

def scatterReduceScalar {selfShape idxShape : Shape} {d idxD : DType}
    (_ : Sig selfShape d) (_ : Sig idxShape idxD) (dim : Nat)
    (_mode : ScatterReduceMode) (_hIdx : idxD.isInt = true)
    (_hCompat : scatterShapeOk selfShape idxShape idxShape dim = true) :
    Sig selfShape d := {}

def maskedSelectPacked {s : Shape} {d : DType}
    (_ : Sig s d) (_ : Sig s .bool) : Sig [Shape.numel s] d × Sig [] .int32 := ({}, {})

def reduce {s : Shape} {d : DType} (_ : Sig s d) (spec : ReduceSpec)
    (_h : ReduceSpec.outShape? spec s = some (Shape.reduce s (spec.normalizedAxes s) spec.keepdim)) :
    Sig (Shape.reduce s (spec.normalizedAxes s) spec.keepdim) d := {}

def cat {shapes : List Shape} {d : DType} (_inputs : List (Sigma fun s => Sig s d)) (axis : Nat)
    (_h : Shape.concatListValid shapes axis = true) :
    Sig (Shape.concatOutList shapes axis) d := {}

def contract2D {s1 s2 out : Shape} {d1 d2 : DType}
    (_ : Sig s1 d1) (_ : Sig s2 d2) (_h : Shape.matmulShape s1 s2 = some out) :
    Sig out (DType.promote d1 d2) := {}

def linear {batch inDim outDim : Nat} {d : DType}
    (_ : Sig [batch, inDim] d) (_ : Sig [inDim, outDim] d) : Sig [batch, outDim] d := {}

def linearBias {batch inDim outDim : Nat} {d : DType}
    (_ : Sig [batch, inDim] d) (_ : Sig [inDim, outDim] d) (_ : Sig [outDim] d) :
    Sig [batch, outDim] d := {}

def conv1d {batch cin cout w kW : Nat} {d : DType}
    (_ : Sig [batch, cin, w] d) (_ : Sig [cout, cin, kW] d)
    (padding stride dilation : Nat) :
    Sig (Shape.conv1dOut [batch, cin, w] [cout, cin, kW] padding stride dilation) d := {}

def conv2d {batch cin cout h w kH kW : Nat} {d : DType}
    (_ : Sig [batch, cin, h, w] d) (_ : Sig [cout, cin, kH, kW] d)
    (padding stride dilation : Nat) :
    Sig (Shape.conv2dOut [batch, cin, h, w] [cout, cin, kH, kW] padding stride dilation) d := {}

def pool2d {batch channels h w : Nat} {d : DType}
    (_ : Sig [batch, channels, h, w] d) (kernelSize padding stride : Nat) :
    Sig (Shape.pool2dShape [batch, channels, h, w] kernelSize padding stride) d := {}

def maxUnpool2dOut {batch channels h w outH outW : Nat} {d idxD : DType}
    (_ : Sig [batch, channels, h, w] d) (_ : Sig [batch, channels, h, w] idxD)
    (_hIdx : idxD.isInt = true) : Sig [batch, channels, outH, outW] d := {}

def batchnormNC {batch channels : Nat}
    (_ : Sig [batch, channels] .float32)
    (_mean : Sig [channels] .float32) (_invstd : Sig [channels] .float32)
    (_weight? : Option (Sig [channels] .float32) := none)
    (_bias? : Option (Sig [channels] .float32) := none) :
    Sig [batch, channels] .float32 := {}

def batchnormNCHW {batch channels height width : Nat}
    (_ : Sig [batch, channels, height, width] .float32)
    (_mean : Sig [channels] .float32) (_invstd : Sig [channels] .float32)
    (_weight? : Option (Sig [channels] .float32) := none)
    (_bias? : Option (Sig [channels] .float32) := none) :
    Sig [batch, channels, height, width] .float32 := {}

end Typed
end Spec
end TinyGrad4
