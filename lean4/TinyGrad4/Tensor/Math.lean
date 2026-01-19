import Float64
import TinyGrad4.Tensor.Tensor
import TinyGrad4.Tensor.Movement

set_option maxHeartbeats 800000

namespace TinyGrad4

namespace StaticTensor

def add {s : List Nat} {d : DType} (t1 t2 : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← UOp.add t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def addB {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) d) := do
  let result ← UOp.add t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def mul {s : List Nat} {d : DType} (t1 t2 : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← UOp.mul t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def mulB {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) d) := do
  let result ← UOp.mul t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def sub {s : List Nat} {d : DType} (t1 t2 : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← UOp.sub t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def subB {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) d) := do
  let result ← UOp.sub t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def div {s : List Nat} {d : DType} (t1 t2 : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← UOp.div t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def divB {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) d) := do
  let result ← UOp.div t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def pow {s : List Nat} {d : DType} (t1 t2 : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← UOp.pow t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def powB {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) d) := do
  let result ← UOp.pow t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def cmplt {s : List Nat} {d : DType} (t1 t2 : StaticTensor s d) : TensorM (StaticTensor s .bool) := do
  let result ← UOp.cmplt t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def cmpltB {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool) := do
  let result ← UOp.cmplt t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def cmpgt {s : List Nat} {d : DType} (t1 t2 : StaticTensor s d) : TensorM (StaticTensor s .bool) := do
  cmplt t2 t1

def cmpgtB {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool) := do
  let result ← UOp.cmplt t2.uop t1.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def cmpeq {s : List Nat} {d : DType} (t1 t2 : StaticTensor s d) : TensorM (StaticTensor s .bool) := do
  let result ← UOp.cmpeq t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def cmpeqB {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool) := do
  let result ← UOp.cmpeq t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def cmpne {s : List Nat} {d : DType} (t1 t2 : StaticTensor s d) : TensorM (StaticTensor s .bool) := do
  let result ← UOp.cmpne t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def cmpneB {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool) := do
  let result ← UOp.cmpne t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def cat {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    (axis : Nat) : TensorM (StaticTensor (Shape.concatOut s1 s2 axis) d) := do
  let out ← UOp.cat [t1.uop, t2.uop] axis
  pure { uop := out, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def catList {d : DType} {shapes : List Shape} (ts : TensorList d shapes) (axis : Nat)
    : TensorM (StaticTensor (Shape.concatOutList shapes axis) d) := do
  let out ← UOp.cat (TensorList.toUOps ts) axis
  let reqGrad := TensorList.anyRequiresGrad ts
  pure { uop := out, requiresGrad := reqGrad, h_shape := sorry_proof }

def bitand {s : List Nat} (t1 t2 : StaticTensor s .bool) : TensorM (StaticTensor s .bool) := do
  let result ← UOp.bitand t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def bitandB {s1 s2 : List Nat} (t1 : StaticTensor s1 .bool) (t2 : StaticTensor s2 .bool)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool) := do
  let result ← UOp.bitand t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def bitor {s : List Nat} (t1 t2 : StaticTensor s .bool) : TensorM (StaticTensor s .bool) := do
  let result ← UOp.bitor t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def bitorB {s1 s2 : List Nat} (t1 : StaticTensor s1 .bool) (t2 : StaticTensor s2 .bool)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool) := do
  let result ← UOp.bitor t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def bitxor {s : List Nat} (t1 t2 : StaticTensor s .bool) : TensorM (StaticTensor s .bool) := do
  let result ← UOp.bitxor t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def bitxorB {s1 s2 : List Nat} (t1 : StaticTensor s1 .bool) (t2 : StaticTensor s2 .bool)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool) := do
  let result ← UOp.bitxor t1.uop t2.uop
  pure { uop := result, requiresGrad := t1.requiresGrad || t2.requiresGrad, h_shape := sorry_proof }

def cast {s : List Nat} {d : DType} (t : StaticTensor s d) (dtype : DType)
    : TensorM (StaticTensor s dtype) := do
  let result ← UOp.cast t.uop dtype
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def bitcast {s : List Nat} {d : DType} (t : StaticTensor s d) (dtype : DType)
    : TensorM (StaticTensor s dtype) := do
  let result ← UOp.bitcast t.uop dtype
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def to {s : List Nat} {d : DType} (t : StaticTensor s d) (dtype : DType)
    : TensorM (StaticTensor s dtype) :=
  cast t dtype

def to_ {s : List Nat} {d : DType} (t : StaticTensor s d) (dtype : DType)
    : TensorM (StaticTensor s dtype) :=
  cast t dtype

def where_ {s1 s2 s3 : List Nat} {d : DType}
    (cond : StaticTensor s1 .bool) (x : StaticTensor s2 d) (y : StaticTensor s3 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 (Shape.broadcastOut s2 s3)) d) := do
  let out ← UOp.where_ cond.uop x.uop y.uop
  pure { uop := out, requiresGrad := x.requiresGrad || y.requiresGrad, h_shape := sorry_proof }

infixl:65 " +. " => addB
infixl:65 " -. " => subB
infixl:70 " *. " => mulB
infixl:70 " /. " => divB

def neg {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← UOp.neg t.uop
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def trunc {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← UOp.trunc t.uop
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def floor {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let truncT ← trunc t
  let isNeg ← UOp.cmplt t.uop truncT.uop
  let one ← UOp.const d 1.0
  let truncMinusOne ← UOp.sub truncT.uop one
  let out ← UOp.where_ isNeg truncMinusOne truncT.uop
  pure { uop := out, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def ceil {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let truncT ← trunc t
  let isPos ← UOp.cmplt truncT.uop t.uop
  let one ← UOp.const d 1.0
  let truncPlusOne ← UOp.add truncT.uop one
  let out ← UOp.where_ isPos truncPlusOne truncT.uop
  pure { uop := out, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def sqrt {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← UOp.sqrt t.uop
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def rsqrt {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let sqrtT ← sqrt t
  let result ← UOp.recip sqrtT.uop
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def exp2 {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← UOp.exp2 t.uop
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def log2 {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← UOp.log2 t.uop
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Sine function -/
def sin {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← UOp.sin t.uop
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Cosine function -/
def cos {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← UOp.cos t.uop
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Tangent function -/
def tan {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← UOp.tan t.uop
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Reciprocal (1/x) -/
def recip {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← UOp.recip t.uop
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def sum {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (Scalar d) := do
  let axes := listRange s.length
  let result ← UOp.sum t.uop axes false
  pure { uop := result, h_shape := sorry_proof }

def max {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (Scalar d) := do
  let axes := listRange s.length
  let result ← UOp.max_ t.uop axes false
  pure { uop := result, h_shape := sorry_proof }

def min {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (Scalar d) := do
  let negT ← neg t
  let maxNeg ← max negT
  neg maxNeg

def mean {shape : List Nat} {d : DType} (t : StaticTensor shape d) : TensorM (Scalar d) := do
  let sumT ← sum t
  let n := listProd shape
  let nConst ← UOp.const d n.toFloat32
  let result ← UOp.div sumT.uop nConst
  pure { uop := result, h_shape := sorry_proof }

-- Constants for exp/log conversion
-- ln(2) ≈ 0.693147
-- log2(e) ≈ 1.442695
def ln2 : Float64 := 0.6931471805599453
def log2e : Float64 := 1.4426950408889634

-- NOTE: We use Float32 for const construction so float32 graphs can stay in Float32/ByteArray land.
def ln2f32 : Float32 := 0.6931471805599453
def log2ef32 : Float32 := 1.4426950408889634

/-- Natural exponential: e^x = 2^(x * log2(e)) -/
def exp {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let log2eConst ← UOp.const d log2ef32
  let scaled ← UOp.mul t.uop log2eConst
  let result ← UOp.exp2 scaled
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Natural logarithm: ln(x) = log2(x) * ln(2) -/
def log {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let log2Result ← UOp.log2 t.uop
  let ln2Const ← UOp.const d ln2f32
  let result ← UOp.mul log2Result ln2Const
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- ReLU: max(0, x) -/
def relu {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let zero ← UOp.const d 0.0
  let result ← UOp.maxBinary t.uop zero
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- ReLU6: min(max(x, 0), 6). -/
def relu6 {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let reluT ← relu t
  let six ← UOp.const d 6.0
  let minus ← UOp.sub t.uop six
  let minusT : StaticTensor s d := { uop := minus, requiresGrad := t.requiresGrad, h_shape := sorry_proof }
  let reluMinus ← relu minusT
  let out ← UOp.sub reluT.uop reluMinus.uop
  pure { uop := out, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Hardsigmoid: relu(alpha*x + beta) - relu(alpha*x + beta - 1). -/
def hardsigmoid {s : List Nat} {d : DType} (t : StaticTensor s d) (alpha : Float32 := 0.16666667)
    (beta : Float32 := 0.5) : TensorM (StaticTensor s d) := do
  let alphaConst ← UOp.const d alpha
  let betaConst ← UOp.const d beta
  let scaled ← UOp.mul t.uop alphaConst
  let shifted ← UOp.add scaled betaConst
  let shiftedT : StaticTensor s d := { uop := shifted, requiresGrad := t.requiresGrad, h_shape := sorry_proof }
  let reluShifted ← relu shiftedT
  let one ← UOp.const d 1.0
  let shiftedMinusOne ← UOp.sub shifted one
  let shiftedMinusOneT : StaticTensor s d := { uop := shiftedMinusOne, requiresGrad := t.requiresGrad, h_shape := sorry_proof }
  let reluShiftedMinusOne ← relu shiftedMinusOneT
  let out ← UOp.sub reluShifted.uop reluShiftedMinusOne.uop
  pure { uop := out, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Sigmoid: 1 / (1 + exp(-x)) -/
def sigmoid {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let negT ← neg t
  let expNeg ← exp negT
  let one ← UOp.const d 1.0
  let denom ← UOp.add expNeg.uop one
  let result ← UOp.div one denom
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Tanh via exp: (e^x - e^-x) / (e^x + e^-x) -/
def tanh {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let negT ← neg t
  let expPos ← exp t
  let expNeg ← exp negT
  let num ← UOp.sub expPos.uop expNeg.uop
  let denom ← UOp.add expPos.uop expNeg.uop
  let result ← UOp.div num denom
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Softplus: log(1 + exp(x)) -/
def softplus {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let expT ← exp t
  let one ← UOp.const d 1.0
  let onePlus ← UOp.add expT.uop one
  let onePlusT : StaticTensor s d := { uop := onePlus, requiresGrad := t.requiresGrad, h_shape := sorry_proof }
  log onePlusT

/-- GELU (tanh approximation). -/
def gelu {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let x2 ← UOp.mul t.uop t.uop
  let x3 ← UOp.mul x2 t.uop
  let c0 ← UOp.const d 0.044715
  let x3Scaled ← UOp.mul x3 c0
  let inner ← UOp.add t.uop x3Scaled
  let c1 ← UOp.const d 0.7978845608
  let scaled ← UOp.mul inner c1
  let scaledT : StaticTensor s d := { uop := scaled, requiresGrad := t.requiresGrad, h_shape := sorry_proof }
  let tanhScaled ← tanh scaledT
  let one ← UOp.const d 1.0
  let onePlus ← UOp.add tanhScaled.uop one
  let half ← UOp.const d 0.5
  let halfOnePlus ← UOp.mul onePlus half
  let result ← UOp.mul t.uop halfOnePlus
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Abs: |x| -/
def abs {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let zero ← UOp.const d 0.0
  let negT ← UOp.neg t.uop
  let isNeg ← UOp.cmplt t.uop zero
  let out ← UOp.where_ isNeg negT t.uop
  pure { uop := out, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Square: x * x. -/
def square {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  mul t t

/-- SiLU / Swish: x * sigmoid(x) -/
def silu {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let sig ← sigmoid t
  mul t sig

/-- Swish alias for SiLU. -/
def swish {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) :=
  silu t

/-- Hardswish: x * relu6(x+3) / 6. -/
def hardswish {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let three ← UOp.const d 3.0
  let tPlusThree ← UOp.add t.uop three
  let tPlusThreeT : StaticTensor s d := { uop := tPlusThree, requiresGrad := t.requiresGrad, h_shape := sorry_proof }
  let relu6T ← relu6 tPlusThreeT
  let mul1 ← UOp.mul t.uop relu6T.uop
  let oneSixth ← UOp.const d 0.16666667
  let out ← UOp.mul mul1 oneSixth
  pure { uop := out, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Leaky ReLU: x if x >= 0, alpha * x otherwise. -/
def leakyRelu {s : List Nat} {d : DType} (t : StaticTensor s d) (alpha : Float32 := 0.01)
    : TensorM (StaticTensor s d) := do
  let zero ← UOp.const d 0.0
  let alphaUop ← UOp.const d alpha
  let isNeg ← UOp.cmplt t.uop zero
  let negOut ← UOp.mul t.uop alphaUop
  let out ← UOp.where_ isNeg negOut t.uop
  pure { uop := out, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- ELU: x if x >= 0, alpha * (exp(x) - 1) otherwise. -/
def elu {s : List Nat} {d : DType} (t : StaticTensor s d) (alpha : Float32 := 1.0)
    : TensorM (StaticTensor s d) := do
  let zero ← UOp.const d 0.0
  let alphaUop ← UOp.const d alpha
  let isNeg ← UOp.cmplt t.uop zero
  let expT ← exp t
  let one ← UOp.const d 1.0
  let expm1 ← UOp.sub expT.uop one
  let negOut ← UOp.mul expm1 alphaUop
  let out ← UOp.where_ isNeg negOut t.uop
  pure { uop := out, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Log-sigmoid: log(sigmoid(x)) -/
def logSigmoid {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let sig ← sigmoid t
  log sig

/-- Clamp values to [lo, hi]. -/
def clamp {s : List Nat} {d : DType} (t : StaticTensor s d) (lo hi : Float32) : TensorM (StaticTensor s d) := do
  let loConst ← UOp.const d lo
  let hiConst ← UOp.const d hi
  let below ← UOp.cmplt t.uop loConst
  let above ← UOp.cmplt hiConst t.uop
  let clippedLo ← UOp.where_ below loConst t.uop
  let clipped ← UOp.where_ above hiConst clippedLo
  pure { uop := clipped, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Clip values to [lo, hi]. Alias for clamp. -/
def clip {s : List Nat} {d : DType} (t : StaticTensor s d) (lo hi : Float32) : TensorM (StaticTensor s d) :=
  clamp t lo hi

/-- Hardtanh clamps values to [minVal, maxVal]. -/
def hardtanh {s : List Nat} {d : DType} (t : StaticTensor s d) (minVal : Float32 := -1.0) (maxVal : Float32 := 1.0)
    : TensorM (StaticTensor s d) := do
  clamp t minVal maxVal

/-- Max along axis with keepdim -/
def maxAxis {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d) := do
  let result ← UOp.max_ t.uop [axis] keepdim
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def minAxis {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d) := do
  let negT ← neg t
  let maxNeg ← maxAxis negT axis keepdim
  neg maxNeg

/-- Max along axis with keepdim (statically checked axis). -/
def maxAxisF {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Fin s.length) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis.val] keepdim) d) := do
  let result ← UOp.max_ t.uop [axis.val] keepdim
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

def minAxisF {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Fin s.length) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis.val] keepdim) d) := do
  let negT ← neg t
  let maxNeg ← maxAxisF negT axis keepdim
  neg maxNeg

/-- Sum along axis with keepdim -/
def sumAxis {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d) := do
  let result ← UOp.sum t.uop [axis] keepdim
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Sum along axis with keepdim (statically checked axis). -/
def sumAxisF {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Fin s.length) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis.val] keepdim) d) := do
  let result ← UOp.sum t.uop [axis.val] keepdim
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Mean along axis with keepdim -/
def meanAxis {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d) := do
  let sumT ← sumAxis t axis keepdim
  let n := listGetD s axis 1
  let nConst ← UOp.const d (Float64.ofNat n).toFloat32
  let result ← UOp.div sumT.uop nConst
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Variance along axis with keepdim -/
def varAxis {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d) := do
  let meanT ← meanAxis t axis true
  let centered ← UOp.sub t.uop meanT.uop
  let centeredT : StaticTensor s d := { uop := centered, requiresGrad := t.requiresGrad, h_shape := sorry_proof }
  let sq ← mul centeredT centeredT
  meanAxis sq axis keepdim

/-- Layer norm over an axis (last axis by default). -/
def layerNorm {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat := s.length - 1)
    (eps : Float32 := 1.0e-5) : TensorM (StaticTensor s d) := do
  let meanT ← meanAxis t axis true
  let centered ← UOp.sub t.uop meanT.uop
  let centeredT : StaticTensor s d := { uop := centered, requiresGrad := t.requiresGrad, h_shape := sorry_proof }
  let sq ← mul centeredT centeredT
  let varT ← meanAxis sq axis true
  let epsConst ← UOp.const d eps
  let varEps ← UOp.add varT.uop epsConst
  let std ← UOp.sqrt varEps
  let invStd ← UOp.recip std
  let out ← UOp.mul centered invStd
  pure { uop := out, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- RMS norm over an axis (last axis by default). -/
def rmsNorm {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat := s.length - 1)
    (eps : Float32 := 1.0e-5) : TensorM (StaticTensor s d) := do
  let sq ← mul t t
  let meanSq ← meanAxis sq axis true
  let epsConst ← UOp.const d eps
  let varEps ← UOp.add meanSq.uop epsConst
  let rms ← UOp.sqrt varEps
  let invRms ← UOp.recip rms
  let out ← UOp.mul t.uop invRms
  pure { uop := out, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

private def classRangeF32 (n : Nat) : Array Float32 := Id.run do
  let mut out := Array.emptyWithCapacity n
  for i in [:n] do
    out := out.push (Float64.ofNat i).toFloat32
  return out

private def resolveDim (dim rank : Nat) : Nat :=
  if dim < rank then dim else
    panic! s!"dim {dim} out of range for rank {rank}"

private def swapLastPerm (rank dim : Nat) : List Nat :=
  let last := rank - 1
  (listRange rank).map fun i =>
    if i == dim then last else if i == last then dim else i

private def replaceLast (s : Shape) (n : Nat) : Shape :=
  if s.isEmpty then [n] else s.take (s.length - 1) ++ [n]

private def gatherShapeOk (shape idxShape : Shape) (dim : Nat) : Bool :=
  shape.length == idxShape.length &&
  listAll (fun i => if i == dim then true else listGetD shape i 0 >= listGetD idxShape i 0) (listRange shape.length)

private def oneHotLastF32 {s : Shape}
    (idx : StaticTensor s .float32) (numClasses : Nat)
    : TensorM (StaticTensor (replaceLast s numClasses) .bool) := do
  let classUop ← UOp.vconstF32 (classRangeF32 numClasses)
  let classes : StaticTensor [numClasses] .float32 := { uop := classUop, h_shape := sorry_proof }
  let eq ← UOp.cmpeq idx.uop classes.uop
  pure { uop := eq, h_shape := sorry_proof }

private def lastSliceShape (s : Shape) (_i : Nat) : Shape :=
  if s.isEmpty then [1] else s.take (s.length - 1) ++ [1]

/-- One-hot encoding for class indices (float32). -/
def oneHotF32 {batch numClasses : Nat}
    (targets : StaticTensor [batch] .float32)
    : TensorM (StaticTensor [batch, numClasses] .float32) := do
  let classUop ← UOp.vconstF32 (classRangeF32 numClasses)
  let classes : StaticTensor [numClasses] .float32 := { uop := classUop, h_shape := sorry_proof }
  let targets2 ← reshapeUnsafe targets [batch, 1]
  let classes2 ← reshapeUnsafe classes [1, numClasses]
  let cmp ← UOp.cmpeq targets2.uop classes2.uop
  let one ← UOp.const .float32 1.0
  let zero ← UOp.const .float32 0.0
  let out ← UOp.where_ cmp one zero
  pure { uop := out, h_shape := sorry_proof }

/-- Gather along an axis using index values (float32 indices). -/
def gatherF32 {s idxShape : Shape}
    (t : StaticTensor s .float32) (dim : Nat)
    (index : StaticTensor idxShape .float32)
    : TensorM (StaticTensor idxShape .float32) := do
  let tShape := t.uop.shape
  let idxShape' := index.uop.shape
  let dim' := resolveDim dim tShape.length
  if !gatherShapeOk tShape idxShape' dim' then
    panic! s!"gather: invalid shapes {tShape} {idxShape'} for dim {dim'}"
  let bounds := (listRange tShape.length).map fun i =>
    if i == dim' then (0, listGetD tShape i 0) else (0, listGetD idxShape' i 0)
  let tShrunk ← shrinkUnsafe t bounds
  let tUnsq ← unsqueezeUnsafe tShrunk tShape.length
  let tPerm ← permuteUnsafe tUnsq (swapLastPerm (tShape.length + 1) dim')
  let idxUnsq ← unsqueezeUnsafe index idxShape'.length
  let numClasses := listGetD tShape dim' 0
  let oneHot ← oneHotLastF32 idxUnsq numClasses
  let zero ← UOp.const .float32 0.0
  let masked ← UOp.where_ oneHot.uop tPerm.uop zero
  let reduced ← UOp.sum masked [idxShape'.length] false
  pure { uop := reduced, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Gather along an axis using int32 indices. -/
def gather {s idxShape : Shape}
    (t : StaticTensor s .float32) (dim : Nat)
    (index : StaticTensor idxShape .int32)
    : TensorM (StaticTensor idxShape .float32) := do
  let indexF ← cast index .float32
  gatherF32 t dim indexF

/-- Gather along the last axis using class indices (float32). -/
def gatherLastF32 {batch numClasses : Nat}
    (x : StaticTensor [batch, numClasses] .float32)
    (targets : StaticTensor [batch] .float32)
    : TensorM (StaticTensor [batch] .float32) := do
  let targets2 ← reshapeUnsafe targets [batch, 1]
  let gathered ← gatherF32 x 1 targets2
  reshapeUnsafe gathered [batch]

def gatherLast {batch numClasses : Nat}
    (x : StaticTensor [batch, numClasses] .float32)
    (targets : StaticTensor [batch] .int32)
    : TensorM (StaticTensor [batch] .float32) := do
  let targetsF ← cast targets .float32
  gatherLastF32 x targetsF

def scatterLastF32 {batch numClasses : Nat}
    (values : StaticTensor [batch] .float32)
    (targets : StaticTensor [batch] .float32)
    : TensorM (StaticTensor [batch, numClasses] .float32) := do
  let oneHot ← oneHotF32 targets
  let values2 ← reshapeUnsafe values [batch, 1]
  let valuesB ← expandUnsafe values2 [batch, numClasses]
  let out ← mul oneHot valuesB
  pure out

def scatterLast {batch numClasses : Nat}
    (values : StaticTensor [batch] .float32)
    (targets : StaticTensor [batch] .int32)
    : TensorM (StaticTensor [batch, numClasses] .float32) := do
  let targetsF ← cast targets .float32
  scatterLastF32 values targetsF

inductive ScatterReduce where
  | sum
  | mean
  | amax
  | amin
  deriving Repr, DecidableEq

private def preScatterF32 {s idxShape srcShape : Shape}
    (self : StaticTensor s .float32) (dim : Nat)
    (index : StaticTensor idxShape .float32) (src : StaticTensor srcShape .float32)
    : TensorM
        (StaticTensor (s ++ [listGetD s dim 0]) .float32 ×
         StaticTensor (s ++ [listGetD s dim 0]) .bool) := do
  let selfShape := self.uop.shape
  let idxShape' := index.uop.shape
  let srcShape' := src.uop.shape
  let dim' := resolveDim dim selfShape.length
  if selfShape.length != idxShape'.length || selfShape.length != srcShape'.length then
    panic! s!"scatter: rank mismatch {selfShape} {idxShape'} {srcShape'}"
  let ok := listAll (fun i =>
    if i == dim' then true
    else (listGetD selfShape i 0 >= listGetD idxShape' i 0) &&
         (listGetD srcShape' i 0 >= listGetD idxShape' i 0)) (listRange selfShape.length)
  if !ok then
    panic! s!"scatter: invalid shapes {selfShape} {idxShape'} {srcShape'} for dim {dim'}"
  let srcBounds := (listRange srcShape'.length).map fun i => (0, listGetD idxShape' i 0)
  let srcShrunk ← shrinkUnsafe src srcBounds
  let srcUnsq ← unsqueezeUnsafe srcShrunk srcShape'.length
  let numClasses := listGetD selfShape dim' 0
  let srcExpanded ← expandUnsafe srcUnsq (srcShrunk.uop.shape ++ [numClasses])
  let perm := swapLastPerm (selfShape.length + 1) dim'
  let srcT ← permuteUnsafe srcExpanded perm
  let idxUnsq ← unsqueezeUnsafe index idxShape'.length
  let maskT ← oneHotLastF32 idxUnsq numClasses
  let maskP ← permuteUnsafe maskT perm
  let padSpec := (listRange selfShape.length).map fun i =>
    if i == dim' then (0, 0)
    else
      let need := listGetD selfShape i 0
      let haveDim := listGetD srcT.uop.shape i 0
      if need < haveDim then (0, 0) else (0, need - haveDim)
  let padSpec := padSpec ++ [(0, 0)]
  let srcP ← padUnsafe srcT padSpec
  let maskP ← padUnsafe maskP padSpec
  let srcOut : StaticTensor (s ++ [listGetD s dim 0]) .float32 := { uop := srcP.uop, h_shape := sorry_proof }
  let maskOut : StaticTensor (s ++ [listGetD s dim 0]) .bool := { uop := maskP.uop, h_shape := sorry_proof }
  pure (srcOut, maskOut)

private def sliceLast {s : Shape} {d : DType}
    (t : StaticTensor s d) (i : Nat)
    : TensorM (StaticTensor (lastSliceShape s i) d) := do
  let shape := t.uop.shape
  if shape.isEmpty then
    panic! "sliceLast: empty shape"
  let last := shape.length - 1
  let bounds := (listRange last).map fun j => (0, listGetD shape j 0)
  let bounds := bounds ++ [(i, i + 1)]
  let sliced ← shrinkUnsafe t bounds
  pure { uop := sliced.uop, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

private def squeezeLast {s : Shape} {d : DType}
    (t : StaticTensor s d) : TensorM (StaticTensor (s.take (s.length - 1)) d) := do
  let shape := t.uop.shape
  if shape.isEmpty then
    panic! "squeezeLast: empty shape"
  let newShape := shape.take (shape.length - 1)
  let reshaped ← UOp.reshape t.uop newShape
  pure { uop := reshaped, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

private def whereSame {s : Shape} {d : DType}
    (cond : StaticTensor s .bool) (x y : StaticTensor s d)
    : TensorM (StaticTensor s d) := do
  let out ← UOp.where_ cond.uop x.uop y.uop
  pure { uop := out, requiresGrad := x.requiresGrad || y.requiresGrad, h_shape := sorry_proof }

private def maskedSetitemLast {s vShape : Shape}
    (target : StaticTensor s .float32)
    (values : StaticTensor vShape .float32)
    (mask : StaticTensor vShape .bool)
    : TensorM (StaticTensor s .float32) := do
  let shape := values.uop.shape
  if shape.isEmpty then
    panic! "maskedSetitemLast: empty shape"
  let lastDim := listGetD shape (shape.length - 1) 0
  if lastDim == 0 then
    pure target
  else
    let accVal0 ← sliceLast values 0
    let accMask0 ← sliceLast mask 0
    let mut accVal := accVal0
    let mut accMask := accMask0
    if lastDim > 1 then
      for i in [:lastDim] do
        if i == 0 then
          pure ()
        else
          let vi ← sliceLast values i
          let mi ← sliceLast mask i
          let accVal' ← whereSame mi vi accVal
          let accMask' ← bitor accMask mi
          accVal := accVal'
          accMask := accMask'
    let accValS ← squeezeLast accVal
    let accMaskS ← squeezeLast accMask
    let accValOut : StaticTensor s .float32 := { uop := accValS.uop, h_shape := sorry_proof }
    let accMaskOut : StaticTensor s .bool := { uop := accMaskS.uop, h_shape := sorry_proof }
    whereSame accMaskOut accValOut target

def scatterReduceF32 {s idxShape srcShape : Shape}
    (self : StaticTensor s .float32) (dim : Nat)
    (index : StaticTensor idxShape .float32) (src : StaticTensor srcShape .float32)
    (reduce : ScatterReduce) (includeSelf : Bool := true)
    : TensorM (StaticTensor s .float32) := do
  let (srcP, maskP) ← preScatterF32 self dim index src
  let lastAxis := srcP.uop.shape.length - 1
  let zero ← UOp.const .float32 0.0
  let one ← UOp.const .float32 1.0
  let maskOnes ← UOp.where_ maskP.uop one zero
  let countU ← UOp.sum maskOnes [lastAxis] false
  let noUpdateU ← UOp.cmpeq countU zero
  let noUpdate : StaticTensor s .bool := { uop := noUpdateU, h_shape := sorry_proof }
  let maskedU ← UOp.where_ maskP.uop srcP.uop zero
  let sumU ← UOp.sum maskedU [lastAxis] false
  let sumT : StaticTensor s .float32 := { uop := sumU, h_shape := sorry_proof }
  match reduce with
  | .sum =>
    if includeSelf then
      add sumT self
    else
      let invU ← UOp.where_ noUpdate.uop self.uop zero
      let invT : StaticTensor s .float32 := { uop := invU, h_shape := sorry_proof }
      add sumT invT
  | .mean =>
    let baseNum ←
      if includeSelf then
        add sumT self
      else
        let invU ← UOp.where_ noUpdate.uop self.uop zero
        let invT : StaticTensor s .float32 := { uop := invU, h_shape := sorry_proof }
        add sumT invT
    let countAddU ←
      if includeSelf then
        UOp.add countU one
      else
        let addOneU ← UOp.where_ noUpdate.uop one zero
        UOp.add countU addOneU
    let countT : StaticTensor s .float32 := { uop := countAddU, h_shape := sorry_proof }
    div baseNum countT
  | .amax =>
    let negInf ← UOp.const .float32 (-1.0e38)
    let maskedMax ← UOp.where_ maskP.uop srcP.uop negInf
    let maxU ← UOp.max_ maskedMax [lastAxis] false
    let maxT : StaticTensor s .float32 := { uop := maxU, h_shape := sorry_proof }
    if includeSelf then
      let outU ← UOp.maxBinary maxT.uop self.uop
      pure { uop := outU, h_shape := sorry_proof }
    else
      let outU ← UOp.where_ noUpdate.uop self.uop maxT.uop
      pure { uop := outU, h_shape := sorry_proof }
  | .amin =>
    let negInf ← UOp.const .float32 (-1.0e38)
    let negSrc ← neg srcP
    let negSelf ← neg self
    let maskedNeg ← UOp.where_ maskP.uop negSrc.uop negInf
    let maxNegU ← UOp.max_ maskedNeg [lastAxis] false
    let maxNegT : StaticTensor s .float32 := { uop := maxNegU, h_shape := sorry_proof }
    let mergedNegU ←
      if includeSelf then
        UOp.maxBinary maxNegT.uop negSelf.uop
      else
        UOp.where_ noUpdate.uop negSelf.uop maxNegT.uop
    let mergedNegT : StaticTensor s .float32 := { uop := mergedNegU, h_shape := sorry_proof }
    neg mergedNegT

def scatterReduce {s idxShape srcShape : Shape}
    (self : StaticTensor s .float32) (dim : Nat)
    (index : StaticTensor idxShape .int32) (src : StaticTensor srcShape .float32)
    (reduce : ScatterReduce) (includeSelf : Bool := true)
    : TensorM (StaticTensor s .float32) := do
  let indexF ← cast index .float32
  scatterReduceF32 self dim indexF src reduce includeSelf

def scatterF32 {s idxShape srcShape : Shape}
    (self : StaticTensor s .float32) (dim : Nat)
    (index : StaticTensor idxShape .float32) (src : StaticTensor srcShape .float32)
    : TensorM (StaticTensor s .float32) := do
  let (srcP, maskP) ← preScatterF32 self dim index src
  maskedSetitemLast self srcP maskP

def scatter {s idxShape srcShape : Shape}
    (self : StaticTensor s .float32) (dim : Nat)
    (index : StaticTensor idxShape .int32) (src : StaticTensor srcShape .float32)
    : TensorM (StaticTensor s .float32) := do
  let indexF ← cast index .float32
  scatterF32 self dim indexF src

/-- Log-sum-exp along axis (numerically stable). -/
def logsumexpAxis {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d) := do
  let maxVal ← UOp.max_ t.uop [axis] true
  let shifted ← UOp.sub t.uop maxVal
  let shiftedT : StaticTensor s d := { uop := shifted, requiresGrad := t.requiresGrad, h_shape := sorry_proof }
  let expShifted ← exp shiftedT
  let sumExp ← UOp.sum expShifted.uop [axis] true
  let sumExpT : StaticTensor (Shape.reduce s [axis] true) d := { uop := sumExp, requiresGrad := t.requiresGrad, h_shape := sorry_proof }
  let logSum ← log sumExpT
  let outKeep ← UOp.add logSum.uop maxVal
  match keepdim with
  | true =>
    pure { uop := outKeep, requiresGrad := t.requiresGrad, h_shape := sorry_proof }
  | false =>
    let outKeepT : StaticTensor (Shape.reduce s [axis] true) d := { uop := outKeep, requiresGrad := t.requiresGrad, h_shape := sorry_proof }
    reshapeUnsafe outKeepT (Shape.reduce s [axis] false)

/-- Log-softmax along an axis (stable). -/
def logSoftmaxAxis {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat) : TensorM (StaticTensor s d) := do
  let logSum ← logsumexpAxis t axis true
  let result ← UOp.sub t.uop logSum.uop
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Softmax along an axis (stable). -/
def softmaxAxis {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat) : TensorM (StaticTensor s d) := do
  let maxVal ← UOp.max_ t.uop [axis] true
  let shifted ← UOp.sub t.uop maxVal
  let shiftedT : StaticTensor s d := { uop := shifted, requiresGrad := t.requiresGrad, h_shape := sorry_proof }
  let expShifted ← exp shiftedT
  let sumExp ← UOp.sum expShifted.uop [axis] true
  let out ← UOp.div expShifted.uop sumExp
  pure { uop := out, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Softmax along last axis: exp(x - max(x)) / sum(exp(x - max(x))) -/
def softmax {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  softmaxAxis t (s.length - 1)

/-- Log-softmax along last axis: x - max(x) - log(sum(exp(x - max(x)))) -/
def logSoftmax {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  logSoftmaxAxis t (s.length - 1)

private def argmaxF32 {batch n : Nat} (t : StaticTensor [batch, n] .float32)
    : TensorM (StaticTensor [batch] .int32) := do
  let maxVal ← UOp.max_ t.uop [1] true
  let eq ← UOp.cmpeq t.uop maxVal
  let eqT : StaticTensor [batch, n] .bool := { uop := eq, h_shape := sorry_proof }
  let eqF ← cast eqT .float32
  let classesUop ← UOp.vconstF32 (classRangeF32 n)
  let classes : StaticTensor [n] .float32 := { uop := classesUop, h_shape := sorry_proof }
  let classes2 ← reshapeUnsafe classes [1, n]
  let classesB ← expandUnsafe classes2 [batch, n]
  let prod ← mul eqF classesB
  let sumC ← sumAxis prod 1 false
  cast sumC .int32

/-- Argmax along last axis - returns indices (non-differentiable). -/
def argmax {batch n : Nat} {d : DType} (t : StaticTensor [batch, n] d)
    : TensorM (StaticTensor [batch] .int32) := do
  let tF ← cast t .float32
  argmaxF32 tF

/-- Argmin along last axis - returns indices (non-differentiable). -/
def argmin {batch n : Nat} {d : DType} (t : StaticTensor [batch, n] d)
    : TensorM (StaticTensor [batch] .int32) := do
  let tF ← cast t .float32
  let negT ← neg tF
  argmaxF32 negT

/-- Scalar multiplication: t * scalar -/
def scale {s : List Nat} {d : DType} (t : StaticTensor s d) (scalar : Float32)
    : TensorM (StaticTensor s d) := do
  let scalarUop ← UOp.const d scalar
  let result ← UOp.mul t.uop scalarUop
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Add scalar: t + scalar -/
def addScalar {s : List Nat} {d : DType} (t : StaticTensor s d) (scalar : Float32)
    : TensorM (StaticTensor s d) := do
  let scalarUop ← UOp.const d scalar
  let result ← UOp.add t.uop scalarUop
  pure { uop := result, requiresGrad := t.requiresGrad, h_shape := sorry_proof }

/-- Cross-entropy loss with log-softmax
    logits: [batch, numClasses], targets: [batch] (class indices as floats)
    Returns scalar loss -/
def crossEntropyLoss {batch numClasses : Nat}
    (logits : StaticTensor [batch, numClasses] .float32)
    (targets : StaticTensor [batch] .float32)
    : TensorM (Scalar .float32) := do
  let logProbs ← logSoftmax logits
  let picked ← gatherLastF32 logProbs targets
  let negPicked ← neg picked
  mean negPicked

/-- Cross-entropy loss with one-hot targets.
    logits: [batch, numClasses], targets: [batch, numClasses] (one-hot)
    Returns scalar loss: mean over batch of -sum(target * log_softmax(logits)) -/
def crossEntropyOneHot {batch numClasses : Nat} {d : DType}
    (logits : StaticTensor [batch, numClasses] d)
    (targets : StaticTensor [batch, numClasses] d)
    : TensorM (Scalar d) := do
  let logProbs ← logSoftmax logits
  let prod ← mul logProbs targets
  let axis := ([batch, numClasses] : List Nat).length - 1
  let sumC ← sumAxis prod axis true
  let negSum ← neg sumC
  mean negSum

/-- Negative log likelihood loss (assumes log_softmax input)
    log_probs: [batch, numClasses], already log-softmax'd
    target_indices: [batch] containing class indices
    For MVP: averages all log probs (placeholder until gather/index support) -/
def nllLoss {batch numClasses : Nat}
    (logProbs : StaticTensor [batch, numClasses] .float32)
    (targets : StaticTensor [batch] .float32)
    : TensorM (Scalar .float32) := do
  let picked ← gatherLastF32 logProbs targets
  let negPicked ← neg picked
  mean negPicked

/-- Smooth L1 (Huber) loss with beta (default 1.0). -/
def smoothL1Loss {s : List Nat} (pred target : StaticTensor s .float32) (beta : Float32 := 1.0)
    : TensorM (Scalar .float32) := do
  let diff ← sub pred target
  let absDiff ← abs diff
  let betaConst ← UOp.const .float32 beta
  let half ← UOp.const .float32 0.5
  let isSmall ← UOp.cmplt absDiff.uop betaConst
  let sq ← mul diff diff
  let sqHalf ← UOp.mul sq.uop half
  let sqScaled ← UOp.div sqHalf betaConst
  let betaHalf ← UOp.mul betaConst half
  let linTerm ← UOp.sub absDiff.uop betaHalf
  let out ← UOp.where_ isSmall sqScaled linTerm
  let outT : StaticTensor s .float32 := { uop := out, requiresGrad := pred.requiresGrad || target.requiresGrad, h_shape := sorry_proof }
  mean outT

/-- Binary cross-entropy loss (expects probabilities in [0, 1]). -/
def binaryCrossEntropy {s : List Nat}
    (pred target : StaticTensor s .float32) (eps : Float32 := 1.0e-7)
    : TensorM (Scalar .float32) := do
  let predClamped ← clamp pred eps (1.0 - eps)
  let logPred ← log predClamped
  let one ← UOp.const .float32 1.0
  let oneMinusPred ← UOp.sub one predClamped.uop
  let oneMinusPredT : StaticTensor s .float32 := { uop := oneMinusPred, requiresGrad := pred.requiresGrad, h_shape := sorry_proof }
  let logOneMinusPred ← log oneMinusPredT
  let oneMinusTarget ← UOp.sub one target.uop
  let oneMinusTargetT : StaticTensor s .float32 := { uop := oneMinusTarget, requiresGrad := target.requiresGrad, h_shape := sorry_proof }
  let term1 ← mul target logPred
  let term2 ← mul oneMinusTargetT logOneMinusPred
  let sumTerms ← add term1 term2
  let negSum ← neg sumTerms
  mean negSum

/-- Binary cross-entropy with logits (numerically stable). -/
def binaryCrossEntropyWithLogits {s : List Nat}
    (logits target : StaticTensor s .float32) : TensorM (Scalar .float32) := do
  let zero ← UOp.const .float32 0.0
  let maxZero ← UOp.maxBinary logits.uop zero
  let absLogits ← abs logits
  let negAbs ← neg absLogits
  let expNegAbs ← exp negAbs
  let one ← UOp.const .float32 1.0
  let onePlus ← UOp.add expNegAbs.uop one
  let onePlusT : StaticTensor s .float32 := { uop := onePlus, requiresGrad := logits.requiresGrad, h_shape := sorry_proof }
  let logOnePlus ← log onePlusT
  let prod ← UOp.mul logits.uop target.uop
  let tmp ← UOp.sub maxZero prod
  let lossUop ← UOp.add tmp logOnePlus.uop
  let lossT : StaticTensor s .float32 := { uop := lossUop, requiresGrad := logits.requiresGrad, h_shape := sorry_proof }
  mean lossT

/-- Mean squared error loss. -/
def mseLoss {s : List Nat} {d : DType}
    (pred target : StaticTensor s d) : TensorM (Scalar d) := do
  let diff ← sub pred target
  let sq ← mul diff diff
  mean sq

/-- Matrix multiplication: [m, k] @ [k, n] -> [m, n] -/
def matmul {m k n : Nat} {d : DType}
    (a : Matrix m k d) (b : Matrix k n d)
    : TensorM (Matrix m n d) := do
  let outUop ← UOp.contract2D a.uop b.uop
  pure {
    uop := outUop
    h_shape := sorry_proof
    requiresGrad := a.requiresGrad || b.requiresGrad
  }

/-- Fully-connected (linear) layer: X @ W -> [batch, out]. -/
def linear {batch inDim outDim : Nat} {d : DType}
    (x : Matrix batch inDim d) (w : Matrix inDim outDim d)
    : TensorM (Matrix batch outDim d) := do
  matmul x w

/-- Fully-connected layer with optional bias: X @ W (+ b) -/
def linearOpt {batch inDim outDim : Nat} {d : DType}
    (x : Matrix batch inDim d) (w : Matrix inDim outDim d)
    (bias : Option (Vector outDim d) := none)
    : TensorM (Matrix batch outDim d) := do
  let y ← matmul x w
  match bias with
  | none => pure y
  | some b =>
    let yb ← addB y b
    pure { uop := yb.uop, h_shape := sorry_proof, requiresGrad := yb.requiresGrad }

/-- Fully-connected layer with bias: X @ W + b (broadcasted over batch). -/
def linearBias {batch inDim outDim : Nat} {d : DType}
    (x : Matrix batch inDim d) (w : Matrix inDim outDim d) (b : Vector outDim d)
    : TensorM (Matrix batch outDim d) := do
  linearOpt x w (some b)

/-- Batched matrix multiplication with broadcast on the batch dim:
    [b1, m, k] @ [b2, k, n] -> [max b1 b2, m, n]. -/
def bmatmul {b1 b2 m k n : Nat} {d : DType}
    (a : BMatrix b1 m k d) (b : BMatrix b2 k n d)
    : TensorM (BMatrix (Nat.max b1 b2) m n d) := do
  let outUop ← UOp.contract2D a.uop b.uop
  pure {
    uop := outUop
    h_shape := sorry_proof
    requiresGrad := a.requiresGrad || b.requiresGrad
  }

-- ============================================================================
-- Initialization
-- ============================================================================

/-- Generate uniform random tensor in [low, high) range.
    Uses: rand * (high - low) + low -/
def uniformInit (shape : Shape) (dt : DType) (low high : Float32) (seed : Nat)
    : TensorM (StaticTensor shape dt) := do
  -- rand produces [0, 1)
  let r ← Tensor.rand shape dt seed
  -- Scale to [low, high): r * (high - low) + low
  let range := high - low
  let rangeT ← Tensor.full shape dt range
  let lowT ← Tensor.full shape dt low
  let scaled ← mul r rangeT
  add scaled lowT

-- ============================================================================
-- Convolution Operations (pool/im2col + matmul)
-- ============================================================================

/-- Placeholder conv2d - returns correctly shaped output tensor.
    Full implementation requires UOp-level pool operation. -/
def conv2dPlaceholder {batch cin cout h w kH kW hOut wOut : Nat} {d : DType}
    (x : StaticTensor [batch, cin, h, w] d)
    (weight : StaticTensor [cout, cin, kH, kW] d)
    (bias : Option (StaticTensor [cout] d) := none)
    (_padding : Nat := 0)
    (_stride : Nat := 1)
    (_dilation : Nat := 1)
    : TensorM (StaticTensor [batch, cout, hOut, wOut] d) := do
  -- Create output buffer with correct shape
  let outShape := [batch, cout, hOut, wOut]
  let out ← UOp.buffer d outShape
  let biasGrad := match bias with | none => false | some b => b.requiresGrad
  let reqGrad := x.requiresGrad || weight.requiresGrad || biasGrad
  pure { uop := out, requiresGrad := reqGrad, h_shape := sorry_proof }

/-- Placeholder maxPool2d - returns correctly shaped output tensor. -/
def maxPool2dPlaceholder {batch cin h w hOut wOut : Nat} {d : DType}
    (x : StaticTensor [batch, cin, h, w] d)
    (_kernelSize : Nat)
    (_stride : Nat := 0)
    : TensorM (StaticTensor [batch, cin, hOut, wOut] d) := do
  let outShape := [batch, cin, hOut, wOut]
  let out ← UOp.buffer d outShape
  pure { uop := out, requiresGrad := x.requiresGrad, h_shape := sorry_proof }

/-- Placeholder avgPool2d - returns correctly shaped output tensor. -/
def avgPool2dPlaceholder {batch cin h w hOut wOut : Nat} {d : DType}
    (x : StaticTensor [batch, cin, h, w] d)
    (_kernelSize : Nat)
    (_stride : Nat := 0)
    : TensorM (StaticTensor [batch, cin, hOut, wOut] d) := do
  let outShape := [batch, cin, hOut, wOut]
  let out ← UOp.buffer d outShape
  pure { uop := out, requiresGrad := x.requiresGrad, h_shape := sorry_proof }

/-- Pad 1D tensor with symmetric padding on W dimension.
    Input:  [batch, channels, width]
    Output: [batch, channels, width + 2*padUnsafe] -/
def pad1d {batch cin w : Nat} {d : DType}
    (x : StaticTensor [batch, cin, w] d)
    (padW : Nat)
    : TensorM (StaticTensor [batch, cin, w + 2*padW] d) := do
  let padding := [(0, 0), (0, 0), (padW, padW)]
  let result ← padUnsafe x padding
  pure { uop := result.uop, requiresGrad := x.requiresGrad, h_shape := sorry_proof }

/-- Pad 2D tensor with symmetric padding on H and W dimensions.
    Input:  [batch, channels, height, width]
    Output: [batch, channels, height + 2*padUnsafe, width + 2*padUnsafe] -/
def pad2d {batch cin h w : Nat} {d : DType}
    (x : StaticTensor [batch, cin, h, w] d)
    (padH padW : Nat)
    : TensorM (StaticTensor [batch, cin, h + 2*padH, w + 2*padW] d) := do
  let padding := [(0, 0), (0, 0), (padH, padH), (padW, padW)]
  let result ← padUnsafe x padding
  pure { uop := result.uop, requiresGrad := x.requiresGrad, h_shape := sorry_proof }

/-- Max pooling 2D operation using pool/im2col + reduce.
    Input:  [batch, channels, height, width]
    Output: [batch, channels, outHeight, outWidth] -/
def maxPool2d {batch cin h w : Nat} {d : DType}
    (x : StaticTensor [batch, cin, h, w] d)
    (kernelSize : Nat)
    (stride : Nat)
    (padding : Nat := 0)
    : TensorM (StaticTensor (Shape.pool2dShape [batch, cin, h, w] kernelSize padding stride) d) := do
  -- Step 1: Pad if needed
  let xPadded ← if padding > 0 then
    let padded ← pad2d x padding padding
    pure { uop := padded.uop, requiresGrad := x.requiresGrad, h_shape := sorry_proof }
  else
    pure x

  -- Step 2: Apply pool/im2col to get patches
  -- Result shape: [batch, cin, hOut, wOut, kH, kW]
  let patches ← pool xPadded [kernelSize, kernelSize] [stride, stride] [1, 1]

  -- Step 3: Take max over the kernel dimensions (last 2 dims)
  -- Reduce over axis -1 (kW) then axis -1 (kH)
  let patchShape := patches.uop.shape
  let axis1 := patchShape.length - 1  -- kW axis
  let reduced1 ← UOp.max_ patches.uop [axis1] false
  let axis2 := patchShape.length - 2  -- kH axis (now shifted)
  let reduced2 ← UOp.max_ reduced1 [axis2] false

  pure { uop := reduced2, requiresGrad := x.requiresGrad, h_shape := sorry_proof }

/-- Average pooling 2D operation using pool/im2col + reduce.
    Input:  [batch, channels, height, width]
    Output: [batch, channels, outHeight, outWidth] -/
def avgPool2d {batch cin h w : Nat} {d : DType}
    (x : StaticTensor [batch, cin, h, w] d)
    (kernelSize : Nat)
    (stride : Nat)
    (padding : Nat := 0)
    : TensorM (StaticTensor (Shape.pool2dShape [batch, cin, h, w] kernelSize padding stride) d) := do
  -- Step 1: Pad if needed
  let xPadded ← if padding > 0 then
    let padded ← pad2d x padding padding
    pure { uop := padded.uop, requiresGrad := x.requiresGrad, h_shape := sorry_proof }
  else
    pure x

  -- Step 2: Apply pool/im2col to get patches
  -- Result shape: [batch, cin, hOut, wOut, kH, kW]
  let patches ← pool xPadded [kernelSize, kernelSize] [stride, stride] [1, 1]

  -- Step 3: Take mean over the kernel dimensions (last 2 dims)
  let patchShape := patches.uop.shape
  let axis1 := patchShape.length - 1  -- kW axis
  let sum1 ← UOp.sum patches.uop [axis1] false
  let axis2 := patchShape.length - 2  -- kH axis (now shifted)
  let sum2 ← UOp.sum sum1 [axis2] false

  -- Divide by kernel area for mean
  let kernelArea := (kernelSize * kernelSize : Nat)
  let divisor ← UOp.const d (Float64.ofNat kernelArea).toFloat32
  let result ← UOp.div sum2 divisor

  pure { uop := result, requiresGrad := x.requiresGrad, h_shape := sorry_proof }

/-- Conv1d operation using pool/im2col + matmul.
    Input:  [batch, inChannels, width]
    Weight: [outChannels, inChannels, kernelW]
    Output: [batch, outChannels, outWidth] -/
def conv1d {batch cin cout w kW : Nat} {d : DType}
    (x : StaticTensor [batch, cin, w] d)
    (weight : StaticTensor [cout, cin, kW] d)
    (bias : Option (StaticTensor [cout] d) := none)
    (padding : Nat := 0)
    (stride : Nat := 1)
    (dilation : Nat := 1)
    : TensorM (StaticTensor (Shape.conv1dOut [batch, cin, w]
                                              [cout, cin, kW]
                                              padding stride dilation) d) := do
  let wOut := Shape.convOutDim w padding dilation kW stride

  -- Step 1: Pad input if needed
  let xPadded ← if padding > 0 then
    let padded ← pad1d x padding
    pure { uop := padded.uop, requiresGrad := x.requiresGrad, h_shape := sorry_proof }
  else
    pure x

  -- Step 2: Apply pool/im2col to get patches
  -- Input: [batch, cin, wPadded]
  -- Output: [batch, cin, wOut, kW]
  let patches ← pool xPadded [kW] [stride] [dilation]

  -- Step 3: Reshape patches for matmul
  -- [batch, cin, wOut, kW] -> [batch * wOut, cin * kW]
  let patchFlat := batch * wOut
  let kernelFlat := cin * kW
  let patchesReshaped ← reshapeUnsafe patches [patchFlat, kernelFlat]

  -- Step 4: Reshape weight
  -- [cout, cin, kW] -> [cout, cin * kW]
  let weightReshaped ← reshapeUnsafe weight [cout, kernelFlat]

  -- Step 5: Transpose weight for matmul: [cout, kernelFlat] -> [kernelFlat, cout]
  let weightT ← T weightReshaped

  -- Step 6: Matmul: [patchFlat, kernelFlat] @ [kernelFlat, cout] = [patchFlat, cout]
  let mm ← matmul patchesReshaped weightT

  -- Step 7: Reshape to [batch, wOut, cout]
  let mmReshaped ← reshapeUnsafe mm [batch, wOut, cout]

  -- Step 8: Permute to [batch, cout, wOut]
  let result ← permuteUnsafe mmReshaped [0, 2, 1]

  -- Step 9: Add bias if present
  let finalUop ← match bias with
  | none => pure result.uop
  | some b =>
    let biasReshaped ← reshapeUnsafe b [1, cout, 1]
    UOp.add result.uop biasReshaped.uop

  let biasGrad := match bias with | none => false | some b => b.requiresGrad
  let reqGrad := x.requiresGrad || weight.requiresGrad || biasGrad
  pure { uop := finalUop, requiresGrad := reqGrad, h_shape := sorry_proof }

/-- Conv2d operation using pool/im2col + matmul.
    Input:  [batch, inChannels, height, width]
    Weight: [outChannels, inChannels, kernelH, kernelW]
    Output: [batch, outChannels, outHeight, outWidth]

    Algorithm:
    1. Pad input if needed
    2. Apply pool/im2col to get patches: [batch, cin, hOut, wOut, kH, kW]
    3. Reshape for matmul: patches -> [batch*hOut*wOut, cin*kH*kW]
    4. Reshape weight: [cout, cin*kH*kW]
    5. Matmul: [batch*hOut*wOut, cin*kH*kW] @ [cin*kH*kW, cout]^T = [batch*hOut*wOut, cout]
    6. Reshape to [batch, hOut, wOut, cout]
    7. Permute to [batch, cout, hOut, wOut]
    8. Add bias if present -/
def conv2d {batch cin cout h w kH kW : Nat} {d : DType}
    (x : StaticTensor [batch, cin, h, w] d)
    (weight : StaticTensor [cout, cin, kH, kW] d)
    (bias : Option (StaticTensor [cout] d) := none)
    (padding : Nat := 0)
    (stride : Nat := 1)
    (dilation : Nat := 1)
    : TensorM (StaticTensor (Shape.conv2dOut [batch, cin, h, w]
                                              [cout, cin, kH, kW]
                                              padding stride dilation) d) := do
  -- Compute output dimensions
  let hOut := Shape.convOutDim h padding dilation kH stride
  let wOut := Shape.convOutDim w padding dilation kW stride

  -- Step 1: Pad input if needed
  let xPadded ← if padding > 0 then
    let padded ← pad2d x padding padding
    pure { uop := padded.uop, requiresGrad := x.requiresGrad, h_shape := sorry_proof }
  else
    pure x

  -- Step 2: Apply pool/im2col to get patches
  -- Input to pool: [batch, cin, hPadded, wPadded]
  -- Output: [batch, cin, hOut, wOut, kH, kW]
  let patches ← pool xPadded [kH, kW] [stride, stride] [dilation, dilation]

  -- Step 3: Reshape patches for matmul
  -- [batch, cin, hOut, wOut, kH, kW] -> [batch * hOut * wOut, cin * kH * kW]
  let patchFlat := batch * hOut * wOut
  let kernelFlat := cin * kH * kW
  let patchesReshaped ← reshapeUnsafe patches [patchFlat, kernelFlat]

  -- Step 4: Reshape weight
  -- [cout, cin, kH, kW] -> [cout, cin * kH * kW]
  let weightReshaped ← reshapeUnsafe weight [cout, kernelFlat]

  -- Step 5: Transpose weight for matmul: [cout, kernelFlat] -> [kernelFlat, cout]
  let weightT ← T weightReshaped

  -- Step 6: Matmul: [patchFlat, kernelFlat] @ [kernelFlat, cout] = [patchFlat, cout]
  let mm ← matmul patchesReshaped weightT

  -- Step 7: Reshape to [batch, hOut, wOut, cout]
  let mmReshaped ← reshapeUnsafe mm [batch, hOut, wOut, cout]

  -- Step 8: Permute to [batch, cout, hOut, wOut]
  let result ← permuteUnsafe mmReshaped [0, 3, 1, 2]

  -- Step 9: Add bias if present
  let finalUop ← match bias with
  | none => pure result.uop
  | some b =>
    -- Reshape bias [cout] -> [1, cout, 1, 1] for broadcasting
    let biasReshaped ← reshapeUnsafe b [1, cout, 1, 1]
    UOp.add result.uop biasReshaped.uop

  let biasGrad := match bias with | none => false | some b => b.requiresGrad
  let reqGrad := x.requiresGrad || weight.requiresGrad || biasGrad
  pure { uop := finalUop, requiresGrad := reqGrad, h_shape := sorry_proof }

/-- Depthwise 2D convolution: each input channel is convolved with its own filter.
    This is a specialized case of grouped convolution where groups = cin = cout.

    Weight shape: [cin, 1, kH, kW] (each channel has one 1×kH×kW filter)

    Implementation using batched matmul (fast, like Python tinygrad):
    1. Pool to get patches: [batch, cin, hOut, wOut, kH, kW]
    2. Reshape patches: [batch, cin, hOut*wOut, kH*kW]
    3. Reshape weight: [cin, 1, kH, kW] -> [1, cin, kH*kW, 1]
    4. Batched matmul: [batch, cin, hOut*wOut, kH*kW] @ [1, cin, kH*kW, 1]
       -> broadcasts batch dims, performs matmul for each (batch, cin)
       -> result: [batch, cin, hOut*wOut, 1]
    5. Squeeze and reshapeUnsafe to [batch, cin, hOut, wOut]

    This uses a single batched CONTRACT operation instead of expandUnsafe+multiply+sum.
-/
def depthwiseConv2d {batch cin h w kH kW : Nat} {d : DType}
    (x : StaticTensor [batch, cin, h, w] d)
    (weight : StaticTensor [cin, 1, kH, kW] d)
    (bias : Option (StaticTensor [cin] d) := none)
    (padding : Nat := 0)
    (stride : Nat := 1)
    (dilation : Nat := 1)
    : TensorM (StaticTensor (Shape.conv2dOut [batch, cin, h, w]
                                              [cin, 1, kH, kW]
                                              padding stride dilation) d) := do
  let hOut := Shape.convOutDim h padding dilation kH stride
  let wOut := Shape.convOutDim w padding dilation kW stride
  let spatialOut := hOut * wOut
  let kernelFlat := kH * kW

  -- Step 1: Pad if needed
  let xPadded ← if padding > 0 then
    let padded ← pad2d x padding padding
    pure { uop := padded.uop, requiresGrad := x.requiresGrad, h_shape := sorry_proof }
  else
    pure x

  -- Step 2: Pool to get patches: [batch, cin, hOut, wOut, kH, kW]
  let patches ← pool xPadded [kH, kW] [stride, stride] [dilation, dilation]

  -- Step 3: Reshape patches for batched matmul: [batch, cin, hOut*wOut, kH*kW]
  let patchesReshaped ← UOp.reshape patches.uop [batch, cin, spatialOut, kernelFlat]

  -- Step 4: Reshape weight for batched matmul: [cin, 1, kH, kW] -> [1, cin, kH*kW, 1]
  -- This allows broadcasting with batch dimension
  let weightReshaped ← UOp.reshape weight.uop [1, cin, kernelFlat, 1]

  -- Step 5: Batched matmul using UOp.contract2D
  -- [batch, cin, hOut*wOut, kH*kW] @ [1, cin, kH*kW, 1]
  -- Batch dims [batch, cin] and [1, cin] broadcast to [batch, cin]
  -- Result: [batch, cin, hOut*wOut, 1]
  let mmResult ← UOp.contract2D patchesReshaped weightReshaped

  -- Step 6: Squeeze and reshapeUnsafe to [batch, cin, hOut, wOut]
  let result ← UOp.reshape mmResult [batch, cin, hOut, wOut]

  -- Step 7: Add bias if present
  let finalUop ← match bias with
  | none => pure result
  | some b =>
    let biasReshaped ← UOp.reshape b.uop [1, cin, 1, 1]
    UOp.add result biasReshaped

  let biasGrad := match bias with | none => false | some b => b.requiresGrad
  let reqGrad := x.requiresGrad || weight.requiresGrad || biasGrad
  pure { uop := finalUop, requiresGrad := reqGrad, h_shape := sorry_proof }

end StaticTensor

end TinyGrad4
