import TinyGrad4.Tensor.Tensor
import TinyGrad4.UOp.Typed
import TinyGrad4.Tensor.Movement

set_option maxHeartbeats 800000

namespace TinyGrad4

namespace StaticTensor

private def ofTU {op : Ops} {s : Shape} {r : Nat} {d : DType} (t : TUOp op s r d) (reqGrad : Bool) :
    StaticTensor s d :=
  StaticTensor.ofTUOp t reqGrad

private def ofTUCast {op : Ops} {s : Shape} {r : Nat} {d : DType} (t : TUOp op s r d) (s' : Shape)
    (reqGrad : Bool) : StaticTensor s' d :=
  StaticTensor.ofTUOp (TUOp.castShape t s') reqGrad

def add {s : List Nat} {d : DType} (t1 t2 : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← TUOp.add t1.tuop t2.tuop
  pure (ofTU result (t1.requiresGrad || t2.requiresGrad))

def addB {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) d) := do
  let result ← TUOp.add t1.tuop t2.tuop
  pure (ofTUCast result (Shape.broadcastOut s1 s2) (t1.requiresGrad || t2.requiresGrad))

def mul {s : List Nat} {d : DType} (t1 t2 : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← TUOp.mul t1.tuop t2.tuop
  pure (ofTU result (t1.requiresGrad || t2.requiresGrad))

def mulB {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) d) := do
  let result ← TUOp.mul t1.tuop t2.tuop
  pure (ofTUCast result (Shape.broadcastOut s1 s2) (t1.requiresGrad || t2.requiresGrad))

def sub {s : List Nat} {d : DType} (t1 t2 : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← TUOp.sub t1.tuop t2.tuop
  pure (ofTU result (t1.requiresGrad || t2.requiresGrad))

def subB {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) d) := do
  let result ← TUOp.sub t1.tuop t2.tuop
  pure (ofTUCast result (Shape.broadcastOut s1 s2) (t1.requiresGrad || t2.requiresGrad))

def div {s : List Nat} {d : DType} (t1 t2 : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← TUOp.div t1.tuop t2.tuop
  pure (ofTU result (t1.requiresGrad || t2.requiresGrad))

def divB {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) d) := do
  let result ← TUOp.div t1.tuop t2.tuop
  pure (ofTUCast result (Shape.broadcastOut s1 s2) (t1.requiresGrad || t2.requiresGrad))

def pow {s : List Nat} {d : DType} (t1 t2 : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← TUOp.binaryOp .POW t1.tuop t2.tuop
  pure (ofTU result (t1.requiresGrad || t2.requiresGrad))

def powB {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) d) := do
  let result ← TUOp.binaryOp .POW t1.tuop t2.tuop
  pure (ofTUCast result (Shape.broadcastOut s1 s2) (t1.requiresGrad || t2.requiresGrad))

def cmplt {s : List Nat} {d : DType} (t1 t2 : StaticTensor s d) : TensorM (StaticTensor s .bool) := do
  let result ← TUOp.cmplt t1.tuop t2.tuop
  pure (ofTU result (t1.requiresGrad || t2.requiresGrad))

def cmpltB {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool) := do
  let result ← TUOp.cmplt t1.tuop t2.tuop
  pure (ofTUCast result (Shape.broadcastOut s1 s2) (t1.requiresGrad || t2.requiresGrad))

def cmpgt {s : List Nat} {d : DType} (t1 t2 : StaticTensor s d) : TensorM (StaticTensor s .bool) := do
  cmplt t2 t1

def cmpgtB {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool) := do
  let result ← TUOp.cmplt t2.tuop t1.tuop
  pure (ofTUCast result (Shape.broadcastOut s1 s2) (t1.requiresGrad || t2.requiresGrad))

def cmpeq {s : List Nat} {d : DType} (t1 t2 : StaticTensor s d) : TensorM (StaticTensor s .bool) := do
  let result ← TUOp.cmpeq t1.tuop t2.tuop
  pure (ofTU result (t1.requiresGrad || t2.requiresGrad))

def cmpeqB {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool) := do
  let result ← TUOp.cmpeq t1.tuop t2.tuop
  pure (ofTUCast result (Shape.broadcastOut s1 s2) (t1.requiresGrad || t2.requiresGrad))

def cmpne {s : List Nat} {d : DType} (t1 t2 : StaticTensor s d) : TensorM (StaticTensor s .bool) := do
  let result ← TUOp.cmpne t1.tuop t2.tuop
  pure (ofTU result (t1.requiresGrad || t2.requiresGrad))

def cmpneB {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool) := do
  let result ← TUOp.cmpne t1.tuop t2.tuop
  pure (ofTUCast result (Shape.broadcastOut s1 s2) (t1.requiresGrad || t2.requiresGrad))

def cat {s1 s2 : List Nat} {d : DType} (t1 : StaticTensor s1 d) (t2 : StaticTensor s2 d)
    (axis : Nat) : TensorM (StaticTensor (Shape.concatOut s1 s2 axis) d) := do
  let out ← UOp.cat [t1.uop, t2.uop] axis
  let outTU := TUOp.castShape (TUOp.ofRaw out) (Shape.concatOut s1 s2 axis)
  pure (ofTU outTU (t1.requiresGrad || t2.requiresGrad))

def catList {d : DType} {shapes : List Shape} (ts : TensorList d shapes) (axis : Nat)
    : TensorM (StaticTensor (Shape.concatOutList shapes axis) d) := do
  let out ← UOp.cat (TensorList.toUOps ts) axis
  let outTU := TUOp.castShape (TUOp.ofRaw out) (Shape.concatOutList shapes axis)
  let reqGrad := TensorList.anyRequiresGrad ts
  pure (ofTU outTU reqGrad)

def bitand {s : List Nat} (t1 t2 : StaticTensor s .bool) : TensorM (StaticTensor s .bool) := do
  let result ← TUOp.binaryOp .AND t1.tuop t2.tuop
  pure (ofTU result (t1.requiresGrad || t2.requiresGrad))

def bitandB {s1 s2 : List Nat} (t1 : StaticTensor s1 .bool) (t2 : StaticTensor s2 .bool)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool) := do
  let result ← TUOp.binaryOp .AND t1.tuop t2.tuop
  pure (ofTUCast result (Shape.broadcastOut s1 s2) (t1.requiresGrad || t2.requiresGrad))

def bitor {s : List Nat} (t1 t2 : StaticTensor s .bool) : TensorM (StaticTensor s .bool) := do
  let result ← TUOp.binaryOp .OR t1.tuop t2.tuop
  pure (ofTU result (t1.requiresGrad || t2.requiresGrad))

def bitorB {s1 s2 : List Nat} (t1 : StaticTensor s1 .bool) (t2 : StaticTensor s2 .bool)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool) := do
  let result ← TUOp.binaryOp .OR t1.tuop t2.tuop
  pure (ofTUCast result (Shape.broadcastOut s1 s2) (t1.requiresGrad || t2.requiresGrad))

def bitxor {s : List Nat} (t1 t2 : StaticTensor s .bool) : TensorM (StaticTensor s .bool) := do
  let result ← TUOp.binaryOp .XOR t1.tuop t2.tuop
  pure (ofTU result (t1.requiresGrad || t2.requiresGrad))

def bitxorB {s1 s2 : List Nat} (t1 : StaticTensor s1 .bool) (t2 : StaticTensor s2 .bool)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool) := do
  let result ← TUOp.binaryOp .XOR t1.tuop t2.tuop
  pure (ofTUCast result (Shape.broadcastOut s1 s2) (t1.requiresGrad || t2.requiresGrad))

def cast {s : List Nat} {d : DType} (t : StaticTensor s d) (dtype : DType)
    : TensorM (StaticTensor s dtype) := do
  let result ← TUOp.cast t.tuop dtype
  pure (ofTU result t.requiresGrad)

def bitcast {s : List Nat} {d : DType} (t : StaticTensor s d) (dtype : DType)
    : TensorM (StaticTensor s dtype) := do
  let result ← TUOp.bitcast t.tuop dtype
  pure (ofTU result t.requiresGrad)

def to {s : List Nat} {d : DType} (t : StaticTensor s d) (dtype : DType)
    : TensorM (StaticTensor s dtype) :=
  cast t dtype

def to_ {s : List Nat} {d : DType} (t : StaticTensor s d) (dtype : DType)
    : TensorM (StaticTensor s dtype) :=
  cast t dtype

def where_ {s1 s2 s3 : List Nat} {d : DType}
    (cond : StaticTensor s1 .bool) (x : StaticTensor s2 d) (y : StaticTensor s3 d)
    : TensorM (StaticTensor (Shape.broadcastOut s1 (Shape.broadcastOut s2 s3)) d) := do
  let out ← TUOp.where_ cond.tuop x.tuop y.tuop
  pure (ofTUCast out (Shape.broadcastOut s1 (Shape.broadcastOut s2 s3)) (x.requiresGrad || y.requiresGrad))

infixl:65 " +. " => addB
infixl:65 " -. " => subB
infixl:70 " *. " => mulB
infixl:70 " /. " => divB

def neg {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← TUOp.neg t.tuop
  pure (ofTU result t.requiresGrad)

def trunc {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← TUOp.unaryOp .TRUNC t.tuop
  pure (ofTU result t.requiresGrad)

def floor {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let truncT ← trunc t
  let isNeg ← TUOp.cmplt t.tuop truncT.tuop
  let one ← TUOp.const d 1.0
  let truncMinusOne ← TUOp.sub truncT.tuop one
  let out ← TUOp.where_ isNeg truncMinusOne truncT.tuop
  pure (ofTU out t.requiresGrad)

def ceil {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let truncT ← trunc t
  let isPos ← TUOp.cmplt truncT.tuop t.tuop
  let one ← TUOp.const d 1.0
  let truncPlusOne ← TUOp.add truncT.tuop one
  let out ← TUOp.where_ isPos truncPlusOne truncT.tuop
  pure (ofTU out t.requiresGrad)

def sqrt {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← TUOp.unaryOp .SQRT t.tuop
  pure (ofTU result t.requiresGrad)

def rsqrt {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let sqrtT ← sqrt t
  let result ← TUOp.unaryOp .RECIPROCAL sqrtT.tuop
  pure (ofTU result t.requiresGrad)

def exp2 {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← TUOp.unaryOp .EXP2 t.tuop
  pure (ofTU result t.requiresGrad)

def log2 {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← TUOp.unaryOp .LOG2 t.tuop
  pure (ofTU result t.requiresGrad)

/-- Sine function -/
def sin {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← TUOp.unaryOp .SIN t.tuop
  pure (ofTU result t.requiresGrad)

/-- Cosine function -/
def cos {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← TUOp.unaryOp .COS t.tuop
  pure (ofTU result t.requiresGrad)

/-- Tangent function -/
def tan {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← TUOp.unaryOp .TAN t.tuop
  pure (ofTU result t.requiresGrad)

/-- Reciprocal (1/x) -/
def recip {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let result ← TUOp.unaryOp .RECIPROCAL t.tuop
  pure (ofTU result t.requiresGrad)

def sum {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (Scalar d) := do
  let axes := listRange s.length
  let result ← TUOp.sum t.tuop axes false
  pure (ofTU result false)

def max {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (Scalar d) := do
  let axes := listRange s.length
  let result ← TUOp.max_ t.tuop axes false
  pure (ofTU result false)

def min {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (Scalar d) := do
  let negT ← neg t
  let maxNeg ← max negT
  neg maxNeg

def mean {shape : List Nat} {d : DType} (t : StaticTensor shape d) : TensorM (Scalar d) := do
  let sumT ← sum t
  let n := listProd shape
  let nConst ← TUOp.const d n.toFloat32
  let result ← TUOp.div sumT.tuop nConst
  pure (ofTU result false)

-- Constants for exp/log conversion
-- ln(2) ≈ 0.693147
-- log2(e) ≈ 1.442695
def ln2 : Float := 0.6931471805599453
def log2e : Float := 1.4426950408889634

-- NOTE: We use Float32 for const construction so float32 graphs can stay in Float32/ByteArray land.
def ln2f32 : Float32 := 0.6931471805599453
def log2ef32 : Float32 := 1.4426950408889634

/-- Natural exponential: e^x = 2^(x * log2(e)) -/
def exp {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let log2eConst ← TUOp.const d log2ef32
  let scaled ← TUOp.mul t.tuop log2eConst
  let result ← TUOp.unaryOp .EXP2 scaled
  pure (ofTU result t.requiresGrad)

/-- Natural logarithm: ln(x) = log2(x) * ln(2) -/
def log {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let log2Result ← TUOp.unaryOp .LOG2 t.tuop
  let ln2Const ← TUOp.const d ln2f32
  let result ← TUOp.mul log2Result ln2Const
  pure (ofTU result t.requiresGrad)

/-- ReLU: max(0, x) -/
def relu {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let zero ← TUOp.const d 0.0
  let result ← TUOp.binaryOp .MAX t.tuop zero
  pure (ofTU result t.requiresGrad)

/-- ReLU6: min(max(x, 0), 6). -/
def relu6 {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let reluT ← relu t
  let six ← TUOp.const d 6.0
  let minus ← TUOp.sub t.tuop six
  let minusT : StaticTensor s d := ofTU minus t.requiresGrad
  let reluMinus ← relu minusT
  let out ← TUOp.sub reluT.tuop reluMinus.tuop
  pure (ofTU out t.requiresGrad)

/-- Hardsigmoid: relu(alpha*x + beta) - relu(alpha*x + beta - 1). -/
def hardsigmoid {s : List Nat} {d : DType} (t : StaticTensor s d) (alpha : Float32 := 0.16666667)
    (beta : Float32 := 0.5) : TensorM (StaticTensor s d) := do
  let alphaConst ← TUOp.const d alpha
  let betaConst ← TUOp.const d beta
  let scaled ← TUOp.mul t.tuop alphaConst
  let shifted ← TUOp.add scaled betaConst
  let shiftedT : StaticTensor s d := ofTU shifted t.requiresGrad
  let reluShifted ← relu shiftedT
  let one ← TUOp.const d 1.0
  let shiftedMinusOne ← TUOp.sub shifted one
  let shiftedMinusOneT : StaticTensor s d := ofTU shiftedMinusOne t.requiresGrad
  let reluShiftedMinusOne ← relu shiftedMinusOneT
  let out ← TUOp.sub reluShifted.tuop reluShiftedMinusOne.tuop
  pure (ofTU out t.requiresGrad)

/-- Sigmoid: 1 / (1 + exp(-x)) -/
def sigmoid {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let negT ← neg t
  let expNeg ← exp negT
  let one ← TUOp.const d 1.0
  let denom ← TUOp.add expNeg.tuop one
  let result ← TUOp.div one denom
  pure (ofTU result t.requiresGrad)

/-- Tanh via exp: (e^x - e^-x) / (e^x + e^-x) -/
def tanh {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let negT ← neg t
  let expPos ← exp t
  let expNeg ← exp negT
  let num ← TUOp.sub expPos.tuop expNeg.tuop
  let denom ← TUOp.add expPos.tuop expNeg.tuop
  let result ← TUOp.div num denom
  pure (ofTU result t.requiresGrad)

/-- Softplus: log(1 + exp(x)) -/
def softplus {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let expT ← exp t
  let one ← TUOp.const d 1.0
  let onePlus ← TUOp.add expT.tuop one
  let onePlusT : StaticTensor s d := ofTU onePlus t.requiresGrad
  log onePlusT

/-- GELU (tanh approximation). -/
def gelu {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let x2 ← TUOp.mul t.tuop t.tuop
  let x3 ← TUOp.mul x2 t.tuop
  let c0 ← TUOp.const d 0.044715
  let x3Scaled ← TUOp.mul x3 c0
  let inner ← TUOp.add t.tuop x3Scaled
  let c1 ← TUOp.const d 0.7978845608
  let scaled ← TUOp.mul inner c1
  let scaledT : StaticTensor s d := ofTU scaled t.requiresGrad
  let tanhScaled ← tanh scaledT
  let one ← TUOp.const d 1.0
  let onePlus ← TUOp.add tanhScaled.tuop one
  let half ← TUOp.const d 0.5
  let halfOnePlus ← TUOp.mul onePlus half
  let result ← TUOp.mul t.tuop halfOnePlus
  pure (ofTU result t.requiresGrad)

/-- Abs: |x| -/
def abs {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let zero ← TUOp.const d 0.0
  let negT ← TUOp.neg t.tuop
  let isNeg ← TUOp.cmplt t.tuop zero
  let out ← TUOp.where_ isNeg negT t.tuop
  pure (ofTU out t.requiresGrad)

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
  let three ← TUOp.const d 3.0
  let tPlusThree ← TUOp.add t.tuop three
  let tPlusThreeT : StaticTensor s d := ofTU tPlusThree t.requiresGrad
  let relu6T ← relu6 tPlusThreeT
  let mul1 ← TUOp.mul t.tuop relu6T.tuop
  let oneSixth ← TUOp.const d 0.16666667
  let out ← TUOp.mul mul1 oneSixth
  pure (ofTU out t.requiresGrad)

/-- Leaky ReLU: x if x >= 0, alpha * x otherwise. -/
def leakyRelu {s : List Nat} {d : DType} (t : StaticTensor s d) (alpha : Float32 := 0.01)
    : TensorM (StaticTensor s d) := do
  let zero ← TUOp.const d 0.0
  let alphaUop ← TUOp.const d alpha
  let isNeg ← TUOp.cmplt t.tuop zero
  let negOut ← TUOp.mul t.tuop alphaUop
  let out ← TUOp.where_ isNeg negOut t.tuop
  pure (ofTU out t.requiresGrad)

/-- ELU: x if x >= 0, alpha * (exp(x) - 1) otherwise. -/
def elu {s : List Nat} {d : DType} (t : StaticTensor s d) (alpha : Float32 := 1.0)
    : TensorM (StaticTensor s d) := do
  let zero ← TUOp.const d 0.0
  let alphaUop ← TUOp.const d alpha
  let isNeg ← TUOp.cmplt t.tuop zero
  let expT ← exp t
  let one ← TUOp.const d 1.0
  let expm1 ← TUOp.sub expT.tuop one
  let negOut ← TUOp.mul expm1 alphaUop
  let out ← TUOp.where_ isNeg negOut t.tuop
  pure (ofTU out t.requiresGrad)

/-- Log-sigmoid: log(sigmoid(x)) -/
def logSigmoid {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  let sig ← sigmoid t
  log sig

/-- Clamp values to [lo, hi]. -/
def clamp {s : List Nat} {d : DType} (t : StaticTensor s d) (lo hi : Float32) : TensorM (StaticTensor s d) := do
  let loConst ← TUOp.const d lo
  let hiConst ← TUOp.const d hi
  let below ← TUOp.cmplt t.tuop loConst
  let above ← TUOp.cmplt hiConst t.tuop
  let clippedLo ← TUOp.where_ below loConst t.tuop
  let clipped ← TUOp.where_ above hiConst clippedLo
  pure (ofTU clipped t.requiresGrad)

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
  let result ← TUOp.max_ t.tuop [axis] keepdim
  pure (ofTU result t.requiresGrad)

def minAxis {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d) := do
  let negT ← neg t
  let maxNeg ← maxAxis negT axis keepdim
  neg maxNeg

/-- Max along axis with keepdim (statically checked axis). -/
def maxAxisF {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Fin s.length) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis.val] keepdim) d) := do
  let result ← TUOp.max_ t.tuop [axis.val] keepdim
  pure (ofTU result t.requiresGrad)

def minAxisF {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Fin s.length) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis.val] keepdim) d) := do
  let negT ← neg t
  let maxNeg ← maxAxisF negT axis keepdim
  neg maxNeg

/-- Sum along axis with keepdim -/
def sumAxis {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d) := do
  let result ← TUOp.sum t.tuop [axis] keepdim
  pure (ofTU result t.requiresGrad)

/-- Sum along axis with keepdim (statically checked axis). -/
def sumAxisF {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Fin s.length) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis.val] keepdim) d) := do
  let result ← TUOp.sum t.tuop [axis.val] keepdim
  pure (ofTU result t.requiresGrad)

/-- Mean along axis with keepdim -/
def meanAxis {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d) := do
  let sumT ← sumAxis t axis keepdim
  let n := listGetD s axis 1
  let nConst ← TUOp.const d (Float.ofNat n).toFloat32
  let result ← TUOp.div sumT.tuop nConst
  pure (ofTU result t.requiresGrad)

/-- Variance along axis with keepdim -/
def varAxis {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d) := do
  let meanT ← meanAxis t axis true
  let centered ← TUOp.sub t.tuop meanT.tuop
  let centeredT : StaticTensor s d := ofTU centered t.requiresGrad
  let sq ← mul centeredT centeredT
  meanAxis sq axis keepdim

/-- Layer norm over an axis (last axis by default). -/
def layerNorm {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat := s.length - 1)
    (eps : Float32 := 1.0e-5) : TensorM (StaticTensor s d) := do
  let meanT ← meanAxis t axis true
  let centered ← TUOp.sub t.tuop meanT.tuop
  let centeredT : StaticTensor s d := ofTU centered t.requiresGrad
  let sq ← mul centeredT centeredT
  let varT ← meanAxis sq axis true
  let epsConst ← TUOp.const d eps
  let varEps ← TUOp.add varT.tuop epsConst
  let std ← TUOp.unaryOp .SQRT varEps
  let invStd ← TUOp.unaryOp .RECIPROCAL std
  let out ← TUOp.mul centered invStd
  pure (ofTU out t.requiresGrad)

/-- RMS norm over an axis (last axis by default). -/
def rmsNorm {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat := s.length - 1)
    (eps : Float32 := 1.0e-5) : TensorM (StaticTensor s d) := do
  let sq ← mul t t
  let meanSq ← meanAxis sq axis true
  let epsConst ← TUOp.const d eps
  let varEps ← TUOp.add meanSq.tuop epsConst
  let rms ← TUOp.unaryOp .SQRT varEps
  let invRms ← TUOp.unaryOp .RECIPROCAL rms
  let out ← TUOp.mul t.tuop invRms
  pure (ofTU out t.requiresGrad)

private def classRangeF32 (n : Nat) : Array Float32 := Id.run do
  let mut out := Array.emptyWithCapacity n
  for i in [:n] do
    out := out.push (Float.ofNat i).toFloat32
  return out

/-- One-hot encoding for class indices (float32). -/
def oneHotF32 {batch numClasses : Nat}
    (targets : StaticTensor [batch] .float32)
    : TensorM (StaticTensor [batch, numClasses] .float32) := do
  let classUop ← UOp.vconstF32 (classRangeF32 numClasses)
  let classesTU := TUOp.castShape (TUOp.ofRaw classUop) [numClasses]
  let classes : StaticTensor [numClasses] .float32 := ofTU classesTU false
  let targets2 ← reshape targets [batch, 1]
  let classes2 ← reshape classes [1, numClasses]
  let cmp ← TUOp.cmpeq targets2.tuop classes2.tuop
  let one ← TUOp.const .float32 1.0
  let zero ← TUOp.const .float32 0.0
  let out ← TUOp.where_ cmp one zero
  pure (ofTUCast out [batch, numClasses] false)

/-- Gather along the last axis using class indices (float32). -/
def gatherLastF32 {batch numClasses : Nat}
    (x : StaticTensor [batch, numClasses] .float32)
    (targets : StaticTensor [batch] .float32)
    : TensorM (StaticTensor [batch] .float32) := do
  let oneHot ← oneHotF32 targets
  let prod ← mul x oneHot
  let sumC ← sumAxis prod 1 false
  pure sumC

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
  let values2 ← reshape values [batch, 1]
  let valuesB ← expand values2 [batch, numClasses]
  let out ← mul oneHot valuesB
  pure out

def scatterLast {batch numClasses : Nat}
    (values : StaticTensor [batch] .float32)
    (targets : StaticTensor [batch] .int32)
    : TensorM (StaticTensor [batch, numClasses] .float32) := do
  let targetsF ← cast targets .float32
  scatterLastF32 values targetsF

/-- Log-sum-exp along axis (numerically stable). -/
def logsumexpAxis {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d) := do
  let maxVal ← TUOp.max_ t.tuop [axis] true
  let shifted ← TUOp.sub t.tuop maxVal
  let shiftedT : StaticTensor s d := ofTU shifted t.requiresGrad
  let expShifted ← exp shiftedT
  let sumExp ← TUOp.sum expShifted.tuop [axis] true
  let sumExpT : StaticTensor (Shape.reduce s [axis] true) d := ofTU sumExp t.requiresGrad
  let logSum ← log sumExpT
  let outKeep ← TUOp.add logSum.tuop maxVal
  match keepdim with
  | true =>
    pure (ofTU outKeep t.requiresGrad)
  | false =>
    let outKeepT : StaticTensor (Shape.reduce s [axis] true) d := ofTU outKeep t.requiresGrad
    reshape outKeepT (Shape.reduce s [axis] false)

/-- Log-softmax along an axis (stable). -/
def logSoftmaxAxis {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat) : TensorM (StaticTensor s d) := do
  let logSum ← logsumexpAxis t axis true
  let result ← TUOp.sub t.tuop logSum.tuop
  pure (ofTU result t.requiresGrad)

/-- Softmax along an axis (stable). -/
def softmaxAxis {s : List Nat} {d : DType} (t : StaticTensor s d) (axis : Nat) : TensorM (StaticTensor s d) := do
  let maxVal ← TUOp.max_ t.tuop [axis] true
  let shifted ← TUOp.sub t.tuop maxVal
  let shiftedT : StaticTensor s d := ofTU shifted t.requiresGrad
  let expShifted ← exp shiftedT
  let sumExp ← TUOp.sum expShifted.tuop [axis] true
  let out ← TUOp.div expShifted.tuop sumExp
  pure (ofTU out t.requiresGrad)

/-- Softmax along last axis: exp(x - max(x)) / sum(exp(x - max(x))) -/
def softmax {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  softmaxAxis t (s.length - 1)

/-- Log-softmax along last axis: x - max(x) - log(sum(exp(x - max(x)))) -/
def logSoftmax {s : List Nat} {d : DType} (t : StaticTensor s d) : TensorM (StaticTensor s d) := do
  logSoftmaxAxis t (s.length - 1)

private def argmaxF32 {batch n : Nat} (t : StaticTensor [batch, n] .float32)
    : TensorM (StaticTensor [batch] .int32) := do
  let maxVal ← TUOp.max_ t.tuop [1] true
  let eq ← TUOp.cmpeq t.tuop maxVal
  let eqT : StaticTensor [batch, n] .bool := ofTUCast eq [batch, n] false
  let eqF ← cast eqT .float32
  let classesUop ← UOp.vconstF32 (classRangeF32 n)
  let classesTU := TUOp.castShape (TUOp.ofRaw classesUop) [n]
  let classes : StaticTensor [n] .float32 := ofTU classesTU false
  let classes2 ← reshape classes [1, n]
  let classesB ← expand classes2 [batch, n]
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
  let scalarUop ← TUOp.const d scalar
  let result ← TUOp.mul t.tuop scalarUop
  pure (ofTU result t.requiresGrad)

/-- Add scalar: t + scalar -/
def addScalar {s : List Nat} {d : DType} (t : StaticTensor s d) (scalar : Float32)
    : TensorM (StaticTensor s d) := do
  let scalarUop ← TUOp.const d scalar
  let result ← TUOp.add t.tuop scalarUop
  pure (ofTU result t.requiresGrad)

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
  let betaConst ← TUOp.const .float32 beta
  let half ← TUOp.const .float32 0.5
  let isSmall ← TUOp.cmplt absDiff.tuop betaConst
  let sq ← mul diff diff
  let sqHalf ← TUOp.mul sq.tuop half
  let sqScaled ← TUOp.div sqHalf betaConst
  let betaHalf ← TUOp.mul betaConst half
  let linTerm ← TUOp.sub absDiff.tuop betaHalf
  let out ← TUOp.where_ isSmall sqScaled linTerm
  let out := TUOp.castDType out .float32
  let outT : StaticTensor s .float32 := ofTU out (pred.requiresGrad || target.requiresGrad)
  mean outT

/-- Binary cross-entropy loss (expects probabilities in [0, 1]). -/
def binaryCrossEntropy {s : List Nat}
    (pred target : StaticTensor s .float32) (eps : Float32 := 1.0e-7)
    : TensorM (Scalar .float32) := do
  let predClamped ← clamp pred eps (1.0 - eps)
  let logPred ← log predClamped
  let one ← TUOp.const .float32 1.0
  let oneMinusPred ← TUOp.sub one predClamped.tuop
  let oneMinusPred := TUOp.castDType oneMinusPred .float32
  let oneMinusPredT : StaticTensor s .float32 := ofTU oneMinusPred pred.requiresGrad
  let logOneMinusPred ← log oneMinusPredT
  let oneMinusTarget ← TUOp.sub one target.tuop
  let oneMinusTarget := TUOp.castDType oneMinusTarget .float32
  let oneMinusTargetT : StaticTensor s .float32 := ofTU oneMinusTarget target.requiresGrad
  let term1 ← mul target logPred
  let term2 ← mul oneMinusTargetT logOneMinusPred
  let sumTerms ← add term1 term2
  let negSum ← neg sumTerms
  mean negSum

/-- Binary cross-entropy with logits (numerically stable). -/
def binaryCrossEntropyWithLogits {s : List Nat}
    (logits target : StaticTensor s .float32) : TensorM (Scalar .float32) := do
  let zero ← TUOp.const .float32 0.0
  let maxZero ← TUOp.binaryOp .MAX logits.tuop zero
  let absLogits ← abs logits
  let negAbs ← neg absLogits
  let expNegAbs ← exp negAbs
  let one ← TUOp.const .float32 1.0
  let onePlus ← TUOp.add expNegAbs.tuop one
  let onePlusT : StaticTensor s .float32 := ofTU onePlus logits.requiresGrad
  let logOnePlus ← log onePlusT
  let prod ← TUOp.mul logits.tuop target.tuop
  let tmp ← TUOp.sub maxZero prod
  let lossUop ← TUOp.add tmp logOnePlus.tuop
  let lossUop := TUOp.castDType lossUop .float32
  let lossT : StaticTensor s .float32 := ofTU lossUop logits.requiresGrad
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
  let outUop ← TUOp.contract2D a.tuop b.tuop
  let outUop := TUOp.castDType outUop d
  pure (ofTUCast outUop [m, n] (a.requiresGrad || b.requiresGrad))

/-- Fully-connected (linear) layer: X @ W -> [batch, out]. -/
def linear {batch inDim outDim : Nat} {d : DType}
    (x : Matrix batch inDim d) (w : Matrix inDim outDim d)
    : TensorM (Matrix batch outDim d) := do
  matmul x w

/-- Fully-connected layer with bias: X @ W + b (broadcasted over batch). -/
def linearBias {batch inDim outDim : Nat} {d : DType}
    (x : Matrix batch inDim d) (w : Matrix inDim outDim d) (b : Vector outDim d)
    : TensorM (Matrix batch outDim d) := do
  let y ← matmul x w
  let yb ← addB y b
  pure { uop := yb.uop, h_shape := sorry_proof, requiresGrad := yb.requiresGrad }

/-- Batched matrix multiplication with broadcast on the batch dim:
    [b1, m, k] @ [b2, k, n] -> [max b1 b2, m, n]. -/
def bmatmul {b1 b2 m k n : Nat} {d : DType}
    (a : BMatrix b1 m k d) (b : BMatrix b2 k n d)
    : TensorM (BMatrix (Nat.max b1 b2) m n d) := do
  let outUop ← TUOp.contract2D a.tuop b.tuop
  let outUop := TUOp.castDType outUop d
  pure (ofTUCast outUop [Nat.max b1 b2, m, n] (a.requiresGrad || b.requiresGrad))

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
  let out ← TUOp.buffer d outShape
  let biasGrad := match bias with | none => false | some b => b.requiresGrad
  let reqGrad := x.requiresGrad || weight.requiresGrad || biasGrad
  pure (ofTU out reqGrad)

/-- Placeholder maxPool2d - returns correctly shaped output tensor. -/
def maxPool2dPlaceholder {batch cin h w hOut wOut : Nat} {d : DType}
    (x : StaticTensor [batch, cin, h, w] d)
    (_kernelSize : Nat)
    (_stride : Nat := 0)
    : TensorM (StaticTensor [batch, cin, hOut, wOut] d) := do
  let outShape := [batch, cin, hOut, wOut]
  let out ← TUOp.buffer d outShape
  pure (ofTU out x.requiresGrad)

/-- Placeholder avgPool2d - returns correctly shaped output tensor. -/
def avgPool2dPlaceholder {batch cin h w hOut wOut : Nat} {d : DType}
    (x : StaticTensor [batch, cin, h, w] d)
    (_kernelSize : Nat)
    (_stride : Nat := 0)
    : TensorM (StaticTensor [batch, cin, hOut, wOut] d) := do
  let outShape := [batch, cin, hOut, wOut]
  let out ← TUOp.buffer d outShape
  pure (ofTU out x.requiresGrad)

/-- Pad 1D tensor with symmetric padding on W dimension.
    Input:  [batch, channels, width]
    Output: [batch, channels, width + 2*pad] -/
def pad1d {batch cin w : Nat} {d : DType}
    (x : StaticTensor [batch, cin, w] d)
    (padW : Nat)
    : TensorM (StaticTensor [batch, cin, w + 2*padW] d) := do
  let padding := [(0, 0), (0, 0), (padW, padW)]
  let result ← pad x padding
  pure { uop := result.uop, requiresGrad := x.requiresGrad, h_shape := sorry_proof }

/-- Pad 2D tensor with symmetric padding on H and W dimensions.
    Input:  [batch, channels, height, width]
    Output: [batch, channels, height + 2*pad, width + 2*pad] -/
def pad2d {batch cin h w : Nat} {d : DType}
    (x : StaticTensor [batch, cin, h, w] d)
    (padH padW : Nat)
    : TensorM (StaticTensor [batch, cin, h + 2*padH, w + 2*padW] d) := do
  let padding := [(0, 0), (0, 0), (padH, padH), (padW, padW)]
  let result ← pad x padding
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
  let reduced1 ← TUOp.max_ patches.tuop [axis1] false
  let axis2 := patchShape.length - 2  -- kH axis (now shifted)
  let reduced2 ← TUOp.max_ reduced1 [axis2] false

  pure (ofTUCast reduced2 (Shape.pool2dShape [batch, cin, h, w] kernelSize padding stride) x.requiresGrad)

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
  let sum1 ← TUOp.sum patches.tuop [axis1] false
  let axis2 := patchShape.length - 2  -- kH axis (now shifted)
  let sum2 ← TUOp.sum sum1 [axis2] false

  -- Divide by kernel area for mean
  let kernelArea := (kernelSize * kernelSize : Nat)
  let divisor ← TUOp.const d (Float.ofNat kernelArea).toFloat32
  let result ← TUOp.div sum2 divisor
  let result := TUOp.castDType result d

  pure (ofTUCast result (Shape.pool2dShape [batch, cin, h, w] kernelSize padding stride) x.requiresGrad)

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
  let outShape := Shape.conv1dOut [batch, cin, w] [cout, cin, kW] padding stride dilation

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
  let patchesReshaped ← reshape patches [patchFlat, kernelFlat]

  -- Step 4: Reshape weight
  -- [cout, cin, kW] -> [cout, cin * kW]
  let weightReshaped ← reshape weight [cout, kernelFlat]

  -- Step 5: Transpose weight for matmul: [cout, kernelFlat] -> [kernelFlat, cout]
  let weightT ← T weightReshaped

  -- Step 6: Matmul: [patchFlat, kernelFlat] @ [kernelFlat, cout] = [patchFlat, cout]
  let mm ← matmul patchesReshaped weightT

  -- Step 7: Reshape to [batch, wOut, cout]
  let mmReshaped ← reshape mm [batch, wOut, cout]

  -- Step 8: Permute to [batch, cout, wOut]
  let result ← permute mmReshaped [0, 2, 1]

  -- Step 9: Add bias if present
  let final ← match bias with
  | none => pure result
  | some b =>
    let biasReshaped ← reshape b [1, cout, 1]
    addB result biasReshaped
  let finalCast : StaticTensor outShape d := ofTUCast final.tuop outShape final.requiresGrad

  let biasGrad := match bias with | none => false | some b => b.requiresGrad
  let reqGrad := x.requiresGrad || weight.requiresGrad || biasGrad
  pure { uop := finalCast.uop, requiresGrad := reqGrad, h_shape := sorry_proof }

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
  let outShape := Shape.conv2dOut [batch, cin, h, w] [cout, cin, kH, kW] padding stride dilation

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
  let patchesReshaped ← reshape patches [patchFlat, kernelFlat]

  -- Step 4: Reshape weight
  -- [cout, cin, kH, kW] -> [cout, cin * kH * kW]
  let weightReshaped ← reshape weight [cout, kernelFlat]

  -- Step 5: Transpose weight for matmul: [cout, kernelFlat] -> [kernelFlat, cout]
  let weightT ← T weightReshaped

  -- Step 6: Matmul: [patchFlat, kernelFlat] @ [kernelFlat, cout] = [patchFlat, cout]
  let mm ← matmul patchesReshaped weightT

  -- Step 7: Reshape to [batch, hOut, wOut, cout]
  let mmReshaped ← reshape mm [batch, hOut, wOut, cout]

  -- Step 8: Permute to [batch, cout, hOut, wOut]
  let result ← permute mmReshaped [0, 3, 1, 2]

  -- Step 9: Add bias if present
  let final ← match bias with
  | none => pure result
  | some b =>
    -- Reshape bias [cout] -> [1, cout, 1, 1] for broadcasting
    let biasReshaped ← reshape b [1, cout, 1, 1]
    addB result biasReshaped
  let finalCast : StaticTensor outShape d := ofTUCast final.tuop outShape final.requiresGrad

  let biasGrad := match bias with | none => false | some b => b.requiresGrad
  let reqGrad := x.requiresGrad || weight.requiresGrad || biasGrad
  pure { uop := finalCast.uop, requiresGrad := reqGrad, h_shape := sorry_proof }

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
    5. Squeeze and reshape to [batch, cin, hOut, wOut]

    This uses a single batched CONTRACT operation instead of expand+multiply+sum.
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
  let outShape := Shape.conv2dOut [batch, cin, h, w] [cin, 1, kH, kW] padding stride dilation

  -- Step 1: Pad if needed
  let xPadded ← if padding > 0 then
    let padded ← pad2d x padding padding
    pure { uop := padded.uop, requiresGrad := x.requiresGrad, h_shape := sorry_proof }
  else
    pure x

  -- Step 2: Pool to get patches: [batch, cin, hOut, wOut, kH, kW]
  let patches ← pool xPadded [kH, kW] [stride, stride] [dilation, dilation]

  -- Step 3: Reshape patches for batched matmul: [batch, cin, hOut*wOut, kH*kW]
  let patchesReshaped ← reshape patches [batch, cin, spatialOut, kernelFlat]

  -- Step 4: Reshape weight for batched matmul: [cin, 1, kH, kW] -> [1, cin, kH*kW, 1]
  -- This allows broadcasting with batch dimension
  let weightReshaped ← reshape weight [1, cin, kernelFlat, 1]

  -- Step 5: Batched matmul using TUOp.contract2D
  -- [batch, cin, hOut*wOut, kH*kW] @ [1, cin, kH*kW, 1]
  -- Batch dims [batch, cin] and [1, cin] broadcast to [batch, cin]
  -- Result: [batch, cin, hOut*wOut, 1]
  let mmResult ← TUOp.contract2D patchesReshaped.tuop weightReshaped.tuop
  let mmResult := TUOp.castDType mmResult d
  let mmResultT : StaticTensor [batch, cin, spatialOut, 1] d :=
    ofTUCast mmResult [batch, cin, spatialOut, 1] (x.requiresGrad || weight.requiresGrad)

  -- Step 6: Squeeze and reshape to [batch, cin, hOut, wOut]
  let result ← reshape mmResultT [batch, cin, hOut, wOut]

  -- Step 7: Add bias if present
  let final ← match bias with
  | none => pure result
  | some b =>
    let biasReshaped ← reshape b [1, cin, 1, 1]
    addB result biasReshaped
  let finalCast : StaticTensor outShape d := ofTUCast final.tuop outShape final.requiresGrad

  let biasGrad := match bias with | none => false | some b => b.requiresGrad
  let reqGrad := x.requiresGrad || weight.requiresGrad || biasGrad
  pure { uop := finalCast.uop, requiresGrad := reqGrad, h_shape := sorry_proof }

end StaticTensor

end TinyGrad4
