import TinyGrad4.Ops
import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph

/-!
# Declarative Fusion Patterns

Composable pattern matchers using Option + do notation.
New patterns can be added without modifying core fusion logic.

## Design

Patterns are functions `UOp → Option α` that compose via:
- `>>=` (bind) for sequential matching
- `<|>` (alternative) for trying multiple patterns
- `do` notation for readable complex patterns

## Example

```lean
def softmax? (u : UOp) : Option SoftmaxInfo := do
  let (numer, denom) ← u.asDiv?
  let exp2Arg ← numer.asExp2?
  let (sumSrc, axes) ← denom.asReduceAdd?
  -- ... more matching
  pure { input, axis }
```

Patterns compose with `<|>`:
```lean
def anyPattern? (u : UOp) : Option FusedPlan :=
  (softmax? u >>= fun x => pure (.softmax x)) <|>
  (matmulRelu? u >>= fun (a,b) => pure (.matmulRelu a b))
```
-/

namespace TinyGrad4.Backend.Pattern

/-! ## Primitive Matchers

These extract structure from individual UOps. Each returns `Option` for composability.
-/

/-- Match a unary op and return its input -/
def asUnary? (u : UOp) (op : Ops) : Option UOp :=
  if u.op == op then
    match u.src with
    | [x] => some x
    | _ => none
  else none

/-- Match a binary op and return both inputs -/
def asBinary? (u : UOp) (op : Ops) : Option (UOp × UOp) :=
  if u.op == op then
    match u.src with
    | [a, b] => some (a, b)
    | _ => none
  else none

/-- Match a ternary op and return all inputs -/
def asTernary? (u : UOp) (op : Ops) : Option (UOp × UOp × UOp) :=
  if u.op == op then
    match u.src with
    | [a, b, c] => some (a, b, c)
    | _ => none
  else none

/-! ### Arithmetic Ops -/

def UOp.asAdd? (u : UOp) : Option (UOp × UOp) := asBinary? u .ADD
def UOp.asSub? (u : UOp) : Option (UOp × UOp) := asBinary? u .SUB
def UOp.asMul? (u : UOp) : Option (UOp × UOp) := asBinary? u .MUL
def UOp.asDiv? (u : UOp) : Option (UOp × UOp) := asBinary? u .FDIV
def UOp.asMax? (u : UOp) : Option (UOp × UOp) := asBinary? u .MAX

def UOp.asNeg? (u : UOp) : Option UOp := asUnary? u .NEG
def UOp.asSqrt? (u : UOp) : Option UOp := asUnary? u .SQRT
def UOp.asRecip? (u : UOp) : Option UOp := asUnary? u .RECIPROCAL
def UOp.asExp2? (u : UOp) : Option UOp := asUnary? u .EXP2
def UOp.asLog2? (u : UOp) : Option UOp := asUnary? u .LOG2
def UOp.asSin? (u : UOp) : Option UOp := asUnary? u .SIN
def UOp.asCos? (u : UOp) : Option UOp := asUnary? u .COS
def UOp.asTan? (u : UOp) : Option UOp := asUnary? u .TAN

def UOp.asWhere? (u : UOp) : Option (UOp × UOp × UOp) := asTernary? u .WHERE
def UOp.asMulAcc? (u : UOp) : Option (UOp × UOp × UOp) := asTernary? u .MULACC

/-! ### Reduce Ops -/

/-- Extract reduce info from UArg -/
def getReduceInfo (arg : UArg) : Option (Ops × List Nat) :=
  match arg with
  | .reduceWithAxes op axes => some (op, axes)
  | _ => none

/-- Match REDUCE_AXIS with specific op, return (source, axes) -/
def UOp.asReduceWith? (u : UOp) (op : Ops) : Option (UOp × List Nat) :=
  if u.op != .REDUCE_AXIS then none
  else match getReduceInfo u.arg with
  | some (reduceOp, axes) =>
    if reduceOp == op then
      match u.src with
      | [src] => some (src, axes)
      | _ => none
    else none
  | none => none

def UOp.asReduceAdd? (u : UOp) : Option (UOp × List Nat) := UOp.asReduceWith? u .ADD
def UOp.asReduceMax? (u : UOp) : Option (UOp × List Nat) := UOp.asReduceWith? u .MAX
def UOp.asReduceMul? (u : UOp) : Option (UOp × List Nat) := UOp.asReduceWith? u .MUL

/-! ### Buffer/Load Ops -/

/-- Check if this is an input buffer or load -/
def UOp.isInput? (u : UOp) : Option UOpId :=
  if u.op == .BUFFER || u.op == .LOAD then some u.uid
  else none

/-- Find input buffer by walking sources -/
partial def UOp.findInputBuffer (u : UOp) : Option UOpId :=
  match UOp.isInput? u with
  | some id => some id
  | none => match u.src with
    | s :: _ => UOp.findInputBuffer s
    | [] => none

/-! ### Constant Matching -/

/-- Get float32 constant value if present -/
def UOp.asConstF32? (u : UOp) : Option Float :=
  if u.op == .CONST then
    match u.arg with
    | .constF32Bits bits => some (Float32.ofBits bits).toFloat
    | _ => none
  else none

/-- Check if constant is approximately a value -/
def UOp.isConstApprox? (u : UOp) (target : Float) (tol : Float := 0.001) : Bool :=
  match UOp.asConstF32? u with
  | some f => (f - target).abs < tol
  | none => false

/-- LOG2E ≈ 1.4427 -/
def UOp.isLog2E? (u : UOp) : Bool := UOp.isConstApprox? u 1.4427

/-- LN2 ≈ 0.6931 -/
def UOp.isLn2? (u : UOp) : Bool := UOp.isConstApprox? u 0.6931

/-! ### Movement Ops -/

def UOp.asReshape? (u : UOp) : Option UOp := asUnary? u .RESHAPE
def UOp.asExpand? (u : UOp) : Option UOp := asUnary? u .EXPAND
def UOp.asPermute? (u : UOp) : Option UOp := asUnary? u .PERMUTE

/-! ## Composite Patterns

Built from primitives using do-notation.
-/

/-- Info extracted from a softmax pattern -/
structure SoftmaxInfo where
  input : UOp
  axis : Nat
  axes : List Nat
  isLog : Bool := false
  deriving Repr

/--
Match stable softmax pattern:
  exp2(log2e * (x - max(x, axis))) / sum(exp2(log2e * (x - max(x, axis))), axis)

Returns the input and axis if matched.
-/
def softmax? (u : UOp) : Option SoftmaxInfo := do
  -- Root must be FDIV
  let (numer, denom) ← UOp.asDiv? u

  -- Numerator should be EXP2(...)
  let exp2Arg ← UOp.asExp2? numer

  -- Denominator should be REDUCE_AXIS(ADD, exp2(...), axes)
  let (sumSrc, axes) ← UOp.asReduceAdd? denom

  -- sumSrc should also be EXP2
  let sumExp2Arg ← UOp.asExp2? sumSrc

  -- Both EXP2 args should be the same
  guard (exp2Arg.uid == sumExp2Arg.uid)

  -- Find the subtraction (x - max(x, axis))
  -- May have LOG2E scaling
  let subPattern ← match UOp.asMul? exp2Arg with
  | some (a, b) =>
    if UOp.isLog2E? a then UOp.asSub? b
    else if UOp.isLog2E? b then UOp.asSub? a
    else UOp.asSub? exp2Arg
  | none => UOp.asSub? exp2Arg

  let (input, maxExpr) := subPattern

  -- maxExpr should be REDUCE_AXIS(MAX, input, same_axes)
  let (maxSrc, maxAxes) ← UOp.asReduceMax? maxExpr

  -- Verify axes match
  guard (axes == maxAxes)

  -- Verify max is over same input (by uid)
  guard (maxSrc.uid == input.uid)

  -- Need at least one axis
  guard (!axes.isEmpty)
  let axis := axes.head!

  pure { input, axis, axes, isLog := false }

/-- Log-softmax pattern: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x)))) -/
def logSoftmax? (u : UOp) : Option SoftmaxInfo := do
  -- Pattern: x - max(x, axis) - log2(sum(exp2(x - max(x, axis)), axis)) * ln2
  -- Root should be SUB: (x - max(x)) - log2(sum(exp2(x - max(x))))
  let (shiftedInput, logSumExp) ← UOp.asSub? u

  -- shiftedInput = x - max(x, axis)
  let (input, maxExpr) ← UOp.asSub? shiftedInput
  let (maxSrc, axes) ← UOp.asReduceMax? maxExpr
  guard (maxSrc.uid == input.uid)

  -- logSumExp = log2(sum(exp2(x - max(x)))) * ln2
  -- or log2(sum(exp2(...)))
  let logArg ← match UOp.asMul? logSumExp with
  | some (a, b) =>
    if UOp.isLn2? a then UOp.asLog2? b
    else if UOp.isLn2? b then UOp.asLog2? a
    else UOp.asLog2? logSumExp
  | none => UOp.asLog2? logSumExp

  -- logArg = sum(exp2(x - max(x)), axis)
  let (sumSrc, sumAxes) ← UOp.asReduceAdd? logArg
  guard (axes == sumAxes)

  -- sumSrc = exp2(x - max(x))
  let exp2Arg ← UOp.asExp2? sumSrc
  let (x2, maxExpr2) ← UOp.asSub? exp2Arg
  guard (x2.uid == input.uid)
  guard (maxExpr2.uid == maxExpr.uid)

  guard (!axes.isEmpty)
  let axis := axes.head!

  pure { input, axis, axes, isLog := true }

/-- Match add followed by activation (fused bias + relu etc) -/
structure BiasActivationInfo where
  input : UOp
  bias : UOp
  activation : Ops  -- The activation op
  deriving Repr

def biasActivation? (u : UOp) (activation : Ops) : Option BiasActivationInfo := do
  let inner ← asUnary? u activation
  let (input, bias) ← UOp.asAdd? inner
  pure { input, bias, activation }

def biasRelu? (u : UOp) : Option BiasActivationInfo := do
  -- Match max(add(x, bias), 0)
  let (inner, zero) ← UOp.asMax? u
  -- Check that one operand is 0
  guard (UOp.isConstApprox? zero 0.0)
  let (input, bias) ← UOp.asAdd? inner
  pure { input, bias, activation := .MAX }

/-- Info extracted from matmul+relu pattern -/
structure MatmulReluInfo where
  a : UOp        -- First input matrix
  b : UOp        -- Second input matrix
  hasBias : Bool := false
  bias : Option UOp := none
  deriving Repr

/--
Match matmul+relu pattern:
  max(contract(a, b), 0)
  max(add(contract(a, b), bias), 0)
-/
def matmulRelu? (u : UOp) : Option MatmulReluInfo := do
  -- Root must be MAX (relu)
  let (inner, zero) ← UOp.asMax? u
  guard (UOp.isConstApprox? zero 0.0)

  -- Inner is either CONTRACT or ADD(CONTRACT, bias)
  match UOp.asAdd? inner with
  | some (contractOp, bias) =>
    -- ADD(CONTRACT, bias) case
    guard (contractOp.op == .CONTRACT)
    match contractOp.src with
    | [a, b] => pure { a, b, hasBias := true, bias := some bias }
    | _ => failure
  | none =>
    -- Direct CONTRACT case
    guard (inner.op == .CONTRACT)
    match inner.src with
    | [a, b] => pure { a, b, hasBias := false, bias := none }
    | _ => failure

/-- Info for GELU pattern -/
structure GeluInfo where
  input : UOp
  approximate : Bool := false  -- True for tanh approximation
  deriving Repr

/--
Match GELU pattern (exact):
  x * 0.5 * (1 + erf(x / sqrt(2)))
Or approximate (tanh):
  x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

For now, match simpler patterns we can detect.
-/
def gelu? (_u : UOp) : Option GeluInfo := do
  -- Complex pattern - defer for now
  failure

/-- Info for layer norm pattern -/
structure LayerNormInfo where
  input : UOp
  axes : List Nat
  gamma : Option UOp := none
  beta : Option UOp := none
  eps : Float := 1e-5
  deriving Repr

/--
Match layer norm pattern:
  gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
-/
def layerNorm? (_u : UOp) : Option LayerNormInfo := do
  -- Complex pattern - defer for now
  failure

/-! ## Utility Functions -/

/-- Collect all node UIDs in a subgraph -/
partial def collectCover (u : UOp) (acc : UOpIdSet := UOpIdSet.mkEmpty) : UOpIdSet :=
  if UOpIdSet.member acc u.uid then acc
  else
    let acc' := UOpIdSet.add acc u.uid
    u.src.foldl (fun (a : UOpIdSet) (s : UOp) => collectCover s a) acc'

/-- Check if two UOps are structurally equal (same uid) -/
def UOp.sameAs (a b : UOp) : Bool := a.uid == b.uid

/-- Check if UOp has specific dtype -/
def UOp.hasType? (u : UOp) (dt : DType) : Option UOp :=
  if u.dtype == dt then some u else none

/-- Require float32 dtype -/
def UOp.asFloat32? (u : UOp) : Option UOp := UOp.hasType? u .float32

end TinyGrad4.Backend.Pattern
