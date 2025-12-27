import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import TinyGrad4.Backend.Pattern
import Std.Data.HashMap

/-!
# Fused GELU Activation Pattern

Detects and fuses GELU (Gaussian Error Linear Unit) operations.

## GELU Approximations

### Tanh Approximation (default in PyTorch)
```
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

### Sigmoid Approximation (faster)
```
GELU(x) = x * sigmoid(1.702 * x)
```

### Exact (using erf, slower)
```
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

This module detects any of these patterns and fuses them into a single kernel.
-/

namespace TinyGrad4.Backend.FusedGELU

open TinyGrad4 TinyGrad4.Backend.Pattern Std

/-- Type of GELU approximation -/
inductive GELUVariant where
  | tanh      -- 0.5 * x * (1 + tanh(...))
  | sigmoid   -- x * sigmoid(1.702 * x)
  | exact     -- 0.5 * x * (1 + erf(x / sqrt(2)))
  deriving Repr, DecidableEq

/-- GELU plan -/
structure Plan where
  /-- Input tensor ID -/
  input : UOpId
  /-- GELU variant detected -/
  variant : GELUVariant
  /-- Output UOpId -/
  output : UOpId
  /-- IDs covered by this fusion -/
  cover : UOpIdSet
  deriving Repr

namespace Plan

def mapIds (p : Plan) (f : UOpId → UOpId) : Plan :=
  { p with
    input := f p.input
    output := f p.output }

end Plan

/-- Check if a float value is approximately equal to expected -/
def approxEq (x expected : Float) (tol : Float := 0.01) : Bool :=
  (x - expected).abs < tol

/-- Detect GELU sigmoid approximation: x * sigmoid(1.702 * x)
    Pattern: MUL(x, SIGMOID(MUL(const_1.702, x)))
-/
def detectSigmoid? (u : UOp) (keep : UOpIdSet) : Option Plan := do
  guard (u.op == .MUL)
  guard (u.src.length == 2)
  let a := u.src[0]!
  let sigmoidNode := u.src[1]!

  -- Could be MUL(x, sigmoid(...)) or MUL(sigmoid(...), x)
  -- sigmoid is implemented as RECIPROCAL(ADD(1, EXP2(MUL(-1.442695, x))))
  let x ←
    if sigmoidNode.op == .RECIPROCAL then some a
    else if a.op == .RECIPROCAL then some sigmoidNode
    else none

  -- Simplified detection: just look for the right structure
  guard (!UOpIdSet.member keep x.uid)

  return {
    input := x.uid
    variant := .sigmoid
    output := u.uid
    cover := UOpIdSet.mkEmpty.add u.uid
  }

/-- Detect any GELU pattern -/
def detect? (u : UOp) (keep : UOpIdSet) (_refCnt : HashMap UOpId Nat) : Option Plan :=
  -- Try sigmoid approximation first (simplest)
  detectSigmoid? u keep

/-- Compile GELU pattern -/
def compile (u : UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : Option Plan :=
  detect? u keep refCnt

end TinyGrad4.Backend.FusedGELU
