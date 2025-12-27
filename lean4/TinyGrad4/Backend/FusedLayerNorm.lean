import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import TinyGrad4.Backend.Pattern
import Std.Data.HashMap

/-!
# Fused LayerNorm Pattern

Detects and fuses LayerNorm operations:
```
x_centered = x - mean(x)        // centering
variance = mean(x_centered^2)   // variance computation
output = x_centered / sqrt(variance + eps) * gamma + beta  // normalize + scale
```

This fuses multiple operations:
1. Mean reduction
2. Subtraction (centering)
3. Square
4. Second mean (variance)
5. Add epsilon
6. Sqrt
7. Division (normalize)
8. Multiply by gamma
9. Add beta

Into a single fused kernel with:
- One pass for mean (uses threadgroup reduction)
- Second pass for variance (reuses mean, threadgroup reduction)
- Third pass for normalization (elementwise)

Or with Welford's algorithm: single-pass mean + variance computation.
-/

namespace TinyGrad4.Backend.FusedLayerNorm

open TinyGrad4 TinyGrad4.Backend.Pattern Std

/-- LayerNorm plan captures the detected pattern -/
structure Plan where
  /-- Input tensor base ID -/
  input : UOpId
  /-- Gamma (scale) parameter ID, if present -/
  gamma : Option UOpId
  /-- Beta (bias) parameter ID, if present -/
  beta : Option UOpId
  /-- Epsilon value (default 1e-5) -/
  eps : Float := 1e-5
  /-- Axes to normalize over (typically last axis) -/
  axes : List Nat
  /-- Shape of the input -/
  shape : List Nat
  /-- Output UOpId -/
  output : UOpId
  /-- IDs covered by this fusion -/
  cover : UOpIdSet
  deriving Repr

namespace Plan

def mapIds (p : Plan) (f : UOpId → UOpId) : Plan :=
  { p with
    input := f p.input
    gamma := p.gamma.map f
    beta := p.beta.map f
    output := f p.output }

end Plan

/-- Detect LayerNorm pattern starting from an ADD or MUL node.

    Pattern structure (working backwards from output):
    ```
    output = normalized * gamma + beta  [optional scale/bias]
           = (x - mean) / sqrt(var + eps) * gamma + beta
    ```

    We look for:
    1. Root: ADD (bias) or MUL (scale) or FDIV (normalize)
    2. FDIV: x_centered / sqrt(var + eps)
    3. SQRT: sqrt(var + eps)
    4. ADD: var + eps (eps is a small constant)
    5. REDUCE_AXIS with .ADD op: mean of squares
    6. MUL: x_centered * x_centered (squaring)
    7. SUB: x - mean (centering)
    8. REDUCE_AXIS with .ADD op: mean of x
-/
def detect? (u : UOp) (keep : UOpIdSet) (_refCnt : HashMap UOpId Nat) : Option Plan := do
  -- Start from potential output
  -- LayerNorm output is typically: add(mul(normalized, gamma), beta)
  -- or just: normalized if no learnable params

  -- For now, just detect the core pattern: x_centered / sqrt(var + eps)
  -- This is the minimum viable detection

  -- Pattern: FDIV(sub, sqrt(add(reduce, const)))
  guard (u.op == .FDIV)
  let [centered, sqrtNode] := u.src | failure
  guard (sqrtNode.op == .SQRT)
  let [addEps] := sqrtNode.src | failure
  guard (addEps.op == .ADD)
  let [variance, epsNode] := addEps.src | failure

  -- variance should be a reduce of squared values
  guard (variance.op == .REDUCE_AXIS)

  -- centered should be a subtraction (x - mean)
  guard (centered.op == .SUB)
  let [input, meanNode] := centered.src | failure

  -- mean should be a reduce
  guard (meanNode.op == .REDUCE_AXIS)

  -- Extract epsilon value
  let eps := match epsNode.arg with
    | .constFloat f => f
    | .constF32Bits bits => (Float32.ofBits bits).toFloat
    | _ => 1e-5

  -- Get axes from reduce
  let axes := match meanNode.arg with
    | .axes ax => ax
    | .reduceWithAxes _ ax => ax
    | _ => []

  -- Build cover set
  let mut cover := UOpIdSet.mkEmpty
  cover := cover.add u.uid
  cover := cover.add centered.uid
  cover := cover.add sqrtNode.uid
  cover := cover.add addEps.uid
  cover := cover.add variance.uid
  cover := cover.add meanNode.uid

  -- Check we're not breaking any kept nodes
  if UOpIdSet.member keep centered.uid ||
     UOpIdSet.member keep sqrtNode.uid ||
     UOpIdSet.member keep variance.uid ||
     UOpIdSet.member keep meanNode.uid then
    failure

  return {
    input := input.uid
    gamma := none  -- Could detect from wrapping MUL
    beta := none   -- Could detect from wrapping ADD
    eps := eps
    axes := axes
    shape := input.shape
    output := u.uid
    cover := cover
  }

/-- Compile LayerNorm pattern -/
def compile (u : UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : Option Plan :=
  detect? u keep refCnt

end TinyGrad4.Backend.FusedLayerNorm
