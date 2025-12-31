import TinyGrad4.Backend.FusedMatmul

namespace TinyGrad4.Backend.FusedMatmulExpr

/-!
# Fused Matmul Expression

An expression-based representation of fused matmul operations,
suitable for semantic comparison against spec definitions.
-/

/-- Expression form of a fused matmul plan.
    Captures the essential computation structure for verification. -/
structure Expr where
  m : Nat
  k : Nat
  n : Nat
  hasBias : Bool
  hasBias2 : Bool
  hasRelu : Bool
  scaleBits : Option UInt32
  deriving Repr

/-- Convert a fused matmul plan to an expression form. -/
def ofPlan (p : FusedMatmul.Plan) : Expr :=
  { m := p.m
    k := p.k
    n := p.n
    hasBias := true  -- FusedMatmul.Plan always has at least one bias
    hasBias2 := p.bias2.isSome
    hasRelu := p.relu
    scaleBits := p.scaleBits }

end TinyGrad4.Backend.FusedMatmulExpr
