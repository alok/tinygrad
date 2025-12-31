import TinyGrad4.Backend.Pattern
import TinyGrad4.UOp.UOp

/-!
# Pattern Matching Tests

Tests for declarative fusion pattern matchers.
-/

namespace TinyGrad4.Test.PatternTest

open TinyGrad4.Backend.Pattern

-- Helper to make test UOps
private def mkConst (id : Nat) (v : Float) : UOp :=
  { uid := ⟨id⟩
    op := .CONST
    dtype := .float32
    src := []
    arg := .constF32Bits v.toFloat32.toBits
    shape := [1] }

private def mkBuffer (id : Nat) (shape : Shape) : UOp :=
  { uid := ⟨id⟩
    op := .BUFFER
    dtype := .float32
    src := []
    arg := .empty
    shape := shape }

private def mkUnary (id : Nat) (op : Ops) (x : UOp) : UOp :=
  { uid := ⟨id⟩
    op := op
    dtype := .float32
    src := [x]
    arg := .empty
    shape := x.shape }

private def mkBinary (id : Nat) (op : Ops) (a b : UOp) : UOp :=
  { uid := ⟨id⟩
    op := op
    dtype := .float32
    src := [a, b]
    arg := .empty
    shape := a.shape }

private def mkReduce (id : Nat) (reduceOp : Ops) (x : UOp) (axes : List Nat) : UOp :=
  -- For simplicity, just keep the shape (reduce output would change shape, but we only care about pattern matching)
  { uid := ⟨id⟩
    op := .REDUCE_AXIS
    dtype := .float32
    src := [x]
    arg := .reduceWithAxes reduceOp axes
    shape := x.shape }

-- Test primitive matchers
#eval do
  let x := mkBuffer 0 [2, 3]
  let y := mkBuffer 1 [2, 3]
  let add := mkBinary 2 .ADD x y

  match UOp.asAdd? add with
  | some (a, b) => IO.println s!"ADD matched: {a.uid} + {b.uid}"
  | none => IO.println "ADD not matched"

#eval do
  let x := mkBuffer 0 [2, 3]
  let neg := mkUnary 1 .NEG x

  match UOp.asNeg? neg with
  | some inner => IO.println s!"NEG matched: neg({inner.uid})"
  | none => IO.println "NEG not matched"

-- Test EXP2
#eval do
  let x := mkBuffer 0 [2, 3]
  let exp := mkUnary 1 .EXP2 x

  match UOp.asExp2? exp with
  | some inner => IO.println s!"EXP2 matched: exp2({inner.uid})"
  | none => IO.println "EXP2 not matched"

-- Test reduce matcher
#eval do
  let x := mkBuffer 0 [2, 3, 4]
  let sumX := mkReduce 1 .ADD x [1]

  match UOp.asReduceAdd? sumX with
  | some (src, axes) => IO.println s!"REDUCE_ADD matched: reduce_add({src.uid}, axes={axes})"
  | none => IO.println "REDUCE_ADD not matched"

-- Test pattern composition with <|>
def testComposition (u : UOp) : Option String :=
  (UOp.asAdd? u >>= fun (a, b) => some s!"add({a.uid}, {b.uid})") <|>
  (UOp.asMul? u >>= fun (a, b) => some s!"mul({a.uid}, {b.uid})") <|>
  (UOp.asNeg? u >>= fun x => some s!"neg({x.uid})") <|>
  some s!"other({repr u.op})"

#eval do
  let x := mkBuffer 0 [2, 3]
  let y := mkBuffer 1 [2, 3]
  let add := mkBinary 2 .ADD x y
  let mul := mkBinary 3 .MUL x y
  let neg := mkUnary 4 .NEG x

  IO.println s!"add: {testComposition add}"
  IO.println s!"mul: {testComposition mul}"
  IO.println s!"neg: {testComposition neg}"
  IO.println s!"buffer: {testComposition x}"

-- Build a softmax-like structure and test matching
#eval do
  -- softmax(x) = exp2(x - max(x)) / sum(exp2(x - max(x)))
  let x := mkBuffer 0 [2, 10]  -- shape [2, 10], softmax over axis 1
  let maxX := mkReduce 1 .MAX x [1]
  let shifted := mkBinary 2 .SUB x maxX
  let expShifted := mkUnary 3 .EXP2 shifted
  let sumExp := mkReduce 4 .ADD expShifted [1]
  let result := mkBinary 5 .FDIV expShifted sumExp

  match softmax? result with
  | some info => IO.println s!"Softmax matched! input={info.input.uid}, axis={info.axis}"
  | none => IO.println "Softmax NOT matched"

  -- Also test the intermediate pieces
  match UOp.asDiv? result with
  | some (n, d) => IO.println s!"  FDIV: numer={n.uid}, denom={d.uid}"
  | none => IO.println "  FDIV not matched"

-- Test that non-softmax doesn't match
#eval do
  let x := mkBuffer 0 [2, 3]
  let y := mkBuffer 1 [2, 3]
  let add := mkBinary 2 .ADD x y

  match softmax? add with
  | some _ => IO.println "ERROR: ADD matched as softmax"
  | none => IO.println "Correctly rejected ADD as softmax"

-- Test matmulRelu pattern
#eval do
  -- matmul_relu(a, b) = max(contract(a, b), 0)
  let a := mkBuffer 0 [2, 3]
  let b := mkBuffer 1 [3, 4]
  let contract : UOp :=
    { uid := ⟨2⟩
      op := .CONTRACT
      dtype := .float32
      src := [a, b]
      arg := .empty
      shape := [2, 4] }
  let zero := mkConst 3 0.0
  let relu := mkBinary 4 .MAX contract zero

  match matmulRelu? relu with
  | some info => IO.println s!"MatmulRelu matched! a={info.a.uid}, b={info.b.uid}, hasBias={info.hasBias}"
  | none => IO.println "MatmulRelu NOT matched"

-- Test matmulRelu with bias
#eval do
  let a := mkBuffer 0 [2, 3]
  let b := mkBuffer 1 [3, 4]
  let bias := mkBuffer 2 [4]
  let contract : UOp :=
    { uid := ⟨3⟩
      op := .CONTRACT
      dtype := .float32
      src := [a, b]
      arg := .empty
      shape := [2, 4] }
  let biased := mkBinary 4 .ADD contract bias
  let zero := mkConst 5 0.0
  let relu := mkBinary 6 .MAX biased zero

  match matmulRelu? relu with
  | some info =>
    let biasId : Option UOpId := info.bias.map (fun (u : UOp) => u.uid)
    IO.println s!"MatmulRelu+Bias matched! a={info.a.uid}, b={info.b.uid}, hasBias={info.hasBias}, bias={biasId}"
  | none => IO.println "MatmulRelu+Bias NOT matched"

-- Test that non-relu max doesn't match matmulRelu
#eval do
  let a := mkBuffer 0 [2, 3]
  let b := mkBuffer 1 [3, 4]
  let contract : UOp :=
    { uid := ⟨2⟩
      op := .CONTRACT
      dtype := .float32
      src := [a, b]
      arg := .empty
      shape := [2, 4] }
  let one := mkConst 3 1.0  -- Not zero!
  let notRelu := mkBinary 4 .MAX contract one

  match matmulRelu? notRelu with
  | some _ => IO.println "ERROR: max(matmul, 1) matched as matmulRelu"
  | none => IO.println "Correctly rejected max(matmul, 1) as matmulRelu"

-- Test biasRelu pattern
#eval do
  let x := mkBuffer 0 [2, 3]
  let bias := mkBuffer 1 [3]
  let added := mkBinary 2 .ADD x bias
  let zero := mkConst 3 0.0
  let relu := mkBinary 4 .MAX added zero

  match biasRelu? relu with
  | some info => IO.println s!"BiasRelu matched! input={info.input.uid}, bias={info.bias.uid}"
  | none => IO.println "BiasRelu NOT matched"

end TinyGrad4.Test.PatternTest
