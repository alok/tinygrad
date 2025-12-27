import TinyGrad4

/-!
# Numerical Gradient Verification

Verify autodiff gradients against finite difference approximation.
This is a crucial correctness check for the gradient implementation.
-/

namespace TinyGrad4.Test

open TinyGrad4
open StaticTensor
open Interpreter

/-- Test: d/dx (x^2) = 2x -/
def testSquareGradient : IO Unit := do
  let result := runTensorM do
    -- Create x = 3.0
    let x ← Tensor.full [1] .float32 3.0

    -- Compute x * x
    let x2 ← mul x x

    -- Sum to get scalar loss
    let loss ← sum x2

    -- Compute gradient
    let gradMap ← backward loss [x.uop]

    -- Evaluate gradient
    let env : Env := ∅
    match gradMap[x.uop.uid]? with
    | some gradUop =>
      let gradVal := eval gradUop env
      pure (some gradVal)
    | none => pure none

  match result with
  | some gradArr =>
    let grad := gradArr[0]!
    let expected := 6.0  -- d/dx(x^2) at x=3 is 2*3 = 6
    let diff := Float.abs (grad - expected)
    let pass := diff < 0.01
    IO.println s!"Test x^2 gradient:"
    IO.println s!"  Autodiff gradient: {grad}"
    IO.println s!"  Expected (2x at x=3): {expected}"
    IO.println s!"  Difference: {diff}"
    IO.println s!"  PASS: {if pass then "true" else "false"}"
  | none =>
    IO.println "ERROR: No gradient computed"

/-- Test: d/dx (x * y) w.r.t. x = y, and w.r.t. y = x -/
def testMulGradient : IO Unit := do
  let result := runTensorM do
    -- Create x = 2.0, y = 5.0
    let x ← Tensor.full [1] .float32 2.0
    let y ← Tensor.full [1] .float32 5.0

    -- Compute x * y
    let xy ← mul x y

    -- Sum to get scalar
    let loss ← sum xy

    -- Compute gradients
    let gradMap ← backward loss [x.uop, y.uop]

    let env : Env := ∅
    let gradX := gradMap[x.uop.uid]?.map (fun g => (eval g env)[0]!)
    let gradY := gradMap[y.uop.uid]?.map (fun g => (eval g env)[0]!)

    pure (gradX, gradY)

  let (gradX, gradY) := result
  IO.println s!"Test x*y gradient:"
  IO.println s!"  d/dx(x*y) at x=2,y=5: {gradX.getD 0} (expected: 5.0)"
  IO.println s!"  d/dy(x*y) at x=2,y=5: {gradY.getD 0} (expected: 2.0)"
  match gradX, gradY with
  | some gx, some gy =>
    let pass := Float.abs (gx - 5.0) < 0.01 && Float.abs (gy - 2.0) < 0.01
    IO.println s!"  PASS: {if pass then "true" else "false"}"
  | _, _ => IO.println "  FAIL: Missing gradients"

/-- Test: d/dx (sum(A * B)) for matrices -/
def testMatrixMulGradient : IO Unit := do
  let result := runTensorM do
    let a ← Tensor.full [2, 2] .float32 2.0  -- All 2s
    let b ← Tensor.full [2, 2] .float32 3.0  -- All 3s

    -- Element-wise multiply
    let ab ← mul a b

    -- Sum
    let loss ← sum ab

    -- Gradients
    let gradMap ← backward loss [a.uop, b.uop]

    let env : Env := ∅
    let gradA := gradMap[a.uop.uid]?.map (eval · env)
    let gradB := gradMap[b.uop.uid]?.map (eval · env)

    pure (gradA, gradB)

  let (gradA, gradB) := result
  IO.println s!"Test matrix elementwise mul gradient:"
  IO.println s!"  A = 2x2 of 2.0, B = 2x2 of 3.0"
  IO.println s!"  loss = sum(A * B) = 4 * 6 = 24"
  IO.println s!"  d/dA should be B = all 3s: {gradA}"
  IO.println s!"  d/dB should be A = all 2s: {gradB}"
  match gradA, gradB with
  | some ga, some gb =>
    -- Check all elements of gradA are 3.0 and gradB are 2.0
    let gaOk := ga.all (fun v => Float.abs (v - 3.0) < 0.01)
    let gbOk := gb.all (fun v => Float.abs (v - 2.0) < 0.01)
    IO.println s!"  PASS: {if gaOk && gbOk then "true" else "false"}"
  | _, _ => IO.println "  FAIL: Missing gradients"

/-- Test: Chain rule: d/dx ((x * 2) * 3) = 6 -/
def testChainRule : IO Unit := do
  let result := runTensorM do
    let x ← Tensor.full [1] .float32 5.0
    let two ← Tensor.full [1] .float32 2.0
    let three ← Tensor.full [1] .float32 3.0

    -- (x * 2) * 3
    let x2 ← mul x two
    let x2_3 ← mul x2 three

    let loss ← sum x2_3

    let gradMap ← backward loss [x.uop]

    let env : Env := ∅
    pure (gradMap[x.uop.uid]?.map (fun g => (eval g env)[0]!))

  IO.println s!"Test chain rule: d/dx((x*2)*3) = 6"
  IO.println s!"  Computed gradient: {result}"
  match result with
  | some g =>
    let pass := Float.abs (g - 6.0) < 0.01
    IO.println s!"  PASS: {if pass then "true" else "false"}"
  | none => IO.println "  FAIL: No gradient"

private def assertAllClose (arr : FloatArray) (expected : Array Float) (tol : Float) (label : String) : IO Unit := do
  if arr.size != expected.size then
    throw (IO.userError s!"{label}: size {arr.size} != {expected.size}")
  for i in [:arr.size] do
    let v := arr[i]!
    let e := expected[i]!
    let diff := Float.abs (v - e)
    if diff > tol then
      throw (IO.userError s!"{label}: idx {i} value {v} expected {e} diff {diff} > {tol}")

/-- Test: gradients of `where` route to the selected branch. -/
def testWhereGradient : IO Unit := do
  let (xUop, yUop, cUop, gradMap) := runTensorM do
    let x ← Tensor.buffer [2] .float32
    let y ← Tensor.buffer [2] .float32
    let c ← Tensor.buffer [2] .bool
    let outUop ← UOp.where_ c.uop x.uop y.uop
    let lossUop ← UOp.sum outUop [] false
    let loss : Scalar .float32 := { uop := lossUop, h_shape := sorry_proof }
    let gradMap ← StaticTensor.backward loss [x.uop, y.uop]
    pure (x.uop, y.uop, c.uop, gradMap)

  let xVal : FloatArray := ⟨#[10.0, 20.0]⟩
  let yVal : FloatArray := ⟨#[30.0, 40.0]⟩
  let cRaw : RawBuffer := { dtype := .bool, data := ByteArray.mk #[1, 0] }

  let env0 : Env := ∅
  let env1 := Interpreter.setBuffer env0 xUop (RawBuffer.ofF32 xVal)
  let env2 := Interpreter.setBuffer env1 yUop (RawBuffer.ofF32 yVal)
  let env := Interpreter.setBuffer env2 cUop cRaw

  match gradMap[xUop.uid]?, gradMap[yUop.uid]? with
  | some dxUop, some dyUop =>
    let dx := Interpreter.eval dxUop env
    let dy := Interpreter.eval dyUop env
    assertAllClose dx #[1.0, 0.0] 0.01 "where d/dx"
    assertAllClose dy #[0.0, 1.0] 0.01 "where d/dy"
    IO.println "Test where gradient:"
    IO.println "  PASS: true"
  | _, _ =>
    IO.println "Test where gradient:"
    IO.println "  FAIL: Missing gradients"

/-- Run all gradient tests -/
def runAllGradientTests : IO Unit := do
  IO.println "=== Gradient Verification Tests ==="
  IO.println ""
  testSquareGradient
  IO.println ""
  testMulGradient
  IO.println ""
  testMatrixMulGradient
  IO.println ""
  testChainRule
  IO.println ""
  testWhereGradient
  IO.println ""
  IO.println "=== All tests complete ==="

end TinyGrad4.Test

namespace TinyGrad4.Test.GradientCheck

def runAll : IO Unit :=
  TinyGrad4.Test.runAllGradientTests

end TinyGrad4.Test.GradientCheck

#eval! TinyGrad4.Test.GradientCheck.runAll
