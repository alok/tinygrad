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

/-- Debug test: check if basic CONST evaluation works -/
def debugConstEval : IO Unit := do
  IO.println "=== Debug: CONST evaluation ==="
  let (constUop, xUop, constSrc) := runTensorM do
    -- Create x = 3.0 (EXPAND(CONST 3.0))
    let x ← Tensor.full [1] .float32 3.0
    -- Also just get the raw CONST
    let c ← UOp.const .float32 3.0
    -- Get the CONST node that's inside EXPAND
    let constSrc := x.uop.src[0]!
    pure (c, x.uop, constSrc)

  IO.println s!"  CONST UOp: op={repr constUop.op}, arg={repr constUop.arg}, shape={constUop.shape}, uid={constUop.uid}"
  IO.println s!"  x UOp (EXPAND): op={repr xUop.op}, shape={xUop.shape}, uid={xUop.uid}"
  IO.println s!"  x.src[0] (inner CONST): uid={constSrc.uid}, arg={repr constSrc.arg}"

  -- Evaluate using evalMany to see the full cache
  let cache := Interpreter.evalMany [constUop] (∅ : Env)
  IO.println s!"  evalMany cache size: {cache.size}"
  for (uid, buf) in cache.toList do
    IO.println s!"    cache[{uid}] = {buf.reprAsString}"

  -- Now evaluate just the EXPAND
  let cache2 := Interpreter.evalMany [xUop] (∅ : Env)
  IO.println s!"  evalMany for EXPAND cache size: {cache2.size}"
  for (uid, buf) in cache2.toList do
    IO.println s!"    cache[{uid}] = {buf.reprAsString}"

  -- Test: manually create bytes for 3.0 and check
  let bits : UInt32 := (3.0 : Float32).toBits
  IO.println s!"  3.0 as Float32 bits: {bits}"

  -- Try creating a buffer manually
  let manualBuf := RawBuffer.ofFloats #[3.0]
  IO.println s!"  Manual buffer: {manualBuf.reprAsString}"

  IO.println ""

/-- Test: d/dx (x^2) = 2x -/
def testSquareGradient : IO Unit := do
  let result := runTensorM do
    -- Create x = 3.0
    let x ← Tensor.full [1] .float32 3.0

    -- Debug: evaluate forward pass first
    let env : Env := ∅
    let xVal := eval x.uop env

    -- Compute x * x
    let x2 ← mul x x
    let x2Val := eval x2.uop env

    -- Sum to get scalar loss
    let loss ← sum x2
    let lossVal := eval loss.uop env

    -- Compute gradient
    let gradMap ← backward loss [x.uop]

    -- Debug: print gradient UOp info
    match gradMap[x.uop.uid]? with
    | some gradUop =>
      let gradVal := eval gradUop env
      -- Return debug info
      pure (some (xVal, x2Val, lossVal, gradVal, gradUop.op, repr gradUop.arg))
    | none => pure none

  match result with
  | some (xVal, x2Val, lossVal, gradVal, gradOp, gradArg) =>
    let grad := gradVal[0]!
    let expected := 6.0  -- d/dx(x^2) at x=3 is 2*3 = 6
    let diff := Float.abs (grad - expected)
    let pass := diff < 0.01
    IO.println s!"Test x^2 gradient:"
    IO.println s!"  Forward x: {xVal.reprAsString}"
    IO.println s!"  Forward x^2: {x2Val.reprAsString}"
    IO.println s!"  Forward loss: {lossVal.reprAsString}"
    IO.println s!"  Gradient UOp: op={repr gradOp}, arg={gradArg}"
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
    let gaOk := (List.range ga.numF32).all fun i => Float.abs (ga.getF32 i - 3.0) < 0.01
    let gbOk := (List.range gb.numF32).all fun i => Float.abs (gb.getF32 i - 2.0) < 0.01
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

private def assertAllClose (arr : RawBuffer) (expected : Array Float) (tol : Float) (label : String) : IO Unit := do
  let n := arr.numF32
  if n != expected.size then
    throw (IO.userError s!"{label}: size {n} != {expected.size}")
  for i in [:n] do
    let v := arr.getF32 i
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

  let xVal := #[10.0, 20.0]
  let yVal := #[30.0, 40.0]
  let cRaw : RawBuffer := { dtype := .bool, data := ByteArray.mk #[1, 0] }

  let env0 : Env := ∅
  let env1 := Interpreter.setBuffer env0 xUop (RawBuffer.ofFloats xVal)
  let env2 := Interpreter.setBuffer env1 yUop (RawBuffer.ofFloats yVal)
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

/-- Test: d/dx sin(x) = cos(x) -/
def testSinGradient : IO Unit := do
  let result := runTensorM do
    let x ← Tensor.full [1] .float32 0.5  -- x = 0.5
    let y ← sin x
    let loss ← sum y
    let gradMap ← backward loss [x.uop]
    let env : Env := ∅
    match gradMap[x.uop.uid]? with
    | some gradUop => pure (some (eval gradUop env))
    | none => pure none

  match result with
  | some gradArr =>
    let grad := gradArr[0]!
    let expected := Float.cos 0.5  -- d/dx sin(x) = cos(x)
    let diff := Float.abs (grad - expected)
    let pass := diff < 0.01
    IO.println s!"Test sin gradient:"
    IO.println s!"  Autodiff: {grad}, Expected cos(0.5): {expected}"
    IO.println s!"  PASS: {if pass then "true" else "false"}"
  | none => IO.println "  FAIL: No gradient"

/-- Test: d/dx cos(x) = -sin(x) -/
def testCosGradient : IO Unit := do
  let result := runTensorM do
    let x ← Tensor.full [1] .float32 0.5
    let y ← cos x
    let loss ← sum y
    let gradMap ← backward loss [x.uop]
    let env : Env := ∅
    match gradMap[x.uop.uid]? with
    | some gradUop => pure (some (eval gradUop env))
    | none => pure none

  match result with
  | some gradArr =>
    let grad := gradArr[0]!
    let expected := -Float.sin 0.5  -- d/dx cos(x) = -sin(x)
    let diff := Float.abs (grad - expected)
    let pass := diff < 0.01
    IO.println s!"Test cos gradient:"
    IO.println s!"  Autodiff: {grad}, Expected -sin(0.5): {expected}"
    IO.println s!"  PASS: {if pass then "true" else "false"}"
  | none => IO.println "  FAIL: No gradient"

/-- Test: d/dx tan(x) = sec^2(x) = 1 + tan^2(x) -/
def testTanGradient : IO Unit := do
  let result := runTensorM do
    let x ← Tensor.full [1] .float32 0.3
    let y ← tan x
    let loss ← sum y
    let gradMap ← backward loss [x.uop]
    let env : Env := ∅
    match gradMap[x.uop.uid]? with
    | some gradUop => pure (some (eval gradUop env))
    | none => pure none

  match result with
  | some gradArr =>
    let grad := gradArr[0]!
    let tanVal := Float.tan 0.3
    let expected := 1.0 + tanVal * tanVal  -- sec^2(x) = 1 + tan^2(x)
    let diff := Float.abs (grad - expected)
    let pass := diff < 0.01
    IO.println s!"Test tan gradient:"
    IO.println s!"  Autodiff: {grad}, Expected 1+tan^2(0.3): {expected}"
    IO.println s!"  PASS: {if pass then "true" else "false"}"
  | none => IO.println "  FAIL: No gradient"

/-- Test: d/db b^e = e * b^(e-1) -/
def testPowGradientBase : IO Unit := do
  let result := runTensorM do
    let b ← Tensor.full [1] .float32 2.0  -- base = 2
    let e ← Tensor.full [1] .float32 3.0  -- exponent = 3 (2^3 = 8)
    let y ← pow b e
    let loss ← sum y
    let gradMap ← backward loss [b.uop]
    let env : Env := ∅
    match gradMap[b.uop.uid]? with
    | some gradUop => pure (some (eval gradUop env))
    | none => pure none

  match result with
  | some gradArr =>
    let grad := gradArr[0]!
    -- d/db b^e = e * b^(e-1) = 3 * 2^2 = 12
    let expected := 3.0 * 4.0
    let diff := Float.abs (grad - expected)
    let pass := diff < 0.1
    IO.println s!"Test pow gradient (d/db):"
    IO.println s!"  Autodiff: {grad}, Expected 3*2^2=12: {expected}"
    IO.println s!"  PASS: {if pass then "true" else "false"}"
  | none => IO.println "  FAIL: No gradient"

/-- Test: d/dx exp2(x) = exp2(x) * ln(2) -/
def testExp2Gradient : IO Unit := do
  let result := runTensorM do
    let x ← Tensor.full [1] .float32 2.0  -- 2^2 = 4
    let y ← exp2 x
    let loss ← sum y
    let gradMap ← backward loss [x.uop]
    let env : Env := ∅
    match gradMap[x.uop.uid]? with
    | some gradUop => pure (some (eval gradUop env))
    | none => pure none

  match result with
  | some gradArr =>
    let grad := gradArr[0]!
    let ln2 := 0.6931471805599453
    let expected := 4.0 * ln2  -- 2^2 * ln(2) ≈ 2.77
    let diff := Float.abs (grad - expected)
    let pass := diff < 0.01
    IO.println s!"Test exp2 gradient:"
    IO.println s!"  Autodiff: {grad}, Expected 4*ln(2): {expected}"
    IO.println s!"  PASS: {if pass then "true" else "false"}"
  | none => IO.println "  FAIL: No gradient"

/-- Test: d/dx log2(x) = 1/(x * ln(2)) -/
def testLog2Gradient : IO Unit := do
  let result := runTensorM do
    let x ← Tensor.full [1] .float32 4.0  -- log2(4) = 2
    let y ← log2 x
    let loss ← sum y
    let gradMap ← backward loss [x.uop]
    let env : Env := ∅
    match gradMap[x.uop.uid]? with
    | some gradUop => pure (some (eval gradUop env))
    | none => pure none

  match result with
  | some gradArr =>
    let grad := gradArr[0]!
    let ln2 := 0.6931471805599453
    let expected := 1.0 / (4.0 * ln2)  -- 1/(4 * ln(2)) ≈ 0.36
    let diff := Float.abs (grad - expected)
    let pass := diff < 0.01
    IO.println s!"Test log2 gradient:"
    IO.println s!"  Autodiff: {grad}, Expected 1/(4*ln2): {expected}"
    IO.println s!"  PASS: {if pass then "true" else "false"}"
  | none => IO.println "  FAIL: No gradient"

/-- Test: d/dx sqrt(x) = 1/(2*sqrt(x)) -/
def testSqrtGradient : IO Unit := do
  let result := runTensorM do
    let x ← Tensor.full [1] .float32 4.0  -- sqrt(4) = 2
    let y ← sqrt x
    let loss ← sum y
    let gradMap ← backward loss [x.uop]
    let env : Env := ∅
    match gradMap[x.uop.uid]? with
    | some gradUop => pure (some (eval gradUop env))
    | none => pure none

  match result with
  | some gradArr =>
    let grad := gradArr[0]!
    let expected := 1.0 / (2.0 * 2.0)  -- 1/(2*sqrt(4)) = 0.25
    let diff := Float.abs (grad - expected)
    let pass := diff < 0.01
    IO.println s!"Test sqrt gradient:"
    IO.println s!"  Autodiff: {grad}, Expected 1/(2*2)=0.25: {expected}"
    IO.println s!"  PASS: {if pass then "true" else "false"}"
  | none => IO.println "  FAIL: No gradient"

/-- Test: d/dx (1/x) = -1/x^2 -/
def testReciprocalGradient : IO Unit := do
  let result := runTensorM do
    let x ← Tensor.full [1] .float32 2.0  -- 1/2 = 0.5
    let y ← recip x
    let loss ← sum y
    let gradMap ← backward loss [x.uop]
    let env : Env := ∅
    match gradMap[x.uop.uid]? with
    | some gradUop => pure (some (eval gradUop env))
    | none => pure none

  match result with
  | some gradArr =>
    let grad := gradArr[0]!
    let expected := -1.0 / 4.0  -- -1/x^2 = -1/4 = -0.25
    let diff := Float.abs (grad - expected)
    let pass := diff < 0.01
    IO.println s!"Test reciprocal gradient:"
    IO.println s!"  Autodiff: {grad}, Expected -1/4=-0.25: {expected}"
    IO.println s!"  PASS: {if pass then "true" else "false"}"
  | none => IO.println "  FAIL: No gradient"

/-- Run all gradient tests -/
def runAllGradientTests : IO Unit := do
  IO.println "=== Gradient Verification Tests ==="
  IO.println ""
  debugConstEval
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
  IO.println "=== Trig/Math Gradient Tests ==="
  IO.println ""
  testSinGradient
  IO.println ""
  testCosGradient
  IO.println ""
  testTanGradient
  IO.println ""
  testExp2Gradient
  IO.println ""
  testLog2Gradient
  IO.println ""
  testSqrtGradient
  IO.println ""
  testReciprocalGradient
  IO.println ""
  testPowGradientBase
  IO.println ""
  IO.println "=== All tests complete ==="

end TinyGrad4.Test

namespace TinyGrad4.Test.GradientCheck

def runAll : IO Unit :=
  TinyGrad4.Test.runAllGradientTests

end TinyGrad4.Test.GradientCheck

def main : IO Unit := TinyGrad4.Test.GradientCheck.runAll
