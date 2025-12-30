import TinyGrad4
import TinyGrad4.NN
import TinyGrad4.Backend.Native

/-!
# Conv2dSmoke

Smoke tests for convolution operations and NN layers.
-/

namespace TinyGrad4.Test.Conv2dSmoke

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

open TinyGrad4
open Interpreter
open StaticTensor
open TinyGrad4.NN
open TinyGrad4.Backend.Native

private def assertShape (got expected : Shape) (label : String) : IO Unit := do
  if got != expected then
    throw (IO.userError s!"{label}: shape {got} != {expected}")

private def assertDType (got expected : DType) (label : String) : IO Unit := do
  if got != expected then
    throw (IO.userError s!"{label}: dtype {repr got} != {repr expected}")

def testConv2dShapeBasic : IO Unit := do
  -- Test conv2d output shape computation
  -- Input: [1, 3, 32, 32], kernel: [16, 3, 3, 3], padding=1, stride=1
  -- Output should be: [1, 16, 32, 32]
  let outShape := Shape.conv2dOut [1, 3, 32, 32] [16, 3, 3, 3] 1 1 1
  assertShape outShape [1, 16, 32, 32] "conv2d basic shape"

def testConv2dShapeStride : IO Unit := do
  -- Test conv2d with stride=2
  -- Input: [1, 3, 32, 32], kernel: [16, 3, 3, 3], padding=1, stride=2
  -- Output should be: [1, 16, 16, 16]
  let outShape := Shape.conv2dOut [1, 3, 32, 32] [16, 3, 3, 3] 1 2 1
  assertShape outShape [1, 16, 16, 16] "conv2d stride=2 shape"

def testConv2dShapeNoPadding : IO Unit := do
  -- Test conv2d without padding
  -- Input: [1, 3, 32, 32], kernel: [16, 3, 3, 3], padding=0, stride=1
  -- Output should be: [1, 16, 30, 30]
  let outShape := Shape.conv2dOut [1, 3, 32, 32] [16, 3, 3, 3] 0 1 1
  assertShape outShape [1, 16, 30, 30] "conv2d no padding shape"

def testConv1dShapeBasic : IO Unit := do
  -- Test conv1d output shape computation
  -- Input: [1, 3, 32], kernel: [16, 3, 3], padding=1, stride=1
  -- Output should be: [1, 16, 32]
  let outShape := Shape.conv1dOut [1, 3, 32] [16, 3, 3] 1 1 1
  assertShape outShape [1, 16, 32] "conv1d basic shape"

def testConv1dShapeStride : IO Unit := do
  -- Test conv1d with stride=2
  -- Input: [1, 3, 32], kernel: [16, 3, 3], padding=1, stride=2
  -- Output should be: [1, 16, 16]
  let outShape := Shape.conv1dOut [1, 3, 32] [16, 3, 3] 1 2 1
  assertShape outShape [1, 16, 16] "conv1d stride=2 shape"

def testConv2dTensor : IO Unit := do
  -- Test conv2d function returns correct shape and dtype
  let y := runTensorM do
    let x ← Tensor.buffer [2, 3, 8, 8] .float32
    let weight ← Tensor.buffer [16, 3, 3, 3] .float32
    conv2d x weight none 1 1 1
  assertShape y.uop.shape [2, 16, 8, 8] "conv2d tensor shape"
  assertDType y.uop.dtype .float32 "conv2d tensor dtype"

def testConv1dTensor : IO Unit := do
  -- Test conv1d function returns correct shape and dtype
  let y := runTensorM do
    let x ← Tensor.buffer [2, 3, 8] .float32
    let weight ← Tensor.buffer [16, 3, 3] .float32
    conv1d x weight none 1 1 1
  assertShape y.uop.shape [2, 16, 8] "conv1d tensor shape"
  assertDType y.uop.dtype .float32 "conv1d tensor dtype"

def testMaxPool2dShape : IO Unit := do
  -- Test maxPool2d output shape using convOutDim directly
  -- Input: [1, 3, 32, 32], kernel: 2, stride: 2, padding: 0
  -- Output spatial: (32 + 0 - 1*(2-1) - 1) / 2 + 1 = (32 - 1) / 2 + 1 = 16
  let hOut := Shape.convOutDim 32 0 1 2 2
  let wOut := Shape.convOutDim 32 0 1 2 2
  if hOut != 16 then
    throw (IO.userError s!"maxPool2d hOut: {hOut} != 16")
  if wOut != 16 then
    throw (IO.userError s!"maxPool2d wOut: {wOut} != 16")

def testMaxPool2dTensor : IO Unit := do
  let y := runTensorM do
    let x ← Tensor.buffer [2, 3, 8, 8] .float32
    maxPool2d x 2 2 0
  assertShape y.uop.shape [2, 3, 4, 4] "maxPool2d tensor shape"
  assertDType y.uop.dtype .float32 "maxPool2d tensor dtype"

def testAvgPool2dTensor : IO Unit := do
  let y := runTensorM do
    let x ← Tensor.buffer [2, 3, 8, 8] .float32
    avgPool2d x 2 2 0
  assertShape y.uop.shape [2, 3, 4, 4] "avgPool2d tensor shape"
  assertDType y.uop.dtype .float32 "avgPool2d tensor dtype"

def testConv2dParams : IO Unit := do
  -- Test Conv2dParams creation
  let params := runTensorM do
    Conv2dParams.create 3 16 3 .float32 true 1 1 1 42
  assertShape params.weight.uop.shape [16, 3, 3, 3] "conv2d weight shape"
  match params.bias with
  | some b => assertShape b.uop.shape [16] "conv2d bias shape"
  | none => throw (IO.userError "conv2d bias should exist")

def testConv2dParamsNoBias : IO Unit := do
  let params := runTensorM do
    Conv2dParams.create 3 16 3 .float32 false 0 1 1 42
  assertShape params.weight.uop.shape [16, 3, 3, 3] "conv2d weight shape (no bias)"
  match params.bias with
  | some _ => throw (IO.userError "conv2d bias should not exist")
  | none => pure ()

def testMaxPool2dParams : IO Unit := do
  let params := MaxPool2dParams.create 2 2 0
  if params.kernelSize != 2 then
    throw (IO.userError s!"maxPool2d kernelSize: {params.kernelSize} != 2")
  if params.stride != 2 then
    throw (IO.userError s!"maxPool2d stride: {params.stride} != 2")
  if params.padding != 0 then
    throw (IO.userError s!"maxPool2d padding: {params.padding} != 0")

def testMaxPool2dParamsDefaultStride : IO Unit := do
  -- When stride=0 is passed, it should default to kernelSize
  let params := MaxPool2dParams.create 3 0 0
  if params.stride != 3 then
    throw (IO.userError s!"maxPool2d default stride: {params.stride} != 3")

def testUniformInit : IO Unit := do
  let t := runTensorM do
    uniformInit [2, 3] .float32 (-1.0) 1.0 42
  assertShape t.uop.shape [2, 3] "uniformInit shape"
  assertDType t.uop.dtype .float32 "uniformInit dtype"

def testTileShape : IO Unit := do
  -- Test tile (repeat) output shape
  let outShape := Shape.repeatOut [2, 3] [2, 3]
  assertShape outShape [4, 9] "tile shape"

def testTileTensor : IO Unit := do
  let y := runTensorM do
    let x ← Tensor.buffer [2, 3] .float32
    tile x [2, 3]
  assertShape y.uop.shape [4, 9] "tile tensor shape"
  assertDType y.uop.dtype .float32 "tile tensor dtype"

def testPoolOutShape : IO Unit := do
  -- Test pool output shape
  -- Input: [1, 3, 8, 8], kernel: [2, 2], stride: [2, 2], dilation: [1, 1]
  let outShape := Shape.poolOut [1, 3, 8, 8] [2, 2] [2, 2] [1, 1]
  -- Output should be: [1, 3, 4, 4, 2, 2]
  assertShape outShape [1, 3, 4, 4, 2, 2] "pool out shape"

-- ============================================================================
-- Value Verification Tests
-- ============================================================================

private def assertEqF32 (raw : RawBuffer) (expected : Array Float) (label : String) : IO Unit := do
  let got := raw.toFloatArray
  if got.size != expected.size then
    throw (IO.userError s!"{label}: size {got.size} != {expected.size}")
  for i in [:expected.size] do
    if Float.abs (got[i]! - expected[i]!) > 0.0001 then
      throw (IO.userError s!"{label}: idx {i} {got[i]!} != {expected[i]!}")

private def assertApproxF32 (raw : RawBuffer) (expected : Array Float) (tol : Float) (label : String) : IO Unit := do
  let got := raw.toFloatArray
  if got.size != expected.size then
    throw (IO.userError s!"{label}: size {got.size} != {expected.size}")
  for i in [:expected.size] do
    if Float.abs (got[i]! - expected[i]!) > tol then
      throw (IO.userError s!"{label}: idx {i} {got[i]!} != {expected[i]!} (tol={tol})")

/-- Test maxPool2d with known values.
    Input: 2x2 tensor [[1, 2], [3, 4]], kernel 2x2, stride 2
    Expected output: [4] (max of all elements) -/
def testMaxPool2dValues : IO Unit := do
  let (base, result) := runTensorM do
    let base ← Tensor.buffer [1, 1, 2, 2] .float32
    let result ← maxPool2d base 2 2 0
    pure (base, result)

  -- Set known values: [1, 2, 3, 4]
  let inputData := packF32FromF64 (FloatArray.mk #[1.0, 2.0, 3.0, 4.0])
  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }

  let resultRaw := evalTensor result env
  -- maxPool 2x2 over [[1,2],[3,4]] should give max = 4
  assertEqF32 resultRaw #[4.0] "maxPool2d values"

/-- Test avgPool2d with known values.
    Input: 2x2 tensor [[1, 2], [3, 4]], kernel 2x2, stride 2
    Expected output: [2.5] (average of all elements) -/
def testAvgPool2dValues : IO Unit := do
  let (base, result) := runTensorM do
    let base ← Tensor.buffer [1, 1, 2, 2] .float32
    let result ← avgPool2d base 2 2 0
    pure (base, result)

  -- Set known values: [1, 2, 3, 4]
  let inputData := packF32FromF64 (FloatArray.mk #[1.0, 2.0, 3.0, 4.0])
  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }

  let resultRaw := evalTensor result env
  -- avgPool 2x2 over [[1,2],[3,4]] should give mean = (1+2+3+4)/4 = 2.5
  assertEqF32 resultRaw #[2.5] "avgPool2d values"

/-- Test conv1d with 1x1 kernel (pointwise convolution).
    Input: [1..5]
    Kernel: 1x1 with weight 2.0 (doubles the input) -/
def testConv1d1x1Kernel : IO Unit := do
  let (base, weight, result) := runTensorM do
    let base ← Tensor.buffer [1, 1, 5] .float32
    let weight ← Tensor.buffer [1, 1, 1] .float32
    let result ← conv1d base weight none 0 1 1
    pure (base, weight, result)

  let inputData := packF32FromF64 (FloatArray.mk #[1.0, 2.0, 3.0, 4.0, 5.0])
  let weightData := packF32FromF64 (FloatArray.mk #[2.0])

  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }
  let env := Interpreter.setBuffer env weight.uop { dtype := .float32, data := weightData }

  let resultRaw := Interpreter.evalTensor result env
  assertEqF32 resultRaw #[2.0, 4.0, 6.0, 8.0, 10.0] "conv1d 1x1 kernel"

/-- Test conv1d with all-ones 3x1 kernel (computes local sums). -/
def testConv1d3KernelSum : IO Unit := do
  let (base, weight, result) := runTensorM do
    let base ← Tensor.buffer [1, 1, 5] .float32
    let weight ← Tensor.buffer [1, 1, 3] .float32
    let result ← conv1d base weight none 0 1 1
    pure (base, weight, result)

  let inputData := packF32FromF64 (FloatArray.mk #[1.0, 2.0, 3.0, 4.0, 5.0])
  let weightData := packF32FromF64 (FloatArray.mk #[1.0, 1.0, 1.0])

  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }
  let env := Interpreter.setBuffer env weight.uop { dtype := .float32, data := weightData }

  let resultRaw := Interpreter.evalTensor result env
  -- Output length = 3: [1+2+3, 2+3+4, 3+4+5]
  assertEqF32 resultRaw #[6.0, 9.0, 12.0] "conv1d 3 kernel sum"

/-- Test conv1d 1x1 with bias. -/
def testConv1d1x1WithBias : IO Unit := do
  let (base, weight, bias, result) := runTensorM do
    let base ← Tensor.buffer [1, 1, 5] .float32
    let weight ← Tensor.buffer [1, 1, 1] .float32
    let bias ← Tensor.buffer [1] .float32
    let result ← conv1d base weight (some bias) 0 1 1
    pure (base, weight, bias, result)

  let inputData := packF32FromF64 (FloatArray.mk #[1.0, 1.0, 1.0, 1.0, 1.0])
  let weightData := packF32FromF64 (FloatArray.mk #[2.0])
  let biasData := packF32FromF64 (FloatArray.mk #[0.5])

  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }
  let env := Interpreter.setBuffer env weight.uop { dtype := .float32, data := weightData }
  let env := Interpreter.setBuffer env bias.uop { dtype := .float32, data := biasData }

  let resultRaw := Interpreter.evalTensor result env
  assertEqF32 resultRaw #[2.5, 2.5, 2.5, 2.5, 2.5] "conv1d 1x1 with bias"

/-- Test conv2d with 1x1 kernel (pointwise convolution).
    Input: 3x3 grid [1..9]
    Kernel: 1x1 with weight 2.0 (doubles the input) -/
def testConv2d1x1Kernel : IO Unit := do
  let (base, weight, result) := runTensorM do
    let base ← Tensor.buffer [1, 1, 3, 3] .float32
    let weight ← Tensor.buffer [1, 1, 1, 1] .float32
    let result ← conv2d base weight none 0 1 1
    pure (base, weight, result)

  let inputData := packF32FromF64 (FloatArray.mk #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
  let weightData := packF32FromF64 (FloatArray.mk #[2.0])

  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }
  let env := Interpreter.setBuffer env weight.uop { dtype := .float32, data := weightData }

  let resultRaw := Interpreter.evalTensor result env
  -- 1x1 conv with weight 2 should double all values
  assertEqF32 resultRaw #[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0] "conv2d 1x1 kernel"

/-- Test conv2d with all-ones 3x3 kernel (computes local sums). -/
def testConv2d3x3SumKernel : IO Unit := do
  let (base, weight, result) := runTensorM do
    let base ← Tensor.buffer [1, 1, 3, 3] .float32
    let weight ← Tensor.buffer [1, 1, 3, 3] .float32
    let result ← conv2d base weight none 0 1 1  -- no padding, stride 1
    pure (base, weight, result)

  -- Input: 3x3 grid of ones
  let inputData := packF32FromF64 (FloatArray.mk #[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
  -- Weight: 3x3 kernel of ones
  let weightData := packF32FromF64 (FloatArray.mk #[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }
  let env := Interpreter.setBuffer env weight.uop { dtype := .float32, data := weightData }

  let resultRaw := evalTensor result env
  -- 3x3 conv with all-ones kernel on 3x3 all-ones input (no padding) = 1x1 output = [9]
  assertEqF32 resultRaw #[9.0] "conv2d 3x3 sum kernel"

/-- Test 2D permute (transpose) with value verification.
    Input [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]] -/
def testPermute2D : IO Unit := do
  let (base, result) := runTensorM do
    let base ← Tensor.buffer [2, 3] .float32
    let result ← permute base [1, 0]  -- transpose
    pure (base, result)

  let inputData := packF32FromF64 (FloatArray.mk #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }

  let resultRaw := evalTensor result env
  -- Transpose [2,3] -> [3,2]: [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
  assertEqF32 resultRaw #[1.0, 4.0, 2.0, 5.0, 3.0, 6.0] "permute 2D values"

/-- Test Native.permuteF32 directly at various ranks. -/
def testPermuteNative : IO Unit := do
  -- 2D transpose
  let inputData := packF32FromF64 (FloatArray.mk #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let result2D := permuteF32 inputData #[2, 3] #[1, 0]
  if ByteArray.size result2D != 24 then
    throw (IO.userError s!"Native permuteF32 2D size: {ByteArray.size result2D} != 24")

  -- 3D permute
  let inputData3D := packF32FromF64 (FloatArray.mk (List.replicate 24 1.0 |>.toArray))
  let result3D := permuteF32 inputData3D #[2, 3, 4] #[2, 0, 1]
  if ByteArray.size result3D != 96 then
    throw (IO.userError s!"Native permuteF32 3D size: {ByteArray.size result3D} != 96")

  -- 6D permute (like what pool produces)
  let inputData6D := packF32FromF64 (FloatArray.mk (List.replicate 9 1.0 |>.toArray))
  let result6D := permuteF32 inputData6D #[1, 1, 1, 3, 1, 3] #[0, 1, 3, 5, 2, 4]
  if ByteArray.size result6D != 36 then
    throw (IO.userError s!"Native permuteF32 6D size: {ByteArray.size result6D} != 36")

/-- Test 3D permute [2, 3, 4] with perm [2, 0, 1] -> [4, 2, 3]. -/
def testPermute3D : IO Unit := do
  let (base, result) := runTensorM do
    let base ← Tensor.buffer [2, 3, 4] .float32
    let result ← permute base [2, 0, 1]
    pure (base, result)

  let inputData := packF32FromF64 (FloatArray.mk (List.replicate 24 1.0 |>.toArray))
  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }

  let resultRaw := evalTensor result env
  if resultRaw.data.size != 96 then  -- 24 floats * 4 bytes
    throw (IO.userError s!"3D permute size: {resultRaw.data.size} != 96")
  assertShape result.uop.shape [4, 2, 3] "permute 3D shape"

/-- Test 4D permute (full reverse). -/
def testPermute4D : IO Unit := do
  let (base, result) := runTensorM do
    let base ← Tensor.buffer [2, 3, 4, 5] .float32
    let result ← permute base [3, 2, 1, 0]
    pure (base, result)

  let inputData := packF32FromF64 (FloatArray.mk (List.replicate 120 1.0 |>.toArray))
  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }

  let resultRaw := evalTensor result env
  if resultRaw.data.size != 480 then  -- 120 floats * 4 bytes
    throw (IO.userError s!"4D permute size: {resultRaw.data.size} != 480")
  assertShape result.uop.shape [5, 4, 3, 2] "permute 4D shape"

/-- Test 6D permute (like pool's intermediate tensor). -/
def testPermute6D : IO Unit := do
  let (base, result) := runTensorM do
    let base ← Tensor.buffer [1, 1, 1, 3, 1, 3] .float32
    let result ← permute base [0, 1, 3, 5, 2, 4]
    pure (base, result)

  let inputData := packF32FromF64 (FloatArray.mk #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }

  let resultRaw := evalTensor result env
  if resultRaw.data.size != 36 then  -- 9 floats * 4 bytes
    throw (IO.userError s!"6D permute size: {resultRaw.data.size} != 36")
  assertShape result.uop.shape [1, 1, 3, 3, 1, 1] "permute 6D shape"

/-- Test conv2d 1x1 with bias. -/
def testConv2d1x1WithBias : IO Unit := do
  let (base, weight, bias, result) := runTensorM do
    let base ← Tensor.buffer [1, 1, 3, 3] .float32
    let weight ← Tensor.buffer [1, 1, 1, 1] .float32
    let bias ← Tensor.buffer [1] .float32
    let result ← conv2d base weight (some bias) 0 1 1
    pure (base, weight, bias, result)

  -- Input: all ones
  let inputData := packF32FromF64 (FloatArray.mk (List.replicate 9 1.0 |>.toArray))
  -- Weight: 1x1 kernel with value 2.0
  let weightData := packF32FromF64 (FloatArray.mk #[2.0])
  -- Bias: 0.5
  let biasData := packF32FromF64 (FloatArray.mk #[0.5])

  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }
  let env := Interpreter.setBuffer env weight.uop { dtype := .float32, data := weightData }
  let env := Interpreter.setBuffer env bias.uop { dtype := .float32, data := biasData }

  let resultRaw := Interpreter.evalTensor result env
  -- 1x1 conv with weight 2 and bias 0.5: input * 2 + 0.5 = 1 * 2 + 0.5 = 2.5
  assertEqF32 resultRaw (List.replicate 9 2.5 |>.toArray) "conv2d 1x1 with bias"

/-- Debug test: trace pad2d operation in isolation. -/
def testPad2dIsolated : IO Unit := do
  let (base, result) := runTensorM do
    let base ← Tensor.buffer [1, 1, 2, 2] .float32
    let result ← pad2d base 1 1  -- pad by 1 on each side -> [1, 1, 4, 4]
    pure (base, result)

  -- Input: [[1,2],[3,4]]
  let inputData := packF32FromF64 (FloatArray.mk #[1.0, 2.0, 3.0, 4.0])
  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }

  let resultRaw := Interpreter.evalTensor result env
  -- Expected 4x4 with zeros around edges:
  -- [[0,0,0,0], [0,1,2,0], [0,3,4,0], [0,0,0,0]]
  -- = [0,0,0,0, 0,1,2,0, 0,3,4,0, 0,0,0,0]
  if resultRaw.data.size != 64 then  -- 16 floats * 4 bytes
    throw (IO.userError s!"pad2d size: {resultRaw.data.size} != 64")
  assertEqF32 resultRaw #[0.0, 0.0, 0.0, 0.0,
                          0.0, 1.0, 2.0, 0.0,
                          0.0, 3.0, 4.0, 0.0,
                          0.0, 0.0, 0.0, 0.0] "pad2d isolated"

/-- Test conv2d with 3x3 kernel and padding=1 (same padding).
    Input: 3x3 grid of ones
    Kernel: 3x3 kernel of ones
    Expected output:
      [[4,6,4], [6,9,6], [4,6,4]]
    where 4 = corner (4 neighbors), 6 = edge (6 neighbors), 9 = center (all 9) -/
def testConv2d3x3WithPadding : IO Unit := do
  let (base, weight, result) := runTensorM do
    let base ← Tensor.buffer [1, 1, 3, 3] .float32
    let weight ← Tensor.buffer [1, 1, 3, 3] .float32
    let result ← conv2d base weight none 1 1 1  -- padding=1
    pure (base, weight, result)

  let inputData := packF32FromF64 (FloatArray.mk (List.replicate 9 1.0 |>.toArray))
  let weightData := packF32FromF64 (FloatArray.mk (List.replicate 9 1.0 |>.toArray))

  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }
  let env := Interpreter.setBuffer env weight.uop { dtype := .float32, data := weightData }

  let resultRaw := Interpreter.evalTensor result env
  assertEqF32 resultRaw #[4.0, 6.0, 4.0, 6.0, 9.0, 6.0, 4.0, 6.0, 4.0] "conv2d 3x3 with padding"

/-- Test conv2d with stride=2.
    Input: 4x4 grid [1..16]
    Kernel: 2x2 with all ones (computes local sums)
    No padding, stride=2 -> 2x2 output
    Expected: [[1+2+5+6, 3+4+7+8], [9+10+13+14, 11+12+15+16]]
            = [[14, 22], [46, 54]] -/
def testConv2dStride2 : IO Unit := do
  let (base, weight, result) := runTensorM do
    let base ← Tensor.buffer [1, 1, 4, 4] .float32
    let weight ← Tensor.buffer [1, 1, 2, 2] .float32
    let result ← conv2d base weight none 0 2 1  -- padding=0, stride=2
    pure (base, weight, result)

  -- Input: 1 to 16
  let inputData := packF32FromF64 (FloatArray.mk #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
  -- Kernel: 2x2 all ones
  let weightData := packF32FromF64 (FloatArray.mk #[1.0, 1.0, 1.0, 1.0])

  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }
  let env := Interpreter.setBuffer env weight.uop { dtype := .float32, data := weightData }

  let resultRaw := Interpreter.evalTensor result env
  assertEqF32 resultRaw #[14.0, 22.0, 46.0, 54.0] "conv2d stride=2"

/-- Test multi-channel conv2d.
    Input: [1, 2, 2, 2] - batch=1, cin=2, h=2, w=2
    Weight: [1, 2, 2, 2] - cout=1, cin=2, kH=2, kW=2
    All input values are 1.0, all weights are 1.0
    Expected: sum of all 2*2*2*2 = 8 values = [8.0] -/
def testConv2dMultiChannel : IO Unit := do
  let (base, weight, result) := runTensorM do
    let base ← Tensor.buffer [1, 2, 2, 2] .float32
    let weight ← Tensor.buffer [1, 2, 2, 2] .float32
    let result ← conv2d base weight none 0 1 1
    pure (base, weight, result)

  -- All 8 input values = 1.0
  let inputData := packF32FromF64 (FloatArray.mk (List.replicate 8 1.0 |>.toArray))
  -- All 8 weight values = 1.0
  let weightData := packF32FromF64 (FloatArray.mk (List.replicate 8 1.0 |>.toArray))

  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }
  let env := Interpreter.setBuffer env weight.uop { dtype := .float32, data := weightData }

  let resultRaw := Interpreter.evalTensor result env
  -- 2x2 kernel on 2x2 input: output is 1x1
  -- Sum: 2 channels * 4 elements per channel = 8
  assertEqF32 resultRaw #[8.0] "conv2d multi-channel"

/-- Test conv2d with multiple output channels.
    Input: [1, 1, 2, 2] - batch=1, cin=1, h=2, w=2
    Weight: [2, 1, 2, 2] - cout=2, cin=1, kH=2, kW=2
    Input: [[1,2],[3,4]]
    Weight[0]: all 1s (sums to 10)
    Weight[1]: all 2s (sums to 20)
    Expected: [10.0, 20.0] -/
def testConv2dMultiOutputChannel : IO Unit := do
  let (base, weight, result) := runTensorM do
    let base ← Tensor.buffer [1, 1, 2, 2] .float32
    let weight ← Tensor.buffer [2, 1, 2, 2] .float32
    let result ← conv2d base weight none 0 1 1
    pure (base, weight, result)

  -- Input: [[1,2],[3,4]] = [1,2,3,4]
  let inputData := packF32FromF64 (FloatArray.mk #[1.0, 2.0, 3.0, 4.0])
  -- Weight: first filter all 1s, second filter all 2s
  let weightData := packF32FromF64 (FloatArray.mk #[1.0, 1.0, 1.0, 1.0,
                                                     2.0, 2.0, 2.0, 2.0])

  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }
  let env := Interpreter.setBuffer env weight.uop { dtype := .float32, data := weightData }

  let resultRaw := Interpreter.evalTensor result env
  -- Filter 0: 1*1 + 1*2 + 1*3 + 1*4 = 10
  -- Filter 1: 2*1 + 2*2 + 2*3 + 2*4 = 20
  assertEqF32 resultRaw #[10.0, 20.0] "conv2d multi-output-channel"

/-- Test batched conv2d.
    Input: [2, 1, 2, 2] - batch=2, cin=1, h=2, w=2
    Weight: [1, 1, 2, 2] - cout=1, cin=1, kH=2, kW=2
    Batch 0: [[1,1],[1,1]] -> sum = 4
    Batch 1: [[2,2],[2,2]] -> sum = 8
    Expected: [[4.0], [8.0]] -/
def testConv2dBatched : IO Unit := do
  let (base, weight, result) := runTensorM do
    let base ← Tensor.buffer [2, 1, 2, 2] .float32
    let weight ← Tensor.buffer [1, 1, 2, 2] .float32
    let result ← conv2d base weight none 0 1 1
    pure (base, weight, result)

  -- Input: batch 0 all 1s, batch 1 all 2s
  let inputData := packF32FromF64 (FloatArray.mk #[1.0, 1.0, 1.0, 1.0,
                                                    2.0, 2.0, 2.0, 2.0])
  -- Weight: all 1s
  let weightData := packF32FromF64 (FloatArray.mk #[1.0, 1.0, 1.0, 1.0])

  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }
  let env := Interpreter.setBuffer env weight.uop { dtype := .float32, data := weightData }

  let resultRaw := Interpreter.evalTensor result env
  -- Batch 0: 1+1+1+1 = 4
  -- Batch 1: 2+2+2+2 = 8
  assertEqF32 resultRaw #[4.0, 8.0] "conv2d batched"

/-- Test maxPool2d with 2x2 kernel on larger input.
    Input: [1, 1, 4, 4] with values 1..16
    Kernel: 2x2, stride 2
    Expected: 2x2 output with max of each 2x2 region -/
def testMaxPool2dLarger : IO Unit := do
  let (base, result) := runTensorM do
    let base ← Tensor.buffer [1, 1, 4, 4] .float32
    let result ← maxPool2d base 2 2 0
    pure (base, result)

  -- Input: 1 to 16
  let inputData := packF32FromF64 (FloatArray.mk #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }

  let resultRaw := Interpreter.evalTensor result env
  -- Max of [[1,2],[5,6]] = 6
  -- Max of [[3,4],[7,8]] = 8
  -- Max of [[9,10],[13,14]] = 14
  -- Max of [[11,12],[15,16]] = 16
  assertEqF32 resultRaw #[6.0, 8.0, 14.0, 16.0] "maxPool2d larger"

/-- Test avgPool2d with 2x2 kernel on larger input.
    Input: 4x4 grid [1..16], kernel 2x2, stride 2
    Expected output: 2x2 grid of local averages. -/
def testAvgPool2dLarger : IO Unit := do
  let (base, result) := runTensorM do
    let base ← Tensor.buffer [1, 1, 4, 4] .float32
    let result ← avgPool2d base 2 2 0
    pure (base, result)

  -- Input: 1 to 16
  let inputData := packF32FromF64 (FloatArray.mk #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }

  let resultRaw := Interpreter.evalTensor result env
  -- Avg of [[1,2],[5,6]] = (1+2+5+6)/4 = 3.5
  -- Avg of [[3,4],[7,8]] = (3+4+7+8)/4 = 5.5
  -- Avg of [[9,10],[13,14]] = (9+10+13+14)/4 = 11.5
  -- Avg of [[11,12],[15,16]] = (11+12+15+16)/4 = 13.5
  assertEqF32 resultRaw #[3.5, 5.5, 11.5, 13.5] "avgPool2d larger"

/-- Test conv2d with dilation.
    Input: 5x5 grid, kernel: 3x3 with dilation=2, no padding
    Dilated kernel covers 5x5 effective receptive field (3 + 2*(3-1) = 5).
    With 5x5 input and 5x5 effective kernel, output is 1x1. -/
def testConv2dDilation : IO Unit := do
  let (base, weight, result) := runTensorM do
    let base ← Tensor.buffer [1, 1, 5, 5] .float32
    let weight ← Tensor.buffer [1, 1, 3, 3] .float32
    -- padding=0, stride=1, dilation=2
    let result ← conv2d base weight none 0 1 2
    pure (base, weight, result)

  -- Input: 5x5 grid of ones
  let inputData := packF32FromF64 (FloatArray.mk (List.replicate 25 1.0 |>.toArray))
  -- Weight: 3x3 kernel of ones
  let weightData := packF32FromF64 (FloatArray.mk (List.replicate 9 1.0 |>.toArray))

  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }
  let env := Interpreter.setBuffer env weight.uop { dtype := .float32, data := weightData }

  assertShape result.uop.shape [1, 1, 1, 1] "conv2d dilation shape"
  let resultRaw := evalTensor result env
  -- With dilation=2, the 3x3 kernel samples at positions:
  -- (0,0), (0,2), (0,4), (2,0), (2,2), (2,4), (4,0), (4,2), (4,4)
  -- All ones, so sum = 9
  assertEqF32 resultRaw #[9.0] "conv2d dilation values"

/-- Test tile operation with values. -/
def testTileValues : IO Unit := do
  let (base, result) := runTensorM do
    let base ← Tensor.buffer [2] .float32
    let result ← tile base [3]
    pure (base, result)

  let inputData := packF32FromF64 (FloatArray.mk #[1.0, 2.0])
  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }

  let resultRaw := evalTensor result env
  -- [1, 2] tiled 3x should be [1, 2, 1, 2, 1, 2]
  assertEqF32 resultRaw #[1.0, 2.0, 1.0, 2.0, 1.0, 2.0] "tile values"

/-- Test depthwise convolution: each channel has its own filter.
    Input: [1, 2, 3, 3] (batch=1, 2 channels, 3x3 spatial)
    Weight: [2, 1, 2, 2] (2 channels, each with 1×2×2 kernel)
    No padding, stride 1 -> output [1, 2, 2, 2]

    Channel 0: sum all 4 elements in each 2x2 window (weight = all 1s)
    Channel 1: weighted sum with kernel [[1,2],[3,4]] -/
def testDepthwiseConv2d : IO Unit := do
  let (base, weight, result) := runTensorM do
    let base ← Tensor.buffer [1, 2, 3, 3] .float32
    let weight ← Tensor.buffer [2, 1, 2, 2] .float32
    let result ← depthwiseConv2d base weight none 0 1 1
    pure (base, weight, result)

  -- Input: 2 channels of 3x3
  -- Channel 0: [[1,2,3],[4,5,6],[7,8,9]]
  -- Channel 1: [[1,1,1],[1,1,1],[1,1,1]]
  let inputData := packF32FromF64 (FloatArray.mk #[
    -- Channel 0
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
    -- Channel 1
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0
  ])

  -- Weight: 2 filters of 2x2
  -- Filter 0: all ones (sum kernel)
  -- Filter 1: [[1,2],[3,4]]
  let weightData := packF32FromF64 (FloatArray.mk #[
    -- Filter 0: sum kernel
    1.0, 1.0,
    1.0, 1.0,
    -- Filter 1: weighted kernel
    1.0, 2.0,
    3.0, 4.0
  ])

  let env := Interpreter.setBuffer (∅ : Env) base.uop { dtype := .float32, data := inputData }
  let env := Interpreter.setBuffer env weight.uop { dtype := .float32, data := weightData }

  assertShape result.uop.shape [1, 2, 2, 2] "depthwise conv2d shape"

  let resultRaw := evalTensor result env

  -- Expected output:
  -- Channel 0 (sum of 2x2 windows on [[1,2,3],[4,5,6],[7,8,9]]):
  --   [[1+2+4+5, 2+3+5+6], [4+5+7+8, 5+6+8+9]] = [[12, 16], [24, 28]]
  -- Channel 1 (dot product with [[1,2],[3,4]] on all-ones):
  --   [[1+2+3+4, 1+2+3+4], [1+2+3+4, 1+2+3+4]] = [[10, 10], [10, 10]]
  assertEqF32 resultRaw #[12.0, 16.0, 24.0, 28.0, 10.0, 10.0, 10.0, 10.0] "depthwise conv2d values"

def runAll : IO Unit := do
  IO.println "=== Conv2dSmoke Tests ==="
  testConv2dShapeBasic
  IO.println "  conv2d shape basic"
  testConv2dShapeStride
  IO.println "  conv2d shape stride"
  testConv2dShapeNoPadding
  IO.println "  conv2d shape no padding"
  testConv1dShapeBasic
  IO.println "  conv1d shape basic"
  testConv1dShapeStride
  IO.println "  conv1d shape stride"
  testConv2dTensor
  IO.println "  conv2d tensor"
  testConv1dTensor
  IO.println "  conv1d tensor"
  testMaxPool2dShape
  IO.println "  maxPool2d shape"
  testMaxPool2dTensor
  IO.println "  maxPool2d tensor"
  testAvgPool2dTensor
  IO.println "  avgPool2d tensor"
  testConv2dParams
  IO.println "  Conv2dParams"
  testConv2dParamsNoBias
  IO.println "  Conv2dParams no bias"
  testMaxPool2dParams
  IO.println "  MaxPool2dParams"
  testMaxPool2dParamsDefaultStride
  IO.println "  MaxPool2dParams default stride"
  testUniformInit
  IO.println "  uniformInit"
  testTileShape
  IO.println "  tile shape"
  testTileTensor
  IO.println "  tile tensor"
  testPoolOutShape
  IO.println "  pool out shape"
  -- Value verification tests
  testMaxPool2dValues
  IO.println "  maxPool2d values"
  testAvgPool2dValues
  IO.println "  avgPool2d values"
  testConv1d1x1Kernel
  IO.println "  conv1d 1x1 kernel"
  testConv1d3KernelSum
  IO.println "  conv1d 3 kernel sum"
  testConv1d1x1WithBias
  IO.println "  conv1d 1x1 with bias"
  testPermute2D
  IO.println "  permute 2D"
  testPermuteNative
  IO.println "  permute native"
  testPermute3D
  IO.println "  permute 3D"
  testPermute4D
  IO.println "  permute 4D"
  testPermute6D
  IO.println "  permute 6D"
  testConv2d1x1Kernel
  IO.println "  conv2d 1x1 kernel"
  testConv2d3x3SumKernel
  IO.println "  conv2d 3x3 sum kernel"
  testConv2d1x1WithBias
  IO.println "  conv2d 1x1 with bias"
  testPad2dIsolated
  IO.println "  pad2d isolated"
  testConv2d3x3WithPadding
  IO.println "  conv2d 3x3 with padding"
  testConv2dStride2
  IO.println "  conv2d stride=2"
  testConv2dMultiChannel
  IO.println "  conv2d multi-channel"
  testConv2dMultiOutputChannel
  IO.println "  conv2d multi-output-channel"
  testConv2dBatched
  IO.println "  conv2d batched"
  testMaxPool2dLarger
  IO.println "  maxPool2d larger"
  testAvgPool2dLarger
  IO.println "  avgPool2d larger"
  testConv2dDilation
  IO.println "  conv2d dilation"
  testTileValues
  IO.println "  tile values"
  testDepthwiseConv2d
  IO.println "  depthwise conv2d"
  IO.println "=== Conv2dSmoke OK ==="

end TinyGrad4.Test.Conv2dSmoke

#eval! TinyGrad4.Test.Conv2dSmoke.runAll
