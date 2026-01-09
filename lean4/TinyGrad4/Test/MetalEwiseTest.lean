import TinyGrad4
import TinyGrad4.Backend.MetalEwise
import TinyGrad4.Backend.DeviceBuffer
import TinyGrad4.Backend.MetalMatmul

/-!
# Metal Elementwise GPU Test

Tests the Metal GPU dispatch for elementwise operations.
-/

-- Disable RawBuffer linter for test files
set_option linter.useRawBuffer false
-- Disable IO.monoNanosNow linter: benchmark timing uses raw monotonic clocks.
set_option linter.monoNanosNow false

namespace TinyGrad4.Test.MetalEwiseTest

open TinyGrad4
open TinyGrad4.Backend
open TinyGrad4.Backend.MetalEwise

/-- Pack float64 array to float32 bytes -/
private def packF32 (data : Array Float) : ByteArray :=
  Native.packF32FromF64 ⟨data⟩

/-- Test direct GPU kernel dispatch -/
def testDirectGPU : IO Unit := do
  IO.println "Testing direct Metal GPU elementwise dispatch..."

  -- Check Metal availability
  let available ← Metal.isAvailable
  IO.println s!"  Metal available: {available}"

  if !available then
    IO.println "  Skipping GPU test (Metal not available)"
    return

  -- Create test input: 4 floats [1.0, 2.0, 3.0, 4.0]
  let input : RawBuffer := { dtype := .float32, data := packF32 #[1.0, 2.0, 3.0, 4.0] }
  IO.println s!"  Input: {input.data.size / 4} floats"

  -- Create a simple add kernel: out[i] = in0[i] + in0[i] = 2 * in0[i]
  let kernelName := "test_double"
  let shader := "#include <metal_stdlib>
using namespace metal;

kernel void test_double(
  device const float* in0 [[buffer(0)]],
  device float* out [[buffer(1)]],
  uint gid [[thread_position_in_grid]]
) {
  if (gid >= 4) return;
  out[gid] = in0[gid] * 2.0f;
}"

  -- Run kernel
  let result ← runEwiseKernel kernelName shader #[input] 4

  -- Check result
  let decoded := result.decode
  IO.println s!"  Output: {decoded.data}"

  -- Verify: should be [2.0, 4.0, 6.0, 8.0]
  let expected := #[2.0, 4.0, 6.0, 8.0]
  let mut maxDiff : Float := 0.0
  for i in [:decoded.data.size] do
    let diff := Float.abs (decoded.data[i]! - expected[i]!)
    if diff > maxDiff then
      maxDiff := diff

  if maxDiff > 0.001 then
    throw (IO.userError s!"Results differ by {maxDiff} > 0.001")

  IO.println s!"  ✓ GPU kernel produced correct result (max diff: {maxDiff})"

/-- Test FusedEwise GPU dispatch via the interpreter path -/
def testFusedEwiseGPU : IO Unit := do
  IO.println "Testing FusedEwise GPU dispatch..."

  let available ← Metal.isAvailable
  if !available then
    IO.println "  Skipping (Metal not available)"
    return

  -- Create a bytecode plan for: x + x = 2*x
  let prog : Array UInt64 := #[
    FusedEwise.instLoad 0,   -- push in0[gid]
    FusedEwise.instLoad 0,   -- push in0[gid]
    FusedEwise.instBinary 0  -- add
  ]

  let plan : FusedEwise.Plan := {
    root := ⟨0⟩
    cover := ∅
    leafBases := #[⟨1⟩]
    leafShapes := #[#[4]]
    leafDtypes := #[0]  -- float32
    leafStrides := #[#[1]]
    leafOffsets := #[0]
    leafMaskStarts := #[#[]]
    leafMaskEnds := #[#[]]
    leafStackShapes := #[]
    leafStackStrides := #[]
    leafStackOffsets := #[]
    leafStackMaskStarts := #[]
    leafStackMaskEnds := #[]
    prog := prog
    kernel := .bytecode
    fast := true
    needsStack := false
  }

  let input : RawBuffer := { dtype := .float32, data := packF32 #[1.0, 2.0, 3.0, 4.0] }

  let result ← runFusedEwiseWithFallback plan #[input] [4]

  let decoded := result.decode
  IO.println s!"  Output: {decoded.data}"

  let expected := #[2.0, 4.0, 6.0, 8.0]
  let mut maxDiff : Float := 0.0
  for i in [:decoded.data.size] do
    let diff := Float.abs (decoded.data[i]! - expected[i]!)
    if diff > maxDiff then
      maxDiff := diff

  if maxDiff > 0.001 then
    throw (IO.userError s!"FusedEwise results differ by {maxDiff} > 0.001")

  IO.println s!"  ✓ FusedEwise GPU produced correct result"

/-- Test GPU dispatch for large tensors (should hit GPU path) -/
def testLargeTensor : IO Unit := do
  IO.println "Testing large tensor GPU dispatch..."

  let available ← Metal.isAvailable
  if !available then
    IO.println "  Skipping (Metal not available)"
    return

  -- Create a large tensor (100K elements)
  let numel := 100000
  let mut inputData : Array Float := #[]
  for i in [:numel] do
    inputData := inputData.push (Float.ofNat (i % 100))

  let input : RawBuffer := { dtype := .float32, data := packF32 inputData }

  -- Create add kernel: out[i] = in0[i] + in0[i]
  let kernelName := "test_large_double"
  let shader := s!"#include <metal_stdlib>
using namespace metal;

kernel void {kernelName}(
  device const float* in0 [[buffer(0)]],
  device float* out [[buffer(1)]],
  uint gid [[thread_position_in_grid]]
) \{
  if (gid >= {numel}) return;
  out[gid] = in0[gid] * 2.0f;
}"

  -- Time GPU kernel
  let startTime ← IO.monoNanosNow
  let result ← runEwiseKernel kernelName shader #[input] numel
  let endTime ← IO.monoNanosNow
  let gpuTimeMs := Float.ofNat (endTime - startTime) / 1000000.0

  -- Verify result
  let decoded := result.decode
  let mut maxDiff : Float := 0.0
  for i in [:min 100 decoded.data.size] do
    let expected := Float.ofNat ((i % 100) * 2)
    let diff := Float.abs (decoded.data[i]! - expected)
    if diff > maxDiff then
      maxDiff := diff

  IO.println s!"  GPU kernel time: {gpuTimeMs} ms for {numel} elements"
  IO.println s!"  Max diff (first 100): {maxDiff}"

  if maxDiff > 0.001 then
    throw (IO.userError s!"Large tensor results differ by {maxDiff}")

  IO.println s!"  ✓ Large tensor GPU dispatch works"

/-- Test persistent GPU buffers - chained operations without CPU copies -/
def testPersistentGPU : IO Unit := do
  IO.println "Testing persistent GPU buffers (no intermediate CPU copies)..."

  let available ← Metal.isAvailable
  if !available then
    IO.println "  Skipping (Metal not available)"
    return

  -- Test DeviceBuffer.fromCPU and ensureGPU
  let numel := 10000
  let mut inputData : Array Float := #[]
  for i in [:numel] do
    inputData := inputData.push (Float.ofNat (i % 10))

  let cpuBuf : RawBuffer := { dtype := .float32, data := packF32 inputData }

  -- Create DeviceBuffer from CPU data
  let dbuf := DeviceBuffer.DeviceBuffer.fromCPU cpuBuf

  -- Upload to GPU (only happens once)
  let startUpload ← IO.monoNanosNow
  let (dbufOnGPU, _) ← DeviceBuffer.DeviceBuffer.toGPU dbuf
  let endUpload ← IO.monoNanosNow
  let uploadMs := Float.ofNat (endUpload - startUpload) / 1000000.0

  IO.println s!"  Upload time: {uploadMs} ms"
  IO.println s!"  Buffer on GPU: {DeviceBuffer.DeviceBuffer.isOnGPU dbufOnGPU}"

  -- Chain 3 operations without copying back to CPU
  -- Op 1: x * 2
  -- Op 2: result + 1
  -- Op 3: result * 0.5
  -- Final: ((x * 2) + 1) * 0.5 = x + 0.5

  let startChain ← IO.monoNanosNow

  -- Generate shader helper
  let mkShader (name : String) (op : String) : String :=
    "#include <metal_stdlib>\n" ++
    "using namespace metal;\n" ++
    s!"kernel void {name}(\n" ++
    "  device const float* in0 [[buffer(0)]],\n" ++
    "  device float* out [[buffer(1)]],\n" ++
    "  uint gid [[thread_position_in_grid]]\n" ++
    ") {\n" ++
    s!"  if (gid >= {numel}) return;\n" ++
    s!"  out[gid] = {op};\n" ++
    "}"

  -- Op 1: Multiply by 2
  let shader1 := mkShader "chain_mul2" "in0[gid] * 2.0f"
  let result1 ← runEwiseKernelDevice "chain_mul2" shader1 #[dbufOnGPU] numel

  -- Op 2: Add 1 (using result1 which stays on GPU)
  let shader2 := mkShader "chain_add1" "in0[gid] + 1.0f"
  let result2 ← runEwiseKernelDevice "chain_add1" shader2 #[result1] numel

  -- Op 3: Multiply by 0.5 (using result2 which stays on GPU)
  let shader3 := mkShader "chain_mul05" "in0[gid] * 0.5f"
  let result3 ← runEwiseKernelDevice "chain_mul05" shader3 #[result2] numel

  let endChain ← IO.monoNanosNow
  let chainMs := Float.ofNat (endChain - startChain) / 1000000.0

  IO.println s!"  Chain of 3 ops (no CPU copies): {chainMs} ms"

  -- Now download final result to CPU (only copy)
  let startDownload ← IO.monoNanosNow
  let finalCPU ← DeviceBuffer.DeviceBuffer.toCPU result3
  let endDownload ← IO.monoNanosNow
  let downloadMs := Float.ofNat (endDownload - startDownload) / 1000000.0

  IO.println s!"  Download time: {downloadMs} ms"

  -- Verify result: ((x * 2) + 1) * 0.5 = x + 0.5
  let decoded := finalCPU.decode
  let mut maxDiff : Float := 0.0
  for i in [:min 100 decoded.data.size] do
    let x := Float.ofNat (i % 10)
    let expected := x + 0.5
    let diff := Float.abs (decoded.data[i]! - expected)
    if diff > maxDiff then
      maxDiff := diff

  IO.println s!"  Max diff (first 100): {maxDiff}"

  if maxDiff > 0.001 then
    throw (IO.userError s!"Chained GPU results differ by {maxDiff}")

  -- Clean up GPU buffers
  DeviceBuffer.DeviceBuffer.release dbufOnGPU
  DeviceBuffer.DeviceBuffer.release result1
  DeviceBuffer.DeviceBuffer.release result2
  DeviceBuffer.DeviceBuffer.release result3

  IO.println s!"  ✓ Persistent GPU buffers work (3 chained ops, 1 upload, 1 download)"

/-- Test GPU-resident matmul using DeviceBuffer API -/
def testMatmulDevice : IO Unit := do
  IO.println "Testing GPU-resident matmul (DeviceBuffer API)..."

  let available ← Metal.isAvailable
  if !available then
    IO.println "  Skipping (Metal not available)"
    return

  -- Create test matrices: A[2,3] @ B[3,2] = C[2,2]
  -- A = [[1,2,3],[4,5,6]]
  -- B = [[1,2],[3,4],[5,6]]
  -- C = [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
  --   = [[22, 28], [49, 64]]

  let aData : Array Float := #[1,2,3, 4,5,6]
  let bData : Array Float := #[1,2, 3,4, 5,6]

  let aBuf : RawBuffer := { dtype := .float32, data := packF32 aData }
  let bBuf : RawBuffer := { dtype := .float32, data := packF32 bData }

  -- Upload to GPU
  let aDevice ← DeviceBuffer.DeviceBuffer.uploadToGPU aBuf
  let bDevice ← DeviceBuffer.DeviceBuffer.uploadToGPU bBuf

  IO.println s!"  A on GPU: {DeviceBuffer.DeviceBuffer.isOnGPU aDevice}"
  IO.println s!"  B on GPU: {DeviceBuffer.DeviceBuffer.isOnGPU bDevice}"

  -- Run matmul on GPU
  let startTime ← IO.monoNanosNow
  let cDevice ← MetalMatmul.matmul2DDevice aDevice bDevice 2 3 2
  let endTime ← IO.monoNanosNow
  let matmulMs := Float.ofNat (endTime - startTime) / 1000000.0

  IO.println s!"  Matmul time: {matmulMs} ms"
  IO.println s!"  Result on GPU: {DeviceBuffer.DeviceBuffer.isOnGPU cDevice}"

  -- Download result
  let cCPU ← DeviceBuffer.DeviceBuffer.toCPU cDevice
  let decoded := cCPU.decode

  IO.println s!"  Result: {decoded.data}"

  -- Verify: [[22, 28], [49, 64]]
  let expected := #[22.0, 28.0, 49.0, 64.0]
  let mut maxDiff : Float := 0.0
  for i in [:decoded.data.size] do
    let diff := Float.abs (decoded.data[i]! - expected[i]!)
    if diff > maxDiff then
      maxDiff := diff

  if maxDiff > 0.001 then
    throw (IO.userError s!"Matmul results differ by {maxDiff} > 0.001")

  -- Clean up
  DeviceBuffer.DeviceBuffer.release aDevice
  DeviceBuffer.DeviceBuffer.release bDevice
  DeviceBuffer.DeviceBuffer.release cDevice

  IO.println s!"  ✓ GPU-resident matmul works (max diff: {maxDiff})"

def runAll : IO Unit := do
  IO.println "=== Metal Ewise GPU Tests ==="
  testDirectGPU
  testFusedEwiseGPU
  testLargeTensor
  testPersistentGPU
  testMatmulDevice
  IO.println "=== Metal GPU Tests OK ==="

end TinyGrad4.Test.MetalEwiseTest

def main : IO Unit := TinyGrad4.Test.MetalEwiseTest.runAll
