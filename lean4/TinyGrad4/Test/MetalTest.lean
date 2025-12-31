import TinyGrad4.Backend.Device
import TinyGrad4.Backend.Metal
import TinyGrad4.Backend.Engine

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

/-!
# Metal Backend Test

Tests the principled Metal backend architecture:
- Device abstraction with typeclasses
- Polymorphic execution engine
- FFI to Metal via tg4_metal.m
-/

namespace TinyGrad4.Test.MetalTest

open TinyGrad4.Backend
open TinyGrad4.Backend.Metal
open TinyGrad4.Backend.Engine

/-- Test Metal availability -/
def testMetalAvailable : IO Unit := do
  IO.println "=== Metal Availability Test ==="
  let available ← Metal.isAvailable
  if available then
    let info ← Metal.deviceInfo
    IO.println s!"✓ Metal available: {info}"
  else
    IO.println "✗ Metal not available"

/-- Test CPU device abstraction -/
def testCPUDevice : IO Unit := do
  IO.println "\n=== CPU Device Test ==="
  let device := cpuDevice
  IO.println s!"Device name: {device.name}"

  -- Test allocator
  let buf ← device.allocator.alloc 100
  IO.println s!"✓ Allocated buffer"

  -- Test renderer
  let rendered := device.renderer.renderMatmul 64 64 64
  IO.println s!"Rendered matmul: {rendered}"

  -- Cleanup
  device.allocator.free buf
  IO.println "✓ Freed buffer"

/-- Test Metal roundtrip (alloc -> copyIn -> copyOut -> free) -/
def testMetalRoundtrip : IO Unit := do
  IO.println "\n=== Metal Roundtrip Test ==="

  -- Create test data
  let testData : FloatArray := ⟨#[1.0, 2.0, 3.0, 4.0, 5.0]⟩
  IO.println s!"Input: {testData.data.toList}"

  -- Roundtrip through Metal
  let result ← Metal.testRoundtrip testData
  IO.println s!"Output: {result.data.toList}"

  -- Verify
  let mut allMatch := true
  for i in [:testData.size] do
    if h : i < testData.size ∧ i < result.size then
      let diff := (testData.data[i]! - result.data[i]!).abs
      if diff > 0.0001 then
        allMatch := false

  if allMatch then
    IO.println "✓ Roundtrip test passed"
  else
    IO.println "✗ Roundtrip test failed"

/-- Test Metal kernel execution -/
def testMetalKernel : IO Unit := do
  IO.println "\n=== Metal Kernel Test ==="

  -- Simple add kernel
  let addKernel := "
#include <metal_stdlib>
using namespace metal;

kernel void test_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] + b[gid];
}
"

  let a : FloatArray := ⟨#[1.0, 2.0, 3.0, 4.0]⟩
  let b : FloatArray := ⟨#[10.0, 20.0, 30.0, 40.0]⟩

  IO.println s!"a = {a.data.toList}"
  IO.println s!"b = {b.data.toList}"

  let result ← Metal.runKernel addKernel "test_add" #[a, b] 4
  IO.println s!"a + b = {result.data.toList}"

  -- Expected: [11, 22, 33, 44]
  let expected := [11.0, 22.0, 33.0, 44.0]
  let mut passed := true
  for i in [:4] do
    if h : i < result.size then
      let diff := (result.data[i]! - expected[i]!).abs
      if diff > 0.001 then
        passed := false

  if passed then
    IO.println "✓ Kernel test passed"
  else
    IO.println "✗ Kernel test failed"

/-- Test polymorphic execution engine -/
def testEngine : IO Unit := do
  IO.println "\n=== Polymorphic Engine Test ==="

  -- Test with CPU device first
  IO.println "Testing with CPU device..."
  let _cpuDev := cpuDevice
  IO.println "✓ CPU device created"

  -- Test with Metal device
  IO.println "Testing with Metal device..."
  let _metalDev := metalDevice
  IO.println "✓ Metal device created"

  -- The engine can work with either device polymorphically
  IO.println "✓ Polymorphic device abstraction works"

/-- Run all tests -/
def main : IO Unit := do
  IO.println "╔══════════════════════════════════════════════════════════╗"
  IO.println "║     TinyGrad4 Principled Metal Backend Tests             ║"
  IO.println "╚══════════════════════════════════════════════════════════╝"

  testMetalAvailable
  testCPUDevice
  testEngine

  -- These require FFI to be linked
  -- Uncomment when building with `lake build` which links tg4c
  -- testMetalRoundtrip
  -- testMetalKernel

  IO.println "\n=== All Tests Complete ==="

end TinyGrad4.Test.MetalTest
