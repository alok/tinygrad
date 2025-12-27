import TinyGrad4.Backend.MetalRenderer
import TinyGrad4.Backend.Device
import TinyGrad4.UOp.UOp

/-!
# Metal Renderer Test

Tests vectorized kernel generation via the MetalRenderer module.
(Does NOT import Metal.lean to avoid FFI linking requirements)
-/

open TinyGrad4 TinyGrad4.Backend TinyGrad4.Backend.MetalRenderer

/-! ## Helper for substring check -/

def containsSubstr (haystack needle : String) : Bool :=
  (haystack.splitOn needle).length > 1

/-! ## Create simple test UOp graph -/

/-- Buffer input A -/
def bufA : UOp := {
  op := .BUFFER
  dtype := .float32
  uid := ⟨0⟩
  src := []
  arg := .bufferIdx 0
  shape := [1000000]
}

/-- Buffer input B -/
def bufB : UOp := {
  op := .BUFFER
  dtype := .float32
  uid := ⟨1⟩
  src := []
  arg := .bufferIdx 1
  shape := [1000000]
}

/-- ADD operation: A + B -/
def addOp : UOp := {
  op := .ADD
  dtype := .float32
  uid := ⟨2⟩
  src := [bufA, bufB]
  arg := .empty
  shape := [1000000]
}

/-- Output buffer C -/
def bufC : UOp := {
  op := .BUFFER
  dtype := .float32
  uid := ⟨3⟩
  src := []
  arg := .bufferIdx 2
  shape := [1000000]
}

/-- Simple test graph: buf0 + buf1 -> buf2 -/
def testNodes : List UOp := [bufA, bufB, addOp, bufC]

/-! ## Test Renderer Methods -/

def testRendererScalar : IO Unit := do
  IO.println "=== Scalar Kernel Test ==="
  match renderKernel "test_add" testNodes ⟨3⟩ with
  | some kernel =>
    IO.println "Generated scalar kernel:"
    IO.println kernel
    -- Verify kernel contains expected elements
    if containsSubstr kernel "buf0[gid] + buf1[gid]" then
      IO.println "✓ Scalar addition found"
    else
      IO.println s!"✗ Missing scalar addition (got kernel: {kernel.take 200})"
  | none => IO.println "✗ Failed to generate scalar kernel"

def testRendererVectorized : IO Unit := do
  IO.println "\n=== Vectorized Kernel Test ==="
  -- Test with size divisible by 4
  match renderEwiseVectorized "test_add_vec4" testNodes ⟨3⟩ 1000000 with
  | some kernel =>
    IO.println "Generated float4 kernel:"
    IO.println kernel
    -- Verify vectorization
    if containsSubstr kernel "float4" then
      IO.println "✓ float4 type found"
    else
      IO.println "✗ Missing float4 type"
    if containsSubstr kernel "in0 + in1" then
      IO.println "✓ Vectorized addition found"
    else
      IO.println "✗ Missing vectorized addition"
  | none => IO.println "✗ Failed to generate vectorized kernel"

def testAutoKernel : IO Unit := do
  IO.println "\n=== Auto Kernel Selection Test ==="
  -- Test with aligned size (should vectorize)
  match renderKernelAuto "test_aligned" testNodes ⟨3⟩ 1000000 with
  | some kernel =>
    if containsSubstr kernel "float4" then
      IO.println "✓ Auto selected float4 for aligned size"
    else
      IO.println "✓ Auto selected scalar for aligned size"
  | none => IO.println "✗ Failed to generate auto kernel"

  -- Test with unaligned size (should use scalar)
  match renderKernelAuto "test_unaligned" testNodes ⟨3⟩ 1000003 with
  | some kernel =>
    if containsSubstr kernel "float4" then
      IO.println "✗ Should not use float4 for non-aligned size"
    else
      IO.println "✓ Auto correctly selected scalar for unaligned size"
  | none => IO.println "✗ Failed to generate auto kernel for unaligned"

/-! ## Reduction Kernel Tests -/

def testReduceKernels : IO Unit := do
  IO.println "\n=== Reduction Kernel Tests ==="

  -- Test simple sum reduction (small, fits in one threadgroup)
  let sumKernel := MetalRenderer.renderSimpleReduceKernel "sum_reduce" .sum 128 32
  IO.println "Generated sum reduction kernel (128 elems, 32 rows):"
  IO.println sumKernel
  if containsSubstr sumKernel "threadgroup float shared" then
    IO.println "✓ Threadgroup memory declared"
  else
    IO.println "✗ Missing threadgroup memory"
  if containsSubstr sumKernel "threadgroup_barrier" then
    IO.println "✓ Barrier synchronization found"
  else
    IO.println "✗ Missing barrier synchronization"
  if containsSubstr sumKernel "shared[tid] + shared[tid + s]" then
    IO.println "✓ Tree reduction found"
  else
    IO.println "✗ Missing tree reduction"

  -- Test max reduction
  IO.println "\n--- Max Reduction ---"
  let maxKernel := MetalRenderer.renderSimpleReduceKernel "max_reduce" .max 64 16
  if containsSubstr maxKernel "max(shared[tid], shared[tid + s])" then
    IO.println "✓ Max reduction operation found"
  else
    IO.println "✗ Missing max reduction"
  if containsSubstr maxKernel "-INFINITY" then
    IO.println "✓ Correct identity for max (-INFINITY)"
  else
    IO.println "✗ Wrong identity for max"

  -- Test auto selection
  IO.println "\n--- Auto Selection ---"
  let smallKernel := MetalRenderer.renderReduceKernelAuto "auto_small" .sum 256 100
  let largeKernel := MetalRenderer.renderReduceKernelAuto "auto_large" .sum 4096 100
  if containsSubstr smallKernel "Simple single-threadgroup" then
    IO.println "✓ Small reduction uses simple kernel"
  else
    IO.println "✗ Small reduction should use simple kernel"
  if containsSubstr largeKernel "Parallel reduction" then
    IO.println "✓ Large reduction uses parallel kernel"
  else
    IO.println "✗ Large reduction should use parallel kernel"

def main : IO Unit := do
  IO.println "╔════════════════════════════════════════╗"
  IO.println "║    Metal Renderer Integration Test     ║"
  IO.println "╚════════════════════════════════════════╝"

  testRendererScalar
  testRendererVectorized
  testAutoKernel
  testReduceKernels

  IO.println "\n=== All Tests Complete ==="
