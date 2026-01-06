import TinyGrad4.Data.Buffer
import TinyGrad4.Data.IndexTransform

/-!
# Buffer Protocol Tests

Tests for Device, Buffer, and IndexTransform modules.
-/

open TinyGrad4.Data

/-! ## Device Tests -/

#guard Device.cpu.isHost == true
#guard Device.gpu.isHost == false
#guard (Device.gpu 0).isGPU == true
#guard Device.cpu.isGPU == false
#guard (Device.remote "s3://bucket").isRemote == true
#guard Device.cpu.name == "cpu"
#guard (Device.gpu 2).name == "gpu:2"

/-! ## RawBuffer Tests -/

-- Test computeCStrides
#guard RawBuffer.computeCStrides #[] == #[]
#guard RawBuffer.computeCStrides #[5] == #[1]
#guard RawBuffer.computeCStrides #[2, 3] == #[3, 1]
#guard RawBuffer.computeCStrides #[2, 3, 4] == #[12, 4, 1]
#guard RawBuffer.computeCStrides #[1, 1, 1] == #[1, 1, 1]
#guard RawBuffer.computeCStrides #[10, 20, 30] == #[600, 30, 1]

-- Test numel
def testRaw1 : RawBuffer := {
  handle := { ptr := 0, device := .cpu }
  dtype := .float32
  shape := #[2, 3, 4]
}
#guard testRaw1.numel == 24
#guard testRaw1.ndim == 3
#guard testRaw1.bytes == 96  -- 24 * 4

-- Test isContiguous
def testRawContiguous : RawBuffer := {
  handle := { ptr := 0, device := .cpu }
  dtype := .float32
  shape := #[2, 3, 4]
  strides := some #[12, 4, 1]
}
#guard testRawContiguous.isContiguous == true

def testRawNonContiguous : RawBuffer := {
  handle := { ptr := 0, device := .cpu }
  dtype := .float32
  shape := #[2, 3, 4]
  strides := some #[24, 8, 2]  -- strided access
}
#guard testRawNonContiguous.isContiguous == false

/-! ## Buffer Tests -/

#guard Buffer.computeCStrides [2, 3, 4] == [12, 4, 1]

def testAlloc : Allocation := {
  handle := { ptr := 0, device := .cpu }
  byteSize := 96
}

def testBuf : Buffer [2, 3, 4] .float32 := {
  allocation := testAlloc
}

#guard testBuf.numel == 24
#guard testBuf.bytes == 96
#guard testBuf.isContiguous == true

def testBufNonContiguous : Buffer [2, 3, 4] .float32 := {
  allocation := testAlloc
  strides := some [24, 8, 2]
}

#guard testBufNonContiguous.isContiguous == false

/-! ## IndexTransform Tests -/

-- Identity
#guard (IndexTransform.id 10).inputLen == 10
#guard (IndexTransform.id 10).outputLen == 10

-- Take
#guard (IndexTransform.take 5 10).outputLen == 5

-- Drop
#guard (IndexTransform.drop 3 10).outputLen == 7

-- Reverse - test structure only, map tested via IO
#guard (IndexTransform.reverse 5).inputLen == 5
#guard (IndexTransform.reverse 5).outputLen == 5

-- Shuffle generates valid permutation
def shuffleT := IndexTransform.shuffle 42 10
#guard shuffleT.inputLen == 10
#guard shuffleT.outputLen == 10

-- Fisher-Yates produces correct size
#guard (IndexTransform.fisherYatesArray 42 0).size == 0
#guard (IndexTransform.fisherYatesArray 42 5).size == 5
#guard (IndexTransform.fisherYatesArray 42 100).size == 100

-- Verify permutation validity
#guard IndexTransform.verifyPermutation (IndexTransform.fisherYatesArray 42 10) 10 == true
#guard IndexTransform.verifyPermutation (IndexTransform.fisherYatesArray 123 50) 50 == true

-- Invalid permutations
#guard IndexTransform.verifyPermutation #[0, 1, 1] 3 == false  -- duplicate
#guard IndexTransform.verifyPermutation #[0, 1, 5] 3 == false  -- out of bounds
#guard IndexTransform.verifyPermutation #[0, 1] 3 == false     -- wrong size

-- Shard size computation
#guard IndexTransform.shardSize 0 4 12 == 3  -- rank 0 of 4, 12 elements: indices 0,4,8
#guard IndexTransform.shardSize 1 4 12 == 3  -- rank 1: indices 1,5,9
#guard IndexTransform.shardSize 0 4 13 == 4  -- rank 0: indices 0,4,8,12
#guard IndexTransform.shardSize 3 4 13 == 3  -- rank 3: indices 3,7,11

/-! ## BufferRegistry Tests -/

def testRegistry : IO Unit := do
  let reg ← BufferRegistry.new

  -- Initial state
  assert! (← reg.activeCount) == 0

  -- Register a buffer
  let id1 ← reg.register .cpu 1024
  assert! (← reg.activeCount) == 1
  assert! (← reg.getRefCount id1) == 1

  -- Acquire increases refcount
  reg.acquire id1
  assert! (← reg.getRefCount id1) == 2

  -- Release decreases refcount
  let freed ← reg.release id1
  assert! freed == false  -- still has refs
  assert! (← reg.getRefCount id1) == 1

  -- Final release frees
  let freed2 ← reg.release id1
  assert! freed2 == true
  assert! (← reg.activeCount) == 0

  IO.println "✓ BufferRegistry tests passed"

/-! ## TrackedBuffer Tests -/

def testTrackedBuffer : IO Unit := do
  let reg ← BufferRegistry.new

  -- Allocate
  let buf ← TrackedBuffer.alloc reg [256] .float32 .cpu
  assert! (← reg.activeCount) == 1
  assert! buf.device == .cpu

  -- withBorrowed pattern
  withBorrowed buf fun view => do
    assert! view.bytes == 256 * 4
    pure ()

  -- Release
  buf.release
  assert! (← reg.activeCount) == 0

  IO.println "✓ TrackedBuffer tests passed"

/-! ## Leak Detection Test -/

def testLeakDetection : IO Unit := do
  let reg ← BufferRegistry.new

  -- Allocate some buffers
  let _b1 ← TrackedBuffer.alloc reg [100] .float32 .cpu
  let b2 ← TrackedBuffer.alloc reg [200] .float32 .cpu
  let _b3 ← TrackedBuffer.alloc reg [300] .float32 .cpu

  -- Only release one
  b2.release

  -- Check for leaks
  let leaks ← reg.checkLeaks
  assert! leaks.size == 2  -- b1 and b3 are still active

  IO.println "✓ Leak detection tests passed"

/-! ## Run All Tests -/

def main : IO Unit := do
  IO.println "Running buffer protocol tests..."
  testRegistry
  testTrackedBuffer
  testLeakDetection
  IO.println "All tests passed!"
