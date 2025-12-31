import TinyGrad4.Backend.Cuda

/-!
# CUDA Compile Test

Minimal test to verify kernel compilation works.
-/

open TinyGrad4.Backend.Cuda

def testKernelSource : String :=
"extern \"C\" __global__ void test_kernel(float* out) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid == 0) {
        out[0] = 42.0f;
    }
}"

def main : IO UInt32 := do
  IO.println "CUDA Compile Test"
  IO.println ""

  IO.println "Step 1: Get device name..."
  let device ← cudaDeviceName
  IO.println s!"  Device: {device}"

  IO.println "Step 2: Allocate buffer..."
  let buf ← cudaAllocBytes 64
  IO.println "  Buffer allocated"

  IO.println "Step 3: Compile kernel..."
  IO.println "  Source:"
  IO.println testKernelSource
  let prog ← cudaCompile "test_kernel" testKernelSource
  IO.println "  Kernel compiled!"

  IO.println "Step 4: Launch kernel..."
  cudaLaunch2D prog #[buf] 1 1 1 1
  cudaSync
  IO.println "  Kernel launched and synced!"

  IO.println "Step 5: Read result..."
  let result ← cudaCopyOutBytes buf 64
  IO.println s!"  Result size: {result.size} bytes"

  IO.println ""
  IO.println "✓ All steps completed!"
  return 0
