# GPU Backend Comparison: Lean TinyGrad4 vs Python tinygrad

This document tracks performance differences between Lean TinyGrad4 and Python tinygrad GPU backends, providing a baseline for future optimization work.

## Current Status

| Backend | Lean TinyGrad4 | Python tinygrad |
|---------|----------------|-----------------|
| Metal   | ✅ Implemented  | ✅ Production    |
| CUDA    | ✅ Implemented  | ✅ Production    |
| OpenCL  | ❌ Not started  | ✅ Available     |

## Metal Results (Local Mac)

### Vector Add Benchmark (1M elements) - Scalar vs Float4

**Test Setup:**
- Device: Apple M4 Max
- Size: 1,000,000 float32 elements (4 MB)
- Operation: `out[i] = a[i] + b[i]`
- Iterations: 100 (warmup 3)
- Date: December 27, 2024

**Vectorization Comparison:**

| Kernel | Time | Bandwidth | Speedup |
|--------|------|-----------|---------|
| Scalar | 171.74 μs | 69.9 GB/s | baseline |
| **Float4** | **30.59 μs** | **392.3 GB/s** | **5.6x** |

**Key Finding**: Float4 vectorization provides **5.6x speedup** on M4 Max, achieving 392 GB/s bandwidth.

### Previous Results (M3 Pro - Scalar Only)

| Metric | Lean TinyGrad4 | Python tinygrad | Gap |
|--------|----------------|-----------------|-----|
| Min Time | 69.89 μs     | 91.51 μs        | 1.31x faster |
| Avg Time | 105.87 μs    | 119.14 μs       | 1.13x faster |
| Best BW  | 172 GB/s     | 131 GB/s        | 1.31x higher |

**Analysis:**
- Float4 vectorization is critical for memory bandwidth
- M4 Max achieves 392 GB/s with float4 (near theoretical max)
- Python has compilation overhead on first call (~20ms)

### Architectural Differences

| Feature | Lean | Python |
|---------|------|--------|
| Dispatch model | `thread_position_in_grid` | `threadgroup_position_in_grid + lid` |
| Vectorization | Scalar (float) | float4 when possible |
| Compilation | Per-launch (no cache) | Module caching |
| Thread sizing | Clamped to 256 | Queries pipeline.maxThreadsPerThreadgroup |
| Shared memory | Not used | 32KB threadgroup memory available |

## CUDA Results (RunPod - NVIDIA RTX A4000)

### Vector Add Benchmark (1M elements)

**Test Setup:**
- GPU: NVIDIA RTX A4000 (16 GB VRAM)
- Size: 1,000,000 float32 elements (4 MB)
- Operation: `out[i] = a[i] + b[i]`
- Iterations: 100 (warmup 3)
- Compilation: NVRTC JIT with `--use_fast_math`
- Date: December 27, 2024

**Measured Results:**

| Backend | Time | Bandwidth | Speedup |
|---------|------|-----------|---------|
| Lean-equivalent CUDA | 34.68 μs | 346 GB/s | **2.05x** |
| Python tinygrad | 71.16 μs | 169 GB/s | baseline |

**Analysis:**
- Lean-equivalent CUDA is **2.05x faster** than Python tinygrad (end-to-end)
- Lean achieves 346 GB/s vs Python's 169 GB/s (end-to-end)
- A4000 theoretical bandwidth: 448 GB/s

### Root Cause: Python Overhead (Not Kernel Quality)

Isolated kernel benchmarks (1000 iterations, C driver):

| Kernel | Grid | Block | Time | Bandwidth |
|--------|------|-------|------|-----------|
| Lean (scalar) | 3907 | 256 | 34.55 μs | 347 GB/s |
| **tinygrad (float4)** | 15625 | 16 | **32.70 μs** | **367 GB/s** |
| Optimal (float4) | 977 | 256 | 32.53 μs | 369 GB/s |

**The tinygrad kernel is actually 5% faster!** The slowdown comes from Python:

| Component | Time | % of Python call |
|-----------|------|------------------|
| CUDA kernel | 32.70 μs | 47% |
| **Python overhead** | **37.21 μs** | **53%** |
| Total | 69.91 μs | 100% |

Python overhead includes:
- TinyJit cache lookup
- Graph traversal / scheduling
- cuLaunchKernel FFI wrapper
- Python function call overhead

**Implication**: Lean's advantage is eliminating interpreter overhead, not generating better GPU code. tinygrad's float4 vectorization is effective.

### CUDA vs Metal Comparison

| Device | Lean Time | Python Time | Speedup |
|--------|-----------|-------------|---------|
| RTX A4000 (CUDA) | 34.68 μs | 71.16 μs | 2.05x |
| M3 Pro (Metal) | 69.89 μs | 91.51 μs | 1.31x |

**Insights:**
- CUDA backend shows larger speedup vs Python than Metal
- RTX A4000 ~2x faster than M3 Pro for this kernel
- Both backends verify correctness (max diff < 1e-7)

## Generated Shader Comparison

### Lean Metal Kernel

```metal
#include <metal_stdlib>
using namespace metal;

kernel void bench_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] + b[gid];
}
```

### Python tinygrad Metal Kernel

```metal
#include <metal_stdlib>
using namespace metal;

kernel void r_1000000(
    device float* data0,
    const device float* data1,
    const device float* data2,
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    int gidx0 = gid.x * 256 + lid.x;
    if (gidx0 < 1000000) {
        float val0 = *(data1 + gidx0);
        float val1 = *(data2 + gidx0);
        *(data0 + gidx0) = val0 + val1;
    }
}
```

**Key Differences:**
1. Python uses hierarchical thread indexing (gid + lid)
2. Python adds bounds check (`if gidx0 < 1000000`)
3. Python uses local variables for intermediate values
4. Python names are auto-generated (`r_1000000`)

### Lean CUDA Kernel

```cuda
extern "C" __global__ void bench_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    out[gid] = a[gid] + b[gid];
}
```

### Python tinygrad CUDA Kernel

*To be captured with `DEBUG=5 CUDA=1 python benchmark.py`*

## Profiling Instructions

### Metal (Local)

```bash
# Capture Metal shader
DEBUG=5 METAL=1 python -c "
from tinygrad import Tensor
a = Tensor.rand(1000000).realize()
b = Tensor.rand(1000000).realize()
c = (a + b).realize()
" 2>&1 | grep -A 20 "kernel void"

# GPU timing (requires Metal System Trace in Instruments)
xcrun xctrace record --template 'Metal System Trace' --output profile.trace ./metal_benchmark
```

### CUDA (RunPod)

```bash
# Create pod
runpod create --name cuda-test --image runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Upload files
scp build/standalone_cuda_benchmark.cu pod:/tmp/
scp build/tinygrad_cuda_benchmark.py pod:/tmp/

# Build and run Lean benchmark
ssh pod 'cd /tmp && nvcc -O3 standalone_cuda_benchmark.cu -o cuda_bench -lcuda -lnvrtc'
ssh pod 'nsys profile --stats=true /tmp/cuda_bench'

# Run Python tinygrad benchmark
ssh pod 'pip install tinygrad && python /tmp/tinygrad_cuda_benchmark.py'

# Deep kernel analysis
ssh pod 'ncu --set basic /tmp/cuda_bench'
```

## Vectorization Framework

A principled, composable vectorization abstraction was added in `TinyGrad4.Backend.Vectorization`.

### Key Types

```lean
-- Supported vector widths
inductive VectorWidth where
  | w1 : VectorWidth  -- Scalar (fallback)
  | w2 : VectorWidth  -- 2-wide
  | w4 : VectorWidth  -- 4-wide (optimal for most GPUs)
  | w8 : VectorWidth  -- 8-wide (some AMD GPUs)

-- Configuration for vectorized kernel generation
structure VectorConfig where
  backend : Backend       -- CUDA, Metal, or OpenCL
  dtype : DType           -- Element type (float32, int32, etc.)
  width : VectorWidth     -- Vector width to use
  totalSize : Nat         -- Total number of elements
  needsBoundsCheck : Bool -- For non-aligned tails
```

### Usage

```lean
-- Choose optimal vectorization for a kernel
let config := VectorConfig.optimal .CUDA .float32 1_000_000

-- Generate vectorized load/store
let load := renderVectorLoad config "data" "idx"
-- => "*(float4*)(data + idx)"

-- Generate complete kernel
let kernel := renderVectorizedBinaryKernel config "my_add" "+"
```

### Benchmark Runners

Both Metal and CUDA benchmarks now include float4 variants:
- `runVectorAdd1MFloat4` - 1M elements with float4 vectorization
- Comparison between scalar and vectorized versions

## Future Optimization Opportunities

### Near-term
- [ ] Cache compiled pipelines/modules
- [ ] Query actual maxThreadsPerBlock from device
- [ ] Add bounds checking to prevent OOB on non-multiple sizes

### Medium-term
- [x] Add float4 vectorization for memory bandwidth (**DONE** - see Vectorization framework)
- [ ] Use threadgroup memory for reductions
- [ ] Implement SIMD/warp-level operations

### Long-term
- [ ] Tensor core / WMMA support for matmul
- [ ] Automatic kernel fusion
- [ ] Auto-tuning thread dimensions

## Running the Benchmarks

### Metal (Local Mac)

```bash
cd lean4
./scripts/build_metal_test.sh
./build/metal_test
```

### CUDA (RunPod)

```bash
cd lean4
python scripts/runpod_benchmark.py --generate-only

# Copy files to RunPod and run
scp build/standalone_cuda_benchmark.cu runpod:/tmp/
scp build/tinygrad_cuda_benchmark.py runpod:/tmp/

ssh runpod 'cd /tmp && nvcc -O3 standalone_cuda_benchmark.cu -o cuda_bench -lcuda -lnvrtc && ./cuda_bench'
ssh runpod 'python tinygrad_cuda_benchmark.py'
```

## References

- [tinygrad Metal backend](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/ops_metal.py)
- [tinygrad CUDA backend](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/ops_cuda.py)
- [NVIDIA Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Metal Best Practices](https://developer.apple.com/documentation/metal/gpu_programming_techniques)
