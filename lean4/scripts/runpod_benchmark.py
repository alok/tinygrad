#!/usr/bin/env python3
"""
RunPod Benchmark Deployment Script

Deploys TinyGrad4 CUDA benchmark to RunPod, runs profiling,
and retrieves results.

Usage:
    python runpod_benchmark.py [--pod-id POD_ID] [--gpu-type GPU_TYPE]

Environment:
    RUNPOD_API_KEY: Your RunPod API key (required)
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

# Import runpod only when needed (avoid error in generate-only mode)
runpod = None

def ensure_runpod():
    global runpod
    if runpod is None:
        try:
            import runpod as rp
            runpod = rp
        except ImportError:
            print("ERROR: runpod not installed. Install with: uv add runpod")
            sys.exit(1)

# Configuration
SCRIPT_DIR = Path(__file__).parent
LEAN4_DIR = SCRIPT_DIR.parent
C_DIR = LEAN4_DIR / "c"

# Files to upload
FILES_TO_UPLOAD = [
    C_DIR / "tg4_cuda.cu",
    LEAN4_DIR / "TinyGrad4" / "Test" / "CudaTestMain.lean",
    LEAN4_DIR / "TinyGrad4" / "Backend" / "Cuda.lean",
    LEAN4_DIR / "TinyGrad4" / "Backend" / "Device.lean",
]

# Standalone CUDA benchmark (no Lean runtime needed)
STANDALONE_CUDA_BENCHMARK = '''
// standalone_cuda_benchmark.cu
// Compile: nvcc -O3 standalone_cuda_benchmark.cu -o cuda_bench -lcuda -lnvrtc

#include <cuda.h>
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK_CU(x) do { \\
    CUresult err = (x); \\
    if (err != CUDA_SUCCESS) { \\
        const char* errStr; \\
        cuGetErrorString(err, &errStr); \\
        fprintf(stderr, "CUDA error: %s\\n", errStr); \\
        exit(1); \\
    } \\
} while(0)

const char* KERNEL_SOURCE = R"(
extern "C" __global__ void bench_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        out[gid] = a[gid] + b[gid];
    }
}
)";

long long get_nanos() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int main() {
    printf("=== TinyGrad4 CUDA Benchmark (Standalone) ===\\n\\n");

    // Initialize CUDA
    CHECK_CU(cuInit(0));

    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));

    char name[256];
    CHECK_CU(cuDeviceGetName(name, sizeof(name), device));
    printf("Device: %s\\n", name);

    CUcontext ctx;
    CHECK_CU(cuCtxCreate(&ctx, 0, device));

    CUstream stream;
    CHECK_CU(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

    // Compile kernel via NVRTC
    printf("Compiling kernel...\\n");
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, KERNEL_SOURCE, "kernel.cu", 0, NULL, NULL);

    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);

    char arch_opt[32];
    snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d%d", major, minor);

    const char* opts[] = { arch_opt, "--use_fast_math" };
    nvrtcCompileProgram(prog, 2, opts);

    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = (char*)malloc(ptxSize);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    CUmodule module;
    CHECK_CU(cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL));
    free(ptx);

    CUfunction kernel;
    CHECK_CU(cuModuleGetFunction(&kernel, module, "bench_add"));

    // Allocate buffers
    int size = 1000000;  // 1M elements
    size_t bytes = size * sizeof(float);
    printf("Size: %d elements (%zu MB)\\n\\n", size, bytes / 1000000);

    CUdeviceptr d_a, d_b, d_out;
    CHECK_CU(cuMemAlloc(&d_a, bytes));
    CHECK_CU(cuMemAlloc(&d_b, bytes));
    CHECK_CU(cuMemAlloc(&d_out, bytes));

    // Initialize host data
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    for (int i = 0; i < size; i++) {
        h_a[i] = (float)(i % 1000) / 1000.0f;
        h_b[i] = (float)((i + 500) % 1000) / 1000.0f;
    }

    CHECK_CU(cuMemcpyHtoD(d_a, h_a, bytes));
    CHECK_CU(cuMemcpyHtoD(d_b, h_b, bytes));

    // Launch params
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    void* args[] = { &d_a, &d_b, &d_out, &size };

    // Warmup
    printf("Warmup...\\n");
    for (int i = 0; i < 3; i++) {
        CHECK_CU(cuLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1, 1, 0, stream, args, NULL));
    }
    CHECK_CU(cuStreamSynchronize(stream));

    // Benchmark
    int iterations = 100;
    printf("Running %d iterations...\\n\\n", iterations);

    long long start = get_nanos();
    for (int i = 0; i < iterations; i++) {
        CHECK_CU(cuLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1, 1, 0, stream, args, NULL));
    }
    CHECK_CU(cuStreamSynchronize(stream));
    long long end = get_nanos();

    double total_ms = (double)(end - start) / 1e6;
    double avg_us = total_ms * 1000.0 / iterations;
    double gflops = ((double)size / avg_us) * 1e6 / 1e9;
    double bandwidth = (3.0 * size * 4.0 / avg_us) * 1e6 / 1e9;

    printf("=== Results ===\\n");
    printf("Time: %.6f us\\n", avg_us);
    printf("Throughput: %.6f GFLOP/s\\n", gflops);
    printf("Bandwidth: %.6f GB/s\\n", bandwidth);

    // Verify
    float* h_out = (float*)malloc(bytes);
    CHECK_CU(cuMemcpyDtoH(h_out, d_out, bytes));

    float max_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float expected = h_a[i] + h_b[i];
        float diff = fabsf(h_out[i] - expected);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max diff: %.9f\\n", max_diff);
    printf("%s\\n", max_diff < 0.0001f ? "OK Results correct" : "ERROR Results incorrect!");

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_out);
    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_out);
    cuModuleUnload(module);
    cuCtxDestroy(ctx);

    return 0;
}
'''

def create_pod(gpu_type: str = "NVIDIA RTX A4000") -> str:
    """Create a new RunPod instance."""
    ensure_runpod()
    print(f"Creating RunPod with GPU: {gpu_type}")

    pod = runpod.create_pod(
        name="tinygrad4-cuda-benchmark",
        image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        gpu_type_id=gpu_type,
        cloud_type="COMMUNITY",  # or "SECURE" for dedicated
        volume_in_gb=10,
        container_disk_in_gb=10,
        min_vcpu_count=4,
        min_memory_in_gb=16,
    )

    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")

    # Wait for pod to be ready
    print("Waiting for pod to be ready...")
    while True:
        status = runpod.get_pod(pod_id)
        if status.get("desiredStatus") == "RUNNING":
            runtime = status.get("runtime", {})
            if runtime.get("uptimeInSeconds", 0) > 0:
                break
        time.sleep(5)
        print(".", end="", flush=True)

    print("\nPod is ready!")
    return pod_id


def run_on_pod(pod_id: str) -> dict:
    """Run benchmark on existing pod."""
    ensure_runpod()
    print(f"\nRunning benchmark on pod: {pod_id}")

    # Get pod info
    pod = runpod.get_pod(pod_id)
    runtime = pod.get("runtime", {})
    ssh_command = runtime.get("sshCommand", "")

    if not ssh_command:
        print("ERROR: No SSH access. Using RunPod execute...")

        # Use RunPod's run_pod API instead
        result = runpod.run_pod(
            pod_id,
            {
                "input": {
                    "command": "bash",
                    "args": ["-c", f"""
cat > /tmp/benchmark.cu << 'CUDA_EOF'
{STANDALONE_CUDA_BENCHMARK}
CUDA_EOF

cd /tmp
nvcc -O3 benchmark.cu -o cuda_bench -lcuda -lnvrtc 2>&1
./cuda_bench

# Run with nsys if available
if command -v nsys &> /dev/null; then
    echo ""
    echo "=== Running with nsys profiling ==="
    nsys profile --stats=true -o /tmp/profile ./cuda_bench 2>&1 | head -100
fi
"""]
                }
            }
        )
        return result

    print(f"SSH: {ssh_command}")
    return {"status": "manual", "ssh": ssh_command}


def run_standalone_benchmark():
    """Generate and run standalone CUDA benchmark locally or print instructions."""
    benchmark_file = LEAN4_DIR / "build" / "standalone_cuda_benchmark.cu"
    benchmark_file.parent.mkdir(parents=True, exist_ok=True)

    with open(benchmark_file, "w") as f:
        f.write(STANDALONE_CUDA_BENCHMARK)

    print(f"Written: {benchmark_file}")
    print()
    print("=== To run on RunPod ===")
    print()
    print("1. Create a pod with CUDA support:")
    print("   runpod create --name cuda-test --image runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel")
    print()
    print("2. SSH into the pod and run:")
    print(f"   scp {benchmark_file} pod:/tmp/")
    print("   ssh pod 'cd /tmp && nvcc -O3 benchmark.cu -o cuda_bench -lcuda -lnvrtc && ./cuda_bench'")
    print()
    print("3. For profiling:")
    print("   ssh pod 'nsys profile --stats=true /tmp/cuda_bench'")
    print()

    # Also generate Python tinygrad comparison
    tinygrad_benchmark = '''#!/usr/bin/env python3
"""Python tinygrad CUDA benchmark for comparison."""

import time
import numpy as np
import os
os.environ["CUDA"] = "1"

from tinygrad import Tensor, TinyJit, Device

@TinyJit
def add_tensors(a: Tensor, b: Tensor):
    return a + b

def benchmark():
    print("=== Python tinygrad CUDA Benchmark ===")
    print(f"Device: {Device.DEFAULT}")

    size = 1_000_000
    print(f"Size: {size} elements ({size * 4 // 1_000_000} MB)")

    a_data = np.array([(i % 1000) / 1000.0 for i in range(size)], dtype=np.float32)
    b_data = np.array([((i + 500) % 1000) / 1000.0 for i in range(size)], dtype=np.float32)

    a = Tensor(a_data).realize()
    b = Tensor(b_data).realize()

    # Compile
    print("Compiling...")
    out = add_tensors(a, b)
    Device[out.device].synchronize()

    # Warmup
    for _ in range(3):
        out = add_tensors(a, b)
        Device[out.device].synchronize()

    # Benchmark
    iterations = 100
    start = time.perf_counter_ns()
    for _ in range(iterations):
        out = add_tensors(a, b)
    Device[out.device].synchronize()
    end = time.perf_counter_ns()

    total_ms = (end - start) / 1e6
    avg_us = total_ms * 1000.0 / iterations
    gflops = (size / avg_us) * 1e6 / 1e9
    bandwidth = (3.0 * size * 4.0 / avg_us) * 1e6 / 1e9

    print(f"\\n=== Results ===")
    print(f"Time: {avg_us:.6f} us")
    print(f"Throughput: {gflops:.6f} GFLOP/s")
    print(f"Bandwidth: {bandwidth:.6f} GB/s")

    # Verify
    result = out.numpy()
    expected = a_data + b_data
    max_diff = np.max(np.abs(result - expected))
    print(f"Max diff: {max_diff}")
    print("OK" if max_diff < 0.0001 else "ERROR")

if __name__ == "__main__":
    benchmark()
'''

    tinygrad_file = LEAN4_DIR / "build" / "tinygrad_cuda_benchmark.py"
    with open(tinygrad_file, "w") as f:
        f.write(tinygrad_benchmark)

    print(f"Written: {tinygrad_file}")
    print("Run on pod: python /tmp/tinygrad_cuda_benchmark.py")


def main():
    parser = argparse.ArgumentParser(description="RunPod CUDA Benchmark")
    parser.add_argument("--pod-id", help="Existing pod ID to use")
    parser.add_argument("--gpu-type", default="NVIDIA RTX A4000", help="GPU type for new pod")
    parser.add_argument("--generate-only", action="store_true", help="Only generate benchmark files")
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key and not args.generate_only:
        print("RUNPOD_API_KEY not set. Generating benchmark files only.")
        args.generate_only = True

    if args.generate_only:
        run_standalone_benchmark()
        return

    ensure_runpod()
    runpod.api_key = api_key

    if args.pod_id:
        result = run_on_pod(args.pod_id)
    else:
        pod_id = create_pod(args.gpu_type)
        result = run_on_pod(pod_id)

    print("\nResult:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
