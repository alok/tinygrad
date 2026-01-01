#!/usr/bin/env python3
"""
RunPod Serverless CUDA Benchmark

Uses RunPod's serverless API to run benchmarks without SSH.
Requires RUNPOD_API_KEY environment variable.

Usage:
    export RUNPOD_API_KEY=your_key_here
    python runpod_serverless.py
"""

import os
import sys
import json
import urllib.request
import urllib.error

# RunPod API endpoint
RUNPOD_API = "https://api.runpod.io/graphql"

# CUDA benchmark code (embedded)
CUDA_BENCHMARK = '''
#include <cuda.h>
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define CHECK_CU(x) { CUresult err = (x); if (err != CUDA_SUCCESS) { const char* s; \
    cuGetErrorString(err, &s); fprintf(stderr, "CUDA error: %s\\n", s); exit(1); } }

const char* KERNEL =
    "extern \\"C\\" __global__ void add(const float* a, const float* b, float* out, int n) "
    "{ int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) out[i] = a[i] + b[i]; }";

int main() {
    printf("=== TinyGrad4 CUDA Benchmark ===\\n");
    CHECK_CU(cuInit(0));
    CUdevice dev; CHECK_CU(cuDeviceGet(&dev, 0));
    char name[256]; cuDeviceGetName(name, 256, dev);
    printf("Device: %s\\n", name);

    CUcontext ctx; CHECK_CU(cuCtxCreate(&ctx, 0, dev));
    CUstream stream; CHECK_CU(cuStreamCreate(&stream, 0));

    nvrtcProgram prog; nvrtcCreateProgram(&prog, KERNEL, "k.cu", 0, 0, 0);
    int maj, min; cuDeviceGetAttribute(&maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    char arch[32]; snprintf(arch, 32, "--gpu-architecture=compute_%d%d", maj, min);
    const char* opts[] = {arch, "--use_fast_math"};
    nvrtcCompileProgram(prog, 2, opts);
    size_t ptxSz; nvrtcGetPTXSize(prog, &ptxSz);
    char* ptx = malloc(ptxSz); nvrtcGetPTX(prog, ptx);
    CUmodule mod; CHECK_CU(cuModuleLoadDataEx(&mod, ptx, 0, 0, 0));
    CUfunction fn; CHECK_CU(cuModuleGetFunction(&fn, mod, "add"));

    int n = 1000000; size_t bytes = n * sizeof(float);
    CUdeviceptr d_a, d_b, d_o;
    CHECK_CU(cuMemAlloc(&d_a, bytes)); CHECK_CU(cuMemAlloc(&d_b, bytes)); CHECK_CU(cuMemAlloc(&d_o, bytes));
    float* h_a = malloc(bytes); float* h_b = malloc(bytes);
    for(int i=0;i<n;i++) { h_a[i]=(i%1000)/1000.0f; h_b[i]=((i+500)%1000)/1000.0f; }
    CHECK_CU(cuMemcpyHtoD(d_a, h_a, bytes)); CHECK_CU(cuMemcpyHtoD(d_b, h_b, bytes));

    void* args[] = {&d_a, &d_b, &d_o, &n};
    int blk=256, grd=(n+blk-1)/blk;
    for(int i=0;i<3;i++) CHECK_CU(cuLaunchKernel(fn,grd,1,1,blk,1,1,0,stream,args,0));
    CHECK_CU(cuStreamSynchronize(stream));

    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    for(int i=0;i<100;i++) CHECK_CU(cuLaunchKernel(fn,grd,1,1,blk,1,1,0,stream,args,0));
    CHECK_CU(cuStreamSynchronize(stream));
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ns = (t1.tv_sec-t0.tv_sec)*1e9 + (t1.tv_nsec-t0.tv_nsec);
    double us = ns/1e3/100;
    printf("Time: %.3f us\\n", us);
    printf("Bandwidth: %.3f GB/s\\n", 3.0*n*4/us*1e6/1e9);

    float* h_o = malloc(bytes); CHECK_CU(cuMemcpyDtoH(h_o, d_o, bytes));
    float maxd=0; for(int i=0;i<n;i++) { float d=fabsf(h_o[i]-h_a[i]-h_b[i]); if(d>maxd)maxd=d; }
    printf("Max diff: %.9f\\n", maxd);
    printf(maxd<0.0001 ? "OK\\n" : "ERROR\\n");
    return 0;
}
'''

PYTHON_BENCHMARK = '''
import time, numpy as np, os
os.environ["CUDA"]="1"
try:
    from tinygrad import Tensor, TinyJit, Device
except:
    import subprocess; subprocess.run(["pip","install","tinygrad"]); from tinygrad import Tensor, TinyJit, Device

@TinyJit
def add(a,b): return a+b

print("=== Python tinygrad CUDA ===")
print(f"Device: {Device.DEFAULT}")
n=1000000
a=Tensor(np.array([(i%1000)/1000 for i in range(n)],dtype=np.float32)).realize()
b=Tensor(np.array([((i+500)%1000)/1000 for i in range(n)],dtype=np.float32)).realize()
o=add(a,b); Device[o.device].synchronize()
for _ in range(3): o=add(a,b); Device[o.device].synchronize()
t0=time.perf_counter_ns()
for _ in range(100): o=add(a,b)
Device[o.device].synchronize()
t1=time.perf_counter_ns()
us=(t1-t0)/1e3/100
print(f"Time: {us:.3f} us")
print(f"Bandwidth: {3*n*4/us*1e6/1e9:.3f} GB/s")
r=o.numpy(); e=a.numpy()+b.numpy(); d=np.max(np.abs(r-e))
print(f"Max diff: {d}")
print("OK" if d<0.0001 else "ERROR")
'''


def graphql_request(query: str, variables: dict = None) -> dict:
    """Make a GraphQL request to RunPod API."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY environment variable not set")
        sys.exit(1)

    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        RUNPOD_API,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code}")
        print(e.read().decode())
        sys.exit(1)


def get_myself():
    """Get current user info to verify API key."""
    query = """
    query {
        myself {
            id
            email
        }
    }
    """
    return graphql_request(query)


def list_pods():
    """List all pods."""
    query = """
    query {
        myself {
            pods {
                id
                name
                desiredStatus
                runtime {
                    uptimeInSeconds
                }
            }
        }
    }
    """
    return graphql_request(query)


def create_pod(name: str = "cuda-benchmark", gpu_type: str = "NVIDIA RTX A4000"):
    """Create a new pod."""
    query = """
    mutation createPod($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id
            name
            desiredStatus
        }
    }
    """
    variables = {
        "input": {
            "name": name,
            "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
            "gpuTypeId": gpu_type,
            "volumeInGb": 10,
            "containerDiskInGb": 10,
            "minVcpuCount": 4,
            "minMemoryInGb": 16,
            "cloudType": "COMMUNITY"
        }
    }
    return graphql_request(query, variables)


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     RunPod Serverless CUDA Benchmark                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Verify API key
    print("Verifying API key...")
    try:
        result = get_myself()
        if "errors" in result:
            print(f"API Error: {result['errors']}")
            sys.exit(1)
        user = result.get("data", {}).get("myself", {})
        print(f"Logged in as: {user.get('email', 'unknown')}")
    except Exception as e:
        print(f"Failed to verify API key: {e}")
        sys.exit(1)

    print()
    print("Listing pods...")
    pods = list_pods()
    pod_list = pods.get("data", {}).get("myself", {}).get("pods", [])

    if pod_list:
        print("Available pods:")
        for pod in pod_list:
            status = pod.get("desiredStatus", "unknown")
            uptime = pod.get("runtime", {}).get("uptimeInSeconds", 0)
            print(f"  - {pod['name']} ({pod['id']}): {status}, uptime: {uptime}s")
    else:
        print("No pods found.")

    print()
    print("=== Benchmark Commands ===")
    print()
    print("To run benchmarks, SSH into a pod and execute:")
    print()
    print("# Compile and run Lean-equivalent benchmark")
    print("cat > /tmp/bench.cu << 'EOF'")
    print(CUDA_BENCHMARK)
    print("EOF")
    print("nvcc -O3 /tmp/bench.cu -o /tmp/bench -lcuda -lnvrtc && /tmp/bench")
    print()
    print("# Run Python tinygrad benchmark")
    print("python3 -c \"\"\"")
    print(PYTHON_BENCHMARK)
    print("\"\"\"")


if __name__ == "__main__":
    main()
