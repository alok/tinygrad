#!/usr/bin/env python3
"""
Raw Metal kernel benchmark - bypassing tinygrad's scheduler.
Uses allocator API for buffer management.
"""

import time
import numpy as np
import os
os.environ["METAL"] = "1"

from tinygrad.runtime.ops_metal import MetalDevice, MetalProgram

KERNEL_SOURCE = """
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
"""

def benchmark_raw_metal():
    print("=== Raw Metal Kernel Benchmark ===")

    dev = MetalDevice("METAL")
    print(f"Device: METAL")

    size = 1_000_000
    byte_size = size * 4
    print(f"Size: {size} elements ({byte_size // 1_000_000} MB)")

    # Create data
    a_data = np.array([(i % 1000) / 1000.0 for i in range(size)], dtype=np.float32)
    b_data = np.array([((i + 500) % 1000) / 1000.0 for i in range(size)], dtype=np.float32)

    # Allocate using allocator
    from tinygrad.device import BufferSpec
    spec = BufferSpec()
    buf_a = dev.allocator._alloc(byte_size, spec)
    buf_b = dev.allocator._alloc(byte_size, spec)
    buf_out = dev.allocator._alloc(byte_size, spec)

    # Copy data using allocator's copy method
    dev.allocator._copyin(buf_a, memoryview(a_data.tobytes()))
    dev.allocator._copyin(buf_b, memoryview(b_data.tobytes()))

    # Compile kernel
    prog = MetalProgram(dev, "bench_add", KERNEL_SOURCE.encode())

    # Warmup
    for _ in range(3):
        prog(buf_a, buf_b, buf_out, global_size=(size, 1, 1), local_size=(256, 1, 1))
    dev.synchronize()

    # Benchmark
    iterations = 100

    start = time.perf_counter_ns()
    for _ in range(iterations):
        prog(buf_a, buf_b, buf_out, global_size=(size, 1, 1), local_size=(256, 1, 1))
    dev.synchronize()
    end = time.perf_counter_ns()

    total_ms = (end - start) / 1e6
    avg_us = total_ms * 1000.0 / iterations
    gflops = (size / avg_us) * 1e6 / 1e9
    bandwidth = (3.0 * size * 4.0 / avg_us) * 1e6 / 1e9

    print(f"\n=== Raw kernel timing (async, sync at end) ===")
    print(f"Time: {avg_us:.6f} μs")
    print(f"Throughput: {gflops:.6f} GFLOP/s")
    print(f"Bandwidth: {bandwidth:.6f} GB/s")

    # Verify
    out_bytes = bytearray(byte_size)
    dev.allocator._copyout(memoryview(out_bytes), buf_out)
    out_data = np.frombuffer(out_bytes, dtype=np.float32)
    expected = a_data + b_data
    max_diff = np.max(np.abs(out_data - expected))
    print(f"Max diff: {max_diff}")
    print("✓ Results correct" if max_diff < 0.0001 else "✗ Results incorrect!")

    # Cleanup
    dev.allocator._free(buf_a, spec)
    dev.allocator._free(buf_b, spec)
    dev.allocator._free(buf_out, spec)

if __name__ == "__main__":
    benchmark_raw_metal()
