#!/usr/bin/env python3
"""
Test tinygrad with TinyJit - reusing compiled kernels like Lean does.
"""

import time
import numpy as np
import os
os.environ["METAL"] = "1"

from tinygrad import Tensor, TinyJit, Device

@TinyJit
def add_tensors(a: Tensor, b: Tensor):
    return a + b

def benchmark():
    print("=== Python tinygrad with TinyJit Benchmark ===")
    print(f"Device: {Device.DEFAULT}")

    size = 1_000_000
    print(f"Size: {size} elements ({size * 4 // 1_000_000} MB)")

    # Create data
    a_data = np.array([(i % 1000) / 1000.0 for i in range(size)], dtype=np.float32)
    b_data = np.array([((i + 500) % 1000) / 1000.0 for i in range(size)], dtype=np.float32)

    a = Tensor(a_data).realize()
    b = Tensor(b_data).realize()

    # First call compiles
    print("Compiling (first call)...")
    start = time.perf_counter()
    out = add_tensors(a, b)
    Device[out.device].synchronize()
    print(f"Compilation time: {(time.perf_counter() - start) * 1000:.2f} ms")

    # Warmup
    for _ in range(3):
        out = add_tensors(a, b)
        Device[out.device].synchronize()

    # Benchmark WITH sync (like Lean's benchmark)
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

    print(f"\n=== TinyJit results (async, sync at end) ===")
    print(f"Time: {avg_us:.6f} μs")
    print(f"Throughput: {gflops:.6f} GFLOP/s")
    print(f"Bandwidth: {bandwidth:.6f} GB/s")

    # Verify
    result = out.numpy()
    expected = a_data + b_data
    max_diff = np.max(np.abs(result - expected))
    print(f"Max diff: {max_diff}")
    print("✓ Results correct" if max_diff < 0.0001 else "✗ Results incorrect!")

if __name__ == "__main__":
    benchmark()
