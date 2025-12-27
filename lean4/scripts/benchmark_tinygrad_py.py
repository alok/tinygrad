#!/usr/bin/env python3
"""
Benchmark Python tinygrad on same task as Lean Metal backend.
1M element vector add, matching the Lean benchmark exactly.
"""

import time
import numpy as np

# Force Metal backend
import os
os.environ["METAL"] = "1"

from tinygrad import Tensor, Device

def benchmark_vector_add():
    print("=== Python tinygrad Vector Add Benchmark ===")
    print(f"Device: {Device.DEFAULT}")

    size = 1_000_000  # 1M elements, same as Lean
    print(f"Size: {size} elements ({size * 4 // 1_000_000} MB)")

    # Create data (same pattern as Lean)
    a_data = np.array([(i % 1000) / 1000.0 for i in range(size)], dtype=np.float32)
    b_data = np.array([((i + 500) % 1000) / 1000.0 for i in range(size)], dtype=np.float32)

    # Create tensors
    a = Tensor(a_data)
    b = Tensor(b_data)

    # Warmup (same as Lean: 3 iterations)
    for _ in range(3):
        out = a + b
        out.realize()

    # Benchmark (same as Lean: 100 iterations)
    iterations = 100

    # Method 1: Total time including realize
    start = time.perf_counter_ns()
    for _ in range(iterations):
        out = a + b
        out.realize()
    end = time.perf_counter_ns()

    total_ms = (end - start) / 1e6
    avg_us = total_ms * 1000.0 / iterations
    gflops = (size / avg_us) * 1e6 / 1e9
    bandwidth = (3.0 * size * 4.0 / avg_us) * 1e6 / 1e9  # read 2, write 1

    print(f"Time: {avg_us:.6f} μs")
    print(f"Throughput: {gflops:.6f} GFLOP/s")
    print(f"Bandwidth: {bandwidth:.6f} GB/s")
    print()

    # Method 2: Just kernel time (more accurate)
    print("=== Kernel-only timing (no graph overhead) ===")

    # Pre-realize inputs
    a.realize()
    b.realize()

    # Create output buffer once
    out = a + b
    out.realize()

    # Now time just the kernel launches
    start = time.perf_counter_ns()
    for _ in range(iterations):
        out = a + b
        out.realize()
    end = time.perf_counter_ns()

    total_ms = (end - start) / 1e6
    avg_us = total_ms * 1000.0 / iterations
    gflops = (size / avg_us) * 1e6 / 1e9
    bandwidth = (3.0 * size * 4.0 / avg_us) * 1e6 / 1e9

    print(f"Time: {avg_us:.6f} μs")
    print(f"Throughput: {gflops:.6f} GFLOP/s")
    print(f"Bandwidth: {bandwidth:.6f} GB/s")
    print()

    # Verify correctness
    result = out.numpy()
    expected = a_data + b_data
    max_diff = np.max(np.abs(result - expected))
    print(f"Max diff from expected: {max_diff}")
    if max_diff < 0.0001:
        print("✓ Results correct")
    else:
        print("✗ Results incorrect!")

if __name__ == "__main__":
    benchmark_vector_add()
