#!/usr/bin/env python3
"""
Fair comparison benchmark: Python tinygrad vs Lean Metal backend.
Uses realized tensors and manual kernel timing.
"""

import time
import numpy as np
import os
os.environ["METAL"] = "1"

from tinygrad import Tensor, Device
from tinygrad.engine.realize import run_schedule
from tinygrad.engine.schedule import create_schedule

def benchmark_raw_kernel():
    """Benchmark just the kernel execution, matching Lean's approach."""
    print("=== Python tinygrad Raw Kernel Benchmark ===")
    print(f"Device: {Device.DEFAULT}")

    size = 1_000_000
    print(f"Size: {size} elements ({size * 4 // 1_000_000} MB)")

    # Create data
    a_data = np.array([(i % 1000) / 1000.0 for i in range(size)], dtype=np.float32)
    b_data = np.array([((i + 500) % 1000) / 1000.0 for i in range(size)], dtype=np.float32)

    a = Tensor(a_data).realize()
    b = Tensor(b_data).realize()

    # Pre-compile by running once
    out = (a + b).realize()

    # Get the schedule for a+b (this is the "compiled kernel")
    out_lazy = a + b

    # Warmup
    for _ in range(3):
        out = (a + b).realize()

    # Benchmark
    iterations = 100

    # Time the full realize loop
    start = time.perf_counter_ns()
    for _ in range(iterations):
        out = (a + b).realize()
    end = time.perf_counter_ns()

    total_ms = (end - start) / 1e6
    avg_us = total_ms * 1000.0 / iterations
    gflops = (size / avg_us) * 1e6 / 1e9
    bandwidth = (3.0 * size * 4.0 / avg_us) * 1e6 / 1e9

    print(f"\n=== Full realize() timing ===")
    print(f"Time: {avg_us:.6f} μs")
    print(f"Throughput: {gflops:.6f} GFLOP/s")
    print(f"Bandwidth: {bandwidth:.6f} GB/s")

    # Verify
    result = out.numpy()
    expected = a_data + b_data
    max_diff = np.max(np.abs(result - expected))
    print(f"Max diff: {max_diff}")
    print("✓ Results correct" if max_diff < 0.0001 else "✗ Results incorrect!")

def benchmark_with_preallocated():
    """Use preallocated output buffer like Lean does."""
    print("\n=== Preallocated buffer benchmark ===")

    size = 1_000_000
    a_data = np.array([(i % 1000) / 1000.0 for i in range(size)], dtype=np.float32)
    b_data = np.array([((i + 500) % 1000) / 1000.0 for i in range(size)], dtype=np.float32)

    a = Tensor(a_data).realize()
    b = Tensor(b_data).realize()

    # Warmup and get output tensor
    out = (a + b).realize()

    iterations = 100

    # Can we reuse the same output buffer? In tinygrad, tensors are immutable
    # so each a+b creates a new output. This is fundamentally different from
    # Lean which reuses buffers.

    start = time.perf_counter_ns()
    for _ in range(iterations):
        out = a + b  # Build graph
        out.realize()  # Execute
    end = time.perf_counter_ns()

    total_ms = (end - start) / 1e6
    avg_us = total_ms * 1000.0 / iterations

    print(f"Time (graph+realize): {avg_us:.6f} μs")
    print(f"Bandwidth: {(3.0 * size * 4.0 / avg_us) * 1e6 / 1e9:.6f} GB/s")

if __name__ == "__main__":
    benchmark_raw_kernel()
    benchmark_with_preallocated()

    print("\n" + "="*60)
    print("NOTE: Python tinygrad creates new graphs/buffers each iteration.")
    print("Lean reuses compiled kernels and preallocated buffers.")
    print("This explains the performance difference.")
    print("="*60)
