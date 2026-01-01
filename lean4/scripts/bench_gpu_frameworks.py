#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "mlx"]
# ///
"""
GPU Framework Comparison: Data loading + normalization + forward pass.

Compares:
- MLX (Apple Silicon native)
- NumPy (CPU baseline)

This measures what actually matters: getting data to GPU and running compute.
"""

import time
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"

def load_raw(max_images: int = 10000) -> bytes:
    path = DATA_DIR / "train-images-idx3-ubyte"
    with open(path, 'rb') as f:
        f.read(16)
        return f.read(max_images * 784)

def bench_mlx(raw_data: bytes, batch_size: int = 64, iterations: int = 10):
    """MLX: Load → GPU → normalize → matmul (simulated forward pass)."""
    try:
        import mlx.core as mx
    except ImportError:
        return None

    num_images = len(raw_data) // 784
    num_batches = num_images // batch_size

    # Pre-create weight matrix for "forward pass"
    W = mx.random.normal((784, 128))
    mx.eval(W)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()

        for i in range(num_batches):
            # Slice raw bytes (zero-copy in Python)
            batch_bytes = raw_data[i * batch_size * 784:(i + 1) * batch_size * 784]

            # Convert to MLX array (goes to unified memory)
            batch_u8 = mx.array(np.frombuffer(batch_bytes, dtype=np.uint8).reshape(batch_size, 784))

            # Normalize on GPU
            batch_f32 = batch_u8.astype(mx.float32) / 255.0

            # Simulated forward pass (matmul)
            out = batch_f32 @ W

            # Force execution
            mx.eval(out)

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "framework": "MLX",
        "num_batches": num_batches,
        "median_ms": np.median(times) * 1000,
        "batches_per_sec": num_batches / np.median(times)
    }

def bench_numpy(raw_data: bytes, batch_size: int = 64, iterations: int = 10):
    """NumPy: Load → normalize → matmul (CPU baseline)."""
    num_images = len(raw_data) // 784
    num_batches = num_images // batch_size

    # Pre-convert to numpy array
    all_data = np.frombuffer(raw_data, dtype=np.uint8).reshape(num_images, 784)

    # Weight matrix
    W = np.random.randn(784, 128).astype(np.float32)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()

        for i in range(num_batches):
            batch = all_data[i * batch_size:(i + 1) * batch_size]
            batch_f32 = batch.astype(np.float32) / 255.0
            _ = batch_f32 @ W

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "framework": "NumPy (CPU)",
        "num_batches": num_batches,
        "median_ms": np.median(times) * 1000,
        "batches_per_sec": num_batches / np.median(times)
    }

def bench_numpy_vectorized(raw_data: bytes, batch_size: int = 64, iterations: int = 10):
    """NumPy: Pre-normalized data (best case CPU)."""
    num_images = len(raw_data) // 784
    num_batches = num_images // batch_size

    # Pre-normalize ALL data upfront
    all_data = np.frombuffer(raw_data, dtype=np.uint8).reshape(num_images, 784).astype(np.float32) / 255.0
    W = np.random.randn(784, 128).astype(np.float32)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()

        for i in range(num_batches):
            batch = all_data[i * batch_size:(i + 1) * batch_size]
            _ = batch @ W

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "framework": "NumPy (pre-norm)",
        "num_batches": num_batches,
        "median_ms": np.median(times) * 1000,
        "batches_per_sec": num_batches / np.median(times)
    }

def main():
    print("=" * 70)
    print("GPU FRAMEWORK BENCHMARK: Data Loading + Forward Pass")
    print("=" * 70)
    print()

    batch_size = 64
    max_images = 10000

    print(f"Config: {max_images} images, batch_size={batch_size}")
    raw_data = load_raw(max_images)
    print(f"Loaded {len(raw_data):,} bytes")
    print()

    results = []

    # MLX
    r = bench_mlx(raw_data, batch_size)
    if r:
        results.append(r)
        print(f"MLX:            {r['median_ms']:6.2f} ms  ({r['batches_per_sec']:,.0f} batch/s)")
    else:
        print("MLX: not available")

    # NumPy
    r = bench_numpy(raw_data, batch_size)
    results.append(r)
    print(f"NumPy (CPU):    {r['median_ms']:6.2f} ms  ({r['batches_per_sec']:,.0f} batch/s)")

    # NumPy pre-normalized
    r = bench_numpy_vectorized(raw_data, batch_size)
    results.append(r)
    print(f"NumPy (pre):    {r['median_ms']:6.2f} ms  ({r['batches_per_sec']:,.0f} batch/s)")

    print()
    print("=" * 70)
    print("WHAT LEAN NEEDS TO COMPETE:")
    print("=" * 70)
    print("""
1. Execute UOp graph on GPU (we have Metal FFI!)
   - Don't loop in Lean, build UOps → compile → execute

2. Zero-copy ByteSlice → Metal buffer
   - Unified memory on Apple Silicon = no copy needed

3. Fused kernels: load+cast+normalize+matmul in one kernel
   - tinygrad already does this in Python

4. For CPU-side ops: SIMD via Accelerate FFI
   - vDSP_sve for sum, vDSP_vfltu8 for cast

The 690x gap is NOT fundamental - it's because we're benchmarking
naive Lean loops instead of the actual tinygrad execution path.
""")

if __name__ == "__main__":
    main()
