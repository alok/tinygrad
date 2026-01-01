#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
Comprehensive data loading benchmark: Python vs Lean.

Tests real workloads:
1. Full batch read (all pixels, not just 2 bytes)
2. Checksum computation (forces memory access)
3. Multiple batch sizes
4. With/without shuffle
5. With/without normalization
"""

import time
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"

def load_raw_bytes(max_images: int = 10000) -> bytes:
    """Load MNIST as raw bytes."""
    path = DATA_DIR / "train-images-idx3-ubyte"
    with open(path, 'rb') as f:
        f.read(16)  # skip header
        return f.read(max_images * 784)

def load_numpy(max_images: int = 10000) -> np.ndarray:
    """Load MNIST as numpy array."""
    path = DATA_DIR / "train-images-idx3-ubyte"
    with open(path, 'rb') as f:
        f.read(16)
        data = np.frombuffer(f.read(max_images * 784), dtype=np.uint8)
        return data.reshape(-1, 784)

def bench_raw_checksum(data: bytes, batch_size: int, iterations: int = 10) -> dict:
    """Checksum all bytes in each batch (raw Python bytes)."""
    num_batches = len(data) // (batch_size * 784)
    bytes_per_batch = batch_size * 784

    times = []
    for _ in range(iterations):
        checksum = 0
        start = time.perf_counter()
        for i in range(num_batches):
            batch_start = i * bytes_per_batch
            batch_end = batch_start + bytes_per_batch
            # Sum all bytes in batch
            for j in range(batch_start, batch_end):
                checksum += data[j]
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "method": "Python bytes loop",
        "batch_size": batch_size,
        "num_batches": num_batches,
        "ms": np.median(times) * 1000,
        "batches_per_sec": num_batches / np.median(times),
        "checksum": checksum
    }

def bench_numpy_checksum(data: np.ndarray, batch_size: int, iterations: int = 10) -> dict:
    """Checksum using numpy (vectorized)."""
    num_batches = len(data) // batch_size

    times = []
    for _ in range(iterations):
        checksum = 0
        start = time.perf_counter()
        for i in range(num_batches):
            batch = data[i * batch_size:(i + 1) * batch_size]
            checksum += batch.sum()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "method": "Numpy vectorized",
        "batch_size": batch_size,
        "num_batches": num_batches,
        "ms": np.median(times) * 1000,
        "batches_per_sec": num_batches / np.median(times),
        "checksum": int(checksum)
    }

def bench_numpy_normalize(data: np.ndarray, batch_size: int, iterations: int = 10) -> dict:
    """Batch + normalize to float32 (realistic training prep)."""
    num_batches = len(data) // batch_size

    times = []
    for _ in range(iterations):
        total = 0.0
        start = time.perf_counter()
        for i in range(num_batches):
            batch = data[i * batch_size:(i + 1) * batch_size]
            # Normalize to [0, 1] float32
            batch_f32 = batch.astype(np.float32) / 255.0
            total += batch_f32.sum()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "method": "Numpy + normalize f32",
        "batch_size": batch_size,
        "num_batches": num_batches,
        "ms": np.median(times) * 1000,
        "batches_per_sec": num_batches / np.median(times),
        "checksum": total
    }

def bench_numpy_shuffle(data: np.ndarray, batch_size: int, iterations: int = 10) -> dict:
    """Shuffle + batch (full epoch simulation)."""
    num_batches = len(data) // batch_size
    indices = np.arange(len(data))

    times = []
    for _ in range(iterations):
        checksum = 0
        start = time.perf_counter()
        np.random.shuffle(indices)
        for i in range(num_batches):
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            batch = data[batch_indices]
            checksum += batch.sum()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "method": "Numpy shuffle + batch",
        "batch_size": batch_size,
        "num_batches": num_batches,
        "ms": np.median(times) * 1000,
        "batches_per_sec": num_batches / np.median(times),
        "checksum": int(checksum)
    }

def main():
    print("=" * 60)
    print("DATA LOADING BENCHMARK: Python vs Lean")
    print("=" * 60)
    print()

    max_images = 10000
    batch_size = 64

    print(f"Config: {max_images} images, batch_size={batch_size}")
    print()

    # Load data
    raw_data = load_raw_bytes(max_images)
    np_data = load_numpy(max_images)
    print(f"Loaded {len(np_data)} images ({len(raw_data):,} bytes)")
    print()

    # Run benchmarks
    results = []

    print("Running benchmarks (10 iterations each, median reported)...")
    print()

    # Raw Python bytes (slow baseline)
    # Skip this - too slow
    # r = bench_raw_checksum(raw_data, batch_size, iterations=1)
    # results.append(r)
    # print(f"  {r['method']}: {r['ms']:.2f} ms ({r['batches_per_sec']:,.0f} batch/s)")

    r = bench_numpy_checksum(np_data, batch_size)
    results.append(r)
    print(f"  {r['method']}: {r['ms']:.2f} ms ({r['batches_per_sec']:,.0f} batch/s)")

    r = bench_numpy_normalize(np_data, batch_size)
    results.append(r)
    print(f"  {r['method']}: {r['ms']:.2f} ms ({r['batches_per_sec']:,.0f} batch/s)")

    r = bench_numpy_shuffle(np_data, batch_size)
    results.append(r)
    print(f"  {r['method']}: {r['ms']:.2f} ms ({r['batches_per_sec']:,.0f} batch/s)")

    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print()
    print(f"{'Method':<30} {'Time (ms)':<12} {'Batch/sec':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['method']:<30} {r['ms']:<12.2f} {r['batches_per_sec']:<15,.0f}")

    print()
    print("=" * 60)
    print("Compare with Lean (run: .lake/build/bin/zero_copy_bench)")
    print("=" * 60)

if __name__ == "__main__":
    main()
