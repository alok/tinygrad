#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
Benchmark Python MNIST data loading for comparison with Lean implementation.

Measures:
1. Raw numpy data loading and batching (CPU baseline)
2. tinygrad Tensor loading with random sampling (GPU-style)
3. Sequential epoch iteration (traditional loader style)
"""

import time
import gzip
import struct
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"

def load_mnist_images(path: Path) -> np.ndarray:
    """Load MNIST images from IDX file."""
    with gzip.open(path, 'rb') if path.suffix == '.gz' else open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols).astype(np.float32) / 255.0

def load_mnist_labels(path: Path) -> np.ndarray:
    """Load MNIST labels from IDX file."""
    with gzip.open(path, 'rb') if path.suffix == '.gz' else open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 2049
        return np.frombuffer(f.read(), dtype=np.uint8)

def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert labels to one-hot encoding."""
    return np.eye(num_classes)[labels]

class NumpyMNISTLoader:
    """Traditional batch loader using numpy (CPU)."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool = True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(images)
        self.indices = np.arange(self.n)

    def __len__(self):
        return self.n // self.batch_size

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for i in range(0, self.n - self.batch_size + 1, self.batch_size):
            idx = self.indices[i:i + self.batch_size]
            yield {
                'pixels': self.images[idx],
                'labels': self.labels[idx],
                'one_hot': one_hot(self.labels[idx])
            }

def benchmark_numpy_loader(images: np.ndarray, labels: np.ndarray,
                           batch_size: int = 64, max_images: int = 10000) -> float:
    """Benchmark numpy-based loader (CPU baseline)."""
    images = images[:max_images]
    labels = labels[:max_images]

    loader = NumpyMNISTLoader(images, labels, batch_size, shuffle=True)

    start = time.perf_counter()
    count = 0
    for batch in loader:
        _ = batch['pixels']  # Access data
        _ = batch['one_hot']
        count += 1
    elapsed = time.perf_counter() - start

    return count, elapsed

def benchmark_numpy_sequential(images: np.ndarray, labels: np.ndarray,
                               batch_size: int = 64, max_images: int = 10000) -> float:
    """Benchmark sequential batching without shuffle."""
    images = images[:max_images]
    labels = labels[:max_images]

    start = time.perf_counter()
    count = 0
    for i in range(0, len(images) - batch_size + 1, batch_size):
        _ = images[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        _ = one_hot(batch_labels)
        count += 1
    elapsed = time.perf_counter() - start

    return count, elapsed

def benchmark_tinygrad_random_sample(batch_size: int = 64, max_images: int = 10000,
                                      num_steps: int = 156) -> float:
    """Benchmark tinygrad-style random sampling (if tinygrad available)."""
    try:
        from tinygrad import Tensor
        from tinygrad.nn.datasets import mnist

        X_train, Y_train, _, _ = mnist()
        X_train = X_train[:max_images]
        Y_train = Y_train[:max_images]

        start = time.perf_counter()
        for _ in range(num_steps):
            samples = Tensor.randint(batch_size, high=X_train.shape[0])
            _ = X_train[samples]
            _ = Y_train[samples]
        elapsed = time.perf_counter() - start

        return num_steps, elapsed
    except ImportError:
        return 0, 0.0

def benchmark_raw_bytes(batch_size: int = 64, max_images: int = 10000) -> tuple[int, float]:
    """Benchmark raw byte access matching Lean zero-copy benchmark."""
    train_images_path = DATA_DIR / "train-images-idx3-ubyte"
    with open(train_images_path, 'rb') as f:
        f.read(16)  # skip header
        data = f.read(max_images * 784)

    num_batches = len(data) // (batch_size * 784)
    bytes_per_batch = batch_size * 784

    # Warm up
    for i in range(num_batches):
        start = i * bytes_per_batch
        _ = data[start]
        _ = data[start + bytes_per_batch - 1]

    # Timed run
    start_time = time.perf_counter()
    total_bytes = 0
    for i in range(num_batches):
        start = i * bytes_per_batch
        # Access first and last byte (matches Lean benchmark)
        _ = data[start]
        _ = data[start + bytes_per_batch - 1]
        total_bytes += bytes_per_batch
    elapsed = time.perf_counter() - start_time

    return num_batches, elapsed, total_bytes

def main():
    print("=== Python MNIST Loader Benchmark ===")
    print()

    # Load data
    print("Loading MNIST data...")
    train_images_path = DATA_DIR / "train-images-idx3-ubyte"
    train_labels_path = DATA_DIR / "train-labels-idx1-ubyte"

    if not train_images_path.exists():
        print(f"Error: {train_images_path} not found")
        return

    images = load_mnist_images(train_images_path)
    labels = load_mnist_labels(train_labels_path)
    print(f"Loaded {len(images)} images")
    print()

    batch_size = 64
    max_images = 10000

    # Benchmark raw byte access (fair comparison with Lean)
    count, elapsed, total_bytes = benchmark_raw_bytes(batch_size, max_images)
    print(f"Raw bytes (zero-copy): {count} batches in {elapsed*1000:.4f} ms")
    print(f"  Throughput: {count/elapsed:,.0f} batches/sec")
    print(f"  Data touched: {total_bytes} bytes")
    print()

    # Benchmark numpy with shuffle
    count, elapsed = benchmark_numpy_loader(images, labels, batch_size, max_images)
    print(f"Numpy loader (shuffled): {count} batches in {elapsed*1000:.2f} ms")
    print(f"  Throughput: {count/elapsed:,.0f} batches/sec")
    print()

    # Benchmark numpy sequential
    count, elapsed = benchmark_numpy_sequential(images, labels, batch_size, max_images)
    print(f"Numpy loader (sequential): {count} batches in {elapsed*1000:.2f} ms")
    print(f"  Throughput: {count/elapsed:,.0f} batches/sec")
    print()

    # Benchmark tinygrad random sampling
    count, elapsed = benchmark_tinygrad_random_sample(batch_size, max_images)
    if count > 0:
        print(f"Tinygrad random sample: {count} batches in {elapsed*1000:.2f} ms")
        print(f"  Throughput: {count/elapsed:,.0f} batches/sec")
    else:
        print("Tinygrad not available for benchmark")
    print()

    print("=== Lean Zero-Copy Results (for comparison) ===")
    print("Zero-copy batch:     30,000,000 batches/sec")
    print("Copy-based batch:     1,200,000 batches/sec")
    print("Dataset per-sample:  16,600,000 samples/sec")

if __name__ == "__main__":
    main()
