#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
Python micro-bench for indexSelect-style gather + cast + sum.
Matches LeanBench defaults (256 images, batch 32) and raw MNIST bytes.
"""

import os
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_mnist_images(max_images: int) -> np.ndarray:
    path = DATA_DIR / "train-images-idx3-ubyte"
    with open(path, "rb") as f:
        f.read(16)  # skip header
        raw = f.read(max_images * 784)
    data = np.frombuffer(raw, dtype=np.uint8)
    return data.reshape(-1, 784)


def bench_numpy(images: np.ndarray, batch_size: int, seed: int) -> tuple[int, float, float]:
    n = len(images)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    batches = n // batch_size

    total = 0.0
    start = time.perf_counter()
    for i in range(batches):
        idx = indices[i * batch_size : (i + 1) * batch_size]
        batch = images[idx]
        total += batch.astype(np.float32).sum()
    elapsed = time.perf_counter() - start
    return batches, elapsed, total


def bench_tinygrad(images: np.ndarray, batch_size: int, seed: int):
    try:
        from tinygrad import Tensor, dtypes
    except Exception:
        return None

    n = len(images)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    batches = n // batch_size

    X = Tensor(images, dtype=dtypes.uint8)
    total = 0.0
    start = time.perf_counter()
    for i in range(batches):
        idx = indices[i * batch_size : (i + 1) * batch_size]
        idx_t = Tensor(idx.astype(np.int32), dtype=dtypes.int32)
        gathered = X[idx_t]
        total += gathered.cast(dtypes.float32).sum().numpy().item()
    elapsed = time.perf_counter() - start
    return batches, elapsed, total


def main() -> None:
    max_images = int(os.getenv("TG4_INDEXSELECT_MAX_IMAGES", "256"))
    batch_size = int(os.getenv("TG4_INDEXSELECT_BATCH", "32"))
    seed = int(os.getenv("TG4_BENCH_SEED", "42"))

    images = load_mnist_images(max_images)

    batches, elapsed, total = bench_numpy(images, batch_size, seed)
    print("=== Python indexSelect micro-bench ===")
    print(f"config: max_images={max_images} batch={batch_size} seed={seed}")
    print(f"numpy: batches={batches} time_s={elapsed:.4f} items_per_s={batches/elapsed:.6f} total={total:.2f}")

    if os.getenv("TG4_PY_SKIP_TINYGRAD", "0") in {"1", "true", "yes"}:
        return

    tg = bench_tinygrad(images, batch_size, seed)
    if tg is None:
        print("tinygrad: not available")
    else:
        batches, elapsed, total = tg
        print(f"tinygrad: batches={batches} time_s={elapsed:.4f} items_per_s={batches/elapsed:.6f} total={total:.2f}")


if __name__ == "__main__":
    main()
