#!/usr/bin/env python3
"""
Compute checksum over MNIST batches for cross-validation with Lean implementation.

The Lean benchmark computes:
  checksum = sum of (first_byte + last_byte) for each batch
where batch is batch_size * 784 bytes of image data.
"""

import struct
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"

def load_mnist_images_raw(path: Path, max_images: int = 10000) -> bytes:
    """Load MNIST images as raw bytes (no normalization)."""
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Invalid magic: {magic}"
        assert rows == 28 and cols == 28
        num_to_read = min(num, max_images)
        return f.read(num_to_read * 784)

def compute_checksum(data: bytes, batch_size: int = 64) -> int:
    """Compute checksum matching Lean implementation."""
    pixels_per_image = 784
    bytes_per_batch = batch_size * pixels_per_image
    num_batches = len(data) // bytes_per_batch

    checksum = 0
    for i in range(num_batches):
        start = i * bytes_per_batch
        end = start + bytes_per_batch
        batch = data[start:end]
        # Match Lean: first byte + last byte
        b0 = batch[0]
        b1 = batch[-1]
        checksum += b0 + b1

    return checksum

def compute_checksum_robust(data: bytes, batch_size: int = 64) -> int:
    """Compute more robust checksum sampling multiple positions."""
    pixels_per_image = 784
    bytes_per_batch = batch_size * pixels_per_image
    num_batches = len(data) // bytes_per_batch

    checksum = 0
    for i in range(num_batches):
        start = i * bytes_per_batch
        # Sample positions: 0, 100, 200, 300, 400, 500
        for offset in [0, 100, 200, 300, 400, 500]:
            if start + offset < len(data):
                checksum += data[start + offset]

    return checksum

def main():
    print("=== MNIST Checksum Cross-Validation ===")
    print()

    train_images_path = DATA_DIR / "train-images-idx3-ubyte"
    if not train_images_path.exists():
        print(f"Error: {train_images_path} not found")
        return

    max_images = 10000
    batch_size = 64

    print(f"Loading {max_images} images from {train_images_path}")
    data = load_mnist_images_raw(train_images_path, max_images)
    print(f"Loaded {len(data)} bytes ({len(data) // 784} images)")

    num_batches = len(data) // (batch_size * 784)
    print(f"Batches: {num_batches} x {batch_size}")
    print()

    checksum_simple = compute_checksum(data, batch_size)
    checksum_robust = compute_checksum_robust(data, batch_size)
    print(f"Python checksum (simple): {checksum_simple}")
    print(f"Python checksum (robust): {checksum_robust}")
    print()

    # Print specific byte values for verification
    # These are interior pixels that should be non-zero
    print("Sample bytes at various positions:")
    for pos in [400, 1000, 5000, 10000, 50000]:
        if pos < len(data):
            print(f"  data[{pos}] = {data[pos]}")

    print()
    print("First image center (should have digit pixels):")
    # Center of first 28x28 image: row 14, col 14 = 14*28 + 14 = 406
    print(f"  data[406] = {data[406]}")

    print()
    print("Compare with Lean output - robust checksum should match!")

if __name__ == "__main__":
    main()
