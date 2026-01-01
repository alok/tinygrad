#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "psutil"]
# ///
"""Big raw data loader benchmark (GB-scale).

Compares:
- raw mmap sequential read
- Grain DataLoader (if available)
- PyTorch DataLoader (if available)
- Ray Data (if available)

Metrics: throughput, latency p50/p90/p99, CPU%, RSS.
"""

from __future__ import annotations

import argparse
import dataclasses
import mmap
import os
import sys
import threading
import time
from typing import Iterable, Optional

import numpy as np
import psutil


@dataclasses.dataclass
class BenchResult:
    name: str
    total_bytes: int
    wall_s: float
    batch_times_s: list[float]
    cpu_pct: float
    rss_bytes: int
    checksum: float


class CPUSampler(threading.Thread):
    def __init__(self, proc: psutil.Process, interval: float = 0.2) -> None:
        super().__init__(daemon=True)
        self.proc = proc
        self.interval = interval
        self._stop_event = threading.Event()
        self.samples: list[float] = []

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.samples.append(self.proc.cpu_percent(interval=self.interval))
            except psutil.Error:
                return

    def stop(self) -> None:
        self._stop_event.set()

    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0


def pct(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    idx = int((p / 100.0) * (len(s) - 1))
    return s[idx]


def _touch_checksum(buf: memoryview) -> int:
    if not buf:
        return 0
    return buf[0] + (buf[-1] << 8)


def checksum_bytes(buf: memoryview) -> int:
    arr = np.frombuffer(buf, dtype=np.uint8)
    return int(arr.sum())


def normalize_sum(buf: memoryview) -> float:
    arr = np.frombuffer(buf, dtype=np.uint8)
    return float((arr.astype(np.float32) / 255.0).sum())


def run_loop(
    name: str,
    batches: Iterable[memoryview],
    mode: str,
) -> BenchResult:
    proc = psutil.Process()
    sampler = CPUSampler(proc)
    sampler.start()

    batch_times: list[float] = []
    checksum = 0.0
    total_bytes = 0

    t_start = time.perf_counter()
    for raw in batches:
        buf = raw if isinstance(raw, memoryview) else memoryview(raw)
        t0 = time.perf_counter()
        if mode == "normalize":
            checksum += normalize_sum(buf)
        elif mode == "noop":
            checksum += _touch_checksum(buf)
        else:
            checksum += checksum_bytes(buf)
        t1 = time.perf_counter()
        batch_times.append(t1 - t0)
        total_bytes += len(buf)
        if isinstance(raw, memoryview):
            raw.release()
    wall = time.perf_counter() - t_start

    sampler.stop()
    sampler.join(timeout=1.0)
    cpu_pct = sampler.mean()
    rss = proc.memory_info().rss

    return BenchResult(name, total_bytes, wall, batch_times, cpu_pct, rss, checksum)


def run_loop_jax(
    name: str,
    batches: Iterable[memoryview],
    mode: str,
    device,
) -> BenchResult:
    import jax
    import jax.numpy as jnp

    if mode == "normalize":
        process = jax.jit(lambda x: jnp.sum(x.astype(jnp.float32) / 255.0), device=device)
    elif mode == "noop":
        process = jax.jit(lambda x: x[0].astype(jnp.float32) + x[-1].astype(jnp.float32), device=device)
    else:
        process = jax.jit(lambda x: jnp.sum(x.astype(jnp.float32)), device=device)

    proc = psutil.Process()
    sampler = CPUSampler(proc)
    sampler.start()

    batch_times: list[float] = []
    checksum = 0.0
    total_bytes = 0

    t_start = time.perf_counter()
    for raw in batches:
        buf = raw if isinstance(raw, memoryview) else memoryview(raw)
        t0 = time.perf_counter()
        host_arr = np.frombuffer(buf, dtype=np.uint8)
        x = jax.device_put(host_arr, device)
        y = process(x)
        y.block_until_ready()
        checksum += float(np.asarray(y))
        t1 = time.perf_counter()
        batch_times.append(t1 - t0)
        total_bytes += len(buf)
        if isinstance(raw, memoryview):
            raw.release()
    wall = time.perf_counter() - t_start

    sampler.stop()
    sampler.join(timeout=1.0)
    cpu_pct = sampler.mean()
    rss = proc.memory_info().rss

    return BenchResult(name, total_bytes, wall, batch_times, cpu_pct, rss, checksum)


def iter_mmap(path: str, chunk_bytes: int, total_bytes: int, iterations: int) -> Iterable[memoryview]:
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), length=total_bytes, access=mmap.ACCESS_READ)
        try:
            for _ in range(iterations):
                offset = 0
                while offset < total_bytes:
                    end = min(offset + chunk_bytes, total_bytes)
                    yield memoryview(mm)[offset:end]
                    offset = end
        finally:
            mm.close()


class MMapDataSource:
    def __init__(self, path: str, chunk_bytes: int, total_bytes: int, return_bytes: bool = False) -> None:
        self.path = path
        self.chunk_bytes = chunk_bytes
        self.total_bytes = total_bytes
        self.return_bytes = return_bytes
        self._open()

    def _open(self) -> None:
        self._fp = open(self.path, "rb")
        self._mm = mmap.mmap(self._fp.fileno(), length=self.total_bytes, access=mmap.ACCESS_READ)

    def __getstate__(self):
        return {
            "path": self.path,
            "chunk_bytes": self.chunk_bytes,
            "total_bytes": self.total_bytes,
            "return_bytes": self.return_bytes,
        }

    def __setstate__(self, state):
        self.path = state["path"]
        self.chunk_bytes = state["chunk_bytes"]
        self.total_bytes = state["total_bytes"]
        self.return_bytes = state.get("return_bytes", False)
        self._open()

    def __len__(self) -> int:
        return self.total_bytes // self.chunk_bytes

    def __getitem__(self, idx: int) -> memoryview:
        start = idx * self.chunk_bytes
        end = min(start + self.chunk_bytes, self.total_bytes)
        if self.return_bytes:
            return self._mm[start:end]
        return memoryview(self._mm)[start:end]


def bench_grain(
    path: str,
    chunk_bytes: int,
    total_bytes: int,
    mode: str,
    worker_count: int,
    worker_buffer_size: int,
    read_threads: Optional[int],
    read_buffer_size: Optional[int],
    shuffle: bool,
) -> Optional[BenchResult]:
    try:
        sys.path.append(os.path.expanduser("~/grain"))
        import grain  # type: ignore
    except Exception:
        return None

    ds = MMapDataSource(path, chunk_bytes, total_bytes, return_bytes=worker_count > 0)
    read_options = None
    if read_threads is not None or read_buffer_size is not None:
        defaults = grain.ReadOptions()
        read_options = grain.ReadOptions(
            num_threads=read_threads if read_threads is not None else defaults.num_threads,
            prefetch_buffer_size=(
                read_buffer_size if read_buffer_size is not None else defaults.prefetch_buffer_size
            ),
        )

    sampler = grain.samplers.IndexSampler(
        num_records=len(ds),
        shuffle=shuffle,
        seed=42,
        num_epochs=1,
        shard_options=grain.sharding.NoSharding(),
    )
    loader = grain.DataLoader(
        data_source=ds,
        sampler=sampler,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        read_options=read_options,
    )

    def batches() -> Iterable[memoryview]:
        for item in loader:
            yield item

    return run_loop(
        name=f"grain(w={worker_count},buf={worker_buffer_size},shuffle={int(shuffle)})",
        batches=batches(),
        mode=mode,
    )


def bench_torch(
    path: str,
    chunk_bytes: int,
    total_bytes: int,
    mode: str,
    num_workers: int,
    prefetch: int,
) -> Optional[BenchResult]:
    try:
        import torch  # type: ignore
        from torch.utils.data import DataLoader, Dataset  # type: ignore
    except Exception:
        return None

    class FileDataset(Dataset):
        def __init__(self, path: str, chunk_bytes: int, total_bytes: int) -> None:
            self.path = path
            self.chunk_bytes = chunk_bytes
            self.total_bytes = total_bytes
            self._open()

        def _open(self) -> None:
            self._fp = open(self.path, "rb")
            self._mm = mmap.mmap(self._fp.fileno(), length=self.total_bytes, access=mmap.ACCESS_READ)

        def __getstate__(self):
            return {
                "path": self.path,
                "chunk_bytes": self.chunk_bytes,
                "total_bytes": self.total_bytes,
            }

        def __setstate__(self, state):
            self.path = state["path"]
            self.chunk_bytes = state["chunk_bytes"]
            self.total_bytes = state["total_bytes"]
            self._open()

        def __len__(self) -> int:
            return self.total_bytes // self.chunk_bytes

        def __getitem__(self, idx: int) -> memoryview:
            start = idx * self.chunk_bytes
            end = min(start + self.chunk_bytes, self.total_bytes)
            return memoryview(self._mm)[start:end]

    ds = FileDataset(path, chunk_bytes, total_bytes)
    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetch if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    return run_loop(
        name=f"torch(w={num_workers},prefetch={prefetch})",
        batches=iter(loader),
        mode=mode,
    )


def bench_ray(
    path: str,
    chunk_bytes: int,
    total_bytes: int,
    mode: str,
) -> Optional[BenchResult]:
    try:
        import ray  # type: ignore
    except Exception:
        return None
    try:
        import ray.data as rd  # type: ignore
    except Exception:
        return None

    # Ray binary file reader expects a directory; single file still works.
    ds = rd.read_binary_files(path)

    def batches() -> Iterable[memoryview]:
        for row in ds.iter_rows():
            data = row["bytes"]
            mv = memoryview(data)
            offset = 0
            while offset < total_bytes:
                end = min(offset + chunk_bytes, total_bytes)
                yield mv[offset:end]
                offset = end

    return run_loop(name="ray", batches=batches(), mode=mode)


def bench_jax(
    path: str,
    chunk_bytes: int,
    total_bytes: int,
    mode: str,
    device_kind: str,
) -> Optional[BenchResult]:
    try:
        import jax
    except Exception:
        return None

    devices = jax.devices(device_kind)
    if not devices:
        return None
    device = devices[0]

    batches = iter_mmap(path, chunk_bytes, total_bytes, 1)
    return run_loop_jax(
        name=f"jax(device={device_kind})",
        batches=batches,
        mode=mode,
        device=device,
    )


def print_result(r: BenchResult) -> None:
    gb = r.total_bytes / 1e9
    gbps = gb / r.wall_s if r.wall_s > 0 else 0.0
    p50 = pct(r.batch_times_s, 50) * 1000
    p90 = pct(r.batch_times_s, 90) * 1000
    p99 = pct(r.batch_times_s, 99) * 1000
    mean = (sum(r.batch_times_s) / len(r.batch_times_s) * 1000) if r.batch_times_s else 0.0
    print(f"{r.name}")
    print(f"  wall_s={r.wall_s:.3f} gbps={gbps:.3f} batches={len(r.batch_times_s)}")
    print(f"  lat_ms mean={mean:.3f} p50={p50:.3f} p90={p90:.3f} p99={p99:.3f}")
    print(f"  cpu_pct={r.cpu_pct:.1f} rss_gb={r.rss_bytes/1e9:.3f} checksum={r.checksum:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="../data/raw_40gb.bin")
    ap.add_argument("--gb", type=int, default=40)
    ap.add_argument("--chunk-mb", type=int, default=4)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--mode", choices=["checksum", "normalize", "noop"], default="checksum")
    ap.add_argument("--grain", action="store_true")
    ap.add_argument("--torch", action="store_true")
    ap.add_argument("--ray", action="store_true")
    ap.add_argument("--jax", action="store_true")
    ap.add_argument("--jax-device", choices=["cpu", "gpu", "tpu"], default="cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--buffer", type=int, default=4)
    ap.add_argument("--read-threads", type=int, default=None)
    ap.add_argument("--read-buffer", type=int, default=None)
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--create", action="store_true")
    args = ap.parse_args()

    path = os.path.abspath(args.path)
    total_bytes = args.gb * (1024**3)
    chunk_bytes = args.chunk_mb * 1024 * 1024

    if args.create and not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.truncate(total_bytes)
        print(f"Created sparse file {path} size={args.gb}GB")

    if not os.path.exists(path):
        raise SystemExit(f"Missing file: {path}")

    file_size = os.path.getsize(path)
    if total_bytes > file_size:
        total_bytes = file_size

    print("=== big dataloader bench ===")
    print(f"path={path}")
    print(f"total_bytes={total_bytes} chunk_bytes={chunk_bytes} iters={args.iters} mode={args.mode}")

    # Baseline mmap
    batches = iter_mmap(path, chunk_bytes, total_bytes, args.iters)
    result = run_loop("mmap", batches, args.mode)
    print_result(result)

    if args.grain:
        r = bench_grain(
            path,
            chunk_bytes,
            total_bytes,
            args.mode,
            worker_count=args.workers,
            worker_buffer_size=args.buffer,
            read_threads=args.read_threads,
            read_buffer_size=args.read_buffer,
            shuffle=args.shuffle,
        )
        if r is None:
            print("grain not available")
        else:
            print_result(r)

    if args.torch:
        r = bench_torch(
            path,
            chunk_bytes,
            total_bytes,
            args.mode,
            num_workers=args.workers,
            prefetch=args.prefetch,
        )
        if r is None:
            print("torch not available")
        else:
            print_result(r)

    if args.ray:
        r = bench_ray(path, chunk_bytes, total_bytes, args.mode)
        if r is None:
            print("ray not available")
        else:
            print_result(r)

    if args.jax:
        r = bench_jax(path, chunk_bytes, total_bytes, args.mode, args.jax_device)
        if r is None:
            print(f"jax device not available: {args.jax_device}")
        else:
            print_result(r)


if __name__ == "__main__":
    main()
