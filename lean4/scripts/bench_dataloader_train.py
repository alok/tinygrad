#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "psutil"]
# ///
"""End-to-end dataloader + training benchmark.

Measures data wait time + GPU compute time for raw mmap, torch DataLoader, and Grain.
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
class TrainResult:
    name: str
    steps: int
    total_bytes: int
    wall_s: float
    data_wait_s: list[float]
    h2d_s: list[float]
    step_s: list[float]
    loop_s: list[float]
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


def iter_mmap(path: str, chunk_bytes: int, total_bytes: int) -> Iterable[memoryview]:
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), length=total_bytes, access=mmap.ACCESS_READ)
        try:
            offset = 0
            while True:
                end = min(offset + chunk_bytes, total_bytes)
                yield memoryview(mm)[offset:end]
                offset = 0 if end == total_bytes else end
        finally:
            mm.close()


class MMapDataset:
    def __init__(self, path: str, chunk_bytes: int, total_bytes: int, return_bytes: bool) -> None:
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
        self.return_bytes = state["return_bytes"]
        self._open()

    def __len__(self) -> int:
        return self.total_bytes // self.chunk_bytes

    def __getitem__(self, idx: int):
        start = idx * self.chunk_bytes
        end = min(start + self.chunk_bytes, self.total_bytes)
        if self.return_bytes:
            return self._mm[start:end]
        return memoryview(self._mm)[start:end]


def make_torch_loader(
    path: str,
    chunk_bytes: int,
    total_bytes: int,
    num_workers: int,
    prefetch: int,
):
    try:
        from torch.utils.data import DataLoader
    except Exception as e:
        print(f"torch import failed: {e}", file=sys.stderr)
        return None

    ds = MMapDataset(path, chunk_bytes, total_bytes, return_bytes=(num_workers > 0))
    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetch if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=lambda x: x,
    )
    return loader


def make_grain_loader(
    path: str,
    chunk_bytes: int,
    total_bytes: int,
    worker_count: int,
    worker_buffer_size: int,
    read_threads: Optional[int],
    read_buffer_size: Optional[int],
):
    try:
        sys.path.append(os.path.expanduser("~/grain"))
        import grain  # type: ignore
    except Exception as e:
        print(f"grain import failed: {e}", file=sys.stderr)
        return None

    ds = MMapDataset(path, chunk_bytes, total_bytes, return_bytes=(worker_count > 0))
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
        shuffle=False,
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
    return loader


def _touch_checksum(buf: memoryview) -> int:
    if not buf:
        return 0
    return buf[0] + (buf[-1] << 8)


def train_loop(
    name: str,
    batches: Iterable[memoryview],
    steps: int,
    device: str,
    feature_bytes: int,
    hidden: int,
    out_dim: int,
    normalize: bool,
) -> TrainResult:
    import torch

    torch.manual_seed(0)

    model = torch.nn.Sequential(
        torch.nn.Linear(feature_bytes, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, out_dim),
    ).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    target = torch.zeros((1, out_dim), device=device)

    proc = psutil.Process()
    sampler = CPUSampler(proc)
    sampler.start()

    data_wait_s: list[float] = []
    h2d_s: list[float] = []
    step_s: list[float] = []
    loop_s: list[float] = []
    checksum = 0.0
    total_bytes = 0

    it = iter(batches)
    t_start = time.perf_counter()
    for _ in range(steps):
        t0 = time.perf_counter()
        raw = next(it)
        t1 = time.perf_counter()
        data_wait_s.append(t1 - t0)

        buf = raw if isinstance(raw, memoryview) else memoryview(raw)
        feature_len = min(feature_bytes, len(buf))
        arr = np.frombuffer(buf, dtype=np.uint8, count=feature_len)
        if normalize:
            arr = (arr.astype(np.float32) / 255.0)
        else:
            checksum += _touch_checksum(buf)
            arr = arr.astype(np.float32)

        t2 = time.perf_counter()
        x = torch.from_numpy(arr).to(device, non_blocking=True).reshape(1, -1)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        h2d_s.append(t3 - t2)

        t4 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        y = model(x)
        loss = (y - target).pow(2).mean()
        loss.backward()
        opt.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t5 = time.perf_counter()
        step_s.append(t5 - t4)
        loop_s.append(t5 - t0)

        total_bytes += len(buf)
        if isinstance(raw, memoryview):
            raw.release()

    wall = time.perf_counter() - t_start

    sampler.stop()
    sampler.join(timeout=1.0)
    cpu_pct = sampler.mean()
    rss = proc.memory_info().rss

    return TrainResult(
        name,
        steps,
        total_bytes,
        wall,
        data_wait_s,
        h2d_s,
        step_s,
        loop_s,
        cpu_pct,
        rss,
        checksum,
    )


def print_result(r: TrainResult) -> None:
    gb = r.total_bytes / (1024 ** 3)
    steps_per_s = r.steps / r.wall_s if r.wall_s else 0.0
    gbps = gb / r.wall_s if r.wall_s else 0.0
    total_wait = sum(r.data_wait_s)
    total_h2d = sum(r.h2d_s)
    total_step = sum(r.step_s)
    total_loop = sum(r.loop_s)
    stall_frac = total_wait / total_loop if total_loop else 0.0
    compute_frac = total_step / total_loop if total_loop else 0.0
    h2d_frac = total_h2d / total_loop if total_loop else 0.0

    print(f"== {r.name} ==")
    print(f"steps: {r.steps}  wall: {r.wall_s:.3f}s  steps/s: {steps_per_s:.2f}  GB/s: {gbps:.2f}")
    print(
        "data_wait p50/p90/p99 (ms): "
        f"{pct(r.data_wait_s, 50)*1e3:.2f} / {pct(r.data_wait_s, 90)*1e3:.2f} / {pct(r.data_wait_s, 99)*1e3:.2f}"
    )
    print(
        "h2d p50/p90/p99 (ms): "
        f"{pct(r.h2d_s, 50)*1e3:.2f} / {pct(r.h2d_s, 90)*1e3:.2f} / {pct(r.h2d_s, 99)*1e3:.2f}"
    )
    print(
        "step p50/p90/p99 (ms): "
        f"{pct(r.step_s, 50)*1e3:.2f} / {pct(r.step_s, 90)*1e3:.2f} / {pct(r.step_s, 99)*1e3:.2f}"
    )
    print(
        "loop p50/p90/p99 (ms): "
        f"{pct(r.loop_s, 50)*1e3:.2f} / {pct(r.loop_s, 90)*1e3:.2f} / {pct(r.loop_s, 99)*1e3:.2f}"
    )
    print(
        "util: wait/step/h2d (%): "
        f"{stall_frac*100:.1f} / {compute_frac*100:.1f} / {h2d_frac*100:.1f}"
    )
    print(f"cpu%: {r.cpu_pct:.1f}  rss: {r.rss_bytes/1e9:.2f} GB  checksum: {r.checksum:.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/raw_40gb.bin")
    ap.add_argument("--total-gb", type=float, default=40.0)
    ap.add_argument("--chunk-mb", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--loader", choices=["raw", "torch", "grain"], default="torch")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--grain-workers", type=int, default=4)
    ap.add_argument("--grain-buffer", type=int, default=4)
    ap.add_argument("--grain-read-threads", type=int, default=None)
    ap.add_argument("--grain-read-buffer", type=int, default=None)
    ap.add_argument("--feature-kb", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out-dim", type=int, default=10)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    chunk_bytes = int(args.chunk_mb * 1024 * 1024)
    total_bytes = int(args.total_gb * 1024 ** 3)
    feature_bytes = args.feature_kb * 1024

    if not os.path.exists(args.path):
        print(f"missing data file: {args.path}", file=sys.stderr)
        return 2

    try:
        import torch
    except Exception as e:
        print(f"torch import failed: {e}", file=sys.stderr)
        return 2

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.loader == "raw":
        batches = iter_mmap(args.path, chunk_bytes, total_bytes)
        r = train_loop(
            name="raw-mmap",
            batches=batches,
            steps=args.steps,
            device=device,
            feature_bytes=feature_bytes,
            hidden=args.hidden,
            out_dim=args.out_dim,
            normalize=args.normalize,
        )
        print_result(r)
        return 0

    if args.loader == "torch":
        loader = make_torch_loader(
            args.path,
            chunk_bytes,
            total_bytes,
            num_workers=args.num_workers,
            prefetch=args.prefetch,
        )
        if loader is None:
            return 2
        r = train_loop(
            name=f"torch(w={args.num_workers},prefetch={args.prefetch})",
            batches=iter(loader),
            steps=args.steps,
            device=device,
            feature_bytes=feature_bytes,
            hidden=args.hidden,
            out_dim=args.out_dim,
            normalize=args.normalize,
        )
        print_result(r)
        return 0

    loader = make_grain_loader(
        args.path,
        chunk_bytes,
        total_bytes,
        worker_count=args.grain_workers,
        worker_buffer_size=args.grain_buffer,
        read_threads=args.grain_read_threads,
        read_buffer_size=args.grain_read_buffer,
    )
    if loader is None:
        return 2
    r = train_loop(
        name=f"grain(w={args.grain_workers},buf={args.grain_buffer})",
        batches=iter(loader),
        steps=args.steps,
        device=device,
        feature_bytes=feature_bytes,
        hidden=args.hidden,
        out_dim=args.out_dim,
        normalize=args.normalize,
    )
    print_result(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
