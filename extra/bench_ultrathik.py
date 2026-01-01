#!/usr/bin/env python3
import argparse, csv, functools, json, subprocess, time
from pathlib import Path

from tinygrad import Device, GlobalCounters, Tensor, TinyJit
from tinygrad.dtype import to_dtype
from tinygrad.helpers import getenv
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters
from tinygrad.engine.realize import run_schedule

from examples.mlperf.initializers import Conv2dHeNormal, Linear
from extra.models import resnet
from test.test_softmax_fusion import single_kernel_softmax

# ultrathik: "real" benches that reuse existing tinygrad models/kernels (no toy ops)

REPO_ROOT = Path(__file__).resolve().parents[1]

def _sync(out=None):
  if out is not None:
    Device[out.device].synchronize()
  else:
    Device[Device.DEFAULT].synchronize()

def _best_of(label, iters, warmup, fn, jitcnt=1):
  for _ in range(warmup):
    fn()
    _sync()

  best = None
  best_stats = None
  for _ in range(iters):
    GlobalCounters.reset()
    st = time.perf_counter()
    out = fn()
    _sync(out)
    et = time.perf_counter()
    tm = (et - st) / max(jitcnt, 1)
    flops = GlobalCounters.global_ops / max(jitcnt, 1)
    mem = GlobalCounters.global_mem / max(jitcnt, 1)
    kernels = GlobalCounters.kernel_count // max(jitcnt, 1)
    if best is None or tm < best:
      best = tm
      best_stats = (flops, mem, kernels)

  if best is None:
    print(f"{label}: no results")
    return None

  flops, mem, kernels = best_stats
  tflops = flops/1e12/best if best > 0 else 0.0
  gbps = mem/1e9/best if best > 0 else 0.0
  print(f"{label}: {best*1e3:8.2f} ms  {tflops:6.2f} tflops  {gbps:6.1f} GB/s  {kernels:5d} kernels")
  return {
    "name": label,
    "time_ms": best*1e3,
    "tflops": tflops,
    "gbps": gbps,
    "kernels": kernels,
    "device": Device.DEFAULT,
  }

def _git(*args):
  try:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
  except Exception:
    return ""

def _run_meta():
  return {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "git_sha": _git("rev-parse", "--short", "HEAD"),
    "git_branch": _git("branch", "--show-current"),
  }

def _write_json(path, meta, results):
  out = {"meta": meta, "results": results}
  p = Path(path).expanduser()
  p.parent.mkdir(parents=True, exist_ok=True)
  with p.open("w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

def _write_csv(path, meta, results):
  rows = [{**meta, **r} for r in results]
  if not rows: return
  p = Path(path).expanduser()
  p.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = sorted({k for row in rows for k in row.keys()})
  write_header = not p.exists()
  with p.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header: writer.writeheader()
    writer.writerows(rows)

def bench_resnet50():
  bs = getenv("RESNET_BS", getenv("BS", 16))
  iters = getenv("RESNET_ITERS", getenv("ITERS", 5))
  warmup = getenv("RESNET_WARMUP", getenv("WARMUP", 2))
  jitcnt = getenv("JITCNT", 1)
  use_assign = getenv("ASSIGN", 1)

  # Match external_benchmark_resnet: init and (optionally) UnsyncedBatchNorm.
  resnet.Conv2d = Conv2dHeNormal
  resnet.Linear = Linear
  if not getenv("SYNCBN"):
    import os
    os.environ.setdefault("BS", str(bs))
    from examples.hlb_cifar10 import UnsyncedBatchNorm
    resnet.BatchNorm = functools.partial(UnsyncedBatchNorm, num_devices=1)

  model = resnet.ResNet50()
  optim = SGD(get_parameters(model), bs / 128 * 1.0)
  Tensor.training = True

  @TinyJit
  def step(x):
    optim.zero_grad()
    x.grad = None
    y = model.forward(x).mean()
    y.backward()
    if use_assign:
      sched, _ = Tensor.schedule_with_vars(y, x.grad, *optim.schedule_step())
    else:
      sched, _ = Tensor.schedule_with_vars(y, x.grad, *[t.grad for t in optim.params])
    for _ in range(jitcnt):
      run_schedule(list(sched))

  x = Tensor.randn(bs, 3, 224, 224, requires_grad=True).realize()
  print(f"ultrathik/resnet50: bs={bs} iters={iters} warmup={warmup} jitcnt={jitcnt} device={Device.DEFAULT}")
  res = _best_of("resnet50_train_step", iters, warmup, lambda: step(x), jitcnt=jitcnt)
  if res is None: return None
  res.update({"bs": bs, "iters": iters, "warmup": warmup, "jitcnt": jitcnt})
  return res

def bench_bert_attention():
  bs = getenv("BERT_BS", getenv("BS", 16))
  seq = getenv("BERT_SEQ", 512)
  heads = getenv("BERT_HEADS", 16)
  head_dim = getenv("BERT_HEAD_DIM", 64)
  iters = getenv("BERT_ITERS", getenv("ITERS", 10))
  warmup = getenv("BERT_WARMUP", getenv("WARMUP", 3))
  use_fused = getenv("FUSED_SOFTMAX", 1)
  acc_dtype = to_dtype(getenv("ACC_DTYPE", "half"))

  q = Tensor.empty(bs, heads, seq, head_dim)
  k = Tensor.empty(bs, heads, seq, head_dim)
  v = Tensor.empty(bs, heads, seq, head_dim)

  @TinyJit
  def attn(q, k, v):
    scores = q @ k.transpose(-1, -2)
    if use_fused:
      probs = single_kernel_softmax(scores, -1, acc_dtype)
    else:
      probs = scores.softmax(-1, dtype=acc_dtype)
    return probs @ v

  print(f"ultrathik/bert_attention: bs={bs} seq={seq} heads={heads} head_dim={head_dim} "
        f"iters={iters} warmup={warmup} fused={use_fused} device={Device.DEFAULT}")
  res = _best_of("bert_attention", iters, warmup, lambda: attn(q, k, v))
  if res is None: return None
  res.update({
    "bs": bs,
    "seq": seq,
    "heads": heads,
    "head_dim": head_dim,
    "iters": iters,
    "warmup": warmup,
    "fused": use_fused,
  })
  return res

def main():
  parser = argparse.ArgumentParser(description="ultrathik tinygrad benches (real models, no toy ops)")
  parser.add_argument("--bench", action="append", choices=["resnet", "bert"], help="bench to run (can repeat)")
  parser.add_argument("--device", help="override Device.DEFAULT (e.g. METAL, CUDA, CPU)")
  parser.add_argument("--json", dest="json_path", help="write results to JSON file")
  parser.add_argument("--csv", dest="csv_path", help="append results to CSV file")
  args = parser.parse_args()

  if args.device: Device.DEFAULT = args.device
  benches = args.bench or ["resnet", "bert"]
  results = []
  if "resnet" in benches:
    if (r := bench_resnet50()) is not None: results.append(r)
  if "bert" in benches:
    if (r := bench_bert_attention()) is not None: results.append(r)
  meta = _run_meta()
  if args.json_path: _write_json(args.json_path, meta, results)
  if args.csv_path: _write_csv(args.csv_path, meta, results)

if __name__ == "__main__":
  main()
