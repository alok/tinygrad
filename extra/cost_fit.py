#!/usr/bin/env python3
import json
import math
import os
import sys

FEATURES = [
  "feat/launches",
  "feat/mem_read",
  "feat/mem_write",
  "feat/mem_view_read",
  "feat/mem_view_write",
  "feat/elem_ops",
  "feat/move_elems",
  "feat/reduce_elems",
  "feat/matmul_muladds",
  "feat/matmul_view_muladds",
]

WEIGHT_KEYS = [
  "kernelOverhead",
  "memReadByte",
  "memWriteByte",
  "memReadViewByte",
  "memWriteViewByte",
  "elem",
  "moveElem",
  "reduceElem",
  "matmulMulAdd",
  "matmulViewMulAdd",
]


def load_samples(paths):
  samples = []
  for path in paths:
    with open(path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          obj = json.loads(line)
        except json.JSONDecodeError:
          continue
        if "ms" not in obj:
          continue
        x = [float(obj.get(k, 0.0)) for k in FEATURES]
        y = float(obj["ms"])
        samples.append((x, y))
  return samples


def solve_normal_eq(samples, ridge=1.0e-8):
  nfeat = len(FEATURES)
  xtx = [[0.0 for _ in range(nfeat)] for _ in range(nfeat)]
  xty = [0.0 for _ in range(nfeat)]
  for x, y in samples:
    for i in range(nfeat):
      xty[i] += x[i] * y
    for i in range(nfeat):
      xi = x[i]
      for j in range(nfeat):
        xtx[i][j] += xi * x[j]
  for i in range(nfeat):
    xtx[i][i] += ridge
  w = gaussian_solve(xtx, xty)
  return w


def gaussian_solve(a, b):
  n = len(b)
  a = [row[:] for row in a]
  b = b[:]
  for i in range(n):
    pivot = i
    for r in range(i + 1, n):
      if abs(a[r][i]) > abs(a[pivot][i]):
        pivot = r
    if abs(a[pivot][i]) < 1.0e-20:
      continue
    if pivot != i:
      a[i], a[pivot] = a[pivot], a[i]
      b[i], b[pivot] = b[pivot], b[i]
    inv = 1.0 / a[i][i]
    for j in range(i, n):
      a[i][j] *= inv
    b[i] *= inv
    for r in range(n):
      if r == i:
        continue
      factor = a[r][i]
      if factor == 0.0:
        continue
      for j in range(i, n):
        a[r][j] -= factor * a[i][j]
      b[r] -= factor * b[i]
  return b


def fit(samples):
  try:
    import numpy as np  # type: ignore
    x = np.array([s[0] for s in samples], dtype=np.float64)
    y = np.array([s[1] for s in samples], dtype=np.float64)
    w, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return w.tolist()
  except Exception:
    return solve_normal_eq(samples)


def main():
  if len(sys.argv) < 2:
    print("usage: cost_fit.py <samples.jsonl> [more.jsonl...]")
    sys.exit(2)
  samples = load_samples(sys.argv[1:])
  if not samples:
    print("no samples found")
    sys.exit(1)
  scale = float(os.environ.get("COST_FIT_SCALE", "1.0"))
  w = fit(samples)
  out = {}
  for key, val in zip(WEIGHT_KEYS, w):
    if math.isnan(val) or math.isinf(val):
      val = 0.0
    if val < 0.0:
      val = 0.0
    out[key] = int(round(val * scale))
  print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
