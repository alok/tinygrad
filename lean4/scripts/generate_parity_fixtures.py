#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinygrad import Tensor


def _flatten(x: Tensor) -> list[float]:
  return [float(v) for v in x.numpy().reshape(-1).tolist()]


def build_fixtures() -> dict:
  Tensor.manual_seed(0)

  x = Tensor.arange(3)
  y = Tensor.linspace(3.0, 6.0, 4)
  mg_ij = x.meshgrid(y, indexing="ij")
  mg_xy = x.meshgrid(y, indexing="xy")

  base = Tensor.arange(6).reshape(2, 3) + 1
  vec = Tensor.arange(6) + 1
  shifted = Tensor.arange(6).float() - 2.5
  logits = Tensor.arange(6).float().reshape(2, 3)

  cases = [
    {
      "id": "eye_3_5",
      "python_ref": "test/test_ops.py::test_eye",
      "shape": [3, 5],
      "expected": _flatten(Tensor.eye(3, 5)),
    },
    {
      "id": "meshgrid_ij_x",
      "python_ref": "test/test_ops.py::test_meshgrid",
      "shape": [3, 4],
      "expected": _flatten(mg_ij[0]),
    },
    {
      "id": "meshgrid_ij_y",
      "python_ref": "test/test_ops.py::test_meshgrid",
      "shape": [3, 4],
      "expected": _flatten(mg_ij[1]),
    },
    {
      "id": "meshgrid_xy_x",
      "python_ref": "test/test_ops.py::test_meshgrid",
      "shape": [4, 3],
      "expected": _flatten(mg_xy[0]),
    },
    {
      "id": "meshgrid_xy_y",
      "python_ref": "test/test_ops.py::test_meshgrid",
      "shape": [4, 3],
      "expected": _flatten(mg_xy[1]),
    },
    {
      "id": "cumsum_axis1_2x3",
      "python_ref": "test/test_ops.py::test_cumsum",
      "shape": [2, 3],
      "expected": _flatten(base.cumsum(axis=1)),
    },
    {
      "id": "cumprod_axis1_2x3",
      "python_ref": "test/test_ops.py::test_cumprod",
      "shape": [2, 3],
      "expected": _flatten(base.cumprod(axis=1)),
    },
    {
      "id": "logsumexp_1d_6",
      "python_ref": "test/test_ops.py::test_logsumexp",
      "shape": [],
      "expected": _flatten(vec.logsumexp()),
    },
    {
      "id": "logcumsumexp_1d_6",
      "python_ref": "test/test_ops.py::test_logcumsumexp",
      "shape": [6],
      "expected": _flatten(vec.logcumsumexp()),
    },
    {
      "id": "linspace_5_10_3",
      "python_ref": "test/test_ops.py::test_linspace",
      "shape": [3],
      "expected": _flatten(Tensor.linspace(5.0, 10.0, 3)),
    },
    {
      "id": "relu_1d_6_shift25",
      "python_ref": "test/test_ops.py::test_relu",
      "shape": [6],
      "expected": _flatten(shifted.relu()),
    },
    {
      "id": "tanh_1d_6_shift25",
      "python_ref": "test/test_ops.py::test_tanh",
      "shape": [6],
      "expected": _flatten(shifted.tanh()),
    },
    {
      "id": "silu_1d_6_shift25",
      "python_ref": "test/test_ops.py::test_silu",
      "shape": [6],
      "expected": _flatten(shifted.silu()),
    },
    {
      "id": "gelu_1d_6_shift25",
      "python_ref": "test/test_ops.py::test_gelu",
      "shape": [6],
      "expected": _flatten(shifted.gelu()),
    },
    {
      "id": "max_1d_6_shift25",
      "python_ref": "test/test_ops.py::test_max",
      "shape": [],
      "expected": _flatten(shifted.max()),
    },
    {
      "id": "min_1d_6_shift25",
      "python_ref": "test/test_ops.py::test_min",
      "shape": [],
      "expected": _flatten(shifted.min()),
    },
    {
      "id": "softmax_last_2x3",
      "python_ref": "test/test_ops.py::test_softmax",
      "shape": [2, 3],
      "expected": _flatten(logits.softmax()),
    },
    {
      "id": "logsoftmax_last_2x3",
      "python_ref": "test/test_ops.py::test_log_softmax",
      "shape": [2, 3],
      "expected": _flatten(logits.log_softmax()),
    },
    {
      "id": "softmax_axis0_2x3",
      "python_ref": "test/test_ops.py::test_softmax_other_axis",
      "shape": [2, 3],
      "expected": _flatten(logits.softmax(axis=0)),
    },
  ]

  return {
    "version": 1,
    "generator": "lean4/scripts/generate_parity_fixtures.py",
    "cases": cases,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Generate TinyGrad Python parity fixtures for Lean tests.")
  parser.add_argument("--out", type=Path, default=Path("lean4/testdata/parity/core_ops.json"))
  args = parser.parse_args()

  payload = build_fixtures()
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(f"Wrote {args.out}")


if __name__ == "__main__":
  main()
