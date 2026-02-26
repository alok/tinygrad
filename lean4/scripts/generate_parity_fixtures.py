#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinygrad import Tensor
import numpy as np


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
  vals7 = Tensor.arange(7).float() * 0.3 - 0.9
  vals7b = Tensor.arange(7).float() * 0.5 - 1.5
  base12 = (Tensor.arange(12).float() + 1).reshape(3, 4)
  vec3 = Tensor.linspace(1.0, 3.0, 3).float()
  mat3 = Tensor.arange(9).float().reshape(3, 3)
  idx_take = Tensor([0, 1, 2, 0, 1], dtype="int")
  take_src = Tensor.arange(6).float().reshape(2, 3)
  mfill_src = Tensor.arange(6).float() + 1
  mfill_mask = Tensor([True, False, True, False, False, True])
  scalar42 = Tensor([42.0]).reshape(())
  mag = Tensor([1.0, -2.0, 0.0, -0.0]).float()
  sign_src = Tensor([-1.0, 2.0, -0.0, 0.0]).float()
  la = Tensor([100.0, -100.0, 1.0]).float()
  lb = Tensor([99.0, 100.0, -2.0]).float()
  ms_src = Tensor.arange(9).float().reshape(3, 3)
  ms_mask = Tensor([[True, False, True], [False, True, False], [False, False, True]])
  ms_sel = ms_src.masked_select(ms_mask)
  ms_packed = np.zeros(ms_src.numel(), dtype=np.float32)
  ms_packed[:ms_sel.numel()] = ms_sel.numpy().reshape(-1)
  scatter_base = Tensor.zeros(1, 1, 16).float()
  scatter_idx = Tensor([5, 7, 13, 15], dtype="int").reshape(1, 1, 4)
  scatter_src = Tensor([6.0, 8.0, 14.0, 16.0]).float().reshape(1, 1, 4)
  scatter_ridx = Tensor([5, 5, 5, 2], dtype="int").reshape(1, 1, 4)
  scatter_rsrc = Tensor([1.0, 2.0, 3.0, 4.0]).float().reshape(1, 1, 4)
  scatter_psrc = Tensor([2.0, 3.0, 4.0, 5.0]).float().reshape(1, 1, 4)
  conv_t_x = (Tensor.arange(4).reshape(1, 1, 2, 2) + 1).float()
  conv_t_w = Tensor.ones(1, 1, 2, 2).float()
  pooled_2x2 = Tensor([6.0, 8.0, 14.0, 16.0]).reshape(1, 1, 2, 2).float()
  unpool_idx_2x2 = Tensor([5, 7, 13, 15], dtype="int").reshape(1, 1, 2, 2)
  bn_nc_x = Tensor.arange(6).float().reshape(2, 3)
  bn_nc_mean = Tensor.linspace(1.5, 3.5, 3).float()
  bn_nc_invstd = Tensor.full((3,), 2.0 / 3.0).float()
  bn_nchw_x = Tensor.arange(8).float().reshape(1, 2, 2, 2)
  bn_nchw_mean = Tensor([1.5, 5.5]).float()
  bn_nchw_invstd = Tensor([1.0, 0.5]).float()
  bn_nchw_weight = Tensor([2.0, -1.0]).float()
  bn_nchw_bias = Tensor([0.5, 1.0]).float()
  drop_src = Tensor.ones(8).float()
  with Tensor.train():
    drop_p1 = drop_src.dropout(1.0)

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
    {
      "id": "round_1d_6_shift25",
      "python_ref": "test/test_ops.py::test_round",
      "shape": [6],
      "expected": _flatten(shifted.round()),
    },
    {
      "id": "sign_1d_6_shift25",
      "python_ref": "test/test_ops.py::test_sign",
      "shape": [6],
      "expected": _flatten(shifted.sign()),
    },
    {
      "id": "lerp_scalar_1d_3",
      "python_ref": "test/test_ops.py::test_lerp",
      "shape": [3],
      "expected": _flatten(Tensor([1.0, 2.0, 3.0]).lerp(Tensor([4.0, 5.0, 6.0]), 0.5)),
    },
    {
      "id": "asin_vals7",
      "python_ref": "test/test_ops.py::test_asin",
      "shape": [7],
      "expected": _flatten(vals7.asin()),
    },
    {
      "id": "acos_vals7",
      "python_ref": "test/test_ops.py::test_acos",
      "shape": [7],
      "expected": _flatten(vals7.acos()),
    },
    {
      "id": "atan_1d_6_shift25",
      "python_ref": "test/test_ops.py::test_atan",
      "shape": [6],
      "expected": _flatten(shifted.atan()),
    },
    {
      "id": "sinh_vals7",
      "python_ref": "test/test_ops.py::test_sinh",
      "shape": [7],
      "expected": _flatten(vals7b.sinh()),
    },
    {
      "id": "cosh_vals7",
      "python_ref": "test/test_ops.py::test_cosh",
      "shape": [7],
      "expected": _flatten(vals7b.cosh()),
    },
    {
      "id": "erf_vals7",
      "python_ref": "test/test_ops.py::test_erf",
      "shape": [7],
      "expected": _flatten(vals7b.erf()),
    },
    {
      "id": "softsign_1d_6_shift25",
      "python_ref": "test/test_ops.py::test_softsign",
      "shape": [6],
      "expected": _flatten(shifted.softsign()),
    },
    {
      "id": "mish_1d_6_shift25",
      "python_ref": "test/test_ops.py::test_mish",
      "shape": [6],
      "expected": _flatten(shifted.mish()),
    },
    {
      "id": "celu_1d_6_shift25",
      "python_ref": "test/test_ops.py::test_celu",
      "shape": [6],
      "expected": _flatten(shifted.celu()),
    },
    {
      "id": "selu_1d_6_shift25",
      "python_ref": "test/test_ops.py::test_selu",
      "shape": [6],
      "expected": _flatten(shifted.selu()),
    },
    {
      "id": "copysign_1d_4",
      "python_ref": "test/test_ops.py::test_copysign",
      "shape": [4],
      "expected": _flatten(mag.copysign(sign_src)),
    },
    {
      "id": "logaddexp_1d_3",
      "python_ref": "test/test_ops.py::test_logaddexp",
      "shape": [3],
      "expected": _flatten(la.logaddexp(lb)),
    },
    {
      "id": "masked_fill_scalar_1d_6",
      "python_ref": "test/test_ops.py::test_masked_fill",
      "shape": [6],
      "expected": _flatten(mfill_src.masked_fill(mfill_mask, -12.0)),
    },
    {
      "id": "take_flat_2x3_idx5",
      "python_ref": "test/test_ops.py::test_take",
      "shape": [5],
      "expected": _flatten(take_src.flatten()[idx_take]),
    },
    {
      "id": "item_scalar_42",
      "python_ref": "test/test_tensor.py::test_item",
      "shape": [],
      "expected": _flatten(scalar42),
    },
    {
      "id": "triu_diag1_3x4",
      "python_ref": "test/test_ops.py::test_triu",
      "shape": [3, 4],
      "expected": _flatten(base12.triu(1)),
    },
    {
      "id": "tril_diag_neg1_3x4",
      "python_ref": "test/test_ops.py::test_tril",
      "shape": [3, 4],
      "expected": _flatten(base12.tril(-1)),
    },
    {
      "id": "diag_vec3",
      "python_ref": "test/test_ops.py::test_diag",
      "shape": [3, 3],
      "expected": _flatten(vec3.diag()),
    },
    {
      "id": "diagonal_mat3",
      "python_ref": "test/test_ops.py::test_diagonal",
      "shape": [3],
      "expected": _flatten(mat3.diagonal()),
    },
    {
      "id": "unfold_8_2_2",
      "python_ref": "test/test_ops.py::test_unfold",
      "shape": [4, 2],
      "expected": _flatten(Tensor.arange(8).float().unfold(0, 2, 2)),
    },
    {
      "id": "masked_select_packed_payload_3x3",
      "python_ref": "test/test_ops.py::test_masked_select",
      "shape": [9],
      "expected": [float(v) for v in ms_packed.reshape(-1).tolist()],
    },
    {
      "id": "masked_select_packed_count_3x3",
      "python_ref": "test/test_ops.py::test_masked_select",
      "shape": [],
      "expected": [float(ms_sel.numel())],
    },
    {
      "id": "scatter_dim_mismatch_1x1x16",
      "python_ref": "test/test_ops.py::test_scatter",
      "shape": [1, 1, 16],
      "expected": _flatten(scatter_base.scatter(2, scatter_idx, scatter_src)),
    },
    {
      "id": "scatter_reduce_sum_dim_mismatch_1x1x16",
      "python_ref": "test/test_ops.py::test_scatter_reduce",
      "shape": [1, 1, 16],
      "expected": _flatten(scatter_base.scatter_reduce(2, scatter_ridx, scatter_rsrc, reduce="sum", include_self=False)),
    },
    {
      "id": "scatter_reduce_mean_dim_mismatch_1x1x16",
      "python_ref": "test/test_ops.py::test_scatter_reduce",
      "shape": [1, 1, 16],
      "expected": _flatten(scatter_base.scatter_reduce(2, scatter_ridx, scatter_rsrc, reduce="mean", include_self=False)),
    },
    {
      "id": "scatter_reduce_amax_dim_mismatch_1x1x16",
      "python_ref": "test/test_ops.py::test_scatter_reduce",
      "shape": [1, 1, 16],
      "expected": _flatten(scatter_base.scatter_reduce(2, scatter_ridx, scatter_rsrc, reduce="amax", include_self=False)),
    },
    {
      "id": "scatter_reduce_amin_dim_mismatch_1x1x16",
      "python_ref": "test/test_ops.py::test_scatter_reduce",
      "shape": [1, 1, 16],
      "expected": _flatten(scatter_base.scatter_reduce(2, scatter_ridx, scatter_rsrc, reduce="amin", include_self=False)),
    },
    {
      "id": "scatter_reduce_prod_dim_mismatch_1x1x16",
      "python_ref": "test/test_ops.py::test_scatter_reduce",
      "shape": [1, 1, 16],
      "expected": _flatten(scatter_base.scatter_reduce(2, scatter_ridx, scatter_psrc, reduce="prod", include_self=False)),
    },
    {
      "id": "conv_transpose2d_core_1x1x2x2",
      "python_ref": "test/test_ops.py::test_output_padded_conv_transpose2d",
      "shape": [1, 1, 5, 5],
      "expected": _flatten(conv_t_x.conv_transpose2d(conv_t_w, stride=2, output_padding=1)),
    },
    {
      "id": "max_unpool2d_default_1x1x2x2",
      "python_ref": "test/test_ops.py::test_max_unpool2d",
      "shape": [1, 1, 4, 4],
      "expected": _flatten(pooled_2x2.max_unpool2d(unpool_idx_2x2, kernel_size=(2, 2), stride=(2, 2))),
    },
    {
      "id": "max_unpool2d_out_1x1x2x2",
      "python_ref": "test/test_ops.py::test_max_unpool2d",
      "shape": [1, 1, 4, 4],
      "expected": _flatten(pooled_2x2.max_unpool2d(unpool_idx_2x2, kernel_size=(2, 2), stride=(2, 2), output_size=(4, 4))),
    },
    {
      "id": "batchnorm_nc_2x3",
      "python_ref": "test/test_nn.py::test_batchnorm_axis",
      "shape": [2, 3],
      "expected": _flatten(bn_nc_x.batchnorm(None, None, bn_nc_mean, bn_nc_invstd)),
    },
    {
      "id": "batchnorm_nchw_1x2x2x2_affine",
      "python_ref": "test/test_nn.py::test_batchnorm_axis",
      "shape": [1, 2, 2, 2],
      "expected": _flatten(bn_nchw_x.batchnorm(bn_nchw_weight, bn_nchw_bias, bn_nchw_mean, bn_nchw_invstd)),
    },
    {
      "id": "dropout_p1_zero_1d8",
      "python_ref": "test/test_edgecases.py::test_dropout_rate_one",
      "shape": [8],
      "expected": _flatten(drop_p1),
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
