# TinyGrad Lean Test Parity Plan

This file tracks Python-to-Lean test migration progress for the Lake test driver (`lake test`).

## Status Legend

- `ported`: implemented in Lean test driver suites
- `partial`: partially covered; additional edge cases still needed
- `deferred`: intentionally postponed

## Wave 1 Sources

### `test/test_tensor.py`

| Python area | Lean case(s) | Status | Notes |
|---|---|---|---|
| zerodim initialization | `tensor.zerodim.initialization` | ported | Runtime scalar shape/value behavior. |
| zeros_like / ones_like behavior | `tensor.like.zeros_ones` | ported | Lean keeps shape/dtype at type level; runtime value checks retained. |
| random seed determinism | `tensor.rand.seed_determinism` | ported | Tests same-seed equality and different-seed divergence. |
| shape/numel invariants | `tensor.prop.numel_triple` | ported | Property-style check with Plausible. |

### `test/test_ops.py`

| Python area | Lean case(s) | Status | Notes |
|---|---|---|---|
| creation ops (`zeros`, `ones`) | `ops.creation.zeros_ones` | ported | Value-level parity. |
| `arange` sequence | `ops.creation.arange` | ported | Basic sequence behavior parity. |
| broadcast add semantics | `ops.broadcast.add` | ported | Representative broadcast execution check. |
| broadcast laws | `ops.prop.broadcastable_comm`, `ops.prop.broadcast_out_refl` | ported | Property-style invariants for shape broadcasting. |

### `test/unit/test_indexing.py`

| Python area | Lean case(s) | Status | Notes |
|---|---|---|---|
| ellipsis/slice shape inference | `indexing.shape.ellipsis_slice` | ported | Spec-layer parity (pure indexing semantics). |
| newaxis/int shape inference | `indexing.shape.newaxis_int` | ported | Spec-layer parity. |
| index normalization examples | `indexing.normalize.examples` | ported | Includes negative and out-of-range cases. |
| normalize bounds invariant | `indexing.prop.normalize_bounds` | ported | Property-style safety invariant. |
| single-int indexing shape | `indexing.prop.single_int_shape` | ported | Property for rank-1 scalar indexing behavior. |
| bool indexing parity | — | deferred | Feature still unsupported in Lean runtime path. |

## Curated Mixed Set

| Area | Lean case(s) | Status | Notes |
|---|---|---|---|
| autodiff scalar gradient | `curated.grad.square` | ported | Mirrors key gradient sanity behavior. |
| where gradient routing | `curated.grad.where_routing` | ported | Branch-specific gradient flow checks. |
| matrix-vector broadcastability | `curated.prop.matrix_vector_broadcast` | ported | Property-style broadcast invariant. |

## Deferred Areas

- GPU/CUDA/Metal/TPU execution-path tests in `lake test` default profiles (CPU-only for Phase 1).
- Heavy model/data tests (MNIST pipelines, large benchmarks).
- Advanced indexing and bool-mask assignment parity not yet implemented in Lean runtime.
