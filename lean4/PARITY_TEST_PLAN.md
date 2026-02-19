# TinyGrad Lean Test Parity Plan

This file tracks Python-to-Lean test migration progress for the Lake test driver (`lake test`).

## Porting Rules

- Prefer runtime semantic checks; do not port assertions that merely restate compile-time shape/dtype facts.
- Every parity case must include `pythonRefs` metadata in Lean test registry entries.
- Keep CPU parity tests in default profiles; device-specific tests remain explicitly tagged.
- For numerics, compare with tolerances and deterministic seeds where randomness is involved.

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
| `randperm` semantics | `tensor.rand.randperm` | partial | Determinism + permutation invariants covered; exact RNG-sequence parity still pending. |
| shape/numel invariants | `tensor.prop.numel_triple` | ported | Property-style check with Plausible. |

### `test/test_ops.py`

| Python area | Lean case(s) | Status | Notes |
|---|---|---|---|
| creation ops (`zeros`, `ones`) | `ops.creation.zeros_ones` | ported | Value-level parity. |
| `arange` sequence | `ops.creation.arange` | ported | Basic sequence behavior parity. |
| `linspace` edge-step semantics | `ops.creation.linspace` | ported | Covers `steps=3/1/0` and int-dtype casting behavior. |
| broadcast add semantics | `ops.broadcast.add` | ported | Representative broadcast execution check. |
| reshape/flatten movement semantics | `ops.move.reshape_flatten_roundtrip` | ported | Round-trip value/shape parity for movement path. |
| transpose/permute movement semantics | `ops.move.permute_transpose` | ported | Matrix transpose parity via both APIs. |
| cat/stack movement semantics | `ops.move.cat_stack` | ported | Validates value ordering across concat and stack axes. |
| expand broadcast semantics | `ops.move.expand` | ported | Broadcasted repeat values from singleton dims. |
| axis reduction semantics (`sum`/`max`) | `ops.reduce.axis_semantics` | ported | `keepdim=true/false` runtime behavior parity. |
| full-tensor `min`/`max` reductions | `ops.reduce.min_max_full` | ported | Scalar reduction outputs with signed inputs. |
| `eye` creation semantics | `ops.creation.eye` | ported | Matrix identity behavior for rectangular shape. |
| `meshgrid` semantics (`ij`/`xy`) | `ops.creation.meshgrid_ij`, `ops.creation.meshgrid_xy` | ported | Both indexing modes covered (2-arg variant). |
| split/chunk/roll/pad-to movement | `ops.move.split_chunk_roll_pad_to` | ported | Includes shape and representative value checks. |
| reduction extensions (`prod/std/var/cum*`) | `ops.reduce.extended` | ported | Core extension coverage (`prod/std/var/cum*/log-sum-exp`). |
| softmax/log-softmax semantics | `ops.softmax.logsoftmax` | ported | Last-axis parity plus explicit axis-0 softmax check. |
| activation family semantics (`relu/tanh/silu/gelu`) | `ops.elemwise.activations` | ported | Deterministic value checks for core nonlinearities. |
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
| conv/pool deterministic smoke | `curated.nn.conv_pool_smoke` | ported | Basic conv/max-pool/avg-pool shape+value parity sanity lane. |
| Python fixture oracle checks | `fixture.core_ops.python_oracle` | ported | Slow-profile deterministic fixtures generated from Python tinygrad. |

## Deferred Areas

- GPU/CUDA/Metal/TPU execution-path tests in `lake test` default profiles (CPU-only for Phase 1).
- Heavy model/data tests (MNIST pipelines, large benchmarks).
- Advanced indexing and bool-mask assignment parity not yet implemented in Lean runtime.
- Exact Python RNG sequence parity for `randperm` is not yet guaranteed (property parity is enforced).

## Mandatory Gates

Every parity PR should pass all three driver profiles locally and in CI:

- `lake test` (fast)
- `lake test -- --profile medium`
- `lake test -- --profile slow`

Current selection counts:

- fast: 26
- medium: 34
- slow: 35
