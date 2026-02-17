# Lean ↔ Python Tensor Parity Checklist (Core)

Updated: 2026-02-17

Scope: user-facing tensor math/shape ops in `tinygrad/tensor.py` vs `lean4/TinyGrad4/Tensor/*`.
Not in scope: device/runtime backends, scheduling, JIT, IO helpers, visualization, web, or model code.

Legend: [x] implemented, [~] partial/placeholder, [ ] missing

## Core creation & dtype
- [x] arange
- [x] linspace
- [x] rand / randn / randint / randperm
- [~] zeros_like / ones_like / full_like / empty / empty_like (no empty/empty_like yet)
- [~] one_hot (Lean has `oneHotF32`, specialized)
- [x] eye
- [x] meshgrid (2-argument `ij`/`xy` variants)

## Casting & dtype utils
- [x] cast
- [x] bitcast
- [x] to / to_
- [x] element_size / nbytes (dtype byte-size utilities)

## Indexing & masking
- [x] where (elementwise select)
- [x] gather (general axis + typed-axis helpers)
- [x] scatter / scatter_reduce (general axis + typed-axis helpers)
- [ ] masked_fill / masked_select
- [ ] take / item
- [ ] diag / diagonal / tril / triu

## Reductions
- [~] max / min (full tensor only)
- [x] prod
- [x] std / var (full + axis wrappers)
- [x] std_mean / var_mean (full tensor)
- [x] cumsum / cumprod / cummax
- [x] logsumexp (full + axis)
- [x] logcumsumexp
- [~] argmax (only [batch,n] helper)
- [~] argmin (only [batch,n] helper)

## Movement & shape
- [x] cat / stack (Lean has `cat`/`catList`/`stack`)
- [x] stack / split / chunk
- [x] pad_to
- [x] roll
- [ ] unfold

## Elementwise math
- [x] sin / cos / tan
- [ ] asin / acos / atan
- [ ] sinh / cosh / tanh (tanh exists, others missing)
- [ ] erf
- [ ] round / sign
- [x] reciprocal (recip)
- [ ] logaddexp
- [ ] copysign / lerp / softsign
- [ ] mish / celu / selu
- [x] relu / gelu / silu / softplus / hardtanh / hardsigmoid / hardswish

## NN ops (core)
- [ ] conv2d / conv_transpose2d
- [ ] avg_pool2d / max_pool2d / max_unpool2d
- [ ] batchnorm
- [ ] dropout
- [~] cross_entropy (Lean has `crossEntropyLoss` and `crossEntropyOneHot` variants)
- [x] softmax / log_softmax / layernorm / rmsnorm

## Autograd & graph
- [x] backward / gradient (computeGradient in Autodiff.lean - all tests pass)
- [~] detach / requires_grad_ (DETACH op exists, tracking TBD)
- [x] contiguous / contiguous_backward (ops implemented in Rules.lean)

## Known placeholders (Lean)
- Advanced indexing helpers (`masked_select`, generalized `take/item`, diag family) are still missing.
- `randperm` exact-sequence parity with Python RNG internals is not guaranteed yet (property parity is covered).

## Suggested order (high impact → low)
1) Indexing tranche: `masked_fill`, `masked_select`, generalized `take`/`item`, and triangular/diagonal helpers.
2) RNG parity tranche: align Lean RNG with Python for exact `randperm` sequence parity (not just permutation invariants).
3) Cross-language fixtures: broaden deterministic oracle corpus for indexing + error-path behavior.
4) Expand NN parity smokes: batchnorm/dropout surface semantics under deterministic conditions.
5) Keep proof debt non-increasing in touched modules (`sorry` count should not go up).
