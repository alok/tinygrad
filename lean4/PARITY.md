# Lean ↔ Python Tensor Parity Checklist (Core)

Updated: 2026-02-17

Scope: user-facing tensor math/shape ops in `tinygrad/tensor.py` vs `lean4/TinyGrad4/Tensor/*`.
Not in scope: device/runtime backends, scheduling, JIT, IO helpers, visualization, web, or model code.

Legend: [x] implemented, [~] partial/placeholder, [ ] missing

## Core creation & dtype
- [x] arange
- [x] linspace
- [~] rand / randn / randint / randperm (no randperm yet)
- [~] zeros_like / ones_like / full_like / empty / empty_like (no empty/empty_like yet)
- [~] one_hot (Lean has `oneHotF32`, specialized)
- [ ] eye
- [ ] meshgrid

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
- [ ] prod
- [~] std / var (axis variants exist; full wrappers missing)
- [ ] std_mean / var_mean
- [ ] cumsum / cumprod / cummax
- [~] logsumexp (axis helper exists; full wrapper missing)
- [ ] logcumsumexp
- [~] argmax (only [batch,n] helper)
- [~] argmin (only [batch,n] helper)

## Movement & shape
- [x] cat / stack (Lean has `cat`/`catList`/`stack`)
- [~] stack / split / chunk (stack only)
- [ ] pad_to
- [ ] roll
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
- Full-tensor wrappers for `std`/`var`/`logsumexp` are not yet exposed.
- Prefix reductions (`cumsum`/`cumprod`/`cummax`) are still missing.

## Suggested order (high impact → low)
1) Creation + movement parity tranche: `eye`, `meshgrid`, `randperm`, `split`, `chunk`, `pad_to`, `roll`.
2) Reductions tranche: `prod`, full `std`/`var`, `std_mean`/`var_mean`, `cumsum`/`cumprod`/`cummax`, full `logsumexp`/`logcumsumexp`.
3) Indexing tranche: `masked_fill`, `masked_select`, `take`/`item`, and triangular/diagonal helpers.
4) Cross-language fixtures: deterministic Python oracle outputs for high-risk tensor semantics.
5) Keep proof debt non-increasing in touched modules (`sorry` count should not go up).
