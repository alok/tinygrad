# Lean ↔ Python Tensor Parity Checklist (Core)

Updated: 2025-12-24

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
- [~] gather (only last-axis helper)
- [~] scatter / scatter_reduce (only last-axis helper)
- [ ] masked_fill / masked_select
- [ ] take / item
- [ ] diag / diagonal / tril / triu

## Reductions
- [~] max / min (full tensor only)
- [ ] prod
- [ ] std / var (full tensor)
- [ ] std_mean / var_mean
- [ ] cumsum / cumprod / cummax
- [ ] logsumexp (full, not only axis helper)
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
- [ ] backward / gradient / detach / requires_grad_
- [ ] contiguous / contiguous_backward

## Known placeholders (Lean)
- `gatherLastF32` is a specialized helper, not a general gather.

## Suggested order (high impact → low)
1) `cast` + `to`/`to_` + dtype utilities (unblocks many ops/tests).
2) General `where` + `gather` + `scatter` (core indexing/parity).
3) `arange`/`linspace`/`rand*` + `_like` constructors (core creation).
4) Reductions: `max`/`min`/`prod` + `std`/`var` + `cumsum`/`cumprod`.
5) Movement: `stack`/`split`/`chunk` + `pad_to` + `roll`.
