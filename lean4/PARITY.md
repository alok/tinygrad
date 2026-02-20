# Lean ↔ Python Tensor Parity Checklist (Core)

Updated: 2026-02-20

Scope: user-facing tensor math/shape ops in `tinygrad/tensor.py` vs `lean4/TinyGrad4/Tensor/*`.
Not in scope: device/runtime backends, scheduling, JIT, IO helpers, visualization, web, or model code.

Legend: [x] implemented, [~] partial/bridge, [ ] missing

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
- [~] masked_fill / masked_select (`maskedFill` exact shape-preserving, `maskedSelectPacked` front-packed bridge + count)
- [x] take / item (`take` static index-shape lane, `item` scalar-only with runtime error for non-scalar)
- [~] diag / diagonal / tril / triu (`tril`/`triu` include diagonal offsets; `diag`/`diagonal` core vector<->matrix forms without offset)

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
- [~] unfold (static last-axis lane backed by existing `pool` path)

## Elementwise math
- [x] sin / cos / tan
- [x] asin / acos / atan
- [x] sinh / cosh / tanh
- [x] erf
- [x] round / sign
- [x] reciprocal (recip)
- [x] logaddexp
- [x] copysign / lerp / softsign
- [x] mish / celu / selu
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

## Known bridges/deferred items (Lean)
- `masked_select` exact dynamic-length output is intentionally deferred while `UOp` tensor shapes stay static (`List Nat`); use `maskedSelectPacked` bridge.
- `diag`/`diagonal` currently ship core vector<->matrix forms (offset variants deferred in this cycle).
- `unfold` currently targets static last-axis lane.
- `randperm` exact-sequence parity with Python RNG internals is not guaranteed yet (property parity is covered).

## Suggested order (high impact → low)
1) Dynamic-shape tranche: graduate `maskedSelectPacked` to exact `masked_select` when symbolic/runtime shape cardinality lands.
2) Offset tranche: extend `diag`/`diagonal` with offsets if shape proof burden remains tractable.
3) RNG parity tranche: align Lean RNG with Python for exact `randperm` sequence parity (not just permutation invariants).
4) Cross-language fixtures: keep deterministic oracle corpus in lockstep with new parity APIs.
5) Expand NN parity smokes: batchnorm/dropout surface semantics under deterministic conditions.
