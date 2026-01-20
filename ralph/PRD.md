# TinyGrad4 Lean — Triton Linear Layers Port

## Goal
Port tinygrad's Triton linear/matmul path from Python to Lean, so the Lean PTX emitter is the primary backend for linear layers (no Python in the hot path), with correctness-first parity to `extra/gemm/triton_nv_matmul.py` and tinygrad's linear semantics.

## Context
We are in the Lean 4 port of tinygrad (TinyGrad4). Recent work added:
- Lean PTX emitter with basic/tiled/smem variants.
- Stride-aware addressing and M/N wrap behavior (matches Triton kernel).
- `emitPtx!`/`ptxSource!` meta helpers for PTX config literals.

The next arc is to *assimilate* Python/Triton code into Lean: correctness parity before performance.

## Scope
In-scope work to unlock Linear/Triton in Lean:
- Match Triton kernel semantics exactly (stride/mask/wrap) for linear layers.
- Lean-only PTX generation for matmul + optional bias/scale/relu epilogue.
- StaticTensor/compile-time shapes actually used in codegen (not ignored).
- Remove reliance on `FloatArray` except legacy compat shims.
- Tests that validate generated PTX (via `trace.compiler.ir`) and runtime correctness.

Out of scope (for now):
- Large-scale perf tuning or kernel auto-tuning.
- Full Triton feature parity beyond matmul + linear epilogue.

## Requirements & Constraints
- Follow `AGENTS.md` + `CLAUDE.md` (2-space indent, 150-char lines, no whitespace-only diffs).
- All functional changes must be tested.
- Lean code should be computable (avoid `sorry` except explicit placeholders).
- Prefer Lean generators/macros over Python runtime.
- Use `uv` for any Python package management.

## Reference Semantics (Triton → Lean parity)
- Strides are row-major for the reference kernel: A stride (am=K, ak=1), B stride (bk=N, bn=1), C stride (cm=N, cn=1).
- A/B loads wrap indices with modulo on M/N (`offs_am`, `offs_bn`); K is stepped in blocks.
- No explicit masks on A/B loads or C stores; correctness relies on M/N/K being block-aligned.
- Bias (when present) uses the same N wrap; Triton uses a mask `offs_bn < N` but it is redundant with modulo.

## Milestones
M0 — Baseline parity scaffolding
- Audit Python reference: `extra/gemm/triton_nv_matmul.py` and tinygrad linear path.
- Enumerate exact semantics to match (strides, wrap/mask, block config).

M1 — Lean PTX parity for linear matmul
- Ensure Lean PTX addressing uses strides + wrap/mask as needed.
- Bias/scale/relu epilogues align with Lean fused matmul path.
- Validate with `trace.compiler.ir` and PTX dumps.

M2 — StaticTensor compile-time shape correctness
- Shapes used at compile time in kernel config and addressing.
- Remove unused `shape` plumbing; enforce shape-carrying types.

M3 — Tests + Linux validation
- Add correctness tests for contiguous + non-contiguous strides.
- Run GPU tests on Linux (ssh ww or runpod) to validate PTX.

## Acceptance Criteria
- Linear matmul + bias/scale/relu is correct on CUDA (Linux) via Lean PTX.
- No Python Triton emitter required for default path.
- `trace.compiler.ir` shows the intended fused kernel shape + epilogue.
- StaticTensor shape is used at compile time (no dead/unused shape params).

## Test Plan
- `~/.elan/bin/lake build emit_lean_ptx`
- `~/.elan/bin/lake build linear_triton_smoke` and `linear_triton_scale_relu_smoke`
- New stride/mask tests for non-contiguous inputs.
- Linux validation on `ssh ww` with PTX dump enabled.

## Work Log
Use `ralph/progress.txt` to record each iteration. Mark completion with:
`<promise>COMPLETE</promise>`
