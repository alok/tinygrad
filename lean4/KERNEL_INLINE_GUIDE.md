# Building `kernel!`: Inline Codegen with a Custom Lean Elaborator

A deep guide to the inline-codegen layer added in `lean4/TinyGrad4/Kernel/`
(`Codegen.lean`, `Inline.lean`, `InlineMetal.lean`), written so you could
rebuild it yourself from an empty buffer. It also covers the grind-powered
proof work that landed alongside it (`Laws.lean`, `Data/Slice.lean`,
`Data/Shard.lean`).

## 1. The problem, in tinygrad terms

Upstream tinygrad's pipeline is: Tensor ops → UOp graph → schedule → codegen
→ runtime. Every stage is a runtime transformation of a data structure, and
the final device source (Metal/CUDA/C) is rendered *at runtime*, then handed
to the driver compiler. tinygrad's philosophy is to keep those layers thin
enough to read.

Lean gives us a lever Python doesn't have: the *elaborator*. Elaboration is
compile time, but it is programmable — an `elab` rule is ordinary Lean code
that runs while the file is being checked, with the whole standard library
available. So a "kernel DSL → device source" translation doesn't have to be
a runtime stage at all. It can happen while the file elaborates, and the
resulting source string can be a literal baked into the binary. That is
`kernel!`:

```lean
def saxpy := kernel! "saxpy" fun a x y => a * x + y
-- saxpy : InlineKernel 3
-- saxpy.metal : String   -- full MSL kernel, already generated
-- saxpy.denote_eq        -- proof: spec ≡ implementation
```

One abstraction layer (runtime rendering) deleted, one artifact (a proof)
gained. This is "cut the layers" applied with a type theory.

## 2. The spec was already there

`Kernel/Spec.lean` defines a small typed expression language:

```lean
inductive Expr : Ty → Type where
  | input (t : Ty) (idx : Nat) : Expr t
  | constF32 (v : Float32) : Expr .f32
  | add : Expr .f32 → Expr .f32 → Expr .f32
  | cmplt : Expr .f32 → Expr .f32 → Expr .bool
  | where_ : Expr .bool → Expr .f32 → Expr .f32 → Expr .f32
  ...
```

with an executable semantics `evalExpr : ScalarOps Float32 → ... → Expr t →
Ty.denote t`. This is a GADT: `Expr .f32` and `Expr .bool` are different
types, so ill-typed kernels (`add` of a boolean) don't exist as values. The
new work *reuses* this as the single source of truth — the elaborator
targets it, the codegen renders it, the proof talks about it. Resist the
urge to invent a second AST "for codegen"; one spec, many interpretations.

## 3. `Codegen.lean`: spec → source, as pure functions

Three pieces:

**`f32Ops : ScalarOps Float32`** — the canonical semantics. Every field is
the eta-reduced native op (`add := (· + ·)`, `max := Max.max`, …). This
matters for proofs: `evalExpr f32Ops` applied to a concrete AST unfolds by
plain delta/beta/projection reduction into ordinary Float32 arithmetic, so
definitional equality (`rfl`) can see through it. If a field were, say,
`add := fun a b => (a.toFloat + b.toFloat).toFloat32` (as an older test
file did), `rfl` proofs would fail and every kernel would need simp.

**`denote`** — `evalExpr f32Ops (readEnv env) e` where
`readEnv : (Fin n → Float32) → (t : Ty) → Nat → t.denote` bridges the
spec's Nat-indexed reads to a sized environment. The `if h : i < n` inside
reduces definitionally for literal `i`/`n` — again keeping `rfl` viable.

**Renderers** — `Expr.toCode` prints a scalar expression that is valid in
*both* MSL and C (inputs as `x{i}`, ternary select, `max()` call — C gets a
`#define max fmaxf` shim). `metalSource`/`cSource` wrap it in a full kernel:
per-thread `out[gid] = expr` for Metal (matching the calling convention of
the existing `MetalEwise.runEwiseKernel`), a sequential loop for C.

Float constants render as **C99/MSL hex-float literals** (`0x1.800000p+0f`)
computed from the bit pattern — bit-exact, no decimal round-trip ambiguity,
and the same literal syntax works in both languages. Handle the special rows
(±0, subnormal `0x0.……p-126f`, ±INFINITY, NAN) explicitly; the mantissa is
23 bits, shifted left once to get six exact hex digits.

## 4. `Inline.lean`: the elaborator

The pipeline for `kernel! "name" fun x y => body`:

1. **Collect binders** → `x ↦ 0, y ↦ 1` (plain idents, or `(x y : T)`
   groups via the `explicitBinder` node).
2. **Reify** the body *syntax* into a working AST `RExpr` by quotation
   pattern matching: `` `($a + $b) ``, `` `(sqrt $a) ``,
   `` `(if $a < $b then $x else $y) ``, literals, variables. Anything else
   is a compile-time error *at the offending subterm* (`throwErrorAt`).
3. **Render three ways** from the one `RExpr`:
   - a `Kernel.Expr` term (the embedded spec),
   - a native `fn` term over `env : Fin n → Float32`,
   - via `RExpr.toKernel : RExpr → Kernel.Expr .f32` (a *meta-level value*,
     not syntax), the Metal and C source strings — computed right there in
     the elaborator and embedded with `Syntax.mkStrLit`.
4. **Emit** the `InlineKernel n` structure literal with
   `denote_eq := by intro env; first | rfl | simp [...] | grind [...]`.

Three design decisions carry the whole thing:

**Syntax-level reification, not Expr-level.** CostExprMeta (the older
`costExpr!`) reifies elaborated `Lean.Expr` via `whnf`. Here we match
surface syntax instead. The win: the DSL is self-contained — `sqrt x` means
`Expr.sqrt` regardless of what `sqrt` resolves to in the user's scope, and
the original lambda is *discarded*; both the spec term and `fn` are rendered
from the reified AST. Reification can't disagree with elaboration because
elaboration of the body never happens.

**Literals keep their original syntax.** `RExpr.lit` carries both the
`Float32` value (computed at meta time with the same `ofScientific`/`OfNat`
functions elaboration would use — needed for source rendering) and the
original token (spliced into both the `constF32 …` spec term and the `fn`
body). Both sides of `denote_eq` therefore contain the *identical* term, and
`rfl` closes arithmetic kernels. If you instead embedded
`Float32.ofBits 0x3FC00000`, the proof would break: `ofBits` is opaque
(`@[extern]`), so it never reduces to what the literal elaborates to.

**Fully qualify names inside quotations.** Quotations pre-resolve
identifiers at the *definition* site. `` `(max $a $b) `` inside the
`Inline` namespace resolved to `RExpr.max` — a genuinely confusing bug
(the error surfaces at the *use* site, in another file). Write `Max.max`,
`Float32.sqrt`, etc.

The conditional case shows the proof-fallback ladder working as designed:
`fn` renders `if a < b then x else y` (Prop-level `<`, `Float32.decLt`
instance) while `denote` unfolds to `if decide (a < b) = true then …`.
`decLt` is opaque, so these are *not* definitionally equal — `rfl` fails,
and `simp` bridges with `decide_eq_true_eq`. `grind` is the third rung for
whatever simp misses. The elaborator doesn't know or care which rung fires;
the ladder is part of the generated term.

## 5. `InlineMetal.lean`: dispatch is someone else's job

`InlineKernel.runMetal` is four lines: arity check, then
`MetalEwise.runEwiseKernel k.name k.metal inputs numel`. The existing
runtime already knew how to compile-with-cache and launch a shader string;
inline codegen just changes *when* the string exists. Keeping this in its
own module keeps `Codegen.lean` backend-free.

Verification and perf, both on this M4 Max:

- `lake exe inline_kernel_smoke` — relu/saxpy/sigmoidish on GPU vs `fn`
  elementwise over 1024-element buffers, plus CPU value checks and a
  runtime `denote`-vs-`fn` spot check (the theorem already guarantees it;
  the run exercises the eval path).
- `lake exe inline_kernel_bench` — steady-state (buffers resident, shader
  compiled once, sync outside the loop): saxpy 94 μs/iter ≈ 178 GB/s at
  2^20 elements; sigmoidish 136 μs/iter (transcendental-bound); per-element
  CPU `fn` ~3 orders slower. Copies excluded deliberately — that's the part
  codegen controls.

## 6. The grind story: type-level equivalences

The goal was to stop hand-writing (or sorrying) the equivalence proofs that
intrinsic typing demands. Three case studies from this change:

**`Data/Slice.lean`** — `RawBufferSlice` carries
`h_valid : (offset + len) * itemsize ≤ size`. The clamped constructors
(`mk'`, `slice`) had `sorry`. The shape of the proof is one nonlinear step
(`x ≤ size/isz → x * isz ≤ size`, i.e. `Nat.div_mul_le_self` +
`Nat.mul_le_mul_right`) with linear residue. Neither `grind` nor `omega`
crosses variable-divisor division alone; a two-line `calc` supplying the
nonlinear step, with `by omega` for the min-arithmetic, is the honest
minimum. Lesson: give the solver the one fact it cannot invent, let it do
the bookkeeping.

**`Data/Shard.lean`** — the `getItem` fallback needed
`i < shardSize cfg n → i < n` across three modes *including degenerate
configs* (`numShards = 0` works out because `n / 0 = 0` vacuously bounds).
Structured as three private arithmetic lemmas (`div`, interleaved-ceiling,
contiguous-with-remainder) dispatched from one `unfold`+`split` proof.
Notable trap: proving arms by `match hk : cfg.numShards with …` inside the
combined proof leaves hypotheses spelled with different atoms
(`k.succ.succ` vs `k + 2`) that omega treats as unrelated; standalone
lemmas with their own variables avoid the mismatch entirely.

**`Kernel/Laws.lean`** — was *stale dead code*: referenced a `Tensor` type
that existed nowhere, wasn't imported by the root module, stated FP algebra
as global `axiom`s, and sorried the tensor lemmas. Rewritten spec-driven:
`Tensor s α := Shape.Index s → α` (function tensors over the intrinsically
typed index — each dimension a `Fin`), extensional `TensorEq`, and *proved*
laws. `grind` shines here: `zip_map_broadcast_self` needs "a size-1 dim can
only hold index 0", which grind derives from `i.isLt : i < 1` after simp
exposes the list structure. The identity `broadcastIndex s s idx = some idx`
falls out of a `toList`/`ofList?` round-trip induction. FP associativity
etc. are now *hypotheses* of the rewrites that need them — no axioms about
`Float32`, so a rewrite is exactly as trustworthy as the law you feed it.
One more trap: simp normalizes `==` to `=` in goals, so state helper lemmas
in the normalized form or they won't fire as rewrite rules.

## 7. Typed docs as regression tests

The Verso essay (`Essay.lean`, built by `tg4_essay_docs`) gained an inline
codegen section whose code blocks are ```lean — Verso *elaborates* them.
The doc defines a kernel with `kernel!` and states `denote_eq` as an
`example`. If the elaborator or the proof automation regresses, the
documentation fails to build. Docs that can't lie.

## 8. Landing: the corrupt-tree constraint

Pushing the full branch history to GitHub fails with
`duplicateEntries: contains duplicate file entries` — a historical tree
object under `lean4/c/` lists `tg4_cuda.cu` twice (an old index corruption,
permanent in history). GitHub's fsck rejects it; this is why the fork's
master was already a squash. Landing therefore = `git commit-tree
HEAD^{tree} -p <fork-master>` (current tree, clean) pushed to `master` and
`main`. Full history stays local. If the history itself ever needs to go up,
`git filter-repo` rewriting that one tree is the tool.

## 9. Extending the DSL: the checklist

To add an op (say `min`), touch in order:

1. `Kernel/Spec.lean` — constructor + `evalExpr` case (only if the spec
   lacks it; prefer composing existing constructors).
2. `Codegen.lean` — `Expr.toCode` case (+ C shim if the function name
   differs), `Expr.arity` case, and an `f32Ops` field *only* for genuinely
   new scalar semantics, defined as the eta-reduced native op.
3. `Inline.lean` — `RExpr` constructor, `toKernel`, `toExprTerm`,
   `toFnTerm` (fully-qualified names!), and a reify pattern.
4. `Test/InlineKernelSmoke.lean` — a kernel using it, checked on CPU and
   GPU.
5. If `rfl` can't close its `denote_eq` (any decide/ite in the rendering),
   confirm the simp/grind rungs do, and add the bridging lemma to the
   generated tactic block if not.

The invariant to preserve throughout: **`fn` and the embedded spec are two
renderings of one reified AST, built from the same tokens.** Everything
else — the proof, the codegen, the GPU agreement — follows from that.

## 10. Round two: semantics hardening and the vectorized win

An adversarial review pass (plus a head-to-head against upstream Python
tinygrad) drove a second round of changes worth understanding:

**Semantics must match the proof, not the platform's favorite intrinsic.**
`max` originally rendered as `max()`/`fmaxf()` — which *discards* NaN, while
the proven `fn` uses Lean's `Max.max` (`maxOfLe`), which *propagates* the
first NaN argument and prefers `-0.0` differently. The generated code now
renders `((a<=b)?b:a)` (scalar) / `select(a,b,a<=b)` (vector): exactly
`maxOfLe`, lane for lane. Same class of fix: `truthy` in the spec was a bit
test (`toBits != 0`, making `-0.0` truthy) while device code value-tests;
the spec now value-tests, matching device and Python semantics. The C
preamble `#define`s `sqrt/exp2/log2/sin` to their `f`-suffixed forms so C
doesn't silently promote through double. Hex-float literals are
parenthesized so `x-(-1.0f)` can't lex as `--`.

**Cache keys are correctness, not hygiene.** The Metal program cache is
keyed by kernel name, so two anonymous kernels sharing the default name
would silently run each other's code. Anonymous kernels now get a
content-derived name (`tg4_k<hash of rendered code>`). Explicitly named
kernels are the user's responsibility, tinygrad-style.

**Vectorize in the generator, not the runtime.** Every kernel now also gets
a float4 variant (`metalVec`, name-suffixed `_v4`, `numel/4` threads) from
the same reified AST — ~25 extra lines of renderer. `runMetal` auto-selects
it when `numel % 4 == 0`. Steady-state on M4 Max, 2^20 elements:

| kernel | scalar | float4 | Python tinygrad (same op) |
|---|---|---|---|
| saxpy | 98 μs, 170 GB/s | **40 μs, 418 GB/s** | ~80 μs wall, 282 GB/s GPU |
| sigmoidish | 81 μs, 104 GB/s | **27 μs, 310 GB/s** | ~78 μs wall |

The elaborated kernels beat upstream Python tinygrad by ~2× wall time on
this op class — the entire runtime cost is one cached-pipeline dispatch,
because everything else already happened at compile time.

**Guards where proofs end.** `runMetal` now validates input byte sizes
(the generated kernel has no `gid` guard — dispatch is exact, so undersized
buffers meant silent OOB), the elaborator rejects duplicate binders and
non-`Float32` binder ascriptions, and `Data/Shard.lean` gained the strong
strided-mode theorem (`localToGlobal_interleaved_lt`) that the earlier
fallback lemma deliberately didn't claim. The last `sorry` in the port
(MNIST one-hot) is gone — implemented as the UOp-graph transcription of the
proven `oneHotF32` pattern (labels-as-column vs class-range-as-row `CMPEQ`,
then `WHERE` 1/0).
