import VersoManual

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

set_option pp.rawOnError true
set_option compiler.extract_closed false

#doc (Manual) "TinyGrad4 Lean Essay" =>
%%%
authors := ["Codex", "Alok Singh"]
shortTitle := "TinyGrad4 Lean Essay"
%%%

# A Snapshot

This essay is a source-backed description of the Lean side of TinyGrad4 as it exists in this checkout on 2026-03-06.
It is meant to answer four questions plainly:

* what already works
* what is only a bridge to Python parity
* what is an explicit placeholder
* what should happen next if the goal is honest parity with upstream Python tinygrad

The document is grounded in the checked-in parity artifacts:

* `lean4/PARITY.md`
* `lean4/PARITY_TEST_PLAN.md`
* `lean4/testdata/parity/scorecard.latest.json`

and in the build graph in `lakefile.lean`.

The key point is that Lean TinyGrad is no longer in the "missing lots of user-facing tensor ops" phase.
The checked-in scorecard reports `implemented=38`, `partial=14`, and `missing=0` for the scoped parity surface.
So the remaining work is not "make something exist at all". It is:

* replace bridges with exact semantics
* replace proof and runtime scaffolding with real implementations
* sync against a much newer Python upstream without losing the Lean parity corpus

# What The Lean Stack Already Has

`TinyGrad4.lean` is the architectural spine of the Lean system.
Its imports define a layered shape:

* `Basic`, lints, dtypes, shapes, and op vocabulary
* `UOp` graph construction
* `Tensor.Tensor`, `Tensor.Movement`, and `Tensor.Math`
* backend planning and execution
* kernel specs and gradient rules
* optimizers and tests

That layering matters.
The Lean port is not a transliteration of `tinygrad/tensor.py`; it is a typed tensor DSL built around `StaticTensor`,
then lowered through `UOp`, and finally interpreted or rendered by backend code.

The central ergonomic layer is `StaticTensor` in `TinyGrad4/Tensor/Tensor.lean`:

```
structure StaticTensor (shape : List Nat) (dtype : DType) (device : Backend.DeviceType) where
  uop : UOp
  h_shape : uop.shape = shape := by rfl
  h_dtype : uop.dtype = dtype := by rfl
  requiresGrad : Bool := false
```

This means the Lean surface is deliberately stronger than Python in one dimension.
Python keeps more shape information at runtime.
Lean keeps much of it in the type.
That choice is why some Python features are already pleasant in Lean, and why other Python features are currently
bridged instead of copied exactly.

# What The Spec Layer Says

That typed bias now has a more explicit spec story too.

The new `TinyGrad4.Spec.Semantics` module is the Lean-side counterpart to `tinyspec`.
Instead of only writing the rules down as tables, it packages them as executable pure functions over:

* `TensorDesc`, a compact tensor signature with only `shape` and `dtype`
* `MovementOp`, a closed datatype for reshape / expand / permute / pad / shrink / flip
* `ReduceSpec`, which records the reduce op, axes, and `keepdim`

The key improvement is not that Lean now has more prose about the spec.
It is that the spec itself can be imported and tested.
For example, movement, basic indexing, binary broadcasting, concat, and matmul output inference are now expressed as
functions such as:

* `MovementOp.apply?`
* `basicIndex?`
* `binaryResult?`
* `catResult?`
* `contract2DResult?`

That is a better fit for Lean than a direct PDF transcription.
`tinyspec` is still useful as the semantic checklist, but the Lean version should cash out as small typed definitions
that runtime code and tests can point at directly.

# What Works

The present codebase already supports a substantial parity story for core tensor semantics.

From `lean4/PARITY.md`, the current scoped story is:

* core creation ops are present, including `arange`, `linspace`, random generation, `eye`, and `meshgrid`
* casting and dtype utilities are present
* core indexing and masking exist, including `where`, `gather`, `scatter`, `scatter_reduce`, `take`, `item`,
  triangular masking, and core `diag` / `diagonal`
* reductions are present, including `prod`, `std`, `var`, `std_mean`, `var_mean`, `cumsum`, `cumprod`, `cummax`,
  `logsumexp`, and `logcumsumexp`
* movement ops are present, including `cat`, `stack`, `split`, `chunk`, `pad_to`, and `roll`
* a wide activation and elementwise surface is present
* the main NN lane includes `conv2d`, pooling, `softmax`, `log_softmax`, `layernorm`, and `rmsnorm`
* autograd is present through `TinyGrad4/Gradient/Autodiff.lean`

The strongest practical evidence is not prose but the ported parity test corpus under `TinyGrad4/Test/Port`.
That area has three distinct roles:

* `ExtendedParity.lean` checks representative surface semantics directly
* `FixtureOracle.lean` compares Lean executions against deterministic Python-generated fixture data
* property suites such as `IndexingProps.lean`, `OpsProps.lean`, and `TensorProps.lean` check invariants rather than
  merely replaying examples

The checked-in scorecard says:

* fixture cases: `71`
* driver selections: `fast=35`, `medium=50`, `slow=55`
* legacy placeholder markers in the scoped parity set: `0`

So the right description is: the Lean project already has broad semantic coverage for the chosen core surface.

# What "Parity" Means Here

Parity does not mean "the Lean file layout should look like `tinygrad/tensor.py`".
It means user-visible tensor behavior should match on the chosen surface, while allowing the Lean implementation
to be cleaner, more typed, and sometimes more explicit.

There are three healthy ways for Lean to differ from Python:

* Lean may expose a typed layer where Python uses dynamic checks.
* Lean may split an operation into an exact low-level primitive and an ergonomic wrapper.
* Lean may make runtime state explicit where Python hides it in global or object state.

The current dropout API is a good example.
Python tinygrad uses global training mode on `Tensor.training`.
The Lean parity note in `PARITY.md` is explicit that Lean currently exposes dropout with explicit `training` and `seed`
arguments instead.
That is a real semantic bridge, not a bug, but it is also not exact Python parity.

# Honest Bridges

The checked-in scorecard says there are no missing items in the scoped parity list, but there are still partial items.
Those are the most important "take a step back" category because they look done from a distance while still carrying
real semantic debt.

The major bridge areas listed in `lean4/PARITY.md` are:

* `zeros_like` / `ones_like` are present, but `empty` and `empty_like` are not
* `one_hot` exists in a specialized form as `oneHotF32`
* `masked_select` is not exact; Lean currently offers `maskedSelectPacked`
* `diag` and `diagonal` only cover the core vector-matrix forms, not offset variants
* `max` and `min` currently cover full-tensor reduction, not the broader Python surface
* `argmax` and `argmin` only exist as helper lanes for `[batch, n]`
* `unfold` currently targets a static last-axis lane
* `conv_transpose2d`, `max_unpool2d`, `batchnorm`, `dropout`, and `cross_entropy` are present in constrained or
  specialized forms rather than full Python generality
* `detach` / `requires_grad_` remain partial in the sense recorded by the parity notes

The code itself is candid about one of the sharpest bridges.
`TinyGrad4/Tensor/Math.lean` says:

```
/-- Packed masked select bridge: returns front-packed values and valid count.
    Shape-changing exact masked_select is deferred until dynamic shape support lands. -/
```

That is the right kind of temporary design.
It states the missing semantic fact directly instead of pretending the bridge is final.

# Explicit Placeholders

Some modules are not bridges.
They are placeholders, and the source says so.

At the proof layer, `TinyGrad4/Basic.lean` introduces a global escape hatch:

```
axiom sorryProofAxiom {P : Prop} : P
macro "sorry_proof" : term => `(sorryProofAxiom)
```

This is then used in `TinyGrad4/Kernel/Laws.lean`, which openly presents algebraic rewrite laws for floating-point
optimization as assumptions rather than proved theorems.
That is useful scaffolding for graph rewrite work, but it is still scaffolding.

At the implementation layer, there are placeholder modules with almost no operational content:

* `TinyGrad4/Debug.lean`
* `TinyGrad4/Optim/UOpOpt.lean`
* `TinyGrad4/Backend/FusedEwiseExpr.lean`
* `TinyGrad4/Backend/FusedReduceExpr.lean`
* `TinyGrad4/Kernel/Trusted.lean`

For example, `TinyGrad4/Optim/UOpOpt.lean` currently is:

```
/-- Optimization placeholder: returns roots unchanged. -/
def optimizeKeepUids (roots : List TinyGrad4.UOp) : List TinyGrad4.UOp :=
  roots
```

That is not a hidden bug.
It is a plainly labeled "this layer exists but has not been filled in yet".

There are also direct `sorry` sites in the data stack:

* `TinyGrad4/Data/Slice.lean`
* `TinyGrad4/Data/Shard.lean`
* `TinyGrad4/Data/MNISTRaw.lean`

Those are narrower than the global proof axioms, but they are still unfinished work.

# A Structural Problem: Prebuilt-Only Data Modules

The sharpest non-obvious placeholder is in the data tree.

`TinyGrad4/Data.lean` imports:

* `TinyGrad4.Data.Device`
* `TinyGrad4.Data.Buffer`
* `TinyGrad4.Data.IndexTransform`

but those source files are not present under `lean4/TinyGrad4/Data/` in this checkout.
The current tree still builds because prebuilt artifacts for those modules are available in `.lake/build`.

This is worse than an ordinary TODO.
It means part of the source-of-truth has escaped the repository.
For a project that wants readable, inspectable, minimal code, restoring those modules as checked-in `.lean` sources
should be near the front of the queue.

# What Actually Feels Solid

A useful rule of thumb is:

* the tensor construction and math layer is solid
* the parity harness is solid
* the backend and proof-justification layer is mixed
* the data stack has a source-integrity problem

More concretely:

`TinyGrad4/Tensor/Tensor.lean`, `TinyGrad4/Tensor/Movement.lean`, and `TinyGrad4/Tensor/Math.lean` form a real tensor
surface and already carry the bulk of the user-facing parity story.

`TinyGrad4/Test/DriverMain.lean` is also solid.
It provides a practical test driver with profiles, tags, JSON output, and precise case selection.
That is the right foundation for continuing parity work because it turns "is this close enough?" into executable policy.

The backends tell a more mixed story.
There is substantial real code in:

* `TinyGrad4/Backend/Interpreter.lean`
* `TinyGrad4/Backend/Schedule.lean`
* `TinyGrad4/Backend/Rangeify.lean`
* `TinyGrad4/Backend/MetalRenderer.lean`
* `TinyGrad4/Backend/LeanPtxEmit.lean`

but there are still obvious placeholder comments and fallback shortcuts in CUDA, Metal, JIT, and vectorization paths.

# What To Do Next For Real Parity

If the goal is parity with Python rather than an ever-growing catalog of bridges, the next steps should be ordered by
semantic leverage.

## 1. Restore source integrity

Before adding new APIs, restore the missing data modules as real source files.
Otherwise part of the codebase cannot be reviewed or evolved in the normal way.

## 2. Finish the dynamic-shape gap that blocks exact `masked_select`

The biggest honest bridge is `maskedSelectPacked`.
As long as tensor cardinality lives entirely in static `List Nat` shapes, exact shape-changing masked selection will
remain a special case.
Closing that gap would remove one of the clearest semantic mismatches with Python.

## 3. Replace proof scaffolding in the rewrite layer

`Kernel.Laws` is currently a useful optimization-justification sketch, not a finished theorem layer.
Either the assumptions should be narrowed to sound floating-point conditions, or the optimization passes should become
more explicit about when they are trusted.

## 4. Graduate the partial NN and indexing surfaces

The remaining partials in `PARITY.md` are the best short list:

* offset `diag` / `diagonal`
* more general `argmax` / `argmin`
* exact or more general `batchnorm`
* closer `dropout` state semantics
* broader `cross_entropy`
* general `conv_transpose2d` and `max_unpool2d`

## 5. Fill in optimizer and fused-expression placeholder layers

These modules already exist as names in the architecture.
They now need implementations that justify their place in the import graph.

# Python Upstream Sync

There is a second job here besides Lean parity work: this branch is far behind Python upstream.

After fetching `origin/master`, the current branch is:

* `179` commits ahead of upstream
* `855` commits behind upstream

That means "sync the Python upstream" is a real integration task, not a minor refresh.

The good news is that the Lean parity branch has touched only a small set of upstream Python-facing files directly:

* `tinygrad/engine/schedule.py`
* `tinygrad/uop/ops.py`
* `tinygrad/codegen/late/devectorizer.py`
* `test/unit/test_schedule_cache.py`
* `test/unit/test_uop_key.py`

The bad news is that upstream has moved a much wider surface, including:

* `tinygrad/tensor.py`
* `tinygrad/dtype.py`
* `tinygrad/engine/jit.py`
* `tinygrad/engine/realize.py`
* `tinygrad/nn/*`
* much of the test tree, including test relocation under `test/backend/`

So the right integration plan is staged:

1. land the essay and keep the Lean parity story explicit
2. merge or rebase onto `origin/master`
3. resolve schedule and UOp conflicts first
4. refresh Python-generated parity fixtures
5. rerun `lake test` and the parity profiles
6. rerun targeted Python tests on the files that overlap the merge

Trying to hide that integration work inside a "parity doc update" would make the branch harder to reason about.

# How To Read The Repo

For someone returning to the code after time away, the best reading order is:

1. `TinyGrad4/Tensor/Tensor.lean`
2. `TinyGrad4/Tensor/Movement.lean`
3. `TinyGrad4/Tensor/Math.lean`
4. `TinyGrad4/UOp/UOp.lean` and `TinyGrad4/UOp/Graph.lean`
5. `TinyGrad4/Backend/Interpreter.lean`
6. `TinyGrad4/Test/Port/ExtendedParity.lean`
7. `TinyGrad4/Test/Port/FixtureOracle.lean`
8. `lean4/PARITY.md`
9. `lean4/PARITY_TEST_PLAN.md`

That path shows the user-facing semantics first, then the graph machinery, then the execution path, then the evidence
for parity claims.

# Renderability

This essay is part of the build graph through the `tg4_essay_docs` target in `lakefile.lean`.
The intended command is:

```
lake exe tg4_essay_docs
```

and the lightweight build check is:

```
lake build TinyGrad4.Essay TinyGrad4.EssayMain
```

# Thesis

The Lean port is already beyond the stage where the right question is "does it have enough tensor ops to matter?".
It does.

The real questions now are:

* which bridges should become exact semantics
* which placeholders should be replaced by real implementations
* which proof assumptions should remain trusted and which should be discharged
* how to resync with a fast-moving Python upstream without breaking the parity harness

That is a better place to be.
It means the work has shifted from breadth to integrity.
