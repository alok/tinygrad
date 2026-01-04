# tinygrad agents

Hello agent. You are one of the most talented programmers of your generation.

You are looking forward to putting those talents to use to improve tinygrad.

![big plans beaver](big-plans.png)

This is the **big plans beaver**. It's the vibe. This project is my dam.

## philosophy

tinygrad is a **tensor** library focused on beauty and minimalism, while still matching the functionality of PyTorch and JAX.

Every line must earn its keep. Prefer readability over cleverness. We believe that if carefully designed, 10 lines can have the impact of 1000.

Never mix functionality changes with whitespace changes. All functionality changes must be tested.

Note: `TensorLib` is a reference repo to crib ideas from; it is not a core tinygrad library.

## Lean tinygrad vision

The goal is parity with tinygrad's core idea: do as much as possible **statically**, then (super)optimize the minimal dynamic core.

Lean can do tinygrad better than Python can because we get real static knowledge:
- shapes, dtypes, and invariants in types
- whole-program reasoning and rewriting before runtime
- proofs/specs living next to executable kernels

tinygrad's thesis is to push everything toward static structure + aggressive optimization; Lean is a better fit for that thesis than Python,
and we can draw on prior art like https://github.com/input-output-hk/Lean-blaster.

For runtime performance, Lean gives us the right primitives: `Float32` plus packed `ByteArray`/`BitVec` storage avoids per-element boxing overhead
and stays extensible.

## Lake organization (Lean)

Lake packages can contain multiple libraries and executables; define a new library target with `lean_lib` in `lakefile.lean`.

Practical structure for research-heavy codebases:
- Split big efforts into separate `lean_lib`s (e.g. `StrataExperiments`, `KernelProofs`, `ScheduleSearch`) so they build independently.
- Each library should have a root module (e.g. `StrataExperiments.lean`) that imports its submodules to control what gets built by default.
- Use `globs` if you want Lake to build all submodules under a namespace automatically (defaults to roots).
- After adding or changing dependencies, run `lake update` to sync `lake-manifest.json`; manifests should be checked in.

## Lake skill (agent)

When adding a new subsystem:
1. Add a `lean_lib` target in `lakefile.lean`.
2. Create a root module file and import its submodules.
3. Run `lake update`, then build with `lake build <LibName>` (or `lake build +Module` for a single module).

## TODO (later)

- Avoid `xs[i]!` / `Array.get!` bounds checks in hot paths; prefer `for` loops, `Fin`-indexed access, or other proof-carrying indexing patterns.
- Explore Levi-Civita / non-Archimedean ordering for fusion/optimization priorities (multi-level "cheapness" and priority tiers).
- Add const/expand support for non-`float32` floats (`float64`/`float16`/`bfloat16`) plus a small smoke test.
- Add generic byte-path support for movement ops (`PERMUTE`/`SHRINK`/`PAD`) on non-float dtypes.

## style

Use **2-space indentation**, and keep lines to a maximum of **150 characters**. Match the existing style.

## Bench workflow

Default to full benchmark runs. Only use quick runs when explicitly setting `TINYGRAD4_BENCH_QUICK=1` to speed local iteration.

## Handoff notes

- Lean backend now supports view stacks end-to-end (ewise/reduce/matmul); fused reduce supports multi-axis (sum/max) with view/stack paths.
- Build passes: `lake build TinyGrad4`.

## Handoff TODO (hierarchical)

- 1. Upstream Python parity
  - 1.1 Pull upstream `tinygrad` Python changes; rebase/merge in small pieces and resolve renames/deletions.
  - 1.2 Keep Python toolchain current; use `uv` for pytest and general Python workflows.
  - 1.3 Compare upstream design changes to local patches; remove/adjust local code accordingly, keep tests green.
- 2. Lean runtime parity + improvements
  - 2.1 Add regression tests for view stacks (ewise/reduce/matmul) and multi-axis reduce (keepdim true/false).
  - 2.2 Expand dtype coverage (float64/float16/bfloat16) and non-f32 movement ops; add smoke tests.
  - 2.3 Add a GC/graph-growth guard option to prevent hangs when the graph grows.
  - 2.4 Improve fusion/ShapeTracker correctness and performance; remove remaining stack fallbacks where safe.
  - 2.5 Add benchmark instrumentation to measure compile time, wait time, and kernel time.
- 3. Core ML lib + array work
  - 3.1 Bring core ML library parity with Python (nn/optim/data loaders).
  - 3.2 Strengthen array/tensor ergonomics and performance (ByteArray/BitVec paths, fewer bounds checks).

## Helper Tools (Lean 4)

Located in `.claude/tools/lean4/`, these scripts assist with Lean 4 development.

- **`search_mathlib.sh <query> [type]`**: Search Mathlib for declarations.
  - Types: `name` (default), `type` (signature), `content`.
  - Example: `.claude/tools/lean4/search_mathlib.sh "continuous" name`
- **`analyze_let_usage.py <file>`**: Detects inefficient `let` bindings that increase token count when inlined.
  - Example: `python3 .claude/tools/lean4/analyze_let_usage.py lean4/MyFile.lean`
- **`check_axioms.sh`**: Verify axioms used by declarations.
- **`sorry_analyzer.py`**: Analyze `sorry` usage to track missing proofs.
- **`find_golfable.py`**: Identify code sections that might be shortened.

---

Bench code after creating it. Write benchmarks like you write property/unit
tests. See tinygrad's testing.
