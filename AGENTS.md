# tinygrad agents

Hello agent. You are one of the most talented programmers of your generation.

You are looking forward to putting those talents to use to improve tinygrad.

## philosophy

tinygrad is a **tensor** library focused on beauty and minimalism, while still matching the functionality of PyTorch and JAX.

Every line must earn its keep. Prefer readability over cleverness. We believe that if carefully designed, 10 lines can have the impact of 1000.

Never mix functionality changes with whitespace changes. All functionality changes must be tested.

## style

Use **2-space indentation**, and keep lines to a maximum of **150 characters**. Match the existing style.

## design signals

Treat ugly names as a design smell, not just a naming problem. If a helper/wrapper name feels awkward, first fix the layer boundary that forced it.

Prefer a clean layered API shape:
- low-level proof/constraint layer can be explicit and unergonomic
- ergonomic layer should read like tensor semantics and build on the low-level layer
- dynamic/runtime-checked paths should stay separate from typed proof paths

## staged specialization

For performance-critical paths, use a staged workflow:
1. Build a parametric tensor program.
2. Plug in datacenter/runtime parameters (device limits, tile configs, launch knobs).
3. Compile the specialized program again.
4. Run the specialized artifact.

The goal is to maximize compile-time optimization and minimize runtime branching in hot paths.

## Resources

- `ssh ww` should give access to 2 A6000, and an env that can run triton (unlike macos).

It also lets you run `triton`, which MacOS doesn't support. (focus on linux support first).
