import Float64
import TinyGrad4.Kernel.Inline
import TinyGrad4.Kernel.InlineMetal

/-!
# InlineKernelSmoke

End-to-end test of the `kernel!` inline-codegen elaborator:
- kernels defined as plain Lean lambdas, compiled at elaboration time
- `fn` (native) checked against expected values
- `denote` (spec) checked against `fn` — backed by the `denote_eq` proof
- generated Metal source compiled and launched on the GPU, results compared
  against the native implementation elementwise
-/

set_option linter.useRawBuffer false

namespace TinyGrad4.Test.InlineKernelSmoke

open TinyGrad4 Kernel Kernel.Inline
open TinyGrad4.Backend

/-! ## Kernels under test (all source generated at compile time) -/

def saxpy := kernel! "tg4_saxpy" fun a x y => a * x + y
def relu := kernel! "tg4_relu" fun x => if x < 0.0 then 0.0 else x
def norm2 := kernel! "tg4_norm2" fun x y => sqrt (x * x + y * y)
def sigmoidish := kernel! "tg4_sigmoidish" fun x => recip (1.0 + exp2 (-1.442695 * x))
def clampMax := kernel! "tg4_clamp" fun x lo => max x lo
def minK := kernel! "tg4_min" fun x y => min x y
def absK := kernel! "tg4_abs" fun x => abs x

/-- The proof fields are real: spec and implementation agree definitionally. -/
example (env : Fin 3 → Float32) : denote saxpy.expr env = saxpy.fn env := saxpy.denote_eq env
example (env : Fin 1 → Float32) : denote relu.expr env = relu.fn env := relu.denote_eq env

/-! ## Helpers -/

private def envOf (xs : Array Float64) : Fin n → Float32 :=
  fun i => (xs.getD i.val 0.0).toFloat32

private def assertClose (got expected : Float64) (msg : String) (tol : Float64 := 1e-5) : IO Unit := do
  if Float64.abs (got - expected) > tol then
    throw (IO.userError s!"{msg}: got={got} expected={expected}")

private def packF32 (data : Array Float64) : RawBuffer :=
  { dtype := .float32, data := Native.packF32FromF64 ⟨data⟩ }

/-! ## CPU checks -/

def testFn : IO Unit := do
  assertClose (saxpy.fn (envOf #[2.0, 3.0, 4.0])).toFloat 10.0 "saxpy.fn"
  assertClose (relu.fn (envOf #[-5.0])).toFloat 0.0 "relu.fn neg"
  assertClose (relu.fn (envOf #[7.5])).toFloat 7.5 "relu.fn pos"
  assertClose (norm2.fn (envOf #[3.0, 4.0])).toFloat 5.0 "norm2.fn"
  assertClose (clampMax.fn (envOf #[-2.0, 0.0])).toFloat 0.0 "clampMax.fn"
  assertClose (minK.fn (envOf #[3.0, -1.5])).toFloat (-1.5) "minK.fn"
  assertClose (absK.fn (envOf #[-4.25])).toFloat 4.25 "absK.fn pos"
  assertClose (absK.fn (envOf #[4.25])).toFloat 4.25 "absK.fn id"
  IO.println "  ✓ fn values"

def testDenoteAgreesAtRuntime : IO Unit := do
  -- denote_eq proves this for all envs; spot-check the evaluation path anyway
  for v in [-3.5, -1.0, 0.0, 0.5, 2.25] do
    let env : Fin 1 → Float32 := envOf #[v]
    if (denote relu.expr env).toBits != (relu.fn env).toBits then
      throw (IO.userError s!"denote/fn mismatch at {v}")
  IO.println "  ✓ denote = fn (runtime spot check)"

/-! ## GPU end-to-end -/

def testMetal : IO Unit := do
  let available ← Metal.isAvailable
  if !available then
    IO.println "  (Metal not available, skipping GPU checks)"
    return
  let numel := 1024
  let xs := (Array.range numel).map fun i => (Float64.ofNat i - 512.0) / 64.0
  let ys := (Array.range numel).map fun i => (Float64.ofNat ((i * 7919) % 1000)) / 250.0 - 2.0
  let as := (Array.range numel).map fun i => (Float64.ofNat ((i * 104729) % 100)) / 50.0 - 1.0

  -- relu (1 input)
  let out ← relu.runMetal #[packF32 xs] numel
  let dec := out.decode
  for i in [:numel] do
    let expected := (relu.fn (envOf #[xs[i]!])).toFloat
    assertClose dec.data[i]! expected s!"relu gpu @{i}" 1e-6
  IO.println s!"  ✓ relu on GPU matches fn ({numel} elements)"

  -- saxpy (3 inputs)
  let out ← saxpy.runMetal #[packF32 as, packF32 xs, packF32 ys] numel
  let dec := out.decode
  for i in [:numel] do
    let expected := (saxpy.fn (envOf #[as[i]!, xs[i]!, ys[i]!])).toFloat
    assertClose dec.data[i]! expected s!"saxpy gpu @{i}" 1e-5
  IO.println s!"  ✓ saxpy on GPU matches fn ({numel} elements)"

  -- sigmoidish (transcendental path: exp2 + recip)
  let out ← sigmoidish.runMetal #[packF32 xs] numel
  let dec := out.decode
  for i in [:numel] do
    let expected := (sigmoidish.fn (envOf #[xs[i]!])).toFloat
    assertClose dec.data[i]! expected s!"sigmoidish gpu @{i}" 1e-5
  IO.println s!"  ✓ sigmoidish on GPU matches fn ({numel} elements)"

  -- min (where_/cmplt composition) and abs (max/neg composition)
  let out ← minK.runMetal #[packF32 xs, packF32 ys] numel
  let dec := out.decode
  for i in [:numel] do
    let expected := (minK.fn (envOf #[xs[i]!, ys[i]!])).toFloat
    assertClose dec.data[i]! expected s!"min gpu @{i}" 1e-6
  IO.println s!"  ✓ min on GPU matches fn ({numel} elements)"

  let out ← absK.runMetal #[packF32 xs] numel
  let dec := out.decode
  for i in [:numel] do
    let expected := (absK.fn (envOf #[xs[i]!])).toFloat
    assertClose dec.data[i]! expected s!"abs gpu @{i}" 1e-6
  IO.println s!"  ✓ abs on GPU matches fn ({numel} elements)"

def runAll : IO Unit := do
  IO.println "InlineKernelSmoke:"
  testFn
  testDenoteAgreesAtRuntime
  testMetal
  IO.println "InlineKernelSmoke: all passed"

end TinyGrad4.Test.InlineKernelSmoke

def main : IO Unit := TinyGrad4.Test.InlineKernelSmoke.runAll
