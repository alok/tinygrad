import Float64
import TinyGrad4.Kernel.Spec

/-!
# Kernel Codegen (spec → device source)

Renders `Kernel.Expr` (the typed spec-level kernel language from `Kernel/Spec.lean`)
directly into device source code: Metal Shading Language and portable C.

This is the "cut abstraction layers" path: the same `Expr` value is
1. evaluated by `evalExpr` (the executable spec),
2. rendered to device source by `metalSource`/`cSource`,
and the inline-codegen elaborator (`Kernel/Inline.lean`) runs step 2 at
*elaboration time*, embedding the source as a string literal in the binary.

Float constants render as C99/MSL hex-float literals (bit-exact round trip,
no decimal parsing ambiguity).

Caveat: the Metal runtime compiles shaders with fast math (`MTLMathModeFast`
in `tg4_metal.m`), so on the GPU subnormals may flush to zero and INFINITY/NAN
propagation through arithmetic is not guaranteed. The C rendering and `fn`
preserve full IEEE semantics; treat non-finite/subnormal literals as
best-effort on Metal.
-/

namespace TinyGrad4.Kernel

/-- Canonical `ScalarOps` over native `Float32`.
    Fields are eta-reduced native ops so `evalExpr f32Ops` unfolds definitionally
    to plain Float32 arithmetic — this is what makes `InlineKernel.denote_eq`
    provable by `rfl`. -/
def f32Ops : ScalarOps Float32 where
  neg := (- ·)
  sqrt := Float32.sqrt
  reciprocal := ((1.0 : Float32) / ·)
  exp2 := Float32.exp2
  log2 := Float32.log2
  sin := Float32.sin
  add := (· + ·)
  sub := (· - ·)
  mul := (· * ·)
  div := (· / ·)
  max := Max.max
  cmplt := fun a b => decide (a < b)
  where_ := fun c x y => if c then x else y
  zero := 0.0
  negInf := (-1.0 : Float32) / 0.0

/-- Read a `Fin n`-indexed float environment; bool inputs and out-of-range reads
    are absent from reified kernels, so they default. -/
def readEnv {n : Nat} (env : Fin n → Float32) : (t : Ty) → Nat → t.denote
  | .f32, i => if h : i < n then env ⟨i, h⟩ else (0.0 : Float32)
  | .bool, _ => false

/-- Reference semantics of a spec expression over an `n`-ary float environment. -/
def denote {n : Nat} (e : Expr .f32) (env : Fin n → Float32) : Float32 :=
  evalExpr f32Ops (readEnv env) e

/-! ## Float32 → hex-float literal (bit-exact) -/

private def hexDigit (n : Nat) : Char := "0123456789abcdef".toList.getD n '0'

private def mantissaHex (mant : Nat) : String :=
  -- 23 mantissa bits, shifted to 24 for six exact hex fraction digits
  let m := mant <<< 1
  String.ofList ((List.range 6).map fun i => hexDigit ((m >>> ((5 - i) * 4)) &&& 0xF))

/-- Render a `Float32` as a C99/MSL hex-float literal, e.g. `0x1.800000p+0f`.
    Bit-exact for all finite values; ±INFINITY/NAN use the standard macros. -/
def f32Lit (x : Float32) : String :=
  let bits := x.toBits
  let sign := if bits >>> 31 == 1 then "-" else ""
  let expo := ((bits >>> 23) &&& 0xFF).toNat
  let mant := (bits &&& 0x7FFFFF).toNat
  -- negative literals are parenthesized: `x-(-1.0f)` must not lex as `x--1.0f`
  if expo == 0xFF then
    if mant == 0 then (if sign == "-" then "(-INFINITY)" else "INFINITY") else "NAN"
  else if expo == 0 then
    if mant == 0 then (if sign == "-" then "(-0.0f)" else "0.0f")
    else s!"({sign}0x0.{mantissaHex mant}p-126f)"  -- subnormal
  else
    let e : Int := Int.ofNat expo - 127
    let esign := if e < 0 then "-" else "+"
    s!"({sign}0x1.{mantissaHex mant}p{esign}{e.natAbs}f)"

/-! ## Scalar expression rendering -/

/-- Render the scalar computation of an `Expr`, reading input `i` as `x{i}`.
    Output is valid in both MSL and C. `max` renders as `((a<=b)?b:a)` — the
    exact semantics of Lean's `Max.max Float32` (`maxOfLe`), including NaN
    propagation and signed zero, which `fmaxf`/`metal::max` do NOT share. -/
def Expr.toCode : Expr t → String
  | .input _ idx => s!"x{idx}"
  | .constBool b => if b then "1" else "0"
  | .constF32 v => f32Lit v
  | .neg a => s!"(-{a.toCode})"
  | .sqrt a => s!"sqrt({a.toCode})"
  | .reciprocal a => s!"(1.0f/{a.toCode})"
  | .exp2 a => s!"exp2({a.toCode})"
  | .log2 a => s!"log2({a.toCode})"
  | .sin a => s!"sin({a.toCode})"
  | .add a b => s!"({a.toCode}+{b.toCode})"
  | .sub a b => s!"({a.toCode}-{b.toCode})"
  | .mul a b => s!"({a.toCode}*{b.toCode})"
  | .div a b => s!"({a.toCode}/{b.toCode})"
  | .max a b => s!"(({a.toCode}<={b.toCode})?{b.toCode}:{a.toCode})"
  | .cmplt a b => s!"({a.toCode}<{b.toCode})"
  | .where_ c x y => s!"({c.toCode}?{x.toCode}:{y.toCode})"
  | .truthy a => s!"({a.toCode}!=0.0f)"

/-- Vectorized (float4) rendering of the same scalar computation. Differences
    from `toCode`: comparisons yield `bool4`, so selection uses MSL `select`
    (componentwise, else-then-cond order) instead of the scalar ternary; `max`
    becomes `select(a, b, a<=b)` — still exactly `maxOfLe` per lane. Literals
    broadcast implicitly. Targets the `kernel!` fragment (no `constBool` in
    condition position). -/
def Expr.toCodeVec : Expr t → String
  | .input _ idx => s!"x{idx}"
  | .constBool b => if b then "1" else "0"
  | .constF32 v => f32Lit v
  | .neg a => s!"(-{a.toCodeVec})"
  | .sqrt a => s!"sqrt({a.toCodeVec})"
  | .reciprocal a => s!"(1.0f/{a.toCodeVec})"
  | .exp2 a => s!"exp2({a.toCodeVec})"
  | .log2 a => s!"log2({a.toCodeVec})"
  | .sin a => s!"sin({a.toCodeVec})"
  | .add a b => s!"({a.toCodeVec}+{b.toCodeVec})"
  | .sub a b => s!"({a.toCodeVec}-{b.toCodeVec})"
  | .mul a b => s!"({a.toCodeVec}*{b.toCodeVec})"
  | .div a b => s!"({a.toCodeVec}/{b.toCodeVec})"
  | .max a b => s!"select({a.toCodeVec},{b.toCodeVec},({a.toCodeVec}<={b.toCodeVec}))"
  | .cmplt a b => s!"({a.toCodeVec}<{b.toCodeVec})"
  | .where_ c x y => s!"select({y.toCodeVec},{x.toCodeVec},{c.toCodeVec})"
  | .truthy a => s!"({a.toCodeVec}!=0.0f)"

/-- Highest float input index + 1 (minimum environment arity). -/
def Expr.arity : Expr t → Nat
  | .input _ idx => idx + 1
  | .constBool _ | .constF32 _ => 0
  | .neg a | .sqrt a | .reciprocal a | .exp2 a | .log2 a | .sin a | .truthy a => a.arity
  | .add a b | .sub a b | .mul a b | .div a b | .max a b | .cmplt a b => Nat.max a.arity b.arity
  | .where_ c x y => Nat.max c.arity (Nat.max x.arity y.arity)

private def loadLines (arity : Nat) (indexVar : String) : String :=
  String.join ((List.range arity).map fun i => s!"  float x{i} = in{i}[{indexVar}];\n")

/-- Full Metal compute kernel: one thread per element, `arity` input buffers,
    one output buffer. Matches the repo's `runEwiseKernel` calling convention. -/
def metalSource (name : String) (arity : Nat) (e : Expr .f32) : String := Id.run do
  let mut params := ""
  for i in [:arity] do
    params := params ++ s!"  device const float* in{i} [[buffer({i})]],\n"
  params := params ++ s!"  device float* out [[buffer({arity})]],\n"
  params := params ++ "  uint gid [[thread_position_in_grid]]"
  s!"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
{params}
) \{
{loadLines arity "gid"}  out[gid] = {e.toCode};
}
"

/-- Vectorized Metal kernel: one thread per FOUR elements via `float4` loads.
    Launch with `numel / 4` threads; requires `numel % 4 == 0`. -/
def metalSourceVec (name : String) (arity : Nat) (e : Expr .f32) : String := Id.run do
  let mut params := ""
  for i in [:arity] do
    params := params ++ s!"  device const float4* in{i} [[buffer({i})]],\n"
  params := params ++ s!"  device float4* out [[buffer({arity})]],\n"
  params := params ++ "  uint gid [[thread_position_in_grid]]"
  let mut loads := ""
  for i in [:arity] do
    loads := loads ++ s!"  float4 x{i} = in{i}[gid];\n"
  s!"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
{params}
) \{
{loads}  out[gid] = {e.toCodeVec};
}
"

/-- Full C kernel: sequential loop over `n` elements. -/
def cSource (name : String) (arity : Nat) (e : Expr .f32) : String := Id.run do
  let mut params := ""
  for i in [:arity] do
    params := params ++ s!"const float* in{i}, "
  s!"#include <math.h>
#define sqrt(x) sqrtf(x)
#define exp2(x) exp2f(x)
#define log2(x) log2f(x)
#define sin(x) sinf(x)

void {name}({params}float* out, unsigned long n) \{
  for (unsigned long i = 0; i < n; i++) \{
{loadLines arity "i"}    out[i] = {e.toCode};
  }
}
"

/-! ## Proof-carrying inline kernel -/

/-- A kernel whose device source was generated at elaboration time by `kernel!`
    (see `Kernel/Inline.lean`), carrying its own spec (`expr`), a direct native
    implementation (`fn`), and the proof that the two agree (`denote_eq`).

    `metal`/`cSrc` are string *literals* in the compiled binary — inline codegen,
    no runtime rendering. -/
structure InlineKernel (n : Nat) where
  name : String
  expr : Expr .f32
  metal : String
  metalVec : String  -- float4 variant (kernel name `{name}_v4`, numel/4 threads)
  cSrc : String
  fn : (Fin n → Float32) → Float32
  denote_eq : ∀ env : Fin n → Float32, denote (n := n) expr env = fn env

end TinyGrad4.Kernel
