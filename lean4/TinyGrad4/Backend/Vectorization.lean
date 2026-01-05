import TinyGrad4.DType

/-!
# Vectorization Abstraction

Principled, composable vectorization for GPU kernels.

## Design Principles

1. **Type-safe vector widths** - Compile-time guarantees about valid widths
2. **Backend-agnostic** - Same abstraction works for CUDA, Metal, OpenCL
3. **Composable** - Vectorization is orthogonal to other kernel properties
4. **Size-aware** - Handles non-aligned sizes correctly

## Usage

```
let config := VectorConfig.optimal backend dtype size
let renderer := vectorizedRenderer baseRenderer config

let load := config.renderLoad "CUDA" "data" "idx"
-- => "*(float4*)(data + idx)"
```
-/

namespace TinyGrad4.Backend.Vectorization

/-! ## Vector Width -/

/-- Supported vector widths for SIMD operations -/
inductive VectorWidth where
  /-- Scalar width (fallback). -/
  | w1 : VectorWidth
  /-- 2-wide vector width (half4 on some GPUs). -/
  | w2 : VectorWidth
  /-- 4-wide vector width (optimal for most GPUs). -/
  | w4 : VectorWidth
  /-- 8-wide vector width (some AMD GPUs). -/
  | w8 : VectorWidth
  deriving Repr, DecidableEq, Inhabited

namespace VectorWidth

/-- Get numeric width -/
abbrev toNat : VectorWidth → Nat
  | w1 => 1
  | w2 => 2
  | w4 => 4
  | w8 => 8

/-- Create from numeric width (clamped to valid) -/
abbrev fromNat : Nat → VectorWidth
  | 1 => w1
  | 2 => w2
  | 4 => w4
  | 8 => w8
  | n => if n >= 8 then w8 else if n >= 4 then w4 else if n >= 2 then w2 else w1

/-- Is this scalar (no vectorization)? -/
abbrev isScalar : VectorWidth → Bool
  | w1 => true
  | _ => false

/-- Can the given size be evenly divided by this width? -/
def divides (w : VectorWidth) (size : Nat) : Bool :=
  size % w.toNat == 0

/-- Largest width that evenly divides the size -/
def maxDivisor (size : Nat) : VectorWidth :=
  if size % 8 == 0 then w8
  else if size % 4 == 0 then w4
  else if size % 2 == 0 then w2
  else w1

end VectorWidth

/-! ## Alignment Predicate -/

/-- Proof that a size is aligned to a vector width (divisible) -/
abbrev VectorAligned (w : VectorWidth) (n : Nat) : Prop := n % w.toNat = 0

/-- VectorAligned is decidable, so `native_decide` works. -/
instance (w : VectorWidth) (n : Nat) : Decidable (VectorAligned w n) :=
  inferInstanceAs (Decidable (n % w.toNat = 0))

/-- Component names indexed by width -/
def VectorWidth.componentNames : VectorWidth → Array String
  | .w1 => #[""]
  | .w2 => #["x", "y"]
  | .w4 => #["x", "y", "z", "w"]
  | .w8 => #["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]

/-- Get component name with bounds proof -/
def VectorWidth.component (w : VectorWidth) (i : Fin w.toNat) : String :=
  match w, i with
  | .w1, ⟨0, _⟩ => ""
  | .w2, ⟨0, _⟩ => "x" | .w2, ⟨1, _⟩ => "y"
  | .w4, ⟨0, _⟩ => "x" | .w4, ⟨1, _⟩ => "y" | .w4, ⟨2, _⟩ => "z" | .w4, ⟨3, _⟩ => "w"
  | .w8, ⟨0, _⟩ => "s0" | .w8, ⟨1, _⟩ => "s1" | .w8, ⟨2, _⟩ => "s2" | .w8, ⟨3, _⟩ => "s3"
  | .w8, ⟨4, _⟩ => "s4" | .w8, ⟨5, _⟩ => "s5" | .w8, ⟨6, _⟩ => "s6" | .w8, ⟨7, _⟩ => "s7"

/-! ## Backend-Specific Syntax -/

/-- Backend identifier for syntax selection -/
inductive Backend where
  /-- CUDA backend. -/
  | CUDA : Backend
  /-- Metal backend. -/
  | Metal : Backend
  /-- OpenCL backend. -/
  | OpenCL : Backend
  deriving Repr, DecidableEq

namespace Backend

/-- Parse backend from string -/
def fromString : String → Backend
  | "CUDA" => CUDA
  | "METAL" => Metal
  | "Metal" => Metal
  | "OPENCL" => OpenCL
  | "OpenCL" => OpenCL
  | _ => CUDA  -- Default

end Backend

/-! ## Vector Type Names -/

/-- Get the vector type name for a given backend, dtype, and width -/
def vectorTypeName (backend : Backend) (dtype : DType) (width : VectorWidth) : String :=
  match backend, dtype, width with
  -- CUDA float types
  | .CUDA, .float32, .w1 => "float"
  | .CUDA, .float32, .w2 => "float2"
  | .CUDA, .float32, .w4 => "float4"
  | .CUDA, .float32, .w8 => "float4"  -- No native float8, use 2x float4
  -- CUDA int types
  | .CUDA, .int32, .w1 => "int"
  | .CUDA, .int32, .w2 => "int2"
  | .CUDA, .int32, .w4 => "int4"
  | .CUDA, .int32, .w8 => "int4"
  -- Metal float types
  | .Metal, .float32, .w1 => "float"
  | .Metal, .float32, .w2 => "float2"
  | .Metal, .float32, .w4 => "float4"
  | .Metal, .float32, .w8 => "float4"
  -- Metal int types
  | .Metal, .int32, .w1 => "int"
  | .Metal, .int32, .w2 => "int2"
  | .Metal, .int32, .w4 => "int4"
  | .Metal, .int32, .w8 => "int4"
  -- Fallback
  | _, _, _ => "float"

/-! ## Vectorized Operations -/

/-- Configuration for vectorized kernel generation (legacy, untyped) -/
structure VectorConfig where
  /-- Target backend -/
  backend : Backend
  /-- Element data type -/
  dtype : DType := .float32
  /-- Vector width to use -/
  width : VectorWidth := .w4
  /-- Total number of elements -/
  totalSize : Nat
  /-- Whether to add bounds checking for non-aligned tail -/
  needsBoundsCheck : Bool := false
  deriving Repr

namespace VectorConfig

/-- Create optimal config for given size -/
def optimal (backend : Backend) (dtype : DType) (size : Nat) : VectorConfig :=
  let width := VectorWidth.maxDivisor size
  { backend, dtype, width, totalSize := size, needsBoundsCheck := !width.divides size }

/-- Number of vector iterations needed -/
def numVectorIters (c : VectorConfig) : Nat :=
  c.totalSize / c.width.toNat

/-- Number of scalar tail elements (if any) -/
def numTailElements (c : VectorConfig) : Nat :=
  c.totalSize % c.width.toNat

/-- Get vector type name -/
def vectorType (c : VectorConfig) : String :=
  vectorTypeName c.backend c.dtype c.width

/-- Get scalar type name -/
def scalarType (c : VectorConfig) : String :=
  vectorTypeName c.backend c.dtype .w1

end VectorConfig

/-! ## Type-Safe Vectorization -/

/-- Type-safe vectorization config with alignment proof -/
structure VectorConfigT (w : VectorWidth) (size : Nat) where
  /-- Target backend -/
  backend : Backend
  /-- Element data type -/
  dtype : DType := .float32
  /-- Proof that size is aligned to width -/
  h_aligned : VectorAligned w size := by native_decide
  deriving Repr

namespace VectorConfigT

variable {w : VectorWidth} {size : Nat}

/-- Number of vector iterations (guaranteed exact due to alignment) -/
def numVectorIters (_c : VectorConfigT w size) : Nat := size / w.toNat

/-- Get vector type name -/
def vectorType (c : VectorConfigT w size) : String :=
  vectorTypeName c.backend c.dtype w

/-- Get scalar type name -/
def scalarType (c : VectorConfigT w size) : String :=
  vectorTypeName c.backend c.dtype .w1

/-- Convert to legacy config for compatibility -/
def toLegacy (c : VectorConfigT w size) : VectorConfig :=
  { backend := c.backend, dtype := c.dtype, width := w, totalSize := size, needsBoundsCheck := false }

end VectorConfigT

/-- maxDivisor always produces aligned width -/
theorem VectorWidth.maxDivisor_aligned (size : Nat) : VectorAligned (maxDivisor size) size := by
  unfold VectorAligned maxDivisor
  split
  · simp only [toNat, Nat.beq_eq_true_eq] at *; assumption
  · split
    · simp only [toNat, Nat.beq_eq_true_eq] at *; assumption
    · split
      · simp only [toNat, Nat.beq_eq_true_eq] at *; assumption
      · simp only [toNat, Nat.mod_one]

/-- Create optimal typed config (chooses width that aligns) -/
def optimalConfig (backend : Backend) (dtype : DType) (size : Nat) :
    Σ w, VectorConfigT w size :=
  ⟨VectorWidth.maxDivisor size, { backend, dtype, h_aligned := VectorWidth.maxDivisor_aligned size }⟩

/-! ## Code Generation -/

/-- Generate a vectorized load expression -/
def renderVectorLoad (c : VectorConfig) (ptr : String) (idx : String) : String :=
  if c.width.isScalar then
    s!"{ptr}[{idx}]"
  else
    let vtype := c.vectorType
    match c.backend with
    | .CUDA => s!"*({vtype}*)({ptr} + {idx})"
    | .Metal => s!"*({vtype}*)({ptr} + {idx})"
    | .OpenCL => s!"vload{c.width.toNat}(0, {ptr} + {idx})"

/-- Generate a vectorized store expression -/
def renderVectorStore (c : VectorConfig) (ptr : String) (idx : String) (val : String) : String :=
  if c.width.isScalar then
    s!"{ptr}[{idx}] = {val}"
  else
    let vtype := c.vectorType
    match c.backend with
    | .CUDA => s!"*({vtype}*)({ptr} + {idx}) = {val}"
    | .Metal => s!"*({vtype}*)({ptr} + {idx}) = {val}"
    | .OpenCL => s!"vstore{c.width.toNat}({val}, 0, {ptr} + {idx})"

/-- Generate vector constructor for element-wise result -/
def renderMakeVector (c : VectorConfig) (elements : List String) : String :=
  if c.width.isScalar then
    elements.head!
  else
    let vtype := c.vectorType
    let args := String.intercalate ", " elements
    match c.backend with
    | .CUDA => s!"make_{vtype}({args})"
    | .Metal => s!"{vtype}({args})"
    | .OpenCL => s!"({vtype})({args})"

/-- Generate element access for a vector (legacy, unbounded) -/
def renderVectorElement (c : VectorConfig) (vec : String) (idx : Nat) : String :=
  if c.width.isScalar then vec
  else
    let suffix := match idx with
      | 0 => "x"
      | 1 => "y"
      | 2 => "z"
      | 3 => "w"
      | _ => s!"s{idx}"  -- OpenCL style for > 4
    s!"{vec}.{suffix}"

/-- Type-safe vector element access with bounds proof -/
def renderVectorElementT (w : VectorWidth) (vec : String) (idx : Fin w.toNat) : String :=
  if w.isScalar then vec
  else s!"{vec}.{w.component idx}"

/-- Component names for iteration -/
def vectorComponents (width : VectorWidth) : List String :=
  match width with
  | .w1 => [""]
  | .w2 => ["x", "y"]
  | .w4 => ["x", "y", "z", "w"]
  | .w8 => ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]

/-! ## Kernel Templates -/

/-- Generate a complete vectorized binary operation kernel -/
def renderVectorizedBinaryKernel
    (c : VectorConfig)
    (kernelName : String)
    (op : String)  -- e.g., "+" or "fmaxf"
    : String :=
  let vtype := c.vectorType
  let stype := c.scalarType
  let width := c.width.toNat
  let numIters := c.numVectorIters

  match c.backend with
  | .CUDA =>
    let _blockSize := 256  -- Used by caller for launch config
    let components := vectorComponents c.width

    -- Build the vector operation
    let vecOp := if c.width.isScalar then
      s!"out[i] = a[i] {op} b[i];"
    else
      let elementOps := components.map fun comp =>
        s!"va.{comp} {op} vb.{comp}"
      let makeVec := s!"make_{vtype}({String.intercalate ", " elementOps})"
      s!"*({vtype}*)(out + i) = {makeVec};"

    s!"extern \"C\" __global__ void {kernelName}(
    {stype}* __restrict__ out,
    const {stype}* __restrict__ a,
    const {stype}* __restrict__ b
) \{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid * {width};
    if (tid < {numIters}) \{
        {vtype} va = *({vtype}*)(a + i);
        {vtype} vb = *({vtype}*)(b + i);
        {vecOp}
    }
}"

  | .Metal =>
    let components := vectorComponents c.width

    let vecOp := if c.width.isScalar then
      s!"out[gid] = a[gid] {op} b[gid];"
    else
      let elementOps := components.map fun comp =>
        s!"va.{comp} {op} vb.{comp}"
      let makeVec := s!"{vtype}({String.intercalate ", " elementOps})"
      s!"*({vtype}*)(out + i) = {makeVec};"

    s!"#include <metal_stdlib>
using namespace metal;

kernel void {kernelName}(
    device {stype}* out [[buffer(0)]],
    device const {stype}* a [[buffer(1)]],
    device const {stype}* b [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) \{
    int i = gid * {width};
    if (gid < {numIters}) \{
        {vtype} va = *({vtype}*)(a + i);
        {vtype} vb = *({vtype}*)(b + i);
        {vecOp}
    }
}"

  | .OpenCL =>
    s!"// OpenCL vectorized kernel (TODO)"

/-- Generate optimized vectorized add kernel (specialized) -/
def renderVectorizedAddKernel (c : VectorConfig) (kernelName : String) : String :=
  renderVectorizedBinaryKernel c kernelName "+"

/-- Generate optimized vectorized mul kernel (specialized) -/
def renderVectorizedMulKernel (c : VectorConfig) (kernelName : String) : String :=
  renderVectorizedBinaryKernel c kernelName "*"

end TinyGrad4.Backend.Vectorization
