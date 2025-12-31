import TinyGrad4.Ops
import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import TinyGrad4.Backend.Pattern

/-!
# Metal Shader Renderer

Generates Metal shader source from UOp graphs.
Uses Pattern module for declarative code generation rules.

## Architecture

```
UOp Graph → linearize → render each UOp → Metal source
```

Similar to tinygrad's cstyle.py renderer but leveraging Lean's
type system for correctness.

## Example Output

```metal
kernel void fused_add_mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float v0 = a[gid];
    float v1 = b[gid];
    float v2 = v0 + v1;
    float v3 = v2 * v1;
    out[gid] = v3;
}
```
-/

namespace TinyGrad4.Backend.MetalRenderer

open TinyGrad4.Backend.Pattern

/-! ## Metal Type Rendering -/

def renderDType : DType → String
  | .float32 => "float"
  | .float64 => "double"
  | .float16 => "half"
  | .int32 => "int"
  | .int64 => "long"
  | .int16 => "short"
  | .int8 => "char"
  | .uint32 => "uint"
  | .uint64 => "ulong"
  | .uint16 => "ushort"
  | .uint8 => "uchar"
  | .bool => "bool"
  | .index => "uint"
  | _ => "float"  -- fallback

/-! ## ALU Operation Rendering -/

def renderBinaryOp : Ops → String → String → String
  | .ADD, a, b => s!"{a} + {b}"
  | .SUB, a, b => s!"{a} - {b}"
  | .MUL, a, b => s!"{a} * {b}"
  | .FDIV, a, b => s!"{a} / {b}"
  | .MAX, a, b => s!"max({a}, {b})"
  | .MOD, a, b => s!"{a} % {b}"
  | .POW, a, b => s!"pow({a}, {b})"
  | .CMPLT, a, b => s!"{a} < {b}"
  | .CMPEQ, a, b => s!"{a} == {b}"
  | .CMPNE, a, b => s!"{a} != {b}"
  | .AND, a, b => s!"{a} & {b}"
  | .OR, a, b => s!"{a} | {b}"
  | .XOR, a, b => s!"{a} ^ {b}"
  | .SHL, a, b => s!"{a} << {b}"
  | .SHR, a, b => s!"{a} >> {b}"
  | _, a, b => s!"/* unknown */ ({a}, {b})"

def renderUnaryOp : Ops → String → String
  | .NEG, x => s!"-{x}"
  | .EXP2, x => s!"exp2({x})"
  | .LOG2, x => s!"log2({x})"
  | .SIN, x => s!"sin({x})"
  | .COS, x => s!"cos({x})"
  | .TAN, x => s!"tan({x})"
  | .SQRT, x => s!"sqrt({x})"
  | .RECIPROCAL, x => s!"(1.0f / {x})"
  | .TRUNC, x => s!"trunc({x})"
  | _, x => s!"/* unknown */ {x}"

def renderTernaryOp : Ops → String → String → String → String
  | .WHERE, c, t, f => s!"({c} ? {t} : {f})"
  | .MULACC, a, b, c => s!"fma({a}, {b}, {c})"
  | _, a, b, c => s!"/* unknown */ ({a}, {b}, {c})"

/-! ## Constant Rendering -/

def renderConst (arg : UArg) (dtype : DType) : String :=
  match arg with
  | .constFloat f =>
    if dtype == .float32 then s!"{f}f"
    else if dtype == .float64 then s!"{f}"
    else s!"({renderDType dtype})({f}f)"
  | .constF32Bits bits =>
    let f := (Float32.ofBits bits).toFloat
    s!"{f}f"
  | .constInt i =>
    if dtype == .int64 then s!"{i}ll"
    else if dtype == .uint64 then s!"{i}ull"
    else s!"{i}"
  | .constBool b => if b then "1" else "0"
  | _ => "0"

/-! ## Variable Naming -/

def varName (uid : UOpId) : String := s!"v{uid.id}"

def bufferName (idx : Nat) : String := s!"buf{idx}"

/-! ## UOp Rendering Context -/

structure RenderCtx where
  /-- Map from UOpId to rendered variable name -/
  vars : Std.HashMap UOpId String := {}
  /-- Buffer index counter -/
  nextBuf : Nat := 0
  /-- Map from buffer UOpId to buffer index -/
  buffers : Std.HashMap UOpId Nat := {}
  /-- Generated code lines -/
  lines : Array String := #[]

def RenderCtx.addLine (ctx : RenderCtx) (line : String) : RenderCtx :=
  { ctx with lines := ctx.lines.push line }

def RenderCtx.setVar (ctx : RenderCtx) (uid : UOpId) (name : String) : RenderCtx :=
  { ctx with vars := ctx.vars.insert uid name }

def RenderCtx.getVar (ctx : RenderCtx) (uid : UOpId) : String :=
  ctx.vars.getD uid s!"/* missing {uid} */"

def RenderCtx.registerBuffer (ctx : RenderCtx) (uid : UOpId) : RenderCtx × Nat :=
  match ctx.buffers.get? uid with
  | some idx => (ctx, idx)
  | none =>
    let idx := ctx.nextBuf
    ({ ctx with buffers := ctx.buffers.insert uid idx, nextBuf := idx + 1 }, idx)

/-! ## Single UOp Rendering -/

/-- Render a single UOp to Metal code -/
def renderUOp (u : UOp) (ctx : RenderCtx) : RenderCtx :=
  let name := varName u.uid
  let dtype := renderDType u.dtype

  match u.op with
  | .CONST =>
    let val := renderConst u.arg u.dtype
    let line := s!"  {dtype} {name} = {val};"
    (ctx.addLine line).setVar u.uid name

  | .BUFFER =>
    let (ctx', idx) := ctx.registerBuffer u.uid
    let bufName := s!"buf{idx}[gid]"
    ctx'.setVar u.uid bufName

  | .ADD | .SUB | .MUL | .FDIV | .MAX | .MOD | .POW
  | .CMPLT | .CMPEQ | .CMPNE | .AND | .OR | .XOR | .SHL | .SHR =>
    match u.src with
    | [a, b] =>
      let aVar := ctx.getVar a.uid
      let bVar := ctx.getVar b.uid
      let expr := renderBinaryOp u.op aVar bVar
      let line := s!"  {dtype} {name} = {expr};"
      (ctx.addLine line).setVar u.uid name
    | _ => ctx.addLine s!"  // ERROR: binary op {repr u.op} with {u.src.length} args"

  | .NEG | .EXP2 | .LOG2 | .SIN | .COS | .TAN | .SQRT | .RECIPROCAL | .TRUNC =>
    match u.src with
    | [x] =>
      let xVar := ctx.getVar x.uid
      let expr := renderUnaryOp u.op xVar
      let line := s!"  {dtype} {name} = {expr};"
      (ctx.addLine line).setVar u.uid name
    | _ => ctx.addLine s!"  // ERROR: unary op {repr u.op} with {u.src.length} args"

  | .WHERE =>
    match u.src with
    | [c, t, f] =>
      let cVar := ctx.getVar c.uid
      let tVar := ctx.getVar t.uid
      let fVar := ctx.getVar f.uid
      let expr := renderTernaryOp u.op cVar tVar fVar
      let line := s!"  {dtype} {name} = {expr};"
      (ctx.addLine line).setVar u.uid name
    | _ => ctx.addLine s!"  // ERROR: ternary op with {u.src.length} args"

  | .RESHAPE | .EXPAND | .PERMUTE =>
    -- Movement ops are identity in elementwise kernels
    match u.src with
    | [x] => ctx.setVar u.uid (ctx.getVar x.uid)
    | _ => ctx

  | _ => ctx.addLine s!"  // TODO: {repr u.op}"

/-! ## Kernel Generation -/

/-- Generate a complete Metal kernel for an elementwise fusion -/
def renderEwiseKernel (name : String) (nodes : List UOp) (outputUid : UOpId) : String := Id.run do
  -- Collect buffers (inputs + output)
  let mut ctx : RenderCtx := {}

  -- First pass: register all buffers
  for u in nodes do
    if u.op == .BUFFER then
      let (ctx', _) := ctx.registerBuffer u.uid
      ctx := ctx'

  -- Render each node
  for u in nodes do
    ctx := renderUOp u ctx

  -- Build kernel signature
  let mut params : Array String := #[]
  let bufList := ctx.buffers.toList.map (fun (uid, idx) => (idx, uid))
  let sortedBufs := bufList.toArray.qsort (fun a b => a.1 < b.1)

  for (idx, _uid) in sortedBufs do
    if idx < sortedBufs.size - 1 then
      params := params.push s!"    device const float* buf{idx} [[buffer({idx})]]"
    else
      params := params.push s!"    device float* buf{idx} [[buffer({idx})]]"

  let paramStr := String.intercalate ",\n" params.toList

  -- Build kernel body
  let bodyLines := ctx.lines.toList
  let bodyStr := String.intercalate "\n" bodyLines

  -- Output assignment
  let outVar := ctx.getVar outputUid
  let outBufIdx := ctx.buffers.size - 1

  s!"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
{paramStr},
    uint gid [[thread_position_in_grid]]
) \{
{bodyStr}
  buf{outBufIdx}[gid] = {outVar};
}"

/-! ## Pattern-Based Kernel Selection -/

/-- Check if a graph is purely elementwise (can use simple gid indexing) -/
def isPureEwise (nodes : List UOp) : Bool :=
  nodes.all fun u =>
    u.op.isALU || u.op == .BUFFER || u.op == .CONST ||
    u.op == .RESHAPE || u.op == .EXPAND

/-- Generate appropriate kernel based on pattern detection -/
def renderKernel (name : String) (nodes : List UOp) (outputUid : UOpId) : Option String := do
  -- For now, only support pure elementwise
  guard (isPureEwise nodes)
  pure (renderEwiseKernel name nodes outputUid)

/-! ## Softmax Kernel (uses Pattern.softmax?) -/

def renderSoftmaxKernel (name : String) (info : SoftmaxInfo) : String :=
  let axis := info.axis
  s!"#include <metal_stdlib>
using namespace metal;

// Fused softmax kernel (axis={axis})
// Pattern detected via Pattern.softmax?
kernel void {name}(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& outer [[buffer(2)]],
    constant uint& inner [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) \{
  uint batch = gid;
  if (batch >= outer) return;

  // Find max for numerical stability
  float max_val = -INFINITY;
  for (uint i = 0; i < inner; i++) \{
    float val = input[batch * inner + i];
    max_val = max(max_val, val);
  }

  // Compute exp and sum
  float sum_exp = 0.0f;
  for (uint i = 0; i < inner; i++) \{
    float val = exp(input[batch * inner + i] - max_val);
    output[batch * inner + i] = val;
    sum_exp += val;
  }

  // Normalize
  float inv_sum = 1.0f / sum_exp;
  for (uint i = 0; i < inner; i++) \{
    output[batch * inner + i] *= inv_sum;
  }
}"

/-! ## Vectorized Kernel Generation -/

/-- Generate a vectorized (float4) Metal kernel for elementwise operations.
    Each thread processes 4 elements, improving memory bandwidth.
    Requires size to be divisible by 4 (checked at runtime). -/
def renderEwiseVectorized (name : String) (nodes : List UOp) (outputUid : UOpId)
    (size : Nat) : Option String := Id.run do
  -- Only support pure elementwise for now
  if !isPureEwise nodes then return none

  -- Collect buffers
  let mut ctx : RenderCtx := {}
  for u in nodes do
    if u.op == .BUFFER then
      let (ctx', _) := ctx.registerBuffer u.uid
      ctx := ctx'

  -- Build vectorized kernel signature
  let bufList := ctx.buffers.toList.map (fun (uid, idx) => (idx, uid))
  let sortedBufs := bufList.toArray.qsort (fun a b => a.1 < b.1)

  let params := sortedBufs.foldl (init := #[]) fun acc (idx, _) =>
    if idx < sortedBufs.size - 1 then
      acc.push s!"    device const float* buf{idx} [[buffer({idx})]]"
    else
      acc.push s!"    device float* buf{idx} [[buffer({idx})]]"

  let paramStr := String.intercalate ",\n" params.toList
  let numVecIters := size / 4
  let outBufIdx := ctx.buffers.size - 1

  -- Generate vectorized body
  let mut bodyLines : Array String := #[]
  bodyLines := bodyLines.push s!"  uint i = gid * 4;"
  bodyLines := bodyLines.push s!"  if (gid >= {numVecIters}) return;"

  -- Load inputs as float4
  for (idx, _) in sortedBufs do
    if idx < sortedBufs.size - 1 then
      bodyLines := bodyLines.push s!"  float4 in{idx} = *((device const float4*)(buf{idx} + i));"

  -- Generate operation based on buffer count
  if sortedBufs.size == 3 then  -- 2 inputs + 1 output (binary op)
    -- Detect the operation from nodes
    let opStr := match nodes.find? (fun u => u.op.isALU && u.src.length == 2) with
      | some u =>
        match u.op with
        | .ADD => "in0 + in1"
        | .SUB => "in0 - in1"
        | .MUL => "in0 * in1"
        | .FDIV => "in0 / in1"
        | .MAX => "max(in0, in1)"
        | .POW => "pow(in0, in1)"
        | _ => "in0 + in1"
      | none => "in0 + in1"
    bodyLines := bodyLines.push s!"  float4 result = {opStr};"
  else if sortedBufs.size == 2 then  -- 1 input + 1 output (unary op)
    let opStr := match nodes.find? (fun u => u.op.isALU && u.src.length == 1) with
      | some u =>
        match u.op with
        | .NEG => "-in0"
        | .EXP2 => "exp2(in0)"
        | .LOG2 => "log2(in0)"
        | .SIN => "sin(in0)"
        | .COS => "cos(in0)"
        | .SQRT => "sqrt(in0)"
        | .RECIPROCAL => "1.0f / in0"
        | _ => "in0"
      | none => "in0"
    bodyLines := bodyLines.push s!"  float4 result = {opStr};"
  else
    bodyLines := bodyLines.push s!"  float4 result = in0;  // fallback"

  -- Store result
  bodyLines := bodyLines.push s!"  *((device float4*)(buf{outBufIdx} + i)) = result;"

  let vecBody := String.intercalate "\n" bodyLines.toList

  return some s!"#include <metal_stdlib>
using namespace metal;

// Vectorized kernel (float4, {numVecIters} iterations)
// Size {size} elements, aligned to 4
kernel void {name}(
{paramStr},
    uint gid [[thread_position_in_grid]]
) \{
{vecBody}
}"

/-- Render kernel with optional vectorization based on size alignment -/
def renderKernelAuto (name : String) (nodes : List UOp) (outputUid : UOpId)
    (size : Nat) : Option String :=
  if !isPureEwise nodes then none
  else if size % 4 = 0 then
    renderEwiseVectorized name nodes outputUid size
  else
    some (renderEwiseKernel name nodes outputUid)

/-! ## Reduction Kernels with Threadgroup Memory -/

/-- Reduction operation type -/
inductive ReduceOp where
  | sum : ReduceOp
  | max : ReduceOp
  | min : ReduceOp
  deriving Repr, DecidableEq

def ReduceOp.identity : ReduceOp → String
  | .sum => "0.0f"
  | .max => "-INFINITY"
  | .min => "INFINITY"

def ReduceOp.combine : ReduceOp → String → String → String
  | .sum, a, b => s!"{a} + {b}"
  | .max, a, b => s!"max({a}, {b})"
  | .min, a, b => s!"min({a}, {b})"

def ReduceOp.atomicOp : ReduceOp → String
  | .sum => "atomic_fetch_add_explicit"
  | .max => "atomic_fetch_max_explicit"
  | .min => "atomic_fetch_min_explicit"

/-- Generate a reduction kernel using threadgroup memory for parallel reduction.

    This kernel performs efficient parallel reduction within threadgroups:
    1. Each thread loads one element into threadgroup memory
    2. Tree-based parallel reduction within threadgroup (log2(N) steps)
    3. First thread writes the threadgroup result to output

    For large arrays, call this kernel multiple times or use a two-pass approach.

    Parameters:
    - name: kernel function name
    - reduceOp: SUM, MAX, or MIN
    - innerSize: number of elements to reduce (the reduction dimension)
    - outerSize: number of independent reductions (batch dimension)
    - threadgroupSize: threads per threadgroup (typically 256 or 1024)

    Generates Metal kernel like:
    ```metal
    kernel void reduce_sum(
        device const float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        threadgroup float* shared [[threadgroup(0)]],
        uint gid [[thread_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint tg_size [[threads_per_threadgroup]]
    ) {
        // Load to threadgroup memory
        shared[tid] = (gid < N) ? input[gid] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Tree reduction in threadgroup memory
        for (uint s = tg_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared[tid] += shared[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // First thread writes result
        if (tid == 0) {
            output[batch] = shared[0];
        }
    }
    ```
-/
def renderReduceKernel (name : String) (reduceOp : ReduceOp) (innerSize outerSize : Nat)
    (threadgroupSize : Nat := 256) : String :=
  let identity := reduceOp.identity
  let combine := reduceOp.combine "shared[tid]" "shared[tid + s]"
  -- Calculate number of threadgroups needed per row
  let threadgroupsPerRow := (innerSize + threadgroupSize - 1) / threadgroupSize

  s!"#include <metal_stdlib>
using namespace metal;

// Parallel reduction with threadgroup memory
// Reduces {innerSize} elements per row, {outerSize} rows
// Uses {threadgroupSize} threads per threadgroup, {threadgroupsPerRow} threadgroups per row
kernel void {name}(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& inner_size [[buffer(2)]],
    constant uint& outer_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) \{
    // Determine which row and position within row
    uint row = tgid / {threadgroupsPerRow};
    uint tg_in_row = tgid % {threadgroupsPerRow};
    uint elem_idx = tg_in_row * tg_size + tid;

    // Bounds check
    if (row >= outer_size) return;

    // Threadgroup shared memory for reduction
    threadgroup float shared[{threadgroupSize}];

    // Load to threadgroup memory (with bounds check)
    uint global_idx = row * inner_size + elem_idx;
    shared[tid] = (elem_idx < inner_size) ? input[global_idx] : {identity};
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree-based parallel reduction
    for (uint s = tg_size / 2; s > 0; s >>= 1) \{
        if (tid < s) \{
            shared[tid] = {combine};
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread of each threadgroup writes partial result
    if (tid == 0) \{
        // For single-threadgroup reduction, write directly to output
        // For multi-threadgroup, would need atomic or second pass
        if ({threadgroupsPerRow} == 1) \{
            output[row] = shared[0];
        } else \{
            // Partial result - would need atomic or second reduction pass
            // For now, use atomic add (works for sum, not ideal for max/min)
            // TODO: Use proper multi-pass reduction for max/min
            output[row * {threadgroupsPerRow} + tg_in_row] = shared[0];
        }
    }
}"

/-- Generate a simple single-pass reduction for small arrays (fits in one threadgroup) -/
def renderSimpleReduceKernel (name : String) (reduceOp : ReduceOp) (innerSize outerSize : Nat) : String :=
  let identity := reduceOp.identity
  let combine := reduceOp.combine "shared[tid]" "shared[tid + s]"
  -- Use threadgroup size that's power of 2 and >= innerSize
  let tgSize := if innerSize ≤ 32 then 32
    else if innerSize ≤ 64 then 64
    else if innerSize ≤ 128 then 128
    else if innerSize ≤ 256 then 256
    else if innerSize ≤ 512 then 512
    else 1024

  s!"#include <metal_stdlib>
using namespace metal;

// Simple single-threadgroup reduction (innerSize={innerSize} fits in tg_size={tgSize})
// Reduces each of {outerSize} rows independently
kernel void {name}(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) \{
    // Each threadgroup handles one row
    uint row = tgid;
    if (row >= {outerSize}) return;

    // Threadgroup shared memory
    threadgroup float shared[{tgSize}];

    // Load element (or identity if out of bounds)
    shared[tid] = (tid < {innerSize}) ? input[row * {innerSize} + tid] : {identity};
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint s = {tgSize / 2}; s > 0; s >>= 1) \{
        if (tid < s) \{
            shared[tid] = {combine};
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread writes result
    if (tid == 0) \{
        output[row] = shared[0];
    }
}"

/-- Choose the appropriate reduction kernel based on problem size -/
def renderReduceKernelAuto (name : String) (reduceOp : ReduceOp)
    (innerSize outerSize : Nat) : String :=
  if innerSize ≤ 1024 then
    -- Small enough for single threadgroup per row
    renderSimpleReduceKernel name reduceOp innerSize outerSize
  else
    -- Need multi-threadgroup reduction
    renderReduceKernel name reduceOp innerSize outerSize 256

/-- Convert Ops to ReduceOp -/
def opsToReduceOp : Ops → Option ReduceOp
  | .ADD => some .sum
  | .MAX => some .max
  | _ => none

/-! ## Matrix Multiplication Kernels -/

/-- Tile sizes for GEMM kernel. These should be tuned for the target GPU.
    - TILE_M, TILE_N: Output tile computed by each threadgroup
    - TILE_K: Reduction dimension chunk size for loading A and B tiles
    - THREADS_M, THREADS_N: Thread layout within threadgroup -/
structure GemmConfig where
  tileM : Nat := 32
  tileN : Nat := 32
  tileK : Nat := 8
  threadsM : Nat := 8
  threadsN : Nat := 8
  deriving Repr

/-- Default GEMM configuration tuned for Apple Silicon -/
def defaultGemmConfig : GemmConfig := {}

/-- Generate a tiled GEMM kernel for C = A @ B.
    A: [M, K], B: [K, N], C: [M, N]

    Each threadgroup computes a TILE_M × TILE_N block of output.
    Uses threadgroup memory to cache tiles of A and B.
    Each thread computes a 4×4 block of output using register blocking.

    This is a standard tiled GEMM suitable for most problem sizes.
    For very small matrices, consider a simpler approach.
-/
def renderGemmKernel (name : String) (m k n : Nat) (cfg : GemmConfig := defaultGemmConfig) : String :=
  let tileM := cfg.tileM
  let tileN := cfg.tileN
  let tileK := cfg.tileK
  let threadsM := cfg.threadsM
  let threadsN := cfg.threadsN
  -- Each thread computes a sub-tile
  let perThreadM := tileM / threadsM  -- e.g., 32/8 = 4
  let perThreadN := tileN / threadsN  -- e.g., 32/8 = 4

  s!"#include <metal_stdlib>
using namespace metal;

// Tiled GEMM: C[{m},{n}] = A[{m},{k}] @ B[{k},{n}]
// Tile: {tileM}x{tileN}, K-tile: {tileK}, Threads: {threadsM}x{threadsN}
// Each thread computes {perThreadM}x{perThreadN} output elements
constant uint M = {m};
constant uint K = {k};
constant uint N = {n};
constant uint TILE_M = {tileM};
constant uint TILE_N = {tileN};
constant uint TILE_K = {tileK};

kernel void {name}(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) \{
    // Threadgroup shared memory for tiles
    threadgroup float As[TILE_M][TILE_K];
    threadgroup float Bs[TILE_K][TILE_N];

    // Output tile position
    uint row_base = tgid.y * TILE_M;
    uint col_base = tgid.x * TILE_N;

    // Thread's position within the tile
    uint local_row = tid.y;  // 0..{threadsM-1}
    uint local_col = tid.x;  // 0..{threadsN-1}

    // Register accumulator for this thread's {perThreadM}x{perThreadN} sub-tile
    float acc[{perThreadM}][{perThreadN}];
    for (uint i = 0; i < {perThreadM}; i++) \{
        for (uint j = 0; j < {perThreadN}; j++) \{
            acc[i][j] = 0.0f;
        }
    }

    // Loop over K dimension in tiles
    for (uint t = 0; t < K; t += TILE_K) \{
        // Cooperatively load A tile [TILE_M x TILE_K] into shared memory
        // Each thread loads multiple elements
        for (uint i = local_row; i < TILE_M; i += {threadsM}) \{
            for (uint kk = local_col; kk < TILE_K; kk += {threadsN}) \{
                uint global_row = row_base + i;
                uint global_k = t + kk;
                As[i][kk] = (global_row < M && global_k < K) ? A[global_row * K + global_k] : 0.0f;
            }
        }

        // Cooperatively load B tile [TILE_K x TILE_N] into shared memory
        for (uint kk = local_row; kk < TILE_K; kk += {threadsM}) \{
            for (uint j = local_col; j < TILE_N; j += {threadsN}) \{
                uint global_k = t + kk;
                uint global_col = col_base + j;
                Bs[kk][j] = (global_k < K && global_col < N) ? B[global_k * N + global_col] : 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot products for this thread's sub-tile
        for (uint kk = 0; kk < TILE_K; kk++) \{
            for (uint i = 0; i < {perThreadM}; i++) \{
                float a_val = As[local_row * {perThreadM} + i][kk];
                for (uint j = 0; j < {perThreadN}; j++) \{
                    float b_val = Bs[kk][local_col * {perThreadN} + j];
                    acc[i][j] += a_val * b_val;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    for (uint i = 0; i < {perThreadM}; i++) \{
        for (uint j = 0; j < {perThreadN}; j++) \{
            uint out_row = row_base + local_row * {perThreadM} + i;
            uint out_col = col_base + local_col * {perThreadN} + j;
            if (out_row < M && out_col < N) \{
                C[out_row * N + out_col] = acc[i][j];
            }
        }
    }
}"

/-- Generate a simple matmul kernel for small matrices (no tiling).
    Good for M, K, N < 64 where tiling overhead isn't worth it. -/
def renderSimpleGemmKernel (name : String) (m k n : Nat) : String :=
  s!"#include <metal_stdlib>
using namespace metal;

// Simple GEMM: C[{m},{n}] = A[{m},{k}] @ B[{k},{n}]
// Each thread computes one output element
kernel void {name}(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) \{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= {m} || col >= {n}) return;

    float sum = 0.0f;
    for (uint k = 0; k < {k}; k++) \{
        sum += A[row * {k} + k] * B[k * {n} + col];
    }
    C[row * {n} + col] = sum;
}"

/-- Generate a batched matmul kernel: C[B, M, N] = A[B, M, K] @ B[B, K, N]
    or with broadcast: A[1, M, K] @ B[B, K, N] -> C[B, M, N]

    This is what conv2d needs - a batch of independent matrix multiplications.
-/
def renderBatchedGemmKernel (name : String) (batch m k n : Nat)
    (aBatch kBatch : Nat := 1) : String :=
  -- aBatch = 1 means A is shared across batch dimension
  -- kBatch = 1 means B is shared across batch dimension
  let aStride := if aBatch == 1 then 0 else m * k
  let bStride := if kBatch == 1 then 0 else k * n

  s!"#include <metal_stdlib>
using namespace metal;

// Batched GEMM: C[{batch},{m},{n}] = A @ B
// A stride: {aStride} (per batch), B stride: {bStride} (per batch)
// Each thread computes one output element
kernel void {name}(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) \{
    uint b = gid.z;   // batch index
    uint row = gid.y; // M index
    uint col = gid.x; // N index

    if (b >= {batch} || row >= {m} || col >= {n}) return;

    // Offset into A and B for this batch
    device const float* A_b = A + b * {aStride};
    device const float* B_b = B + b * {bStride};

    float sum = 0.0f;
    for (uint kk = 0; kk < {k}; kk++) \{
        sum += A_b[row * {k} + kk] * B_b[kk * {n} + col];
    }
    C[b * {m * n} + row * {n} + col] = sum;
}"

/-- Choose appropriate GEMM kernel based on matrix sizes -/
def renderGemmKernelAuto (name : String) (m k n : Nat) : String :=
  if m ≤ 64 && k ≤ 64 && n ≤ 64 then
    -- Small matrices: simple approach
    renderSimpleGemmKernel name m k n
  else
    -- Larger matrices: tiled approach
    renderGemmKernel name m k n

end TinyGrad4.Backend.MetalRenderer
