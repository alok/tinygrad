import TinyGrad4.UOp.UOp
import TinyGrad4.Backend.MetalRenderer

/-!
# Benchmark Kernel Generation

Generates Metal shaders for benchmarks using the actual MetalRenderer.
This ensures benchmarks test the real codegen path.
-/

namespace TinyGrad4.Benchmark.Kernels

open TinyGrad4
open TinyGrad4.Backend.MetalRenderer

/-! ## UOp Graph Construction for Benchmarks -/

/-- Build a simple binary elementwise UOp graph: out = a op b -/
def mkBinaryEwiseGraph (op : Ops) : List UOp × UOpId := Id.run do
  -- Buffer a (input 0)
  let bufA : UOp := {
    uid := ⟨0⟩
    op := .BUFFER
    dtype := .float32
    src := []
    arg := .bufferIdx 0
    shape := []
  }
  -- Buffer b (input 1)
  let bufB : UOp := {
    uid := ⟨1⟩
    op := .BUFFER
    dtype := .float32
    src := []
    arg := .bufferIdx 1
    shape := []
  }
  -- Buffer out (output 2) - needed for proper 3-buffer signature
  let bufOut : UOp := {
    uid := ⟨2⟩
    op := .BUFFER
    dtype := .float32
    src := []
    arg := .bufferIdx 2
    shape := []
  }
  -- Operation: a op b
  let result : UOp := {
    uid := ⟨3⟩
    op := op
    dtype := .float32
    src := [bufA, bufB]
    arg := .empty
    shape := []
  }
  ([bufA, bufB, bufOut, result], ⟨3⟩)

/-- Build a unary elementwise UOp graph: out = op(a) -/
def mkUnaryEwiseGraph (op : Ops) : List UOp × UOpId := Id.run do
  -- Buffer a (input 0)
  let bufA : UOp := {
    uid := ⟨0⟩
    op := .BUFFER
    dtype := .float32
    src := []
    arg := .bufferIdx 0
    shape := []
  }
  -- Buffer out (output 1)
  let bufOut : UOp := {
    uid := ⟨1⟩
    op := .BUFFER
    dtype := .float32
    src := []
    arg := .bufferIdx 1
    shape := []
  }
  -- Operation: op(a)
  let result : UOp := {
    uid := ⟨2⟩
    op := op
    dtype := .float32
    src := [bufA]
    arg := .empty
    shape := []
  }
  ([bufA, bufOut, result], ⟨2⟩)

/-- Build a fused add-mul graph: out = (a + b) * c -/
def mkFusedAddMulGraph : List UOp × UOpId := Id.run do
  let bufA : UOp := { uid := ⟨0⟩, op := .BUFFER, dtype := .float32, src := [], arg := .bufferIdx 0, shape := [] }
  let bufB : UOp := { uid := ⟨1⟩, op := .BUFFER, dtype := .float32, src := [], arg := .bufferIdx 1, shape := [] }
  let bufC : UOp := { uid := ⟨2⟩, op := .BUFFER, dtype := .float32, src := [], arg := .bufferIdx 2, shape := [] }
  let bufOut : UOp := { uid := ⟨3⟩, op := .BUFFER, dtype := .float32, src := [], arg := .bufferIdx 3, shape := [] }
  let add : UOp := { uid := ⟨4⟩, op := .ADD, dtype := .float32, src := [bufA, bufB], arg := .empty, shape := [] }
  let mul : UOp := { uid := ⟨5⟩, op := .MUL, dtype := .float32, src := [add, bufC], arg := .empty, shape := [] }
  ([bufA, bufB, bufC, bufOut, add, mul], ⟨5⟩)

/-! ## Kernel Types for Benchmarks -/

/-- Benchmark kernel specification -/
inductive BenchKernel where
  | add       : BenchKernel  -- a + b
  | mul       : BenchKernel  -- a * b
  | sub       : BenchKernel  -- a - b
  | div       : BenchKernel  -- a / b
  | max       : BenchKernel  -- max(a, b)
  | neg       : BenchKernel  -- -a
  | exp2      : BenchKernel  -- exp2(a)
  | sin       : BenchKernel  -- sin(a)
  | sqrt      : BenchKernel  -- sqrt(a)
  | recip     : BenchKernel  -- 1/a
  | addMul    : BenchKernel  -- (a + b) * c (fused)
  | reduceSum : BenchKernel  -- sum(a)
  | reduceMax : BenchKernel  -- max(a)
  deriving Repr, DecidableEq

def BenchKernel.name : BenchKernel → String
  | .add => "add"
  | .mul => "mul"
  | .sub => "sub"
  | .div => "div"
  | .max => "max_ew"
  | .neg => "neg"
  | .exp2 => "exp2"
  | .sin => "sin"
  | .sqrt => "sqrt"
  | .recip => "recip"
  | .addMul => "fused_add_mul"
  | .reduceSum => "reduce_sum"
  | .reduceMax => "reduce_max"

/-- Number of input buffers for a kernel -/
def BenchKernel.numInputs : BenchKernel → Nat
  | .add | .mul | .sub | .div | .max => 2
  | .neg | .exp2 | .sin | .sqrt | .recip => 1
  | .addMul => 3
  | .reduceSum | .reduceMax => 1

/-- Number of FLOPs per element -/
def BenchKernel.flopsPerElement : BenchKernel → Nat
  | .add | .mul | .sub | .div | .max | .neg => 1
  | .exp2 | .sin | .sqrt | .recip => 1  -- Counted as 1 for simplicity
  | .addMul => 2  -- 1 add + 1 mul
  | .reduceSum | .reduceMax => 1

/-! ## Shader Generation -/

/-- Generate Metal shader for a benchmark kernel (elementwise) -/
def generateEwiseShader (kernel : BenchKernel) (size : Nat) : Option String := do
  match kernel with
  | .add =>
    let (nodes, outId) := mkBinaryEwiseGraph .ADD
    renderKernelAuto "add" nodes outId size
  | .mul =>
    let (nodes, outId) := mkBinaryEwiseGraph .MUL
    renderKernelAuto "mul" nodes outId size
  | .sub =>
    let (nodes, outId) := mkBinaryEwiseGraph .SUB
    renderKernelAuto "sub" nodes outId size
  | .div =>
    let (nodes, outId) := mkBinaryEwiseGraph .FDIV
    renderKernelAuto "div" nodes outId size
  | .max =>
    let (nodes, outId) := mkBinaryEwiseGraph .MAX
    renderKernelAuto "max_ew" nodes outId size
  | .neg =>
    let (nodes, outId) := mkUnaryEwiseGraph .NEG
    renderKernelAuto "neg" nodes outId size
  | .exp2 =>
    let (nodes, outId) := mkUnaryEwiseGraph .EXP2
    renderKernelAuto "exp2" nodes outId size
  | .sin =>
    let (nodes, outId) := mkUnaryEwiseGraph .SIN
    renderKernelAuto "sin" nodes outId size
  | .sqrt =>
    let (nodes, outId) := mkUnaryEwiseGraph .SQRT
    renderKernelAuto "sqrt" nodes outId size
  | .recip =>
    let (nodes, outId) := mkUnaryEwiseGraph .RECIPROCAL
    renderKernelAuto "recip" nodes outId size
  | .addMul =>
    let (nodes, outId) := mkFusedAddMulGraph
    renderKernelAuto "fused_add_mul" nodes outId size
  | .reduceSum | .reduceMax => none  -- Use reduce-specific generator

/-- Generate Metal shader for a reduce kernel -/
def generateReduceShader (kernel : BenchKernel) (innerSize outerSize : Nat) : Option String := do
  match kernel with
  | .reduceSum => some (renderReduceKernelAuto "reduce_sum" .sum innerSize outerSize)
  | .reduceMax => some (renderReduceKernelAuto "reduce_max" .max innerSize outerSize)
  | _ => none

/-- Generate Metal shader for any benchmark kernel -/
def generateShader (kernel : BenchKernel) (size : Nat) : Option String :=
  match kernel with
  | .reduceSum | .reduceMax => generateReduceShader kernel size 1  -- Single reduction
  | k => generateEwiseShader k size

/-! ## Convenience: Print shader for testing -/

def printShader (kernel : BenchKernel) (size : Nat := 1000000) : IO Unit := do
  match generateShader kernel size with
  | some shader => IO.println shader
  | none => IO.println s!"Failed to generate shader for {repr kernel}"

end TinyGrad4.Benchmark.Kernels
