import TinyGrad4.Backend.MetalRenderer
import TinyGrad4.Backend.Pattern
import TinyGrad4.UOp.UOp

/-!
# Metal Renderer Tests

Tests for generating Metal shader source from UOp graphs.
-/

namespace TinyGrad4.Test.MetalRenderTest

open TinyGrad4.Backend.MetalRenderer
open TinyGrad4.Backend.Pattern

-- Helper to make test UOps
private def mkConst (id : Nat) (v : Float) : UOp :=
  { uid := ⟨id⟩
    op := .CONST
    dtype := .float32
    src := []
    arg := .constF32Bits v.toFloat32.toBits
    shape := [1] }

private def mkBuffer (id : Nat) (shape : Shape) : UOp :=
  { uid := ⟨id⟩
    op := .BUFFER
    dtype := .float32
    src := []
    arg := .empty
    shape := shape }

private def mkUnary (id : Nat) (op : Ops) (x : UOp) : UOp :=
  { uid := ⟨id⟩
    op := op
    dtype := .float32
    src := [x]
    arg := .empty
    shape := x.shape }

private def mkBinary (id : Nat) (op : Ops) (a b : UOp) : UOp :=
  { uid := ⟨id⟩
    op := op
    dtype := .float32
    src := [a, b]
    arg := .empty
    shape := a.shape }

def runAll : IO Unit := do
  -- Test simple add kernel
  do
    let a := mkBuffer 0 [1024]
    let b := mkBuffer 1 [1024]
    let c := mkBinary 2 .ADD a b
    let nodes := [a, b, c]

    let kernel := renderEwiseKernel "test_add" nodes ⟨2⟩
    IO.println "=== Simple Add Kernel ==="
    IO.println kernel

  -- Test fused add+mul kernel
  do
    let a := mkBuffer 0 [1024]
    let b := mkBuffer 1 [1024]
    let c := mkBinary 2 .ADD a b
    let d := mkBinary 3 .MUL c b
    let nodes := [a, b, c, d]

    let kernel := renderEwiseKernel "test_add_mul" nodes ⟨3⟩
    IO.println "=== Fused Add+Mul Kernel ==="
    IO.println kernel

  -- Test unary ops (neg, exp2, sin)
  do
    let a := mkBuffer 0 [1024]
    let b := mkUnary 1 .NEG a
    let c := mkUnary 2 .EXP2 b
    let d := mkUnary 3 .SIN c
    let nodes := [a, b, c, d]

    let kernel := renderEwiseKernel "test_unary_chain" nodes ⟨3⟩
    IO.println "=== Unary Chain Kernel ==="
    IO.println kernel

  -- Test complex expression: (a + b) * exp2(a - b)
  do
    let a := mkBuffer 0 [1024]
    let b := mkBuffer 1 [1024]
    let sum := mkBinary 2 .ADD a b
    let diff := mkBinary 3 .SUB a b
    let expDiff := mkUnary 4 .EXP2 diff
    let result := mkBinary 5 .MUL sum expDiff
    let nodes := [a, b, sum, diff, expDiff, result]

    let kernel := renderEwiseKernel "test_complex" nodes ⟨5⟩
    IO.println "=== Complex Expression Kernel ==="
    IO.println kernel

  -- Test softmax kernel generation
  do
    let info : SoftmaxInfo := {
      input := mkBuffer 0 [32, 1024]
      axis := 1
      axes := [1]
      isLog := false
    }
    let kernel := renderSoftmaxKernel "test_softmax" info
    IO.println "=== Softmax Kernel ==="
    IO.println kernel

  -- Test with constants
  do
    let a := mkBuffer 0 [1024]
    let two := mkConst 1 2.0
    let scaled := mkBinary 2 .MUL a two
    let nodes := [a, two, scaled]

    let kernel := renderEwiseKernel "test_scale" nodes ⟨2⟩
    IO.println "=== Scale by Constant Kernel ==="
    IO.println kernel

  -- Test relu: max(x, 0)
  do
    let a := mkBuffer 0 [1024]
    let zero := mkConst 1 0.0
    let relu := mkBinary 2 .MAX a zero
    let nodes := [a, zero, relu]

    let kernel := renderEwiseKernel "test_relu" nodes ⟨2⟩
    IO.println "=== ReLU Kernel ==="
    IO.println kernel

  IO.println "=== MetalRenderTest OK ==="

def main : IO Unit := runAll

end TinyGrad4.Test.MetalRenderTest
