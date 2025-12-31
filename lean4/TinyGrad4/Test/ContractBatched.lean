import TinyGrad4

/-!
# Test: Batched CONTRACT (Matmul)

Checks that `.CONTRACT` supports broadcasting over leading (batch) dims:
  A: [b, m, k], B: [1, k, n] -> Out: [b, m, n]
-/

namespace TinyGrad4.Test

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

open TinyGrad4
open StaticTensor
open Interpreter

def batchedContractForward : IO Unit := do
  let ok := runTensorM do
    let env : Env := ∅

    -- B broadcast: [2,3,4] @ [1,4,5] -> [2,3,5]
    let a1 ← Tensor.full [2, 3, 4] .float32 1.0
    let b1 ← Tensor.full [1, 4, 5] .float32 2.0
    let out1 ← bmatmul a1 b1
    let outArr1 := eval out1.uop env

    -- A broadcast: [1,3,4] @ [2,4,5] -> [2,3,5]
    let a2 ← Tensor.full [1, 3, 4] .float32 1.0
    let b2 ← Tensor.full [2, 4, 5] .float32 2.0
    let out2 ← bmatmul a2 b2
    let outArr2 := eval out2.uop env

    -- No broadcast: [2,3,4] @ [2,4,5] -> [2,3,5]
    let a3 ← Tensor.full [2, 3, 4] .float32 1.0
    let b3 ← Tensor.full [2, 4, 5] .float32 2.0
    let out3 ← bmatmul a3 b3
    let outArr3 := eval out3.uop env

    let close8 := fun x => Float.abs (x - 8.0) < 0.0001
    pure (outArr1.all close8 && outArr2.all close8 && outArr3.all close8)

  if !ok then
    throw (IO.userError "Batched CONTRACT Forward Test: FAILED")
  IO.println "=== Batched CONTRACT Forward Test: ok ==="

#eval! TinyGrad4.Test.batchedContractForward

def batchedContractRankMismatch : IO Unit := do
  let (aU, bU, outU) := runTensorM do
    let a ← Tensor.buffer [2, 3, 4] .float32
    let b ← Tensor.buffer [3, 2, 4, 2] .float32
    let out ← UOp.contract2D a.uop b.uop
    pure (a.uop, b.uop, out)

  let aData : FloatArray :=
    ⟨(Array.replicate 12 1.0) ++ (Array.replicate 12 2.0)⟩
  let bData : FloatArray := ⟨Array.replicate (3 * 2 * 4 * 2) 1.0⟩
  let mut env : Env := ∅
  env := setBuffer env aU (RawBuffer.ofF32 aData)
  env := setBuffer env bU (RawBuffer.ofF32 bData)

  let out := eval outU env
  if out.size != 3 * 2 * 3 * 2 then
    throw (IO.userError s!"Batched CONTRACT RankMismatch: wrong size {out.size}")

  let mut ok := true
  for i0 in [:3] do
    let base := i0 * 12
    for j in [:6] do
      ok := ok && Float.abs (out[base + j]! - 4.0) < 0.0001
    for j in [:6] do
      ok := ok && Float.abs (out[base + 6 + j]! - 8.0) < 0.0001

  if !ok then
    throw (IO.userError "Batched CONTRACT RankMismatch: FAILED")
  IO.println "=== Batched CONTRACT RankMismatch Test: ok ==="

#eval! TinyGrad4.Test.batchedContractRankMismatch

end TinyGrad4.Test
