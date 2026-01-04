import TinyGrad4

/-!
# Fused Matmul Kernel Tests

Directly exercises the portable C matmul+bias kernel:
- bias broadcasting to [m, n]
- matches separate matmul + add result
-/

namespace TinyGrad4.Test.FusedMatmul

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

open TinyGrad4
open StaticTensor
open Interpreter
open Backend

/-- Pack float64 array to float32 bytes -/
private def packF32 (data : Array Float) : ByteArray :=
  Native.packF32FromF64 ⟨data⟩

private def assertAllClose (arr : Array Float) (expected : Array Float) (tol : Float) (label : String) : IO Unit := do
  if arr.size != expected.size then
    throw (IO.userError s!"{label}: size {arr.size} != {expected.size}")
  for i in [:arr.size] do
    let v := arr[i]!
    let e := expected[i]!
    let diff := Float.abs (v - e)
    if diff > tol then
      throw (IO.userError s!"{label}: idx {i} value {v} expected {e} diff {diff} > {tol}")

private def testMatmulBias : IO Unit := do
  let a : Array Float := #[1.0, 1.0, 1.0,  1.0, 1.0, 1.0]  -- [2,3]
  let b : Array Float := #[0.5, 0.5,  0.5, 0.5,  0.5, 0.5]  -- [3,2]
  let bias : Array Float := #[0.1, 0.2]                     -- [2]
  let ab := packF32 a
  let bb := packF32 b
  let biasb := packF32 bias

  let outBytes := Native.matmulBiasF32 ab bb biasb #[2] 2 3 2
  let out := (RawBuffer.decode { dtype := .float32, data := outBytes }).data

  -- A@B = 1.5 for each output; add bias [0.1,0.2]
  assertAllClose out #[1.6, 1.7, 1.6, 1.7] 0.0001 "matmul+bias"

private def testMatmulBiasRelu : IO Unit := do
  let a : Array Float := #[1.0, 1.0, 1.0,  1.0, 1.0, 1.0]   -- [2,3]
  let b : Array Float := #[0.5, 0.5,  0.5, 0.5,  0.5, 0.5]   -- [3,2]
  let bias : Array Float := #[-2.0, 0.2]                     -- [2]
  let ab := packF32 a
  let bb := packF32 b
  let biasb := packF32 bias

  let outBytes := Native.matmulBiasReluF32 ab bb biasb #[2] 2 3 2
  let out := (RawBuffer.decode { dtype := .float32, data := outBytes }).data

  -- A@B = 1.5; add bias [-2.0,0.2] -> [-0.5, 1.7]; relu -> [0.0, 1.7]
  assertAllClose out #[0.0, 1.7, 0.0, 1.7] 0.0001 "matmul+bias+relu"

private def testMatmulBias2 : IO Unit := do
  let a : Array Float := #[1.0, 1.0, 1.0,  1.0, 1.0, 1.0]  -- [2,3]
  let b : Array Float := #[0.5, 0.5,  0.5, 0.5,  0.5, 0.5]  -- [3,2]
  let bias0 : Array Float := #[0.1, 0.2]                    -- [2]
  let bias1 : Array Float := #[1.0, 2.0,  3.0, 4.0]         -- [2,2]
  let ab := packF32 a
  let bb := packF32 b
  let b0 := packF32 bias0
  let b1 := packF32 bias1

  let outBytes := Native.matmulBias2F32 ab bb b0 #[2] b1 #[2, 2] 2 3 2
  let out := (RawBuffer.decode { dtype := .float32, data := outBytes }).data

  -- A@B = 1.5; +bias0 (broadcast [2]) +bias1 ([2,2])
  assertAllClose out #[2.6, 3.7, 4.6, 5.7] 0.0001 "matmul+bias2"

private def testMatmulBias2Relu : IO Unit := do
  let a : Array Float := #[1.0, 1.0, 1.0,  1.0, 1.0, 1.0]   -- [2,3]
  let b : Array Float := #[0.5, 0.5,  0.5, 0.5,  0.5, 0.5]   -- [3,2]
  let bias0 : Array Float := #[-2.0, 0.2]                    -- [2]
  let bias1 : Array Float := #[0.0]                          -- []
  let ab := packF32 a
  let bb := packF32 b
  let b0 := packF32 bias0
  let b1 := packF32 bias1

  let outBytes := Native.matmulBias2ReluF32 ab bb b0 #[2] b1 #[] 2 3 2
  let out := (RawBuffer.decode { dtype := .float32, data := outBytes }).data

  -- A@B = 1.5; +bias0 [-2.0,0.2] -> [-0.5, 1.7]; relu -> [0.0, 1.7]
  assertAllClose out #[0.0, 1.7, 0.0, 1.7] 0.0001 "matmul+bias2+relu"

private def testMatmulBatchedBias : IO Unit := do
  let a : Array Float := #[1.0, 1.0, 1.0,  1.0, 1.0, 1.0,  2.0, 2.0, 2.0,  2.0, 2.0, 2.0] -- [2,2,3]
  let b : Array Float := #[0.5, 0.5,  0.5, 0.5,  0.5, 0.5]                                  -- [1,3,2]
  let bias : Array Float := #[0.1, 0.2]                                                      -- [2]
  let ab := packF32 a
  let bb := packF32 b
  let biasb := packF32 bias

  let aStarts : Array Nat := #[0, 24]
  let bStarts : Array Nat := #[0, 0]
  let biasStarts : Array Nat := #[0, 0]
  let outBytes := Native.matmulBatchedBiasF32 ab bb biasb #[2] aStarts bStarts biasStarts 2 3 2
  let out := (RawBuffer.decode { dtype := .float32, data := outBytes }).data

  -- batch0: A@B=1.5, batch1: A@B=3.0; add bias [0.1,0.2]
  assertAllClose out #[1.6, 1.7, 1.6, 1.7, 3.1, 3.2, 3.1, 3.2] 0.0001 "batched matmul+bias"

private def testMatmulBatchedBias2 : IO Unit := do
  let a : Array Float := #[1.0, 1.0, 1.0,  1.0, 1.0, 1.0,  2.0, 2.0, 2.0,  2.0, 2.0, 2.0] -- [2,2,3]
  let b : Array Float := #[0.5, 0.5,  0.5, 0.5,  0.5, 0.5]                                  -- [1,3,2]
  let bias0 : Array Float := #[0.1, 0.2]                                                     -- [2]
  let bias1 : Array Float := #[1.0, 2.0,  3.0, 4.0,  5.0, 6.0,  7.0, 8.0]                    -- [2,2,2]
  let ab := packF32 a
  let bb := packF32 b
  let b0 := packF32 bias0
  let b1 := packF32 bias1

  let aStarts : Array Nat := #[0, 24]
  let bStarts : Array Nat := #[0, 0]
  let b0Starts : Array Nat := #[0, 0]
  let b1Starts : Array Nat := #[0, 16]
  let outBytes := Native.matmulBatchedBias2F32 ab bb b0 #[2] b1 #[2, 2] aStarts bStarts b0Starts b1Starts 2 3 2
  let out := (RawBuffer.decode { dtype := .float32, data := outBytes }).data

  assertAllClose out #[2.6, 3.7, 4.6, 5.7, 8.1, 9.2, 10.1, 11.2] 0.0001 "batched matmul+bias2"

private def testPlannerBias2 : IO Unit := do
  let root := runTensorM do
    let a ← Tensor.full [2, 3] .float32 1.0
    let b ← Tensor.full [3, 2] .float32 0.5
    let bias0 ← Tensor.full [2] .float32 0.1
    let bias1 ← Tensor.full [2, 2] .float32 0.2
    let ab ← UOp.contract2D a.uop b.uop
    let t ← UOp.add ab bias0.uop
    let y ← UOp.add t bias1.uop
    pure y

  let nodes := UOp.toposort root
  let keepIds := UOpIdSet.add UOpIdSet.mkEmpty root.uid

  let refCnt0 : Std.HashMap UOpId Nat := Id.run do
    let mut refCnt0 : Std.HashMap UOpId Nat := ∅
    for u in nodes do
      for s in u.src do
        refCnt0 := refCnt0.insert s.uid (refCnt0.getD s.uid 0 + 1)
    return refCnt0

  match FusedMatmul.compile root keepIds refCnt0 with
  | none => throw (IO.userError "planner bias2: expected fusion plan")
  | some plan =>
    if plan.bias2.isSome then
      pure ()
    else
      throw (IO.userError "planner bias2: expected bias2")

private def testPlannerReluNoBias : IO Unit := do
  let root := runTensorM do
    let a ← Tensor.full [2, 3] .float32 1.0
    let b ← Tensor.full [3, 2] .float32 0.5
    let ab ← UOp.contract2D a.uop b.uop
    let zero ← UOp.const .float32 0.0
    let out ← UOp.maxBinary ab zero
    pure out

  let nodes := UOp.toposort root
  let keepIds := UOpIdSet.add UOpIdSet.mkEmpty root.uid

  let refCnt0 : Std.HashMap UOpId Nat := Id.run do
    let mut refCnt0 : Std.HashMap UOpId Nat := ∅
    for u in nodes do
      for s in u.src do
        refCnt0 := refCnt0.insert s.uid (refCnt0.getD s.uid 0 + 1)
    return refCnt0

  match FusedMatmul.compile root keepIds refCnt0 with
  | none => throw (IO.userError "planner relu: expected fusion plan")
  | some plan =>
    if plan.relu && plan.bias2.isNone then
      pure ()
    else
      throw (IO.userError "planner relu: expected relu plan (no bias2)")

private def testPlannerBatchedBias : IO Unit := do
  let root := runTensorM do
    let a ← Tensor.full [2, 2, 3] .float32 1.0
    let b ← Tensor.full [1, 3, 2] .float32 0.5
    let bias ← Tensor.full [2] .float32 0.1
    let ab ← UOp.contract2D a.uop b.uop
    let y ← UOp.add ab bias.uop
    pure y

  let nodes := UOp.toposort root
  let keepIds := UOpIdSet.add UOpIdSet.mkEmpty root.uid

  let refCnt0 : Std.HashMap UOpId Nat := Id.run do
    let mut refCnt0 : Std.HashMap UOpId Nat := ∅
    for u in nodes do
      for s in u.src do
        refCnt0 := refCnt0.insert s.uid (refCnt0.getD s.uid 0 + 1)
    return refCnt0

  match FusedMatmul.compile root keepIds refCnt0 with
  | none => throw (IO.userError "planner batched bias: expected fusion plan")
  | some plan =>
    if !plan.aStarts.isEmpty then
      pure ()
    else
      throw (IO.userError "planner batched bias: expected batched starts")

private def testPlannerBatchedBiasExpandedA : IO Unit := do
  let root := runTensorM do
    let a0 ← Tensor.full [1, 2, 3] .float32 1.0
    let a ← StaticTensor.expand a0 [2, 2, 3]
    let b ← Tensor.full [1, 3, 2] .float32 0.5
    let bias ← Tensor.full [2] .float32 0.1
    let ab ← UOp.contract2D a.uop b.uop
    let y ← UOp.add ab bias.uop
    pure y

  let nodes := UOp.toposort root
  let keepIds := UOpIdSet.add UOpIdSet.mkEmpty root.uid

  let refCnt0 : Std.HashMap UOpId Nat := Id.run do
    let mut refCnt0 : Std.HashMap UOpId Nat := ∅
    for u in nodes do
      for s in u.src do
        refCnt0 := refCnt0.insert s.uid (refCnt0.getD s.uid 0 + 1)
    return refCnt0

  match FusedMatmul.compile root keepIds refCnt0 with
  | none => throw (IO.userError "planner batched bias expand: expected fusion plan")
  | some plan =>
    if plan.aStarts == #[0, 0] then
      pure ()
    else
      throw (IO.userError "planner batched bias expand: expected broadcasted aStarts")

private def testPlannerBatchedBiasExpandedB : IO Unit := do
  let root := runTensorM do
    let a ← Tensor.full [2, 2, 3] .float32 1.0
    let b0 ← Tensor.full [1, 3, 2] .float32 0.5
    let b ← StaticTensor.expand b0 [2, 3, 2]
    let bias ← Tensor.full [2] .float32 0.1
    let ab ← UOp.contract2D a.uop b.uop
    let y ← UOp.add ab bias.uop
    pure y

  let nodes := UOp.toposort root
  let keepIds := UOpIdSet.add UOpIdSet.mkEmpty root.uid

  let refCnt0 : Std.HashMap UOpId Nat := Id.run do
    let mut refCnt0 : Std.HashMap UOpId Nat := ∅
    for u in nodes do
      for s in u.src do
        refCnt0 := refCnt0.insert s.uid (refCnt0.getD s.uid 0 + 1)
    return refCnt0

  match FusedMatmul.compile root keepIds refCnt0 with
  | none => throw (IO.userError "planner batched bias expand B: expected fusion plan")
  | some plan =>
    if plan.bStarts == #[0, 0] then
      pure ()
    else
      throw (IO.userError "planner batched bias expand B: expected broadcasted bStarts")

private def testPlannerBatchedBiasExpandedBias : IO Unit := do
  let root := runTensorM do
    let a ← Tensor.full [2, 2, 3] .float32 1.0
    let b ← Tensor.full [1, 3, 2] .float32 0.5
    let bias0 ← Tensor.full [1, 2] .float32 0.1
    let bias ← StaticTensor.expand bias0 [2, 2]
    let ab ← UOp.contract2D a.uop b.uop
    let y ← UOp.add ab bias.uop
    pure y

  let nodes := UOp.toposort root
  let keepIds := UOpIdSet.add UOpIdSet.mkEmpty root.uid

  let refCnt0 : Std.HashMap UOpId Nat := Id.run do
    let mut refCnt0 : Std.HashMap UOpId Nat := ∅
    for u in nodes do
      for s in u.src do
        refCnt0 := refCnt0.insert s.uid (refCnt0.getD s.uid 0 + 1)
    return refCnt0

  match FusedMatmul.compile root keepIds refCnt0 with
  | none => throw (IO.userError "planner batched bias expand bias: expected fusion plan")
  | some plan =>
    if plan.biasStarts == #[0, 0] then
      pure ()
    else
      throw (IO.userError "planner batched bias expand bias: expected broadcasted biasStarts")

private def testPlannerBatchedBias2ExpandedBias0 : IO Unit := do
  let root := runTensorM do
    let a ← Tensor.full [2, 2, 3] .float32 1.0
    let b ← Tensor.full [1, 3, 2] .float32 0.5
    let bias0Base ← Tensor.buffer [1, 2] .float32
    let bias0 ← StaticTensor.expand bias0Base [2, 2]
    let bias1 ← Tensor.buffer [2, 2, 2] .float32
    let ab ← UOp.contract2D a.uop b.uop
    let t ← UOp.add ab bias0.uop
    let y ← UOp.add t bias1.uop
    pure y

  let nodes := UOp.toposort root
  let keepIds := UOpIdSet.add UOpIdSet.mkEmpty root.uid

  let refCnt0 : Std.HashMap UOpId Nat := Id.run do
    let mut refCnt0 : Std.HashMap UOpId Nat := ∅
    for u in nodes do
      for s in u.src do
        refCnt0 := refCnt0.insert s.uid (refCnt0.getD s.uid 0 + 1)
    return refCnt0

  match FusedMatmul.compile root keepIds refCnt0 with
  | none => throw (IO.userError "planner batched bias2 expand bias0: expected fusion plan")
  | some plan =>
    if plan.biasStarts == #[0, 0] && plan.bias2Starts == #[0, 16] then
      pure ()
    else
      throw (IO.userError s!"planner batched bias2 expand bias0: biasStarts={repr plan.biasStarts} bias2Starts={repr plan.bias2Starts}")

private def testPlannerBatchedBias2ExpandedBias1 : IO Unit := do
  let root := runTensorM do
    let a ← Tensor.full [2, 2, 3] .float32 1.0
    let b ← Tensor.full [1, 3, 2] .float32 0.5
    let bias0 ← Tensor.full [2] .float32 0.1
    let bias1Base ← Tensor.buffer [1, 2, 2] .float32
    let bias1 ← StaticTensor.expand bias1Base [2, 2, 2]
    let ab ← UOp.contract2D a.uop b.uop
    let t ← UOp.add ab bias0.uop
    let y ← UOp.add t bias1.uop
    pure y

  let nodes := UOp.toposort root
  let keepIds := UOpIdSet.add UOpIdSet.mkEmpty root.uid

  let refCnt0 : Std.HashMap UOpId Nat := Id.run do
    let mut refCnt0 : Std.HashMap UOpId Nat := ∅
    for u in nodes do
      for s in u.src do
        refCnt0 := refCnt0.insert s.uid (refCnt0.getD s.uid 0 + 1)
    return refCnt0

  match FusedMatmul.compile root keepIds refCnt0 with
  | none => throw (IO.userError "planner batched bias2 expand bias1: expected fusion plan")
  | some plan =>
    if plan.bias2Starts == #[0, 0] then
      pure ()
    else
      throw (IO.userError s!"planner batched bias2 expand bias1: bias2Starts={repr plan.bias2Starts}")

def runAll : IO Unit := do
  IO.println "=== FusedMatmul Tests ==="
  testMatmulBias
  IO.println "✓ matmul+bias"
  testMatmulBiasRelu
  IO.println "✓ matmul+bias+relu"
  testMatmulBias2
  IO.println "✓ matmul+bias2"
  testMatmulBias2Relu
  IO.println "✓ matmul+bias2+relu"
  testMatmulBatchedBias
  IO.println "✓ batched matmul+bias"
  testMatmulBatchedBias2
  IO.println "✓ batched matmul+bias2"
  testPlannerBias2
  IO.println "✓ planner bias2"
  testPlannerReluNoBias
  IO.println "✓ planner relu (no bias)"
  testPlannerBatchedBias
  IO.println "✓ planner batched bias"
  testPlannerBatchedBiasExpandedA
  IO.println "✓ planner batched bias (expanded A)"
  testPlannerBatchedBiasExpandedB
  IO.println "✓ planner batched bias (expanded B)"
  testPlannerBatchedBiasExpandedBias
  IO.println "✓ planner batched bias (expanded bias)"
  testPlannerBatchedBias2ExpandedBias0
  IO.println "✓ planner batched bias2 (expanded bias0)"
  testPlannerBatchedBias2ExpandedBias1
  IO.println "✓ planner batched bias2 (expanded bias1)"
  IO.println "=== FusedMatmul OK ==="

end TinyGrad4.Test.FusedMatmul

