import TinyGrad4

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

/-!
# ViewFusionSmoke

Ensures movement ops can be virtualized as strided/masked views inside fused kernels, so we can fuse
across `PERMUTE`/`PAD` without materializing intermediate buffers.
-/

namespace TinyGrad4.Test.ViewFusionSmoke

open Std
open TinyGrad4
open Interpreter
open StaticTensor
open Backend

private def assertClose (got expected : Float) (tol : Float) (label : String) : IO Unit := do
  let diff := Float.abs (got - expected)
  if diff > tol then
    throw (IO.userError s!"{label}: got {got} expected {expected} diff {diff} > {tol}")

private def shrink3d (data : FloatArray) (shape : Shape) (bounds : List (Nat × Nat)) : FloatArray := Id.run do
  let newShape := Shape.shrink shape bounds
  let numel := listProd newShape
  let mut out := FloatArray.emptyWithCapacity numel
  for i in [:numel] do
    let idx := Interpreter.unflattenIndex i newShape
    let idx' := (idx.zip bounds).map (fun (v, (s, _)) => v + s)
    let flat := Interpreter.flattenIndex idx' shape
    out := out.push data[flat]!
  return out

private def maxOf (arr : FloatArray) : Float := Id.run do
  if arr.size == 0 then
    return 0.0
  let mut m := arr[0]!
  for i in [1:arr.size] do
    let v := arr[i]!
    if v > m then
      m := v
  return m

private def testPermutePadIntoFusedReduceMaxAll : IO Unit := do
  let xData : FloatArray := ⟨#[
    -2.0, -3.0, -4.0,
    -5.0, -6.0, -7.0
  ]⟩

  let (xU, outU) := runTensorM do
    let x ← Tensor.buffer [2, 3] .float32
    let xp ← StaticTensor.permute x [1, 0]
    let xpad ← StaticTensor.pad xp [(1, 1), (0, 0)]
    let c ← UOp.const .float32 1.0
    let y ← UOp.add xpad.uop c
    let out ← UOp.max_ y [] false
    pure (x.uop, out)

  let env : Env :=
    let env0 : Env := ∅
    Interpreter.setBuffer env0 xU (RawBuffer.ofF32 xData)

  let compiled := Interpreter.compile outU
  match compiled.implMap[outU.uid]? with
  | some (.fusedReduce _) => pure ()
  | _ => throw (IO.userError "expected root to select fusedReduce")

  let bad := compiled.nodes.filter fun u =>
    u.op == .PAD || u.op == .PERMUTE || u.op == .ADD
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"expected movement/alU nodes to be virtualized into kernel, found {bad.length}\n{msg}")

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outArr := (outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))).decode
  if outArr.size != 1 then
    throw (IO.userError s!"expected scalar output, got size {outArr.size}")

  let got := outArr[0]!
  assertClose got 1.0 1.0e-6 "view fusion reduce max(all)"

private def testShrinkReshapeFoldIntoFusedReduceMaxAll : IO Unit := do
  let xData : FloatArray := ⟨#[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                             12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]⟩

  let (xU, outU) := runTensorM do
    let x ← Tensor.buffer [2, 3, 4] .float32
    let xs ← StaticTensor.shrink x [(0, 2), (0, 2), (0, 4)]
    let xr ← StaticTensor.reshape xs [2, 8]
    let c ← UOp.const .float32 1.0
    let y ← UOp.add xr.uop c
    let out ← UOp.max_ y [] false
    pure (x.uop, out)

  let env : Env :=
    Interpreter.setBuffer (∅ : Env) xU (RawBuffer.ofF32 xData)

  let compiled := Interpreter.compile outU
  match compiled.implMap[outU.uid]? with
  | some (.fusedReduce plan) =>
    if plan.ewise.needsStack then
      throw (IO.userError "expected shrink→reshape fold to avoid a view stack")
  | _ => throw (IO.userError "expected root to select fusedReduce")

  let bad := compiled.nodes.filter fun u =>
    u.op == .RESHAPE || u.op == .SHRINK || u.op == .ADD
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"expected movement/alU nodes to be virtualized, found {bad.length}\n{msg}")

  let shrinked := shrink3d xData [2, 3, 4] [(0, 2), (0, 2), (0, 4)]
  let expected := maxOf (FloatArray.map (fun v => v + 1.0) shrinked)

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outArr := (outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))).decode
  if outArr.size != 1 then
    throw (IO.userError s!"expected scalar output, got size {outArr.size}")
  assertClose outArr[0]! expected 1.0e-6 "view fusion reduce max(all) fold"

private def testPadReshapeFoldIntoFusedReduceMaxAll : IO Unit := do
  let xData : FloatArray := ⟨#[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                             12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]⟩

  let (xU, outU) := runTensorM do
    let x ← Tensor.buffer [2, 3, 4] .float32
    let xp ← StaticTensor.pad x [(1, 1), (0, 0), (0, 0)]
    let xr ← StaticTensor.reshape xp [4, 12]
    let c ← UOp.const .float32 1.0
    let y ← UOp.add xr.uop c
    let out ← UOp.max_ y [] false
    pure (x.uop, out)

  let env : Env :=
    Interpreter.setBuffer (∅ : Env) xU (RawBuffer.ofF32 xData)

  let compiled := Interpreter.compile outU
  match compiled.implMap[outU.uid]? with
  | some (.fusedReduce plan) =>
    if plan.ewise.needsStack then
      throw (IO.userError "expected pad→reshape fold to avoid a view stack")
  | _ => throw (IO.userError "expected root to select fusedReduce")

  let bad := compiled.nodes.filter fun u =>
    u.op == .PAD || u.op == .RESHAPE || u.op == .ADD
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"expected movement/ALU nodes to be virtualized, found {bad.length}\n{msg}")

  let expected := 24.0
  let outCache := Interpreter.evalCompiledRaw compiled env
  let outArr := (outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))).decode
  if outArr.size != 1 then
    throw (IO.userError s!"expected scalar output, got size {outArr.size}")
  assertClose outArr[0]! expected 1.0e-6 "view fusion reduce max(all) pad/reshape fold"

private def testShrinkReshapeGapIntoFusedReduceMaxAll : IO Unit := do
  let xData : FloatArray := ⟨#[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                             12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]⟩

  let (xU, outU) := runTensorM do
    let x ← Tensor.buffer [2, 3, 4] .float32
    let xs ← StaticTensor.shrink x [(0, 2), (0, 2), (0, 4)]
    let xr ← StaticTensor.reshape xs [4, 4]
    let c ← UOp.const .float32 1.0
    let y ← UOp.add xr.uop c
    let out ← UOp.max_ y [] false
    pure (x.uop, out)

  let env : Env :=
    Interpreter.setBuffer (∅ : Env) xU (RawBuffer.ofF32 xData)

  let compiled := Interpreter.compile outU
  match compiled.implMap[outU.uid]? with
  | some (.fusedReduce plan) =>
    if !plan.ewise.needsStack then
      throw (IO.userError "expected shrink→reshape gap to require a view stack")
  | _ => throw (IO.userError "expected root to select fusedReduce")

  let bad := compiled.nodes.filter fun u =>
    u.op == .RESHAPE || u.op == .SHRINK || u.op == .ADD
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"expected movement/alU nodes to be virtualized, found {bad.length}\n{msg}")

  let shrinked := shrink3d xData [2, 3, 4] [(0, 2), (0, 2), (0, 4)]
  let expected := maxOf (FloatArray.map (fun v => v + 1.0) shrinked)

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outArr := (outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))).decode
  if outArr.size != 1 then
    throw (IO.userError s!"expected scalar output, got size {outArr.size}")
  assertClose outArr[0]! expected 1.0e-6 "view fusion reduce max(all) gap"

def runAll : IO Unit := do
  IO.println "=== ViewFusionSmoke Tests ==="
  testPermutePadIntoFusedReduceMaxAll
  IO.println "✓ permute/pad virtualized in fused reduce"
  testShrinkReshapeFoldIntoFusedReduceMaxAll
  IO.println "✓ shrink→reshape fold in fused reduce"
  testPadReshapeFoldIntoFusedReduceMaxAll
  IO.println "✓ pad→reshape fold in fused reduce"
  testShrinkReshapeGapIntoFusedReduceMaxAll
  IO.println "✓ shrink→reshape gap in fused reduce"
  IO.println "=== ViewFusionSmoke OK ==="

end TinyGrad4.Test.ViewFusionSmoke

