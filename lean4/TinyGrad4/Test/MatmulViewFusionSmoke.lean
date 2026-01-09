import TinyGrad4

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

/-!
# MatmulViewFusionSmoke

Ensures fused matmul can virtualize movement inputs (especially `PERMUTE` for transpose) as
strided views, so we can fuse `q @ kᵀ + bias` without materializing `kᵀ`.
-/

namespace TinyGrad4.Test.MatmulViewFusionSmoke

open Std
open TinyGrad4
open Interpreter
open Backend
open StaticTensor

private def startsBytes (batch : Nat) (matNumel : Nat) : Array Nat := Id.run do
  let mut out : Array Nat := Array.emptyWithCapacity batch
  for i in [:batch] do
    out := out.push (i * matNumel * 4)
  return out

private def shrink3d (data : Array Float) (shape : Shape) (bounds : List (Nat × Nat)) : Array Float := Id.run do
  let newShape := Shape.shrink shape bounds
  let numel := listProd newShape
  let mut out : Array Float := Array.emptyWithCapacity numel
  for i in [:numel] do
    let idx := Interpreter.unflattenIndex i newShape
    let idx' := (idx.zip bounds).map (fun (v, (s, _)) => v + s)
    let flat := Interpreter.flattenIndex idx' shape
    out := out.push data[flat]!
  return out

private def packF32 (data : Array Float) : ByteArray :=
  Native.packF32FromF64 ⟨data⟩

private def assertAllClose (arr : RawBuffer) (expected : RawBuffer) (tol : Float) (label : String) : IO Unit := do
  let a := arr.decode
  let e := expected.decode
  if a.size != e.size then
    throw (IO.userError s!"{label}: size {a.size} != {e.size}")
  for i in [:a.size] do
    let v := a[i]!
    let ev := e[i]!
    let diff := Float.abs (v - ev)
    if diff > tol then
      throw (IO.userError s!"{label}: idx {i} value {v} expected {ev} diff {diff} > {tol}")

private def getFusedMatmul (compiled : Interpreter.Compiled) (root : UOp) : IO Backend.FusedMatmul.Plan := do
  match compiled.implMap[root.uid]? with
  | some (.fusedMatmul plan) => pure plan
  | _ => throw (IO.userError "expected root to select fusedMatmul")

private def testPermuteIntoFusedMatmul : IO Unit := do
  let b := 2
  let t := 3
  let d := 2

  let qData : Array Float := #[
    0.1, 0.2,  0.3, 0.4,  0.5, 0.6,
    0.7, 0.8,  0.9, 1.0,  1.1, 1.2
  ]
  let kData : Array Float := #[
    1.0, 0.9,  0.8, 0.7,  0.6, 0.5,
    0.4, 0.3,  0.2, 0.1,  0.0, -0.1
  ]
  let maskData : Array Float := #[
    0.0, -0.1, -0.2,
    0.0, -0.1, -0.2,
    0.0, -0.1, -0.2
  ]

  let qb := packF32 qData
  let kb := packF32 kData
  let maskb := packF32 maskData

  let (qU, kU, maskU, scoresMaskedU) := runTensorM do
    let q ← Tensor.buffer [b, t, d] .float32
    let k ← Tensor.buffer [b, t, d] .float32
    let mask ← Tensor.buffer [t, t] .float32
    let kT ← StaticTensor.permute k [0, 2, 1]
    let scores ← UOp.contract2D q.uop kT.uop
    let scoresMasked ← UOp.add scores mask.uop
    pure (q.uop, k.uop, mask.uop, scoresMasked)

  let env : Env :=
    let env0 : Env := ∅
    let env1 := Interpreter.setBuffer env0 qU { dtype := .float32, data := qb }
    let env2 := Interpreter.setBuffer env1 kU { dtype := .float32, data := kb }
    Interpreter.setBuffer env2 maskU { dtype := .float32, data := maskb }

  let compiled := Interpreter.compile scoresMaskedU
  match compiled.implMap[scoresMaskedU.uid]? with
  | some (.fusedMatmul _) => pure ()
  | _ => throw (IO.userError "expected root to select fusedMatmul")

  let bad := compiled.nodes.filter fun u =>
    u.op == .PERMUTE || u.op == .CONTRACT
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"expected permute/contract to be virtualized into kernel, found {bad.length}\n{msg}")

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outBuf := outCache.getD scoresMaskedU.uid (RawBuffer.zeros scoresMaskedU.dtype (listProd scoresMaskedU.shape))

  let kTBytes := Native.permuteF32 kb #[b, t, d] #[0, 2, 1]
  let qStarts := startsBytes b (t * d)
  let kStarts := startsBytes b (d * t)
  let scoresBytes := Native.matmulBatchedF32 qb kTBytes qStarts kStarts t d t
  let scoresMaskedBytes := Native.addBcastF32 scoresBytes maskb #[b, t, t] #[t, t] #[b, t, t]
  let expected : RawBuffer := { dtype := .float32, data := scoresMaskedBytes }

  assertAllClose outBuf expected 0.001 "matmul view fusion"

private def testShrinkReshapeFoldIntoFusedMatmul : IO Unit := do
  let m := 3
  let k := 2
  let n := 8

  let aData : Array Float := #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  let bData : Array Float := #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
  let biasData : Array Float := #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

  let ab := packF32 aData
  let bb := packF32 bData
  let biasb := packF32 biasData

  let (aU, bU, _brU, biasU, outU) := runTensorM do
    let a ← Tensor.buffer [m, k] .float32
    let b ← Tensor.buffer [2, 3, 4] .float32
    let bias ← Tensor.buffer [n] .float32
    let bs ← StaticTensor.shrink b [(0, 2), (0, 2), (0, 4)]
    let br ← StaticTensor.reshape bs [k, n]
    let out0 ← UOp.contract2D a.uop br.uop
    let out ← UOp.add out0 bias.uop
    pure (a.uop, b.uop, br.uop, bias.uop, out)

  let env : Env :=
    let env0 : Env := ∅
    let env1 := Interpreter.setBuffer env0 aU { dtype := .float32, data := ab }
    let env2 := Interpreter.setBuffer env1 bU { dtype := .float32, data := bb }
    Interpreter.setBuffer env2 biasU { dtype := .float32, data := biasb }

  let compiled := Interpreter.compile outU
  let plan ← getFusedMatmul compiled outU
  if plan.bBase != bU.uid then
    throw (IO.userError "expected shrink→reshape fold to virtualize into base buffer")

  let bad := compiled.nodes.filter fun u =>
    u.op == .SHRINK || u.op == .RESHAPE || u.op == .CONTRACT
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"expected shrink/reshape/contract to be virtualized, found {bad.length}\n{msg}")

  let bShr := shrink3d bData [2, 3, 4] [(0, 2), (0, 2), (0, 4)]
  let bShrBytes := packF32 bShr
  let outBytes := Native.matmulF32 ab bShrBytes m k n
  let outBiasBytes := Native.addBcastF32 outBytes biasb #[m, n] #[n] #[m, n]
  let expected : RawBuffer := { dtype := .float32, data := outBiasBytes }

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outBuf := outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))
  assertAllClose outBuf expected 0.001 "matmul shrink→reshape fold"

private def testShrinkReshapeGapIntoFusedMatmul : IO Unit := do
  let m := 3
  let k := 4
  let n := 4

  let aData : Array Float := #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
  let bData : Array Float := #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
  let biasData : Array Float := #[0.1, 0.2, 0.3, 0.4]

  let ab := packF32 aData
  let bb := packF32 bData
  let biasb := packF32 biasData

  let (aU, bU, _brU, biasU, outU) := runTensorM do
    let a ← Tensor.buffer [m, k] .float32
    let b ← Tensor.buffer [2, 3, 4] .float32
    let bias ← Tensor.buffer [n] .float32
    let bs ← StaticTensor.shrink b [(0, 2), (0, 2), (0, 4)]
    let br ← StaticTensor.reshape bs [k, n]
    let out0 ← UOp.contract2D a.uop br.uop
    let out ← UOp.add out0 bias.uop
    pure (a.uop, b.uop, br.uop, bias.uop, out)

  let env : Env :=
    let env0 : Env := ∅
    let env1 := Interpreter.setBuffer env0 aU { dtype := .float32, data := ab }
    let env2 := Interpreter.setBuffer env1 bU { dtype := .float32, data := bb }
    Interpreter.setBuffer env2 biasU { dtype := .float32, data := biasb }

  let compiled := Interpreter.compile outU
  let plan ← getFusedMatmul compiled outU
  if plan.bBase != bU.uid || !plan.needsStack then
    throw (IO.userError "expected shrink→reshape gap to require view stack and virtualize into base buffer")

  let bad := compiled.nodes.filter fun u =>
    u.op == .SHRINK || u.op == .RESHAPE || u.op == .CONTRACT
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"expected shrink/reshape/contract to be virtualized, found {bad.length}\n{msg}")

  let bShr := shrink3d bData [2, 3, 4] [(0, 2), (0, 2), (0, 4)]
  let bShrBytes := packF32 bShr
  let outBytes := Native.matmulF32 ab bShrBytes m k n
  let outBiasBytes := Native.addBcastF32 outBytes biasb #[m, n] #[n] #[m, n]
  let expected : RawBuffer := { dtype := .float32, data := outBiasBytes }

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outBuf := outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))
  assertAllClose outBuf expected 0.001 "matmul shrink→reshape gap"

private def testPadReshapeIntoFusedMatmul : IO Unit := do
  let m := 3
  let k := 4
  let n := 4

  let aData : Array Float := #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
    0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
  let bData : Array Float := #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  let biasData : Array Float := #[0.1, 0.2, 0.3, 0.4]

  let ab := packF32 aData
  let bb := packF32 bData
  let biasb := packF32 biasData

  let (aU, bU, _brU, biasU, outU) := runTensorM do
    let a ← Tensor.buffer [m, k] .float32
    let b ← Tensor.buffer [2, 2, 2] .float32
    let bias ← Tensor.buffer [n] .float32
    let bp ← StaticTensor.pad b [(0, 0), (0, 0), (0, 2)]
    let br ← StaticTensor.reshape bp [k, n]
    let out0 ← UOp.contract2D a.uop br.uop
    let out ← UOp.add out0 bias.uop
    pure (a.uop, b.uop, br.uop, bias.uop, out)

  let env : Env :=
    let env0 : Env := ∅
    let env1 := Interpreter.setBuffer env0 aU { dtype := .float32, data := ab }
    let env2 := Interpreter.setBuffer env1 bU { dtype := .float32, data := bb }
    Interpreter.setBuffer env2 biasU { dtype := .float32, data := biasb }

  let compiled := Interpreter.compile outU
  let plan ← getFusedMatmul compiled outU
  if plan.bBase != bU.uid then
    throw (IO.userError "expected pad→reshape to virtualize into base buffer")

  let bad := compiled.nodes.filter fun u =>
    u.op == .PAD || u.op == .RESHAPE || u.op == .CONTRACT
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"expected pad/reshape/contract to be virtualized, found {bad.length}\n{msg}")

  let bPadBytes := Native.padF32 bb #[2, 2, 2] #[0, 0, 0] #[0, 0, 2]
  let outBytes := Native.matmulF32 ab bPadBytes m k n
  let outBiasBytes := Native.addBcastF32 outBytes biasb #[m, n] #[n] #[m, n]
  let expected : RawBuffer := { dtype := .float32, data := outBiasBytes }

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outBuf := outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))
  assertAllClose outBuf expected 0.001 "matmul pad→reshape"

private def testPadReshapeStackIntoFusedMatmul : IO Unit := do
  let m := 2
  let k := 2
  let n := 6

  let aData : Array Float := #[0.1, 0.2, 0.3, 0.4]
  let bData : Array Float := #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  let biasData : Array Float := #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

  let ab := packF32 aData
  let bb := packF32 bData
  let biasb := packF32 biasData

  let (aU, bU, biasU, outU) := runTensorM do
    let a ← Tensor.buffer [m, k] .float32
    let b ← Tensor.buffer [2, 3] .float32
    let bias ← Tensor.buffer [n] .float32
    let bp ← StaticTensor.pad b [(1, 1), (0, 0)]
    let br ← StaticTensor.reshape bp [k, n]
    let out0 ← UOp.contract2D a.uop br.uop
    let out ← UOp.add out0 bias.uop
    pure (a.uop, b.uop, bias.uop, out)

  let env : Env :=
    let env0 : Env := ∅
    let env1 := Interpreter.setBuffer env0 aU { dtype := .float32, data := ab }
    let env2 := Interpreter.setBuffer env1 bU { dtype := .float32, data := bb }
    Interpreter.setBuffer env2 biasU { dtype := .float32, data := biasb }

  let compiled := Interpreter.compile outU
  let plan ← getFusedMatmul compiled outU
  if plan.bBase != bU.uid || !plan.needsStack then
    throw (IO.userError "expected pad→reshape to require view stack and virtualize into base buffer")

  let bad := compiled.nodes.filter fun u =>
    u.op == .PAD || u.op == .RESHAPE || u.op == .CONTRACT
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"expected pad/reshape/contract to be virtualized, found {bad.length}\n{msg}")

  let bPadBytes := Native.padF32 bb #[2, 3] #[1, 0] #[1, 0]
  let outBytes := Native.matmulF32 ab bPadBytes m k n
  let outBiasBytes := Native.addBcastF32 outBytes biasb #[m, n] #[n] #[m, n]
  let expected : RawBuffer := { dtype := .float32, data := outBiasBytes }

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outBuf := outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))
  assertAllClose outBuf expected 0.001 "matmul pad→reshape stack"

private def testFlipReshapeIntoFusedMatmul : IO Unit := do
  let m := 2
  let k := 4
  let n := 4

  let aData : Array Float := #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  let bData : Array Float := #[
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0
  ]
  let biasData : Array Float := #[0.05, 0.1, 0.15, 0.2]

  let ab := packF32 aData
  let bb := packF32 bData
  let biasb := packF32 biasData

  let (aU, bU, _brU, biasU, outU) := runTensorM do
    let a ← Tensor.buffer [m, k] .float32
    let b ← Tensor.buffer [2, 2, 4] .float32
    let bias ← Tensor.buffer [n] .float32
    let bf ← StaticTensor.flip b [2]
    let br ← StaticTensor.reshape bf [k, n]
    let out0 ← UOp.contract2D a.uop br.uop
    let out ← UOp.add out0 bias.uop
    pure (a.uop, b.uop, br.uop, bias.uop, out)

  let env : Env :=
    let env0 : Env := ∅
    let env1 := Interpreter.setBuffer env0 aU { dtype := .float32, data := ab }
    let env2 := Interpreter.setBuffer env1 bU { dtype := .float32, data := bb }
    Interpreter.setBuffer env2 biasU { dtype := .float32, data := biasb }

  let compiled := Interpreter.compile outU
  let plan ← getFusedMatmul compiled outU
  if plan.bBase != bU.uid then
    throw (IO.userError "expected flip→reshape to virtualize into base buffer")

  let bad := compiled.nodes.filter fun u =>
    u.op == .FLIP || u.op == .RESHAPE || u.op == .CONTRACT
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"expected flip/reshape/contract to be virtualized, found {bad.length}\n{msg}")

  let bFlipBytes := Native.flipF32 bb #[2, 2, 4] #[2]
  let outBytes := Native.matmulF32 ab bFlipBytes m k n
  let outBiasBytes := Native.addBcastF32 outBytes biasb #[m, n] #[n] #[m, n]
  let expected : RawBuffer := { dtype := .float32, data := outBiasBytes }

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outBuf := outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))
  assertAllClose outBuf expected 0.001 "matmul flip→reshape"

def runAll : IO Unit := do
  IO.println "=== MatmulViewFusionSmoke Tests ==="
  testPermuteIntoFusedMatmul
  IO.println "✓ permute virtualized in fused matmul"
  testShrinkReshapeFoldIntoFusedMatmul
  IO.println "✓ shrink→reshape fold virtualized in fused matmul"
  testShrinkReshapeGapIntoFusedMatmul
  IO.println "✓ shrink→reshape gap handled in fused matmul"
  testPadReshapeIntoFusedMatmul
  IO.println "✓ pad→reshape virtualized in fused matmul"
  testPadReshapeStackIntoFusedMatmul
  IO.println "✓ pad→reshape stack virtualized in fused matmul"
  testFlipReshapeIntoFusedMatmul
  IO.println "✓ flip→reshape virtualized in fused matmul"
  IO.println "=== MatmulViewFusionSmoke OK ==="

end TinyGrad4.Test.MatmulViewFusionSmoke

