import Float64
import TinyGrad4

-- Disable RawBuffer linter for test files that need Array Float64 literals
set_option linter.useRawBuffer false

/-!
# ContractViewFusionSmoke

Ensures plain `CONTRACT(a, PERMUTE(b))` and non-affine move chains like `PERMUTE → RESHAPE` can be executed without
materializing movement intermediates, by selecting the view-aware fused contract kernel.
-/

namespace TinyGrad4.Test.ContractViewFusionSmoke

open Std
open TinyGrad4
open Interpreter
open Backend
open StaticTensor

private def assertAllClose (arr : FloatArray) (expected : FloatArray) (tol : Float64) (label : String) : IO Unit := do
  if arr.size != expected.size then
    throw (IO.userError s!"{label}: size {arr.size} != {expected.size}")
  for i in [:arr.size] do
    let v := arr[i]!
    let e := expected[i]!
    let diff := Float64.abs (v - e)
    if diff > tol then
      throw (IO.userError s!"{label}: idx {i} value {v} expected {e} diff {diff} > {tol}")

private def shrink3d (data : FloatArray) (shape : Shape) (bounds : List (Nat × Nat)) : FloatArray := Id.run do
  let newShape := Shape.shrinkUnsafe shape bounds
  let numel := listProd newShape
  let mut out := FloatArray.emptyWithCapacity numel
  for i in [:numel] do
    let idx := Interpreter.unflattenIndex i newShape
    let idx' := (idx.zip bounds).map (fun (v, (s, _)) => v + s)
    let flat := Interpreter.flattenIndex idx' shape
    out := out.push data[flat]!
  return out

private def testPermuteIntoFusedContract : IO Unit := do
  let m := 2
  let k := 3
  let n := 4

  let aData : FloatArray := ⟨#[0.1, 0.2, 0.3,  0.4, 0.5, 0.6]⟩
  let bData : FloatArray := ⟨#[
    1.0, 0.9, 0.8,
    0.7, 0.6, 0.5,
    0.4, 0.3, 0.2,
    0.1, 0.0, -0.1
  ]⟩

  let ab := Native.packF32FromF64 aData
  let bb := Native.packF32FromF64 bData

  let (aU, bU, outU) := runTensorM do
    let a ← Tensor.buffer [m, k] .float32
    let b ← Tensor.buffer [n, k] .float32
    let bT ← StaticTensor.permuteUnsafe b [1, 0]
    let out ← UOp.contract2D a.uop bT.uop
    pure (a.uop, b.uop, out)

  let env : Env :=
    let env0 : Env := ∅
    let env1 := Interpreter.setBuffer env0 aU { dtype := .float32, data := ab }
    Interpreter.setBuffer env1 bU { dtype := .float32, data := bb }

  let compiled := Interpreter.compile outU
  match compiled.implMap[outU.uid]? with
  | some (.fusedContract _) => pure ()
  | _ => throw (IO.userError "expected root to select fusedContract")

  let bad := compiled.nodes.filter fun u =>
    u.op == .PERMUTE || u.op == .CONTRACT
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"expected permuteUnsafe/contract to be virtualized into kernel, found {bad.length}\n{msg}")

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outArr := (outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))).decode

  let bTBytes := Native.permuteF32 bb #[n, k] #[1, 0]
  let expectedBytes := Native.matmulF32 ab bTBytes m k n
  let expected := RawBuffer.decode { dtype := .float32, data := expectedBytes }

  assertAllClose outArr expected 0.001 "contract view fusion"

private def testPermuteReshapeIntoFusedContract : IO Unit := do
  let m := 5
  let k := 6
  let n := 4

  let aData : FloatArray := ⟨#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
    0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
    1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
    1.9, 2.0, 2.1, 2.2, 2.3, 2.4,
    2.5, 2.6, 2.7, 2.8, 2.9, 3.0]⟩

  let bData : FloatArray := ⟨#[1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0,
    17.0, 18.0, 19.0, 20.0,
    21.0, 22.0, 23.0, 24.0]⟩

  let ab := Native.packF32FromF64 aData
  let bb := Native.packF32FromF64 bData

  let (aU, bU, outU) := runTensorM do
    let a ← Tensor.buffer [m, k] .float32
    let b ← Tensor.buffer [2, 3, 4] .float32
    let bp ← StaticTensor.permuteUnsafe b [1, 0, 2]
    let br ← StaticTensor.reshapeUnsafe bp [k, n]
    let out ← UOp.contract2D a.uop br.uop
    pure (a.uop, b.uop, out)

  let env : Env :=
    let env0 : Env := ∅
    let env1 := Interpreter.setBuffer env0 aU { dtype := .float32, data := ab }
    Interpreter.setBuffer env1 bU { dtype := .float32, data := bb }

  let compiled := Interpreter.compile outU
  match compiled.implMap[outU.uid]? with
  | some (.fusedContract plan) =>
    if !plan.needsStack then
      throw (IO.userError "expected fusedContract to require a view stackUnsafe (permuteUnsafe→reshapeUnsafe)")
  | _ => throw (IO.userError "expected root to select fusedContract")

  let bad := compiled.nodes.filter fun u =>
    u.op == .PERMUTE || u.op == .RESHAPE || u.op == .CONTRACT
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"expected permuteUnsafe/reshapeUnsafe/contract to be virtualized into kernel, found {bad.length}\n{msg}")

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outArr := (outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))).decode

  let bpBytes := Native.permuteF32 bb #[2, 3, 4] #[1, 0, 2]
  let expectedBytes := Native.matmulF32 ab bpBytes m k n
  let expected := RawBuffer.decode { dtype := .float32, data := expectedBytes }

  assertAllClose outArr expected 0.001 "contract view-stackUnsafe fusion"

private def testShrinkReshapeFoldIntoFusedContract : IO Unit := do
  let m := 3
  let k := 2
  let n := 8

  let aData : FloatArray := ⟨#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]⟩
  let bData : FloatArray := ⟨#[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]⟩

  let ab := Native.packF32FromF64 aData
  let bShr := shrink3d bData [2, 3, 4] [(0, 2), (0, 2), (0, 4)]
  let bb := Native.packF32FromF64 bShr

  let (aU, bU, outU) := runTensorM do
    let a ← Tensor.buffer [m, k] .float32
    let b ← Tensor.buffer [2, 3, 4] .float32
    let bs ← StaticTensor.shrinkUnsafe b [(0, 2), (0, 2), (0, 4)]
    let br ← StaticTensor.reshapeUnsafe bs [k, n]
    let out ← UOp.contract2D a.uop br.uop
    pure (a.uop, b.uop, out)

  let env : Env :=
    let env0 : Env := ∅
    let env1 := Interpreter.setBuffer env0 aU { dtype := .float32, data := ab }
    Interpreter.setBuffer env1 bU { dtype := .float32, data := Native.packF32FromF64 bData }

  let compiled := Interpreter.compile outU
  match compiled.implMap[outU.uid]? with
  | some (.fusedContract plan) =>
    if plan.needsStack then
      throw (IO.userError "expected shrinkUnsafe→reshapeUnsafe fold to avoid a view stackUnsafe")
  | _ => throw (IO.userError "expected root to select fusedContract")

  let bad := compiled.nodes.filter fun u =>
    u.op == .SHRINK || u.op == .RESHAPE || u.op == .CONTRACT
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"expected shrinkUnsafe/reshapeUnsafe/contract to be virtualized, found {bad.length}\n{msg}")

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outArr := (outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))).decode

  let expectedBytes := Native.matmulF32 ab bb m k n
  let expected := RawBuffer.decode { dtype := .float32, data := expectedBytes }
  assertAllClose outArr expected 0.001 "contract shrinkUnsafe→reshapeUnsafe fold"

private def testShrinkReshapeGapIntoFusedContract : IO Unit := do
  let m := 2
  let k := 4
  let n := 4

  let aData : FloatArray := ⟨#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]⟩
  let bData : FloatArray := ⟨#[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]⟩

  let ab := Native.packF32FromF64 aData
  let bShr := shrink3d bData [2, 3, 4] [(0, 2), (0, 2), (0, 4)]
  let bb := Native.packF32FromF64 bShr

  let (aU, bU, outU) := runTensorM do
    let a ← Tensor.buffer [m, k] .float32
    let b ← Tensor.buffer [2, 3, 4] .float32
    let bs ← StaticTensor.shrinkUnsafe b [(0, 2), (0, 2), (0, 4)]
    let br ← StaticTensor.reshapeUnsafe bs [k, n]
    let out ← UOp.contract2D a.uop br.uop
    pure (a.uop, b.uop, out)

  let env : Env :=
    let env0 : Env := ∅
    let env1 := Interpreter.setBuffer env0 aU { dtype := .float32, data := ab }
    Interpreter.setBuffer env1 bU { dtype := .float32, data := Native.packF32FromF64 bData }

  let compiled := Interpreter.compile outU
  match compiled.implMap[outU.uid]? with
  | some (.fusedContract plan) =>
    if !plan.needsStack then
      throw (IO.userError "expected shrinkUnsafe→reshapeUnsafe gap to require a view stackUnsafe")
  | _ => throw (IO.userError "expected root to select fusedContract")

  let bad := compiled.nodes.filter fun u =>
    u.op == .SHRINK || u.op == .RESHAPE || u.op == .CONTRACT
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"expected shrinkUnsafe/reshapeUnsafe/contract to be virtualized, found {bad.length}\n{msg}")

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outArr := (outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))).decode

  let expectedBytes := Native.matmulF32 ab bb m k n
  let expected := RawBuffer.decode { dtype := .float32, data := expectedBytes }
  assertAllClose outArr expected 0.001 "contract shrinkUnsafe→reshapeUnsafe gap"

private def testPadReshapeIntoFusedContract : IO Unit := do
  let m := 3
  let k := 4
  let n := 4

  let aData : FloatArray := ⟨#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
    0.7, 0.8, 0.9, 1.0, 1.1, 1.2]⟩
  let bData : FloatArray := ⟨#[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]⟩

  let ab := Native.packF32FromF64 aData
  let bb := Native.packF32FromF64 bData
  let bPadBytes := Native.padF32 bb #[2, 2, 2] #[0, 0, 0] #[0, 0, 2]

  let (aU, bU, outU) := runTensorM do
    let a ← Tensor.buffer [m, k] .float32
    let b ← Tensor.buffer [2, 2, 2] .float32
    let bp ← StaticTensor.padUnsafe b [(0, 0), (0, 0), (0, 2)]
    let br ← StaticTensor.reshapeUnsafe bp [k, n]
    let out ← UOp.contract2D a.uop br.uop
    pure (a.uop, b.uop, out)

  let env : Env :=
    let env0 : Env := ∅
    let env1 := Interpreter.setBuffer env0 aU { dtype := .float32, data := ab }
    Interpreter.setBuffer env1 bU { dtype := .float32, data := bb }

  let compiled := Interpreter.compile outU
  match compiled.implMap[outU.uid]? with
  | some (.fusedContract _) =>
    pure ()
  | _ => throw (IO.userError "expected root to select fusedContract")

  let bad := compiled.nodes.filter fun u =>
    u.op == .PAD || u.op == .RESHAPE || u.op == .CONTRACT
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"expected padUnsafe/reshapeUnsafe/contract to be virtualized, found {bad.length}\n{msg}")

  let expectedBytes := Native.matmulF32 ab bPadBytes m k n
  let expected := RawBuffer.decode { dtype := .float32, data := expectedBytes }

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outArr := (outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))).decode
  assertAllClose outArr expected 0.001 "contract padUnsafe→reshapeUnsafe"

private def testPadReshapeStackIntoFusedContract : IO Unit := do
  let m := 2
  let k := 2
  let n := 6

  let aData : FloatArray := ⟨#[0.1, 0.2, 0.3, 0.4]⟩
  let bData : FloatArray := ⟨#[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]⟩

  let ab := Native.packF32FromF64 aData
  let bb := Native.packF32FromF64 bData
  let bPadBytes := Native.padF32 bb #[2, 3] #[1, 0] #[1, 0]

  let (aU, bU, outU) := runTensorM do
    let a ← Tensor.buffer [m, k] .float32
    let b ← Tensor.buffer [2, 3] .float32
    let bp ← StaticTensor.padUnsafe b [(1, 1), (0, 0)]
    let br ← StaticTensor.reshapeUnsafe bp [k, n]
    let out ← UOp.contract2D a.uop br.uop
    pure (a.uop, b.uop, out)

  let env : Env :=
    let env0 : Env := ∅
    let env1 := Interpreter.setBuffer env0 aU { dtype := .float32, data := ab }
    Interpreter.setBuffer env1 bU { dtype := .float32, data := bb }

  let compiled := Interpreter.compile outU
  match compiled.implMap[outU.uid]? with
  | some (.fusedContract plan) =>
    if !plan.needsStack then
      throw (IO.userError "expected padUnsafe→reshapeUnsafe to require a view stackUnsafe")
  | _ => throw (IO.userError "expected root to select fusedContract")

  let bad := compiled.nodes.filter fun u =>
    u.op == .PAD || u.op == .RESHAPE || u.op == .CONTRACT
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"expected padUnsafe/reshapeUnsafe/contract to be virtualized, found {bad.length}\n{msg}")

  let expectedBytes := Native.matmulF32 ab bPadBytes m k n
  let expected := RawBuffer.decode { dtype := .float32, data := expectedBytes }

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outArr := (outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))).decode
  assertAllClose outArr expected 0.001 "contract padUnsafe→reshapeUnsafe stackUnsafe"

private def testFlipReshapeIntoFusedContract : IO Unit := do
  let m := 3
  let k := 2
  let n := 4

  let aData : FloatArray := ⟨#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]⟩
  let bData : FloatArray := ⟨#[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]⟩

  let ab := Native.packF32FromF64 aData
  let bb := Native.packF32FromF64 bData
  let bFlipBytes := Native.flipF32 bb #[2, 2, 2] #[2]

  let (aU, bU, outU) := runTensorM do
    let a ← Tensor.buffer [m, k] .float32
    let b ← Tensor.buffer [2, 2, 2] .float32
    let bf ← StaticTensor.flipUnsafe b [2]
    let br ← StaticTensor.reshapeUnsafe bf [k, n]
    let out ← UOp.contract2D a.uop br.uop
    pure (a.uop, b.uop, out)

  let env : Env :=
    let env0 : Env := ∅
    let env1 := Interpreter.setBuffer env0 aU { dtype := .float32, data := ab }
    Interpreter.setBuffer env1 bU { dtype := .float32, data := bb }

  let compiled := Interpreter.compile outU
  match compiled.implMap[outU.uid]? with
  | some (.fusedContract plan) =>
    if !plan.needsStack then
      throw (IO.userError "expected flipUnsafe→reshapeUnsafe to require a view stackUnsafe")
  | _ => throw (IO.userError "expected root to select fusedContract")

  let bad := compiled.nodes.filter fun u =>
    u.op == .FLIP || u.op == .RESHAPE || u.op == .CONTRACT
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"expected flipUnsafe/reshapeUnsafe/contract to be virtualized, found {bad.length}\n{msg}")

  let expectedBytes := Native.matmulF32 ab bFlipBytes m k n
  let expected := RawBuffer.decode { dtype := .float32, data := expectedBytes }

  let outCache := Interpreter.evalCompiledRaw compiled env
  let outArr := (outCache.getD outU.uid (RawBuffer.zeros outU.dtype (listProd outU.shape))).decode
  assertAllClose outArr expected 0.001 "contract flipUnsafe→reshapeUnsafe"

def runAll : IO Unit := do
  IO.println "=== ContractViewFusionSmoke Tests ==="
  testPermuteIntoFusedContract
  IO.println "✓ permuteUnsafe virtualized in fused contract"
  testPermuteReshapeIntoFusedContract
  IO.println "✓ permuteUnsafe→reshapeUnsafe virtualized in fused contract"
  testShrinkReshapeFoldIntoFusedContract
  IO.println "✓ shrinkUnsafe→reshapeUnsafe fold virtualized in fused contract"
  testShrinkReshapeGapIntoFusedContract
  IO.println "✓ shrinkUnsafe→reshapeUnsafe gap virtualized in fused contract"
  testPadReshapeIntoFusedContract
  IO.println "✓ padUnsafe→reshapeUnsafe virtualized in fused contract"
  testPadReshapeStackIntoFusedContract
  IO.println "✓ padUnsafe→reshapeUnsafe stackUnsafe virtualized in fused contract"
  testFlipReshapeIntoFusedContract
  IO.println "✓ flipUnsafe→reshapeUnsafe virtualized in fused contract"
  IO.println "=== ContractViewFusionSmoke OK ==="

end TinyGrad4.Test.ContractViewFusionSmoke

#eval! TinyGrad4.Test.ContractViewFusionSmoke.runAll
