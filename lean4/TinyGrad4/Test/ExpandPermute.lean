import TinyGrad4

/-!
# Expand/Permute Interpreter Tests

Smoke tests for movement ops that are performance-critical:
- `EXPAND` for non-scalar broadcasting
- `PERMUTE` for rank > 2 tensors
- bool expand/permute (sanity)
-/

namespace TinyGrad4.Test.ExpandPermute

-- Disable RawBuffer linter for test files that need Array Float literals
set_option linter.useRawBuffer false

open TinyGrad4
open Interpreter

private def assertAllClose (arr : Array Float) (expected : Array Float) (tol : Float) (label : String) : IO Unit := do
  if arr.size != expected.size then
    throw (IO.userError s!"{label}: size {arr.size} != {expected.size}")
  for i in [:arr.size] do
    let v := arr[i]!
    let e := expected[i]!
    let diff := Float.abs (v - e)
    if diff > tol then
      throw (IO.userError s!"{label}: idx {i} value {v} expected {e} diff {diff} > {tol}")

private def mkRange (n : Nat) : Array Float := Id.run do
  let mut out := Array.emptyWithCapacity n
  for i in [:n] do
    out := out.push (Float.ofNat i)
  return out

def testExpandRow : IO Unit := do
  let (buf, expanded) := runTensorM do
    let b ← UOp.buffer .float32 [1, 3]
    let e ← UOp.expand b [2, 3]
    pure (b, e)

  let inp : Array Float := #[10.0, 11.0, 12.0]
  let env := Interpreter.setBuffer (∅ : Env) buf (RawBuffer.ofFloats inp)
  let out := (Interpreter.eval expanded env).toFloatArray.data
  assertAllClose out #[10.0, 11.0, 12.0, 10.0, 11.0, 12.0] 0.0001 "expandRow"

def testPermute3D : IO Unit := do
  let (buf, permuted) := runTensorM do
    let b ← UOp.buffer .float32 [2, 3, 4]
    let p ← UOp.permute b [2, 1, 0]
    pure (b, p)

  let inp := mkRange (2 * 3 * 4)
  let env := Interpreter.setBuffer (∅ : Env) buf (RawBuffer.ofFloats inp)
  let out := (Interpreter.eval permuted env).toFloatArray.data

  if permuted.shape != [4, 3, 2] then
    throw (IO.userError s!"permute3D: shape {permuted.shape} != [4,3,2]")
  if out.size != 4 * 3 * 2 then
    throw (IO.userError s!"permute3D: size {out.size} != {4 * 3 * 2}")

  -- Input values are v[idx] = idx (as Float), so expected output is just the mapped flat index.
  for i in [:out.size] do
    let outIdx := Interpreter.unflattenIndex i permuted.shape
    let inIdx := [outIdx.getD 2 0, outIdx.getD 1 0, outIdx.getD 0 0]
    let expectedIdx := Interpreter.flattenIndex inIdx [2, 3, 4]
    let expected : Float := Float.ofNat expectedIdx
    let v := out[i]!
    let diff := Float.abs (v - expected)
    if diff > 0.0001 then
      throw (IO.userError s!"permute3D: idx {i} value {v} expected {expected} diff {diff}")

def testShrink2D : IO Unit := do
  let (buf, shrunk) := runTensorM do
    let b ← UOp.buffer .float32 [3, 4]
    let s ← UOp.shrink b [(1, 3), (0, 2)]
    pure (b, s)

  let inp := mkRange (3 * 4)
  let env := Interpreter.setBuffer (∅ : Env) buf (RawBuffer.ofFloats inp)
  let out := (Interpreter.eval shrunk env).toFloatArray.data
  if shrunk.shape != [2, 2] then
    throw (IO.userError s!"shrink2D: shape {shrunk.shape} != [2,2]")
  assertAllClose out #[4.0, 5.0, 8.0, 9.0] 0.0001 "shrink2D"

def testPad2D : IO Unit := do
  let (buf, padded) := runTensorM do
    let b ← UOp.buffer .float32 [1, 2]
    let p ← UOp.pad b [(1, 1), (2, 0)]
    pure (b, p)

  let inp : Array Float := #[1.0, 2.0]
  let env := Interpreter.setBuffer (∅ : Env) buf (RawBuffer.ofFloats inp)
  let out := (Interpreter.eval padded env).toFloatArray.data
  if padded.shape != [3, 4] then
    throw (IO.userError s!"pad2D: shape {padded.shape} != [3,4]")
  assertAllClose out
    #[0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 2.0,
      0.0, 0.0, 0.0, 0.0]
    0.0001
    "pad2D"

def testFlip2D : IO Unit := do
  let (buf, flipped1, flipped2) := runTensorM do
    let b ← UOp.buffer .float32 [2, 3]
    let f1 ← UOp.flip b [1]
    let f2 ← UOp.flip b [0, 1]
    pure (b, f1, f2)

  let inp := mkRange (2 * 3)
  let env := Interpreter.setBuffer (∅ : Env) buf (RawBuffer.ofFloats inp)
  let out1 := (Interpreter.eval flipped1 env).toFloatArray.data
  let out2 := (Interpreter.eval flipped2 env).toFloatArray.data
  assertAllClose out1 #[2.0, 1.0, 0.0, 5.0, 4.0, 3.0] 0.0001 "flip axis1"
  assertAllClose out2 #[5.0, 4.0, 3.0, 2.0, 1.0, 0.0] 0.0001 "flip axes0,1"

def testBoolExpandPermute : IO Unit := do
  let (buf, expanded, permuted) := runTensorM do
    let b ← UOp.buffer .bool [1, 2]
    let e ← UOp.expand b [2, 2]
    let p ← UOp.permute e [1, 0]
    pure (b, e, p)

  let raw : RawBuffer := { dtype := .bool, data := ByteArray.mk #[1, 0] }
  let env := Interpreter.setBuffer (∅ : Env) buf raw
  let outE : RawBuffer ← Interpreter.evalCached expanded env
  let outP : RawBuffer ← Interpreter.evalCached permuted env

  if outE.dtype != DType.bool || outE.data.size != 4 then
    throw (IO.userError s!"boolExpand: bad out dtype/size (size={outE.data.size})")
  if outP.dtype != DType.bool || outP.data.size != 4 then
    throw (IO.userError s!"boolPermute: bad out dtype/size (size={outP.data.size})")

  -- expanded from [1,2] [[1,0]] -> [[1,0],[1,0]]
  let eExpected := ByteArray.mk #[1, 0, 1, 0]
  if outE.data != eExpected then
    throw (IO.userError s!"boolExpand: {outE.data.toList} != {eExpected.toList}")

  -- permute [1,0] on [2,2]: [[1,0],[1,0]] -> [[1,1],[0,0]]
  let pExpected := ByteArray.mk #[1, 1, 0, 0]
  if outP.data != pExpected then
    throw (IO.userError s!"boolPermute: {outP.data.toList} != {pExpected.toList}")

def runAll : IO Unit := do
  IO.println "=== Expand/Permute Tests ==="
  testExpandRow
  IO.println "✓ expand non-scalar"
  testPermute3D
  IO.println "✓ permute 3D"
  testShrink2D
  IO.println "✓ shrink 2D"
  testPad2D
  IO.println "✓ pad 2D"
  testFlip2D
  IO.println "✓ flip 2D"
  testBoolExpandPermute
  IO.println "✓ bool expand+permute"
  IO.println "=== Expand/Permute OK ==="

end TinyGrad4.Test.ExpandPermute

