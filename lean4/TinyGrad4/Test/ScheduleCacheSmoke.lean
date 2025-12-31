import TinyGrad4

/-!
# ScheduleCacheSmoke

Checks the optional schedule cache and the GC-style size cap.
-/

namespace TinyGrad4.Test.ScheduleCacheSmoke

open TinyGrad4
open Interpreter

private def assertEqNat (got expected : Nat) (label : String) : IO Unit := do
  if got != expected then
    throw (IO.userError s!"{label}: got {got}, expected {expected}")

private def assertLeNat (got limit : Nat) (label : String) : IO Unit := do
  if got > limit then
    throw (IO.userError s!"{label}: got {got}, expected <= {limit}")

private def mkGraphs : UOp × UOp := runTensorM do
  let x ← Tensor.buffer [2, 2] .float32
  let y ← Tensor.buffer [2, 2] .float32
  let add ← UOp.add x.uop y.uop
  let mul ← UOp.mul x.uop y.uop
  pure (add, mul)

def testScheduleCacheSize : IO Unit := do
  Interpreter.clearScheduleCache
  Interpreter.setScheduleCacheLimit 0
  let (u1, u2) := mkGraphs
  let _ ← Interpreter.compileManyCached [u1]
  let _ ← Interpreter.compileManyCached [u2]
  let sz ← Interpreter.getScheduleCacheSize
  assertEqNat sz 2 "cache size (unbounded)"
  Interpreter.clearScheduleCache

def testScheduleCacheLimit : IO Unit := do
  Interpreter.clearScheduleCache
  Interpreter.setScheduleCacheLimit 1
  let (u1, u2) := mkGraphs
  let _ ← Interpreter.compileManyCached [u1]
  let _ ← Interpreter.compileManyCached [u2]
  let sz ← Interpreter.getScheduleCacheSize
  assertLeNat sz 1 "cache size (limit 1)"
  Interpreter.setScheduleCacheLimit 0
  Interpreter.clearScheduleCache

def runAll : IO Unit := do
  IO.println "ScheduleCacheSmoke"
  testScheduleCacheSize
  testScheduleCacheLimit

end TinyGrad4.Test.ScheduleCacheSmoke

#eval! TinyGrad4.Test.ScheduleCacheSmoke.runAll
