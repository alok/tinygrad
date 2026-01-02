import TinyGrad4

/-!
# TUOp Smoke Test

Ensures the typed TUOp builder can construct a simple graph and pass validation.
-/

namespace TinyGrad4.Test.TUOpSmoke

open TinyGrad4

private def assertValid (roots : List UOp) (label : String) : IO Unit := do
  let errs := UOp.validateMany roots
  if errs.size != 0 then
    let shown := errs.toList.take 10 |>.map (fun e => e.render)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"{label}: {errs.size} validation errors\n{msg}")

private def testTUOpBasic : IO Unit := do
  let outUop : UOp := runUOpM do
    let a ← TUOp.buffer .float32 [2, 3]
    let b ← TUOp.buffer .float32 [1, 3]
    let sum ← TUOp.add a b
    let neg ← TUOp.neg sum
    pure neg.raw
  assertValid [outUop] "typed uop basic"

def runAll : IO Unit := do
  IO.println "=== TUOpSmoke Tests ==="
  testTUOpBasic
  IO.println "✓ typed UOp basic graph"
  IO.println "=== TUOpSmoke OK ==="

end TinyGrad4.Test.TUOpSmoke

#eval! TinyGrad4.Test.TUOpSmoke.runAll
