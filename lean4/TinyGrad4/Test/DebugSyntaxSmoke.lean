import TinyGrad4

/-!
# DebugSyntaxSmoke

Ensures the `tg.debug.*` macros expand and type-check.
-/

namespace TinyGrad4.Test.DebugSyntaxSmoke

open TinyGrad4

private def testPrint : IO Unit := do
  let _ : Unit := tg.debug.print "hello"
  let _ : Unit := tg.debug.print[ordered] "ordered"
  let _ : Unit := tg.debug.print[partitioned] "partitioned"
  let _ : Unit := tg.debug.print[ordered, partitioned] "both"
  pure (f := IO) ()

private def testCallback : IO Unit := do
  let f : Nat → Unit := fun _ => ()
  let _ : Unit := tg.debug.callback f 1
  let _ : Unit := tg.debug.callback[ordered] f 2
  let _ : Unit := tg.debug.callback[partitioned] f 3
  let _ : Unit := tg.debug.callback[ordered, partitioned] f 4
  pure (f := IO) ()

private def testBreakpoint : IO Unit := do
  let _ : Unit := (tg.debug.breakpoint)
  let _ : Unit := tg.debug.breakpoint "here"
  pure (f := IO) ()

def runAll : IO Unit := do
  IO.println "=== DebugSyntaxSmoke Tests ==="
  testPrint
  IO.println "✓ debug.print"
  testCallback
  IO.println "✓ debug.callback"
  testBreakpoint
  IO.println "✓ debug.breakpoint"
  IO.println "=== DebugSyntaxSmoke OK ==="

end TinyGrad4.Test.DebugSyntaxSmoke

