import TinyGrad4

/-!
# TagSyntaxSmoke

Ensures the `view/move/device/fusion/cost` tag syntax elaborates.
-/

namespace TinyGrad4.Test.TagSyntaxSmoke

open TinyGrad4

private def testTags : IO Unit := do
  let x := (1 : Int)
  let _ := view[x]
  let _ := move[x]
  let _ := device["CPU"] x
  let _ := fusion["softmax"] x
  let _ := cost[123] x
  pure ()

def runAll : IO Unit := do
  IO.println "=== TagSyntaxSmoke Tests ==="
  testTags
  IO.println "✓ tag syntax"
  IO.println "=== TagSyntaxSmoke OK ==="

end TinyGrad4.Test.TagSyntaxSmoke

