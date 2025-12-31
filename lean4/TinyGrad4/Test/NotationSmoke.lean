import TinyGrad4

/-!
# NotationSmoke

Checks we can use the `Float32^[n, m, k]` tensor type notation and `^[...]` shape literals.
-/

namespace TinyGrad4.Test.NotationSmoke

open TinyGrad4

private def testTensorTypeNotation : IO Unit := do
  let (_ : Float32^[2, 3]) := runTensorM do
    let x ← Tensor.zeros ^[2, 3] .float32
    pure x
  pure ()

def runAll : IO Unit := do
  IO.println "=== NotationSmoke Tests ==="
  testTensorTypeNotation
  IO.println "✓ tensor type notation"
  IO.println "=== NotationSmoke OK ==="

end TinyGrad4.Test.NotationSmoke

#eval! TinyGrad4.Test.NotationSmoke.runAll

