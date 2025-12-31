import TinyGrad4

/-!
# OptimInterfaceSmoke

Checks the optimizer interface builds fused-friendly graphs.
-/

namespace TinyGrad4.Test.OptimInterfaceSmoke

open TinyGrad4
open StaticTensor
open Interpreter

private def testSGDFusion : IO Unit := do
  let cfg : TinyGrad4.Optim.SGD := { learningRate := 0.1, momentum := 0.0 }
  let newUop := runTensorM do
    let w ← Tensor.buffer [4] .float32
    let g ← Tensor.buffer [4] .float32
    let res ← TinyGrad4.Optim.SGD.optimizer.step w g [] cfg
    pure res.param.uop

  let compiled ← Interpreter.compileManyCached [newUop]
  match compiled.implMap[newUop.uid]? with
  | some (.fusedSGD _) => pure ()
  | _ => throw (IO.userError "expected fusedSGD for SGD optimizer step (no momentum)")

private def testSGDMomentumGraph : IO Unit := do
  let cfg : TinyGrad4.Optim.SGD := { learningRate := 0.1, momentum := 0.9 }
  let (newW, newV) := runTensorM do
    let w ← Tensor.buffer [4] .float32
    let g ← Tensor.buffer [4] .float32
    let v ← Tensor.buffer [4] .float32
    let res ← TinyGrad4.Optim.SGD.optimizer.step w g [v] cfg
    match res.state with
    | [v'] => pure (res.param.uop, v'.uop)
    | _ => panic! "SGD momentum state mismatch"

  let _ ← Interpreter.compileManyCached [newW, newV]
  pure ()

private def testAdamWGraph : IO Unit := do
  let cfg : TinyGrad4.Optim.AdamW := { lr := 0.001, step := 1 }
  let (newW, newM, newV) := runTensorM do
    let w ← Tensor.buffer [4] .float32
    let g ← Tensor.buffer [4] .float32
    let m ← Tensor.buffer [4] .float32
    let v ← Tensor.buffer [4] .float32
    let res ← TinyGrad4.Optim.AdamW.optimizer.step w g [m, v] cfg
    match res.state with
    | [m', v'] => pure (res.param.uop, m'.uop, v'.uop)
    | _ => panic! "AdamW state mismatch"

  let _ ← Interpreter.compileManyCached [newW, newM, newV]
  pure ()

def runAll : IO Unit := do
  IO.println "=== OptimInterfaceSmoke Tests ==="
  testSGDFusion
  IO.println "✓ sgd interface fuses"
  testSGDMomentumGraph
  IO.println "✓ sgd momentum interface builds"
  testAdamWGraph
  IO.println "✓ adamw interface builds"
  IO.println "=== OptimInterfaceSmoke OK ==="

end TinyGrad4.Test.OptimInterfaceSmoke

#eval! TinyGrad4.Test.OptimInterfaceSmoke.runAll
