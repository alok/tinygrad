import Float64
import TinyGrad4
import TinyGrad4.Test.Case
import TinyGrad4.Test.Profiles
import TinyGrad4.Test.Assertions

namespace TinyGrad4.Test.Port.CuratedMixed

open TinyGrad4
open TinyGrad4.Interpreter
open TinyGrad4.StaticTensor
open TinyGrad4.Test
open TinyGrad4.Test.Assertions

def testSquareGradient : IO Unit := do
  let grad? := runTensorM do
    let x ← Tensor.full [1] .float32 3.0
    let x2 ← mul x x
    let loss ← sum x2
    let gradMap ← backward loss [x.uop]
    pure (gradMap[x.uop.uid]?)
  match grad? with
  | none =>
    throw <| IO.userError "backward did not produce gradient for x"
  | some gradUop =>
    let gradRaw := eval gradUop (∅ : Env)
    assertRawAllClose gradRaw #[6.0] 0.01 "d/dx x^2 at x=3"

def testWhereGradientRouting : IO Unit := do
  let (xUop, yUop, cUop, gradMap) := runTensorM do
    let x ← Tensor.buffer [2] .float32
    let y ← Tensor.buffer [2] .float32
    let c ← Tensor.buffer [2] .bool
    let outUop ← UOp.where_ c.uop x.uop y.uop
    let lossUop ← UOp.sum outUop [] false
    let loss : Scalar .float32 := StaticTensor.ofUOp lossUop
    let gradMap ← StaticTensor.backward loss [x.uop, y.uop]
    pure (x.uop, y.uop, c.uop, gradMap)

  let xVal : RawBuffer := RawBuffer.ofFloats #[10.0, 20.0]
  let yVal : RawBuffer := RawBuffer.ofFloats #[30.0, 40.0]
  let cVal : RawBuffer := { dtype := .bool, data := ByteArray.mk #[1, 0] }

  let env0 : Env := ∅
  let env1 := Interpreter.setBuffer env0 xUop xVal
  let env2 := Interpreter.setBuffer env1 yUop yVal
  let env := Interpreter.setBuffer env2 cUop cVal

  match gradMap[xUop.uid]?, gradMap[yUop.uid]? with
  | some dxUop, some dyUop =>
    let dx := Interpreter.eval dxUop env
    let dy := Interpreter.eval dyUop env
    assertRawAllClose dx #[1.0, 0.0] 0.01 "where d/dx routes to selected branch"
    assertRawAllClose dy #[0.0, 1.0] 0.01 "where d/dy routes to selected branch"
  | _, _ =>
    throw <| IO.userError "missing gradients for where branches"

def testMatrixVectorBroadcastPlausible (cfg : RunConfig) : IO Unit :=
  assertPlausible cfg "matrix/vector broadcastability"
    (∀ (batch dim : Nat), Shape.broadcastable [batch, dim] [dim] = true)

def cases : List TestCase :=
  [
    {
      name := "curated.grad.square"
      group := "curated"
      minProfile := .medium
      suite := fun _ => ioTest "autodiff: d/dx x^2 = 2x" testSquareGradient
    },
    {
      name := "curated.grad.where_routing"
      group := "curated"
      minProfile := .medium
      suite := fun _ => ioTest "autodiff: where routes gradients correctly" testWhereGradientRouting
    },
    {
      name := "curated.prop.matrix_vector_broadcast"
      group := "curated"
      minProfile := .fast
      suite := fun cfg =>
        ioTest "Plausible: matrix/vector broadcastability" (testMatrixVectorBroadcastPlausible cfg)
    }
  ]

end TinyGrad4.Test.Port.CuratedMixed
