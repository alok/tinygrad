import Float64
import TinyGrad4
import TinyGrad4.Test.Case
import TinyGrad4.Test.Profiles
import TinyGrad4.Test.Assertions

namespace TinyGrad4.Test.Port.OpsProps

open TinyGrad4
open TinyGrad4.Interpreter
open TinyGrad4.StaticTensor
open TinyGrad4.Test
open TinyGrad4.Test.Assertions

def testZerosOnesCreation : IO Unit := do
  let (z, o) := runTensorM do
    let z ← Tensor.zeros [2, 2] .float32
    let o ← Tensor.ones [2, 2] .float32
    pure (z, o)
  assertRawAllClose (evalTensor z) (Array.replicate 4 0.0) 0.0001 "Tensor.zeros values"
  assertRawAllClose (evalTensor o) (Array.replicate 4 1.0) 0.0001 "Tensor.ones values"

def testArangeValues : IO Unit := do
  let t := runTensorM do
    Tensor.arange 6 .float32
  assertShape t.uop.shape [6] "Tensor.arange shape"
  assertRawAllClose (evalTensor t) #[0.0, 1.0, 2.0, 3.0, 4.0, 5.0] 0.0001 "Tensor.arange values"

def testBroadcastAddValues : IO Unit := do
  let out := runTensorM do
    let a ← Tensor.full [2, 1] .float32 2.0
    let b ← Tensor.full [1, 3] .float32 3.0
    addBroadcast a b (by native_decide)
  assertShape out.uop.shape [2, 3] "addBroadcast output shape"
  assertRawAllClose (evalTensor out) (Array.replicate 6 5.0) 0.0001 "addBroadcast output values"

def testBroadcastableCommPlausible (cfg : RunConfig) : IO Unit :=
  assertPlausible cfg "broadcastable symmetry"
    (∀ s1 s2 : List Nat, Shape.broadcastable s1 s2 = Shape.broadcastable s2 s1)

def testBroadcastOutReflPlausible (cfg : RunConfig) : IO Unit :=
  assertPlausible cfg "broadcastOut reflexivity"
    (∀ s : List Nat, Shape.broadcastOut s s = s)

def cases : List TestCase :=
  [
    {
      name := "ops.creation.zeros_ones"
      group := "ops"
      minProfile := .fast
      suite := fun _ => ioTest "Tensor.zeros/Tensor.ones values" testZerosOnesCreation
    },
    {
      name := "ops.creation.arange"
      group := "ops"
      minProfile := .fast
      suite := fun _ => ioTest "Tensor.arange sequence" testArangeValues
    },
    {
      name := "ops.broadcast.add"
      group := "ops"
      minProfile := .fast
      suite := fun _ => ioTest "addBroadcast computes expected matrix" testBroadcastAddValues
    },
    {
      name := "ops.prop.broadcastable_comm"
      group := "ops"
      minProfile := .medium
      suite := fun cfg =>
        ioTest "Plausible: broadcastable symmetry" (testBroadcastableCommPlausible cfg)
    },
    {
      name := "ops.prop.broadcast_out_refl"
      group := "ops"
      minProfile := .fast
      suite := fun cfg =>
        ioTest "Plausible: broadcastOut reflexivity" (testBroadcastOutReflPlausible cfg)
    }
  ]

end TinyGrad4.Test.Port.OpsProps
