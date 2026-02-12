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

def testReshapeFlattenRoundTrip : IO Unit := do
  let (reshaped, flat) := runTensorM do
    let base ← Tensor.arange 6 .float32
    let reshaped ← reshapeUnsafe base [2, 3]
    let flat ← flatten reshaped
    pure (reshaped, flat)
  assertShape reshaped.uop.shape [2, 3] "reshapeUnsafe output shape"
  assertShape flat.uop.shape [6] "flatten output shape"
  assertRawAllClose (evalTensor flat) #[0.0, 1.0, 2.0, 3.0, 4.0, 5.0] 0.0001 "reshape/flatten round-trip values"

def testPermuteAndTransposeValues : IO Unit := do
  let (permuted, transposed) := runTensorM do
    let base ← Tensor.arange 6 .float32
    let mat ← reshapeUnsafe base [2, 3]
    let permuted ← permuteUnsafe mat [1, 0]
    let transposed ← T mat
    pure (permuted, transposed)
  assertShape permuted.uop.shape [3, 2] "permuteUnsafe output shape"
  assertShape transposed.uop.shape [3, 2] "transpose output shape"
  let expected := #[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
  assertRawAllClose (evalTensor permuted) expected 0.0001 "permuteUnsafe values"
  assertRawAllClose (evalTensor transposed) expected 0.0001 "transpose values"

def testExpandBroadcastValues : IO Unit := do
  let expanded := runTensorM do
    let base ← Tensor.arange 2 .float32
    let col ← reshapeUnsafe base [2, 1]
    expandUnsafe col [2, 3]
  assertShape expanded.uop.shape [2, 3] "expandUnsafe output shape"
  assertRawAllClose (evalTensor expanded) #[0.0, 0.0, 0.0, 1.0, 1.0, 1.0] 0.0001 "expandUnsafe broadcast values"

def testReduceAxisSemantics : IO Unit := do
  let (sumKeep, sumDrop, maxDrop) := runTensorM do
    let base ← Tensor.arange 6 .float32
    let mat ← reshapeUnsafe base [2, 3]
    let sumKeep ← sumAxis mat 1 true
    let sumDrop ← sumAxis mat 1 false
    let maxDrop ← maxAxis mat 1 false
    pure (sumKeep, sumDrop, maxDrop)
  assertShape sumKeep.uop.shape [2, 1] "sumAxis keepdim shape"
  assertShape sumDrop.uop.shape [2] "sumAxis no-keepdim shape"
  assertShape maxDrop.uop.shape [2] "maxAxis no-keepdim shape"
  assertRawAllClose (evalTensor sumKeep) #[3.0, 12.0] 0.0001 "sumAxis keepdim values"
  assertRawAllClose (evalTensor sumDrop) #[3.0, 12.0] 0.0001 "sumAxis no-keepdim values"
  assertRawAllClose (evalTensor maxDrop) #[2.0, 5.0] 0.0001 "maxAxis no-keepdim values"

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
      name := "ops.move.reshape_flatten_roundtrip"
      group := "ops"
      minProfile := .fast
      suite := fun _ => ioTest "reshapeUnsafe/flatten round-trip semantics" testReshapeFlattenRoundTrip
    },
    {
      name := "ops.move.permute_transpose"
      group := "ops"
      minProfile := .fast
      suite := fun _ => ioTest "permuteUnsafe and transpose matrix semantics" testPermuteAndTransposeValues
    },
    {
      name := "ops.move.expand"
      group := "ops"
      minProfile := .fast
      suite := fun _ => ioTest "expandUnsafe broadcast semantics" testExpandBroadcastValues
    },
    {
      name := "ops.reduce.axis_semantics"
      group := "ops"
      minProfile := .fast
      suite := fun _ => ioTest "sumAxis/maxAxis keepdim semantics" testReduceAxisSemantics
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
