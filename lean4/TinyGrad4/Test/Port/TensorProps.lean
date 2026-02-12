import Float64
import TinyGrad4
import TinyGrad4.Test.Case
import TinyGrad4.Test.Profiles
import TinyGrad4.Test.Assertions

namespace TinyGrad4.Test.Port.TensorProps

open TinyGrad4
open TinyGrad4.Interpreter
open TinyGrad4.StaticTensor
open TinyGrad4.Test
open TinyGrad4.Test.Assertions

def testZeroDimInitialization : IO Unit := do
  let scalar := runTensorM do
    Tensor.full [] .float32 55.0
  assertShape scalar.uop.shape [] "Tensor.full scalar shape"
  let raw := evalTensor scalar
  assertRawAllClose raw #[55.0] 0.0001 "Tensor.full scalar value"

def testZerosLikeOnesLike : IO Unit := do
  let (z, o) := runTensorM do
    let base ← Tensor.full [2, 3] .float32 7.0
    let z ← Tensor.zerosLike base
    let o ← Tensor.onesLike base
    pure (z, o)
  assertShape z.uop.shape [2, 3] "zerosLike shape"
  assertShape o.uop.shape [2, 3] "onesLike shape"
  assertRawAllClose (evalTensor z) (Array.replicate 6 0.0) 0.0001 "zerosLike values"
  assertRawAllClose (evalTensor o) (Array.replicate 6 1.0) 0.0001 "onesLike values"

def testRandSeedDeterminism : IO Unit := do
  let (a, b, c) := runTensorM do
    let a ← Tensor.rand [8] .float32 1337
    let b ← Tensor.rand [8] .float32 1337
    let c ← Tensor.rand [8] .float32 1338
    pure (a, b, c)
  let ra := evalTensor a
  let rb := evalTensor b
  let rc := evalTensor c
  if ra.data != rb.data then
    throw <| IO.userError "Tensor.rand should be deterministic for identical seeds"
  if ra.data == rc.data then
    throw <| IO.userError "Tensor.rand should produce different outputs for different seeds"

def testNumelTriplePlausible (cfg : RunConfig) : IO Unit :=
  assertPlausible cfg "numel triple product"
    (∀ a b c : Nat, Shape.numel [a, b, c] = a * b * c)

def testBroadcastableReflPlausible (cfg : RunConfig) : IO Unit :=
  assertPlausible cfg "broadcastable reflexivity"
    (∀ s : List Nat, Shape.broadcastable s s = true)

def cases : List TestCase :=
  [
    {
      name := "tensor.zerodim.initialization"
      group := "tensor"
      minProfile := .fast
      suite := fun _ => ioTest "Tensor scalar initialization" testZeroDimInitialization
    },
    {
      name := "tensor.like.zeros_ones"
      group := "tensor"
      minProfile := .fast
      suite := fun _ => ioTest "zerosLike/onesLike preserve shape and values" testZerosLikeOnesLike
    },
    {
      name := "tensor.rand.seed_determinism"
      group := "tensor"
      minProfile := .fast
      suite := fun _ => ioTest "Tensor.rand deterministic by seed" testRandSeedDeterminism
    },
    {
      name := "tensor.prop.numel_triple"
      group := "tensor"
      minProfile := .medium
      suite := fun cfg =>
        ioTest "Plausible: numel triple product" (testNumelTriplePlausible cfg)
    },
    {
      name := "tensor.prop.broadcastable_refl"
      group := "tensor"
      minProfile := .fast
      suite := fun cfg =>
        ioTest "Plausible: broadcastable reflexivity" (testBroadcastableReflPlausible cfg)
    }
  ]

end TinyGrad4.Test.Port.TensorProps
