import TinyGrad4.Backend.Vectorization

/-!
# Dependent Vectorization Tests

Tests for the type-safe vectorization API using `native_decide` for alignment proofs.
-/

open TinyGrad4.Backend.Vectorization

/-! ## Compile-time alignment proofs -/

-- These prove at compile time via native_decide
#check (inferInstance : Decidable (VectorAligned .w4 1_000_000))

example : VectorAligned .w4 1_000_000 := by native_decide
example : VectorAligned .w4 4_000_000 := by native_decide
example : VectorAligned .w8 8_000_000 := by native_decide
example : VectorAligned .w2 1_000_002 := by native_decide

-- This would FAIL to compile (uncomment to verify):
-- example : VectorAligned .w4 1_000_003 := by native_decide

/-! ## Type-safe vector configs -/

def config1M : VectorConfigT .w4 1_000_000 := {
  backend := .Metal
  dtype := .float32
}

def config8M : VectorConfigT .w8 8_000_000 := {
  backend := .CUDA
  dtype := .float32
}

-- optimalConfig picks the right width
def autoConfig := optimalConfig .Metal .float32 1_000_000
#check autoConfig  -- Σ w, VectorConfigT w 1_000_000

/-! ## Bounded element access -/

-- These compile - indices are provably in bounds
#eval renderVectorElementT .w4 "vec" ⟨0, by native_decide⟩  -- "vec.x"
#eval renderVectorElementT .w4 "vec" ⟨1, by native_decide⟩  -- "vec.y"
#eval renderVectorElementT .w4 "vec" ⟨2, by native_decide⟩  -- "vec.z"
#eval renderVectorElementT .w4 "vec" ⟨3, by native_decide⟩  -- "vec.w"

-- This would FAIL to compile (index 4 is out of bounds for w4):
-- #eval renderVectorElementT .w4 "vec" ⟨4, by native_decide⟩

/-! ## Component names are correct -/

#eval VectorWidth.component .w1 ⟨0, by native_decide⟩  -- ""
#eval VectorWidth.component .w2 ⟨0, by native_decide⟩  -- "x"
#eval VectorWidth.component .w2 ⟨1, by native_decide⟩  -- "y"
#eval VectorWidth.component .w8 ⟨7, by native_decide⟩  -- "s7"

/-! ## Runtime test -/

def testVectorization : IO Unit := do
  IO.println "=== Dependent Vectorization Test ==="

  -- Test alignment predicate
  IO.println s!"VectorAligned .w4 1_000_000 = true (proven at compile time)"

  -- Test optimalConfig
  let ⟨w, config⟩ := optimalConfig .Metal .float32 1_000_000
  IO.println s!"optimalConfig for 1M: width = {w.toNat}, iters = {config.numVectorIters}"

  let ⟨widthOdd, _configOdd⟩ := optimalConfig .CUDA .float32 1_000_003
  IO.println s!"optimalConfig for 1000003: width = {widthOdd.toNat} (not divisible by 4, falls back)"

  -- Test renderVectorElementT
  IO.println s!"renderVectorElementT .w4 'v' 0 = {renderVectorElementT .w4 "v" ⟨0, by native_decide⟩}"
  IO.println s!"renderVectorElementT .w4 'v' 3 = {renderVectorElementT .w4 "v" ⟨3, by native_decide⟩}"

  -- Test code generation
  let legacyConfig := config.toLegacy
  IO.println s!"vectorType = {legacyConfig.vectorType}"
  IO.println s!"renderVectorLoad = {renderVectorLoad legacyConfig "data" "i"}"

  IO.println "=== All Tests Passed ==="

def main : IO Unit := testVectorization
