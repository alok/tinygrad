import Lean

open Lean Meta

/-!
# Custom Simp Sets for TinyGrad4

Modeled after peephole-formal's approach, we define specialized simp sets
for different domains within TinyGrad4.

## Usage

Tag lemmas with the appropriate attribute:
```lean
@[simp_shape]
theorem listProd_singleton (n : Nat) : listProd [n] = n := by simp [listProd]
```

Use in proofs:
```lean
simp [simp_shape]
```
-/

/-- Simp lemmas for shape calculations (listProd, broadcast, strides) -/
initialize simpShape : SimpExtension ←
  registerSimpAttr `simp_shape
    "simp lemmas for shape calculations"

/-- Simp lemmas for dtype conversions and properties -/
initialize simpDtype : SimpExtension ←
  registerSimpAttr `simp_dtype
    "simp lemmas for dtype operations"

/-- Simp lemmas for UOp graph transformations -/
initialize simpUop : SimpExtension ←
  registerSimpAttr `simp_uop
    "simp lemmas for UOp graph transformations"

/-- Simp lemmas for cost model calculations -/
initialize simpCost : SimpExtension ←
  registerSimpAttr `simp_cost
    "simp lemmas for cost model calculations"

/-- Simp lemmas for view and ShapeTracker operations -/
initialize simpView : SimpExtension ←
  registerSimpAttr `simp_view
    "simp lemmas for view operations"
