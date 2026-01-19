import Float64
/-
Test file for the Float64 explicit linter.

This file demonstrates preferred `Float64` usage (no linter warnings).
-/
import TinyGrad4.Linter.FloatLinter

namespace FloatLinterTest

-- Preferred usage (Float64)
def badExample (x : Float64) : Float64 := x * 2.0

-- NOTE: The linter helps with code clarity. When you explicitly write Float64
-- in type signatures, the 64-bit precision is unambiguous to readers.

-- Disable the linter in a section
section SilencedSection
set_option linter.floatExplicit false
def silencedExample (x : Float64) : Float64 := x * 2.0  -- no warning here
end SilencedSection

-- Multiple Float64 usages are fine
def multipleFloats (x : Float64) (y : Float64) : Float64 := x + y

end FloatLinterTest
