/-
Test file for the Float64 explicit linter.

This file demonstrates how the linter warns about `Float` usage.
-/
import TinyGrad4.Linter.FloatLinter

namespace FloatLinterTest

-- This should trigger a warning (uses Float)
def badExample (x : Float) : Float := x * 2.0

-- NOTE: The linter helps with code clarity. When you explicitly write Float64
-- in type signatures, the 64-bit precision is unambiguous to readers.

-- Disable the linter in a section
section SilencedSection
set_option linter.floatExplicit false
def silencedExample (x : Float) : Float := x * 2.0  -- no warning here
end SilencedSection

-- Multiple Float usages should each trigger a warning
def multipleFloats (x : Float) (y : Float) : Float := x + y

end FloatLinterTest
