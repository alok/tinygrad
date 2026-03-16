import Plausible
import TinyGrad4
import TinyGrad4.Spec
import TinyGrad4.Test.Profiles

/-!
Compile-time Plausible property checks.

These are intentionally compile-time guards: they fail the build if Plausible
finds a counter-example, and they complement the runtime LSpec suites.
-/

namespace TinyGrad4.Test.PlausibleProps

open TinyGrad4
open TinyGrad4.Test

private def fastCfg : Plausible.Configuration :=
  plausibleConfig { profile := .fast, seed := 1337 }

#eval Plausible.Testable.check (∀ a b c : Nat, Shape.numel [a, b, c] = a * b * c) fastCfg
#eval Plausible.Testable.check (∀ s : List Nat, Shape.broadcastable s s = true) fastCfg
#eval Plausible.Testable.check (∀ s1 s2 : List Nat, Shape.broadcastable s1 s2 = Shape.broadcastable s2 s1) fastCfg

end TinyGrad4.Test.PlausibleProps
