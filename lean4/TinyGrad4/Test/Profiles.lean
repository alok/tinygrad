import Plausible

namespace TinyGrad4.Test

inductive TestProfile where
  | fast
  | medium
  | slow
  deriving Repr, BEq, Inhabited

structure RunConfig where
  profile : TestProfile := .fast
  seed : Nat := 1337
  deriving Repr

def TestProfile.rank : TestProfile → Nat
  | .fast => 0
  | .medium => 1
  | .slow => 2

def TestProfile.allows (requested minProfile : TestProfile) : Bool :=
  minProfile.rank <= requested.rank

def TestProfile.parse? (s : String) : Option TestProfile :=
  match s.toLower with
  | "fast" => some .fast
  | "medium" => some .medium
  | "slow" => some .slow
  | _ => none

def plausibleConfig (cfg : RunConfig) : Plausible.Configuration :=
  let base : Plausible.Configuration :=
    match cfg.profile with
    | .fast => { numInst := 64, maxSize := 24, numRetries := 20, quiet := true }
    | .medium => { numInst := 256, maxSize := 64, numRetries := 30, quiet := true }
    | .slow => { numInst := 1024, maxSize := 128, numRetries := 50, quiet := true }
  { base with randomSeed := some cfg.seed }

end TinyGrad4.Test
