import Float64
import TinyGrad4.Backend.LeanPtxEmit

open Lean PrettyPrinter Delaborator

syntax (name := tg4Ptx) "ptx! " term : term

macro_rules
  | `(ptx! $t) => `(TinyGrad4.Backend.LeanPtxEmit.ptxSource $t)

/-- Print `ptxSource cfg` as `ptx! cfg`. -/
@[app_unexpander TinyGrad4.Backend.LeanPtxEmit.ptxSource]
def unexpandPtxSource : Lean.PrettyPrinter.Unexpander
  | `($_ $t) => `(ptx! $t)
  | _ => throw ()

namespace TinyGrad4

-- Placeholder for pretty-printers.

end TinyGrad4
