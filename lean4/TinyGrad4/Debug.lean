namespace TinyGrad4

/-- Debug helper: no-op placeholder until shared-structure tracing is implemented. -/
def dbgTraceIfShared {α} (_msg : String) (arr : Array α) : Array α :=
  arr

end TinyGrad4
