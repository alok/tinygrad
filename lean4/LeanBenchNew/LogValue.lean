namespace LeanBenchNew

/-- Loggable scalar values for benchmarking backends. -/
inductive LogValue where
  | str (value : String)
  | nat (value : Nat)
  | int (value : Int)
  | float (value : Float)
  | bool (value : Bool)
  deriving Repr, Inhabited

end LeanBenchNew
