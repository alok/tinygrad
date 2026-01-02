import TinyGrad4.Data.Profile

-- Disable IO.monoNanosNow linter: this module defines the timing monad instance.
set_option linter.monoNanosNow false

namespace TinyGrad4.Data

/-! ## MonadTimeNS -/

/-- Monotonic time source in nanoseconds. -/
class MonadTimeNS (m : Type → Type) where
  monoNs : m Nat

namespace MonadTimeNS

def monoNs [MonadTimeNS m] : m Nat :=
  MonadTimeNS.monoNs

end MonadTimeNS

instance : MonadTimeNS IO where
  monoNs := IO.monoNanosNow

/-! ## MonadTiming -/

/-- Provides access to a profiler for timing spans. -/
class MonadTiming (m : Type → Type) where
  getProfiler : m Profiler

namespace MonadTiming

def getProfiler [MonadTiming m] : m Profiler :=
  MonadTiming.getProfiler

end MonadTiming

/-! ## TimingT -/

/-- Reader-based timing transformer. -/
abbrev TimingT (m : Type → Type) : Type → Type :=
  ReaderT Profiler m

namespace TimingT

def run (p : Profiler) (x : TimingT m α) : m α :=
  x p

end TimingT

instance [Monad m] : MonadTiming (TimingT m) where
  getProfiler := read

instance [Monad m] [MonadTimeNS m] : MonadTimeNS (TimingT m) where
  monoNs := lift MonadTimeNS.monoNs

/-! ## Timing helpers -/

/-- Record a named span around an action using the current profiler. -/
def timeSpan [Monad m] [MonadTimeNS m] [MonadTiming m] [MonadLiftT IO m]
    (name : String) (action : m α) : m α := do
  let start ← MonadTimeNS.monoNs
  let result ← action
  let stop ← MonadTimeNS.monoNs
  let p ← MonadTiming.getProfiler
  liftM (m := IO) <| p.recordSampleSpan name start stop
  pure result

/-- Time a raw IO action inside a timing monad. -/
def timeIO [Monad m] [MonadTimeNS m] [MonadTiming m] [MonadLiftT IO m]
    (name : String) (action : IO α) : m α := do
  timeSpan name (liftM (m := IO) action)

end TinyGrad4.Data
