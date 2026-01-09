import TinyGrad4.Timing
import TinyGrad4.Data.Profile

namespace TinyGrad4.Data

/-! ## MonadTiming -/

/-- Provides access to a profiler for timing spans. -/
class MonadTiming (m : Type → Type) where
  getProfiler : m Profiler

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

instance [Monad m] [TinyGrad4.MonadTimeNS m] : TinyGrad4.MonadTimeNS (TimingT m) where
  monoNs := fun _ => TinyGrad4.MonadTimeNS.monoNs

/-! ## Timing helpers -/

/-- Record a named span around an action using the current profiler. -/
def timeSpan [Monad m] [TinyGrad4.MonadTimeNS m] [MonadTiming m] [MonadLiftT IO m]
    (name : String) (action : m α) : m α := do
  let start ← TinyGrad4.MonadTimeNS.monoNs
  let result ← action
  let stop ← TinyGrad4.MonadTimeNS.monoNs
  let p ← MonadTiming.getProfiler
  liftM (m := IO) <| p.recordSampleSpan name start stop
  pure result

/-- Time a raw IO action inside a timing monad. -/
def timeIO [Monad m] [TinyGrad4.MonadTimeNS m] [MonadTiming m] [MonadLiftT IO m]
    (name : String) (action : IO α) : m α := do
  timeSpan name (liftM (m := IO) action)

end TinyGrad4.Data
