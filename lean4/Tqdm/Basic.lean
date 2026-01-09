import Std

/-!
# Tqdm - Minimal progress bar

Lean-native progress bar inspired by Python's tqdm, with a small API surface
and no runtime dependencies beyond Std.
-/

namespace Tqdm

structure Config where
  /-- Total items (if known). -/
  total? : Option Nat := none
  /-- Optional label. -/
  desc : String := ""
  /-- Progress bar width. -/
  barWidth : Nat := 30
  /-- Minimum interval between prints (ms). -/
  minIntervalMs : Nat := 100
  /-- Minimum iterations between prints. -/
  minIters : Nat := 1
  /-- Unit label for rate display. -/
  unit : String := "it"
  /-- Show items/second rate. -/
  showRate : Bool := true
  /-- Show ETA if total is known. -/
  showEta : Bool := true
  /-- Show percent if total is known. -/
  showPercent : Bool := true
  /-- Show count (and total if known). -/
  showCount : Bool := true
  /-- ASCII bar (portable). -/
  ascii : Bool := true
  deriving Repr

structure State where
  count : Nat := 0
  startNs : Nat
  lastPrintNs : Nat
  lastPrintCount : Nat := 0
  finished : Bool := false
  deriving Repr

structure Tqdm where
  cfg : Config
  state : IO.Ref State

private def nsToSeconds (ns : Nat) : Float :=
  ns.toFloat / 1_000_000_000.0

private def repeatChar (c : Char) (n : Nat) : String :=
  String.ofList (List.replicate n c)

private def formatFloat (x : Float) (digits : Nat := 2) : String :=
  let s := x.toString
  match s.splitOn "." with
  | [a, b] => a ++ "." ++ b.take digits
  | _ => s

private def formatSeconds (secs : Nat) : String :=
  let s := secs % 60
  let m := (secs / 60) % 60
  let h := secs / 3600
  if h > 0 then
    s!"{h}h{m}m{s}s"
  else if m > 0 then
    s!"{m}m{s}s"
  else
    s!"{s}s"

private def render (cfg : Config) (st : State) (now : Nat) : String :=
  let elapsedNs := now - st.startNs
  let elapsedSec := nsToSeconds elapsedNs
  let rate := if elapsedSec > 0.0 then st.count.toFloat / elapsedSec else 0.0
  let elapsedStr := formatSeconds (elapsedNs / 1_000_000_000)
  let parts0 : List String := if cfg.desc != "" then [cfg.desc] else []
  let parts1 :=
    match cfg.total? with
    | some total =>
        let totalSafe := if total == 0 then 1 else total
        let filled := Nat.min cfg.barWidth ((st.count * cfg.barWidth) / totalSafe)
        let rest := cfg.barWidth - filled
        let fillChar := if cfg.ascii then '=' else '█'
        let restChar := if cfg.ascii then '.' else ' '
        let headChar := if cfg.ascii then '>' else '█'
        let bar :=
          if filled == 0 then
            repeatChar restChar cfg.barWidth
          else if filled >= cfg.barWidth then
            repeatChar fillChar cfg.barWidth
          else
            repeatChar fillChar (filled - 1) ++ String.singleton headChar ++ repeatChar restChar rest
        let base : List String := [s!"[{bar}]"]
        let base := if cfg.showPercent then
          let pct := (st.count * 100) / totalSafe
          base ++ [s!"{pct}%"]
        else
          base
        let base := if cfg.showCount then base ++ [s!"{st.count}/{total}"] else base
        let base := if cfg.showEta && rate > 0.0 then
          let remaining := if st.count >= total then 0 else total - st.count
          let etaSec := (remaining.toFloat / rate).toUInt64.toNat
          base ++ [s!"eta={formatSeconds etaSec}"]
        else
          base
        base
    | none =>
        if cfg.showCount then [s!"{st.count}"] else []
  let parts2 := parts0 ++ parts1
  let parts3 := if cfg.showRate then parts2 ++ [s!"{formatFloat rate} {cfg.unit}/s"] else parts2
  let parts4 := parts3 ++ [s!"elapsed={elapsedStr}"]
  String.intercalate " " parts4

private def shouldPrint (cfg : Config) (st : State) (now : Nat) (force : Bool) : Bool :=
  if force then
    true
  else
    let minNs := cfg.minIntervalMs * 1_000_000
    let timeOk := now - st.lastPrintNs >= minNs
    let countOk := st.count - st.lastPrintCount >= cfg.minIters
    timeOk && countOk

def new (cfg : Config := {}) (total? : Option Nat := none) : IO Tqdm := do
  let start ← IO.monoNanosNow
  let total := match total? with
    | some v => some v
    | none => cfg.total?
  let cfg' := { cfg with total? := total }
  let state ← IO.mkRef { startNs := start, lastPrintNs := start }
  pure { cfg := cfg', state }

def refresh (tq : Tqdm) (force : Bool := false) : IO Unit := do
  let now ← IO.monoNanosNow
  let doPrint ← tq.state.modifyGet fun st =>
    let should := shouldPrint tq.cfg st now force
    let st' := if should then { st with lastPrintNs := now, lastPrintCount := st.count } else st
    (should, st')
  if doPrint then
    let st ← tq.state.get
    let line := render tq.cfg st now
    IO.print s!"\r{line}"
    let out ← IO.getStdout
    out.flush

def set (tq : Tqdm) (count : Nat) : IO Unit := do
  tq.state.modify fun st => { st with count }
  refresh tq

def add (tq : Tqdm) (delta : Nat := 1) : IO Unit := do
  tq.state.modify fun st => { st with count := st.count + delta }
  refresh tq

def tick (tq : Tqdm) : IO Unit :=
  add tq 1

def finish (tq : Tqdm) : IO Unit := do
  refresh tq (force := true)
  IO.println ""
  tq.state.modify fun st => { st with finished := true }

def withTqdm (cfg : Config := {}) (total? : Option Nat := none) (f : Tqdm → IO α) : IO α := do
  let tq ← new cfg total?
  let result ← f tq
  finish tq
  pure result

def forInArray (arr : Array α) (cfg : Config := {}) (f : α → IO Unit) : IO Unit := do
  let tq ← new { cfg with total? := some arr.size }
  for x in arr do
    f x
    tick tq
  finish tq

def forInList (xs : List α) (cfg : Config := {}) (f : α → IO Unit) : IO Unit := do
  let tq ← new { cfg with total? := some xs.length }
  for x in xs do
    f x
    tick tq
  finish tq

def forInRange (n : Nat) (cfg : Config := {}) (f : Nat → IO Unit) : IO Unit := do
  let tq ← new { cfg with total? := some n }
  for i in [:n] do
    f i
    tick tq
  finish tq

end Tqdm
