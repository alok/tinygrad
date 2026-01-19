import Float64
import Std

namespace LeanBenchNew

private def insertFloatAsc (x : Float64) (xs : List Float64) : List Float64 :=
  match xs with
  | [] => [x]
  | y :: ys => if x <= y then x :: y :: ys else y :: insertFloatAsc x ys

private def sortFloatAsc (xs : List Float64) : List Float64 :=
  xs.foldl (fun acc x => insertFloatAsc x acc) []

private def listGetD (xs : List Float64) (idx : Nat) (default : Float64) : Float64 :=
  match xs, idx with
  | [], _ => default
  | x :: _, 0 => x
  | _ :: xs', Nat.succ k => listGetD xs' k default

/-- Mean of a Float64 array. -/
def mean (xs : Array Float64) : Float64 :=
  if xs.size == 0 then
    0.0
  else
    xs.foldl (init := 0.0) (fun acc x => acc + x) / (Float64.ofNat xs.size)

/-- Median of a Float64 array. -/
def median (xs : Array Float64) : Float64 :=
  if xs.size == 0 then
    0.0
  else
    let sorted := sortFloatAsc xs.toList
    let n := sorted.length
    let mid := n / 2
    if n % 2 == 0 then
      let a := listGetD sorted (mid - 1) 0.0
      let b := listGetD sorted mid 0.0
      (a + b) / 2.0
    else
      listGetD sorted mid 0.0

/-- Time an action over trials with inner iterations. Prints per-iteration ms samples. -/
def timeTrials (label : String) (trials iters : Nat) (act : IO Unit) : IO (Array Float64) := do
  let t := if trials == 0 then 1 else trials
  let iters' := if iters == 0 then 1 else iters
  let mut times : Array Float64 := #[]
  for _ in [:t] do
    let start ← IO.monoNanosNow
    for _ in [:iters'] do
      act
    let stop ← IO.monoNanosNow
    let dtNs : Nat := stop - start
    let totalMs : Float64 := (Float64.ofNat dtNs) / 1.0e6
    let perMs : Float64 := totalMs / (Float64.ofNat iters')
    times := times.push perMs
  IO.println s!"{label}: trials={t} mean={mean times} ms/iter median={median times} ms/iter"
  IO.println s!"{label}: samples={times.toList}"
  return times

end LeanBenchNew
