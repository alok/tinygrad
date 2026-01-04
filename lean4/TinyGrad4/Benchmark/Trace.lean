import Lean.Data.Json
import Lean.Data.Json.Elab
import TinyGrad4.UOp.Graph

/-!
# Benchmark Trace Events

Lightweight trace schema for benchmarking and observability.
Emits JSON for schedule/codegen/render/compile/launch phases.
-/

namespace TinyGrad4.Benchmark.Trace

open TinyGrad4
open Lean

structure TraceEvent where
  phase : String
  startNs : Nat
  durationNs : Nat
  uopStats : Option _root_.TinyGrad4.UOp.UOpStats := none
  metrics : Option Lean.Json := none
  deriving Inhabited

structure TraceMeta where
  backend : String
  device : String
  kernel : String
  size : Nat
  iterations : Nat
  timestampNs : Nat
  deriving Inhabited

structure TraceReport where
  metaInfo : TraceMeta
  events : Array TraceEvent
  deriving Inhabited

instance : ToJson TraceEvent where
  toJson e := Id.run do
    let mut fields : List (Prod String Lean.Json) := [
      ("phase", toJson e.phase),
      ("start_ns", toJson e.startNs),
      ("duration_ns", toJson e.durationNs)
    ]
    match e.uopStats with
    | some stats => fields := fields ++ [("uop", toJson stats)]
    | none => pure ()
    match e.metrics with
    | some metrics => fields := fields ++ [("metrics", metrics)]
    | none => pure ()
    return Lean.Json.mkObj fields

instance : ToJson TraceMeta where
  toJson m := Lean.Json.mkObj [
    ("backend", Lean.Json.str m.backend),
    ("device", Lean.Json.str m.device),
    ("kernel", Lean.Json.str m.kernel),
    ("size", toJson m.size),
    ("iterations", toJson m.iterations),
    ("timestamp_ns", toJson m.timestampNs)
  ]

instance : ToJson TraceReport where
  toJson r := Lean.Json.mkObj [
    ("meta", toJson r.metaInfo),
    ("events", Lean.Json.arr (r.events.map toJson))
  ]

def writeReport (path : String) (report : TraceReport) : IO Unit := do
  let json := (toJson report).pretty
  IO.FS.writeFile path json

def traceOutPath? : IO (Option String) :=
  IO.getEnv "TG4_TRACE_OUT"

end TinyGrad4.Benchmark.Trace
