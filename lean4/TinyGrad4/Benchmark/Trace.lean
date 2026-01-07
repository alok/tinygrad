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

/-! ## Global Trace Event Collection

Similar to profileEvents in Instrumentation.lean, but for pipeline-level tracing.
Enable with `TG4_TRACE=1` environment variable.
-/

-- Disable monoNanosNow linter for timing code
set_option linter.monoNanosNow false

/-- Global trace events -/
initialize traceEvents : IO.Ref (Array TraceEvent) ← IO.mkRef #[]

/-- Whether tracing is enabled (checked from TG4_TRACE env once at init) -/
initialize traceEnabled : IO.Ref Bool ← IO.mkRef false

/-- Initialize trace enabled flag from environment -/
initialize do
  let env ← IO.getEnv "TG4_TRACE"
  traceEnabled.set env.isSome

/-- Check if tracing is enabled -/
def isTracingEnabled : IO Bool := traceEnabled.get

/-- Trace a pipeline phase with timing.
    Records TraceEvent when TG4_TRACE=1 is set. -/
def withTrace (phase : String) (uopStats : Option UOp.UOpStats := none)
    (action : IO α) : IO α := do
  if !(← traceEnabled.get) then
    return ← action
  let startNs ← IO.monoNanosNow
  let result ← action
  let endNs ← IO.monoNanosNow
  traceEvents.modify (·.push {
    phase, startNs, durationNs := endNs - startNs, uopStats
  })
  return result

/-- Trace a pure computation (wraps in IO) -/
def withTracePure (phase : String) (uopStats : Option UOp.UOpStats := none)
    (action : Unit → α) : IO α := do
  withTrace phase uopStats (pure (action ()))

/-- Get all recorded trace events -/
def getTraceEvents : IO (Array TraceEvent) := traceEvents.get

/-- Reset/clear trace events -/
def resetTraceEvents : IO Unit := traceEvents.set #[]

/-- Print trace events summary to stdout -/
def printTraceEvents : IO Unit := do
  let events ← getTraceEvents
  if events.isEmpty then return
  IO.println "=== Trace Events ==="
  for e in events do
    let ms := e.durationNs.toFloat / 1000000.0
    let statsStr := match e.uopStats with
      | some s => s!" (nodes={s.total})"
      | none => ""
    IO.println s!"{e.phase}: {ms} ms{statsStr}"
  IO.println "===================="

/-- Export trace events as JSON array -/
def traceEventsToJson : IO Lean.Json := do
  let events ← getTraceEvents
  return .arr (events.map toJson)

/-- Build and optionally write trace report if TG4_TRACE_OUT is set -/
def maybeWriteTraceReport (meta : TraceMeta) : IO Unit := do
  match ← traceOutPath? with
  | none => return
  | some path =>
    let events ← getTraceEvents
    let report : TraceReport := { metaInfo := meta, events }
    writeReport path report

end TinyGrad4.Benchmark.Trace
