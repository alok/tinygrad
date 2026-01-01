import TinyGrad4
import LeanBenchNew.Stats
import Wandb.Json

namespace TinyGrad4Bench

open TinyGrad4
open Interpreter
open Std
open Wandb.Json

structure KernelAgg where
  sumMs : Float
  count : Nat
  deriving Repr

private def kernelKey (t : Interpreter.KernelTiming) : String :=
  s!"tag={t.tag} shape={t.shape} numel={t.numel} dtype={repr t.dtype}"

private def addKernelAgg (agg : HashMap String KernelAgg) (t : Interpreter.KernelTiming) : HashMap String KernelAgg :=
  let key := kernelKey t
  let cur := agg.getD key { sumMs := 0.0, count := 0 }
  agg.insert key { sumMs := cur.sumMs + t.ms, count := cur.count + 1 }

private def addKernelAggs (agg : HashMap String KernelAgg) (times : Array Interpreter.KernelTiming)
    : HashMap String KernelAgg := Id.run do
  let mut a := agg
  for t in times do
    a := addKernelAgg a t
  return a

def dumpKernelAgg (agg : HashMap String KernelAgg) : IO Unit := do
  for (key, v) in agg.toList do
    let avg := if v.count == 0 then 0.0 else v.sumMs / (Float.ofNat v.count)
    IO.println s!"  kernel.agg {key} total_ms={v.sumMs} avg_ms={avg} runs={v.count}"

def dumpKernelAggFromRuns (c : Interpreter.Compiled) (env : Env) (runs : Nat) : IO Unit := do
  let runs' := if runs == 0 then 1 else runs
  let mut agg : HashMap String KernelAgg := ∅
  for _ in [:runs'] do
    let (_, times) ← Interpreter.evalCompiledRawTimed c env
    agg := addKernelAggs agg times
  IO.println s!"fused kernel times (aggregate, runs={runs'}):"
  dumpKernelAgg agg

private def shapeJson (shape : Shape) : J :=
  arr (shape.map (fun d => nat d))

private def featureFields (f : Backend.CostFeatures) : List (String × J) :=
  [ ("feat/launches", nat f.launches)
  , ("feat/mem_read", nat f.memRead)
  , ("feat/mem_write", nat f.memWrite)
  , ("feat/mem_view_read", nat f.memViewRead)
  , ("feat/mem_view_write", nat f.memViewWrite)
  , ("feat/elem_ops", nat f.elemOps)
  , ("feat/move_elems", nat f.moveElems)
  , ("feat/reduce_elems", nat f.reduceElems)
  , ("feat/matmul_muladds", nat f.matmulMulAdds)
  , ("feat/matmul_view_muladds", nat f.matmulViewMulAdds)
  ]

def dumpKernelSamplesWithCost (c : Interpreter.Compiled) (env : Env) (runs : Nat) (path : String) : IO Unit := do
  let runs' := if runs == 0 then 1 else runs
  let nodeMap : HashMap UOpId UOp :=
    c.nodes.foldl (init := (∅ : HashMap UOpId UOp)) fun m u => m.insert u.uid u
  let h ← IO.FS.Handle.mk path .append
  for _ in [:runs'] do
    let (_, times) ← Interpreter.evalCompiledRawTimed c env
    for t in times do
      let u := nodeMap.getD t.uid default
      let impl := c.implMap.getD t.uid (.node (u.src.map (fun s => s.uid)))
      let feats := Backend.Fusion.implCostFeatures u impl
      let baseFields := [
        ("tag", str t.tag),
        ("ms", float t.ms),
        ("numel", nat t.numel),
        ("shape", shapeJson t.shape),
        ("dtype", str (toString (repr t.dtype)))
      ]
      h.putStrLn (render (obj (baseFields ++ featureFields feats)))
  IO.println s!"cost trace appended to {path} (runs={runs'})"

def dumpKernels (c : Interpreter.Compiled) : IO Unit := do
  let mut idx := 0
  for u in c.nodes do
    if u.op == .KERNEL then
      let tag := match c.implMap[u.uid]? with
        | some impl => impl.tag
        | none => "node"
      IO.println s!"  kernel[{idx}] tag={tag} shape={u.shape} numel={u.numel} dtype={repr u.dtype} srcs={u.src.length}"
      idx := idx + 1

def mean := LeanBenchNew.mean
def median := LeanBenchNew.median
def timeTrials := LeanBenchNew.timeTrials

end TinyGrad4Bench
