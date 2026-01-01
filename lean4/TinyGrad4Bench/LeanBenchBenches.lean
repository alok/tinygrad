import TinyGrad4.Data.Buffer
import TinyGrad4.Data.IndexTransform
import LeanBench

/-!
LeanBench (external) benchmarks for TinyGrad4.
These are lightweight CPU benches to exercise the runner and plan/sharding.
-/

open LeanBench
open TinyGrad4.Data

def mkCfg (suiteName : String) (tags : List String := []) : BenchConfig :=
  { suite := some suiteName, tags := tags }

bench_suite "buffer"
bench_suite "data"

bench "buffer/computeCStrides-4d" (mkCfg "buffer" ["buffer", "strides"]) do
  let strides := RawBuffer.computeCStrides #[32, 3, 224, 224]
  if strides.size == 0 then
    IO.println ""

bench "data/shuffle-60k" (mkCfg "data" ["data", "shuffle"]) do
  let shuffle := IndexTransform.shuffle 42 60000
  if shuffle.outputLen == 0 then
    IO.println ""

bench "data/shuffle-lookup-1k" (mkCfg "data" ["data", "shuffle", "lookup"]) do
  let shuffle := IndexTransform.shuffle 42 60000
  let mut acc := 0
  for i in [:1000] do
    if hi : i < shuffle.outputLen then
      let idx := shuffle.map ⟨i, hi⟩
      acc := acc + idx.val
  if acc == 0 then
    IO.println ""
