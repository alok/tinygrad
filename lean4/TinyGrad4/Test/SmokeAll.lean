import TinyGrad4

import TinyGrad4.Test.BroadcastOpSmoke
import TinyGrad4.Test.AttentionSmoke
import TinyGrad4.Test.CompiledTrainingSmoke
import TinyGrad4.Test.FusedEwiseExprSmoke
import TinyGrad4.Test.FusedMatmulExprSmoke
import TinyGrad4.Test.FusedReduceExprSmoke
import TinyGrad4.Test.FusedSoftmaxExprSmoke
import TinyGrad4.Test.GradientCheck
import TinyGrad4.Test.CostExprSmoke
import TinyGrad4.Test.CostExprMetaSmoke
import TinyGrad4.Test.DebugSyntaxSmoke
import TinyGrad4.Test.EndToEndLoopSmoke
import TinyGrad4.Test.TagSyntaxSmoke
import TinyGrad4.Test.KernelExprSmoke
import TinyGrad4.Test.KernelizeSmoke
import TinyGrad4.Test.ScheduleCacheSmoke
import TinyGrad4.Test.ContractViewFusionSmoke
import TinyGrad4.Test.DataLoaderSmoke
import TinyGrad4.Test.DataArrayNSmallDtypeSmoke
import TinyGrad4.Test.MathOpsSmoke
import TinyGrad4.Test.MatmulViewFusionSmoke
import TinyGrad4.Test.NoopMovementOptSmoke
import TinyGrad4.Test.NotationSmoke
import TinyGrad4.Test.OptimInterfaceSmoke
import TinyGrad4.Test.ReduceAxisFinSmoke
import TinyGrad4.Test.ValidateSmoke
import TinyGrad4.Test.ViewFusionSmoke
import TinyGrad4.Test.ViewStackFusionSmoke
import TinyGrad4.Test.WeightInitSmoke
import TinyGrad4.Test.SpecBroadcastIndexingSmoke
import TinyGrad4.Test.ShapeTrackerSmoke
import TinyGrad4.Test.Conv2dSmoke

/-!
# SmokeAll

Single entry point to run the fast Lean TinyGrad4 smoke suite.

Run with:
`cd lean4 && lake build TinyGrad4.Test.SmokeAll`
-/

namespace TinyGrad4.Test.SmokeAll

def runAll : IO Unit := do
  IO.println "=== TinyGrad4 SmokeAll ==="
  TinyGrad4.Test.ValidateSmoke.runAll
  TinyGrad4.Test.NotationSmoke.runAll
  TinyGrad4.Test.OptimInterfaceSmoke.runAll
  TinyGrad4.Test.GradientCheck.runAll
  TinyGrad4.Test.CostExprSmoke.runAll
  TinyGrad4.Test.CostExprMetaSmoke.runAll
  TinyGrad4.Test.DebugSyntaxSmoke.runAll
  TinyGrad4.Test.EndToEndLoopSmoke.runAll
  TinyGrad4.Test.TagSyntaxSmoke.runAll
  TinyGrad4.Test.KernelExprSmoke.runAll
  TinyGrad4.Test.NoopMovementOptSmoke.runAll
  TinyGrad4.Test.ReduceAxisFinSmoke.runAll
  TinyGrad4.Test.KernelizeSmoke.runAll
  TinyGrad4.Test.ScheduleCacheSmoke.runAll
  TinyGrad4.Test.FusedEwiseExprSmoke.runAll
  TinyGrad4.Test.FusedReduceExprSmoke.runAll
  TinyGrad4.Test.FusedSoftmaxExprSmoke.runAll
  TinyGrad4.Test.FusedMatmulExprSmoke.runAll
  TinyGrad4.Test.BroadcastOpSmoke.runAll
  TinyGrad4.Test.AttentionSmoke.runAll
  TinyGrad4.Test.ContractViewFusionSmoke.runAll
  TinyGrad4.Test.DataLoaderSmoke.runAll
  TinyGrad4.Test.DataArrayNSmallDtypeSmoke.runAll
  TinyGrad4.Test.MathOpsSmoke.runAll
  TinyGrad4.Test.MatmulViewFusionSmoke.runAll
  TinyGrad4.Test.WeightInitSmoke.runAll
  TinyGrad4.Test.ViewFusionSmoke.runAll
  TinyGrad4.Test.ViewStackFusionSmoke.runAll
  TinyGrad4.Test.CompiledTrainingSmoke.runAll
  TinyGrad4.Test.SpecBroadcastIndexingSmoke.runAll
  TinyGrad4.Test.ShapeTrackerSmoke.runAll
  TinyGrad4.Test.Conv2dSmoke.runAll
  IO.println "=== SmokeAll OK ==="

end TinyGrad4.Test.SmokeAll

#eval! TinyGrad4.Test.SmokeAll.runAll
