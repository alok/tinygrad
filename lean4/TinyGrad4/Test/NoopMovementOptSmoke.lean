import TinyGrad4

/-!
# No-op Movement Optimization Smoke Test

Ensures our "keep uid" optimizer rewrites no-op movement operations into cheap identity nodes,
so the interpreter doesn't accidentally allocate/copy for cases like:
- permute with identity permutation
- pad with all-zero padding
- shrink with full bounds
- flip with empty axes
-/

namespace TinyGrad4.Test.NoopMovementOptSmoke

open TinyGrad4
open StaticTensor

private def assertNoOpMovementRewritten : IO Unit := do
  let outU := runTensorM do
    let x ← Tensor.buffer [2, 3] .float32
    let x1 ← StaticTensor.permute x [0, 1]
    let x2 ← StaticTensor.pad x1 [(0, 0), (0, 0)]
    let x3 ← StaticTensor.shrink x2 [(0, 2), (0, 3)]
    let x4 ← StaticTensor.flip x3 []
    pure x4.uop

  let rootsOpt := TinyGrad4.Optim.optimizeKeepUids [outU]
  let nodesOpt := UOp.toposortMany rootsOpt

  let bad := nodesOpt.filter fun u =>
    u.op == .PERMUTE || u.op == .PAD || u.op == .SHRINK || u.op == .FLIP || u.op == .EXPAND
  if !bad.isEmpty then
    let shown := bad.take 10 |>.map (fun u => u.pretty)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"expected no-op movement nodes to be rewritten, found {bad.length}\n{msg}")

  let errs := UOp.validateMany rootsOpt
  if errs.size != 0 then
    let shown := errs.toList.take 10 |>.map (fun e => e.render)
    let msg := String.intercalate "\n" shown
    throw (IO.userError s!"optimized graph failed validation ({errs.size} errors)\n{msg}")

def runAll : IO Unit := do
  IO.println "=== NoopMovementOptSmoke Tests ==="
  assertNoOpMovementRewritten
  IO.println "✓ rewrites no-op movement ops"
  IO.println "=== NoopMovementOptSmoke OK ==="

end TinyGrad4.Test.NoopMovementOptSmoke

#eval! TinyGrad4.Test.NoopMovementOptSmoke.runAll

