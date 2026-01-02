import TinyGrad4
-- Disable IO.monoNanosNow linter: benchmark timing uses raw monotonic clocks.
set_option linter.monoNanosNow false

/-!
# AttentionBench

Portable CPU benchmark for the attention "scores" step:

  scores = (q @ kᵀ) * scale + mask

We report three variants:
- baseline: explicit permute + matmul + mul + add
- permute+fuse: explicit permute + fused (matmul*scale + bias)
- view+fuse: no permute, using fused view kernel from the planner
-/

namespace TinyGrad4Bench.AttentionBench

open TinyGrad4
open Interpreter
open Backend

private def startsBytes (batch : Nat) (matNumel : Nat) : Array Nat := Id.run do
  let mut out : Array Nat := Array.emptyWithCapacity batch
  for i in [:batch] do
    out := out.push (i * matNumel * 4)
  return out

private def timeIt (label : String) (iters : Nat) (act : IO Unit) : IO Unit := do
  let start ← IO.monoNanosNow
  for _ in [:iters] do
    act
  let stop ← IO.monoNanosNow
  let dtNs : Nat := stop - start
  let totalMs : Float := (Float.ofNat dtNs) / 1.0e6
  let perMs : Float := totalMs / (Float.ofNat iters)
  IO.println s!"{label}: {perMs} ms/iter ({totalMs} ms total, iters={iters})"

private def benchScores (b t d iters : Nat) : IO Unit := do
  let invSqrtD : Float32 := (1.0 / (Float.sqrt (Float.ofNat d))).toFloat32
  let attScaleBits := invSqrtD.toBits
  let attScaleBytes := Native.fullF32Bits 1 attScaleBits

  let qBytes := Native.fullF32Bits (b * t * d) ((0.01 : Float).toFloat32.toBits)
  let kBytes := Native.fullF32Bits (b * t * d) ((0.02 : Float).toFloat32.toBits)
  let maskBytes := Native.fullF32Bits (t * t) ((0.0 : Float).toFloat32.toBits)

  let qStarts := startsBytes b (t * d)
  let kStarts := startsBytes b (d * t)
  let maskStarts : Array Nat := Array.replicate b 0

  let scoresMaskedU := runTensorM do
    let q ← Tensor.buffer [b, t, d] .float32
    let k ← Tensor.buffer [b, t, d] .float32
    let mask ← Tensor.buffer [t, t] .float32

    let kT ← StaticTensor.permute k [0, 2, 1]
    let scores ← UOp.contract2D q.uop kT.uop
    let scale ← UOp.const .float32 invSqrtD
    let scoresScaled ← UOp.mul scores scale
    let scoresMasked ← UOp.add scoresScaled mask.uop
    pure scoresMasked

  let compiled := Interpreter.compileMany [scoresMaskedU]
  let plan ←
    match compiled.implMap[scoresMaskedU.uid]? with
    | some (.fusedMatmul plan) => pure plan
    | _ => throw (IO.userError "bench: expected fusedMatmul plan")

  let rep := Backend.Fusion.report [scoresMaskedU]
  let chosen :=
    rep.chosen.toList.filter (fun c => c.root == scoresMaskedU.uid)
  match chosen with
  | [c0] =>
    IO.println s!"predicted: baseTime={c0.baseTime} implTime={c0.implTime} gainTime={c0.gainTime}"
    IO.println s!"  launches base={c0.baseLaunches} impl={c0.implLaunches}"
    IO.println s!"  bytes base r={c0.baseReadBytes} w={c0.baseWriteBytes}  impl r={c0.implReadBytes} w={c0.implWriteBytes}"
  | _ =>
    IO.println "predicted: (no single chosen candidate entry found)"

  match plan.scaleBits with
  | some bits =>
    if bits != attScaleBits then
      throw (IO.userError s!"bench: fusedMatmul scaleBits mismatch got {bits}, expected {attScaleBits}")
  | none =>
    throw (IO.userError "bench: expected fusedMatmul scaleBits")

  let sink ← IO.mkRef (0 : UInt32)
  let mix (bytes : ByteArray) : IO Unit := do
    if bytes.size == 0 then
      sink.modify (· + 1)
    else
      let b0 := bytes.get! 0
      let b1 := bytes.get! (bytes.size - 1)
      let x : UInt32 := UInt32.ofNat (b0.toNat) + (UInt32.ofNat (b1.toNat) <<< 8)
      sink.modify (fun acc => acc + x + 1)

  let baseline : IO Unit := do
    let kTBytes := Native.permuteF32 kBytes #[b, t, d] #[0, 2, 1]
    let scoresBytes := Native.matmulBatchedF32 qBytes kTBytes qStarts kStarts t d t
    let scoresScaled := Native.mulBcastF32 scoresBytes attScaleBytes #[b, t, t] #[] #[b, t, t]
    let out := Native.addBcastF32 scoresScaled maskBytes #[b, t, t] #[t, t] #[b, t, t]
    mix out

  let permuteFuse : IO Unit := do
    let kTBytes := Native.permuteF32 kBytes #[b, t, d] #[0, 2, 1]
    let out :=
      Native.matmulBatchedBiasScaleF32
        qBytes kTBytes maskBytes #[t, t]
        qStarts kStarts maskStarts
        t d t
        attScaleBits
    mix out

  let viewFuse : IO Unit := do
    let out :=
      Native.matmulViewBiasScaleF32
        qBytes kBytes maskBytes
        plan.aStrides plan.aOffset plan.aMaskStarts plan.aMaskEnds
        plan.bStrides plan.bOffset plan.bMaskStarts plan.bMaskEnds
        plan.biasStrides plan.biasOffset plan.biasMaskStarts plan.biasMaskEnds
        scoresMaskedU.shape.toArray
        plan.k
        attScaleBits
    mix out

  IO.println s!"=== AttentionBench scores: b={b} t={t} d={d} ==="
  timeIt "baseline (permute + matmul + mul + add)" iters baseline
  timeIt "permute+fuse (permute + fused matmul*scale + bias)" iters permuteFuse
  timeIt "view+fuse (no permute, fused view kernel)" iters viewFuse
  IO.println s!"sink: {← sink.get}"

def run : IO Unit := do
  -- Keep sizes modest for a fast, portable run.
  benchScores (b := 8) (t := 128) (d := 64) (iters := 2000)

end TinyGrad4Bench.AttentionBench

#eval! TinyGrad4Bench.AttentionBench.run
