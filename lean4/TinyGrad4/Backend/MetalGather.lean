import TinyGrad4.Backend.Metal
import TinyGrad4.Backend.Buffer
import TinyGrad4.Backend.FusedGather
import TinyGrad4.Backend.GatherKernel
import TinyGrad4.Backend.Native
import TinyGrad4.Shape
import TinyGrad4.Benchmark.Instrumentation
import TinyGrad4.Timing

/-!
# GPU Gather Execution via Metal

Executes fused gather kernels on Metal GPU, falling back to CPU when unavailable.
This is a temporary codegen path until ProgramSpec v2 lands.
-/

namespace TinyGrad4.Backend.MetalGather

open TinyGrad4
open TinyGrad4.Backend
open TinyGrad4.Backend.Metal
open TinyGrad4.Backend.GatherKernel
open TinyGrad4.Benchmark (withProfile)
open TinyGrad4 (MonadTimeNS)

/-! ## Local Helpers -/

private def zerosRaw (dtype : DType) (numel : Nat) : RawBuffer :=
  { dtype, data := ByteArray.mk (Array.replicate (numel * dtype.itemsize) 0) }

/-! ## Program Cache -/

initialize programCache : IO.Ref (Std.HashMap String MetalProgram) ← IO.mkRef ∅

def getOrCompile (name : String) (shader : String) : IO MetalProgram := do
  let cache ← programCache.get
  match cache[name]? with
  | some prog => return prog
  | none =>
    let prog ← metalCompile name shader
    programCache.modify (·.insert name prog)
    return prog

/-! ## Timing Stats -/

structure GatherKernelStats where
  launches : Nat := 0
  kernelNs : Nat := 0
  totalNs : Nat := 0
  minKernelNs : Nat := 0
  maxKernelNs : Nat := 0
  minTotalNs : Nat := 0
  maxTotalNs : Nat := 0
  deriving Repr

instance : Inhabited GatherKernelStats := ⟨{}⟩

initialize gatherKernelStatsRef : IO.Ref GatherKernelStats ← IO.mkRef {}

def clearGatherKernelStats : IO Unit :=
  gatherKernelStatsRef.set {}

def getGatherKernelStats : IO GatherKernelStats :=
  gatherKernelStatsRef.get

private def updateGatherKernelStats (kernelNs totalNs : Nat) : IO Unit := do
  gatherKernelStatsRef.modify fun s =>
    let launches := s.launches + 1
    let minKernel := if s.launches == 0 then kernelNs else Nat.min s.minKernelNs kernelNs
    let maxKernel := if s.launches == 0 then kernelNs else Nat.max s.maxKernelNs kernelNs
    let minTotal := if s.launches == 0 then totalNs else Nat.min s.minTotalNs totalNs
    let maxTotal := if s.launches == 0 then totalNs else Nat.max s.maxTotalNs totalNs
    { s with
      launches
      kernelNs := s.kernelNs + kernelNs
      totalNs := s.totalNs + totalNs
      minKernelNs := minKernel
      maxKernelNs := maxKernel
      minTotalNs := minTotal
      maxTotalNs := maxTotal }

/-! ## Kernel Dispatch -/

def runGatherKernel (name : String) (shader : String) (x idx : RawBuffer)
    (outShape : Shape) (dtype : DType) : IO RawBuffer := do
  let numel := listProd outShape
  if numel == 0 then
    return zerosRaw dtype 0

  let totalStart ← MonadTimeNS.monoNs

  let elemSize := dtype.itemsize
  let outBytes := numel * elemSize
  let xBytes := x.data.size
  let idxBytes := idx.data.size

  let xBuf ← metalAllocBytes xBytes
  metalCopyInBytes xBuf x.data
  let idxBuf ← metalAllocBytes idxBytes
  metalCopyInBytes idxBuf idx.data

  let outBuf ← metalAllocBytes outBytes

  let prog ← withProfile "METAL" "gather_compile" (getOrCompile name shader)
  let threadsPerGroup : Nat := 256
  let totalThreads := numel
  let kernelStart ← MonadTimeNS.monoNs
  withProfile "METAL" "gather_launch" do
    metalLaunch prog #[xBuf, idxBuf, outBuf] totalThreads 1 1 threadsPerGroup 1 1
    metalSync
  let kernelStop ← MonadTimeNS.monoNs

  let outBytes' ← metalCopyOutBytes outBuf outBytes
  let totalStop ← MonadTimeNS.monoNs

  metalFree xBuf
  metalFree idxBuf
  metalFree outBuf

  updateGatherKernelStats (kernelStop - kernelStart) (totalStop - totalStart)

  return { dtype := dtype, data := outBytes' }

/-! ## Fused Gather Entry Points -/

def runFusedGather (plan : FusedGather.Plan) (x idx : RawBuffer)
    (outShape : Shape) (dtype : DType) : IO RawBuffer := do
  let elemSize := dtype.itemsize
  let idxElemSize := plan.idxItemsize
  let xNumel := if elemSize == 0 then 0 else x.data.size / elemSize
  let idxNumel := if idxElemSize == 0 then 0 else idx.data.size / idxElemSize

  let progHash := hash
    (plan.xView.strides, plan.xView.offset, plan.xView.maskStart, plan.xView.maskEnd,
     plan.idxView.strides, plan.idxView.offset, plan.idxView.maskStart, plan.idxView.maskEnd,
     outShape.toArray, plan.maskShape, plan.reduceAxis, elemSize, idxElemSize, plan.idxSigned, xNumel, idxNumel)
  let name := s!"fused_gather_{progHash}"

  let shader := renderGatherKernel .metal name plan outShape xNumel idxNumel elemSize idxElemSize plan.idxSigned
  runGatherKernel name shader x idx outShape dtype

private def runGatherCPU (plan : FusedGather.Plan) (x idx : RawBuffer)
    (outShape : Shape) (dtype : DType) : RawBuffer :=
  let classDim := plan.maskShape.getD plan.reduceAxis 0
  let outBytes :=
    Native.gatherView x.data idx.data outShape.toArray
      plan.xView.strides plan.xView.offset plan.xView.maskStart plan.xView.maskEnd
      plan.idxView.strides plan.idxView.offset plan.idxView.maskStart plan.idxView.maskEnd
      plan.reduceAxis classDim dtype.itemsize plan.idxItemsize (if plan.idxSigned then 1 else 0)
  { dtype := dtype, data := outBytes }

def runFusedGatherWithFallback (plan : FusedGather.Plan) (x idx : RawBuffer)
    (outShape : Shape) (dtype : DType) : IO RawBuffer := do
  let numel := listProd outShape
  if numel > 4294967295 then
    return runGatherCPU plan x idx outShape dtype
  let available ← Metal.isAvailable
  if available then
    try
      runFusedGather plan x idx outShape dtype
    catch _ =>
      return runGatherCPU plan x idx outShape dtype
  else
    return runGatherCPU plan x idx outShape dtype

end TinyGrad4.Backend.MetalGather
