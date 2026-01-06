import TinyGrad4.Backend.Cuda
import TinyGrad4.Backend.Buffer
import TinyGrad4.Backend.FusedGather
import TinyGrad4.Backend.GatherKernel
import TinyGrad4.Backend.Native
import TinyGrad4.Shape

/-!
# GPU Gather Execution via CUDA

Executes fused gather kernels on CUDA GPU, falling back to CPU when unavailable.
This is a temporary codegen path until ProgramSpec v2 lands.
-/

namespace TinyGrad4.Backend.CudaGather

open TinyGrad4
open TinyGrad4.Backend
open TinyGrad4.Backend.Cuda
open TinyGrad4.Backend.GatherKernel

/-! ## Local Helpers -/

private def zerosRaw (dtype : DType) (numel : Nat) : RawBuffer :=
  { dtype, data := ByteArray.mk (Array.replicate (numel * dtype.itemsize) 0) }

/-! ## Program Cache -/

initialize programCache : IO.Ref (Std.HashMap String CUDAProgram) ← IO.mkRef ∅

def getOrCompile (name : String) (shader : String) : IO CUDAProgram := do
  let cache ← programCache.get
  match cache[name]? with
  | some prog => return prog
  | none =>
    let prog ← cudaCompile name shader
    programCache.modify (·.insert name prog)
    return prog

/-! ## Kernel Dispatch -/

def runGatherKernel (name : String) (shader : String) (x idx : RawBuffer)
    (outShape : Shape) (dtype : DType) : IO RawBuffer := do
  let numel := listProd outShape
  if numel == 0 then
    return zerosRaw dtype 0

  let elemSize := dtype.itemsize
  let outBytes := numel * elemSize
  let xBytes := x.data.size
  let idxBytes := idx.data.size

  let xBuf ← cudaAllocBytes xBytes
  cudaCopyInBytes xBuf x.data
  let idxBuf ← cudaAllocBytes idxBytes
  cudaCopyInBytes idxBuf idx.data
  let outBuf ← cudaAllocBytes outBytes

  let prog ← getOrCompile name shader
  let threadsPerBlock : Nat := 256
  let totalThreads := numel
  cudaLaunch2D prog #[xBuf, idxBuf, outBuf] totalThreads 1 threadsPerBlock 1
  cudaSync

  let outBytes' ← cudaCopyOutBytes outBuf outBytes

  cudaFree xBuf
  cudaFree idxBuf
  cudaFree outBuf

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
     outShape.toArray, plan.maskShape, plan.reduceAxis, elemSize, idxElemSize, xNumel, idxNumel)
  let name := s!"fused_gather_{progHash}"

  let shader := renderGatherKernel .cuda name plan outShape xNumel idxNumel elemSize idxElemSize
  runGatherKernel name shader x idx outShape dtype

private def runGatherCPU (plan : FusedGather.Plan) (x idx : RawBuffer)
    (outShape : Shape) (dtype : DType) : RawBuffer :=
  let classDim := plan.maskShape.getD plan.reduceAxis 0
  let outBytes :=
    Native.gatherView x.data idx.data outShape.toArray
      plan.xView.strides plan.xView.offset plan.xView.maskStart plan.xView.maskEnd
      plan.idxView.strides plan.idxView.offset plan.idxView.maskStart plan.idxView.maskEnd
      plan.reduceAxis classDim dtype.itemsize plan.idxItemsize
  { dtype := dtype, data := outBytes }

def runFusedGatherWithFallback (plan : FusedGather.Plan) (x idx : RawBuffer)
    (outShape : Shape) (dtype : DType) : IO RawBuffer := do
  let available ← Cuda.isAvailable
  if available then
    try
      runFusedGather plan x idx outShape dtype
    catch _ =>
      return runGatherCPU plan x idx outShape dtype
  else
    return runGatherCPU plan x idx outShape dtype

end TinyGrad4.Backend.CudaGather
