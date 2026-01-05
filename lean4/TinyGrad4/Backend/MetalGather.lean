import TinyGrad4.Backend.Metal
import TinyGrad4.Backend.Buffer
import TinyGrad4.Backend.FusedGather
import TinyGrad4.Backend.Native
import TinyGrad4.Shape

/-!
# GPU Gather Execution via Metal

Executes fused gather kernels on Metal GPU, falling back to CPU when unavailable.
This is a temporary codegen path until ProgramSpec v2 lands.
-/

namespace TinyGrad4.Backend.MetalGather

open TinyGrad4
open TinyGrad4.Backend
open TinyGrad4.Backend.Metal

/-! ## Local Helpers -/

private def zerosRaw (dtype : DType) (numel : Nat) : RawBuffer :=
  { dtype, data := ByteArray.mk (Array.replicate (numel * dtype.itemsize) 0) }

private def renderInt64 (v : Int64) : String :=
  s!"{v}l"

private def renderNat (v : Nat) : String :=
  s!"{v}u"

private def renderZeroStore (elemSize : Nat) (indent : String) : String := Id.run do
  let mut out := ""
  if elemSize == 1 then
    out := out ++ s!"{indent}out[gid] = 0;\n"
  else
    for i in [:elemSize] do
      out := out ++ s!"{indent}out[out_byte + {renderNat i}] = 0;\n"
  out

private def renderCopyStore (elemSize : Nat) (indent : String) : String := Id.run do
  let mut out := ""
  if elemSize == 1 then
    out := out ++ s!"{indent}out[gid] = x[(uint)x_off];\n"
  else
    out := out ++ s!"{indent}uint x_byte = (uint)x_off * {renderNat elemSize};\n"
    for i in [:elemSize] do
      out := out ++ s!"{indent}out[out_byte + {renderNat i}] = x[x_byte + {renderNat i}];\n"
  out

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

/-! ## Shader Rendering -/

private def renderGatherKernelFromPlan (name : String) (plan : FusedGather.Plan)
    (outShape : Shape) (xNumel idxNumel : Nat) (elemSize idxElemSize : Nat) : String := Id.run do
  let outRank := outShape.length
  let maskRank := plan.maskShape.size
  let numel := listProd outShape
  let axis := plan.reduceAxis
  let classDim := plan.maskShape.getD axis 0
  let idxNumel64 := Int64.ofNat idxNumel
  let xNumel64 := Int64.ofNat xNumel
  let outStrides :=
    (Shape.unitStrides outShape).map (fun i => Int.toNat i) |>.toArray
  let idxType :=
    match idxElemSize with
    | 1 => "char"
    | 2 => "short"
    | 4 => "int"
    | 8 => "long"
    | _ => "int"

  let mut lines := ""
  lines := lines ++ s!"  if (gid >= {renderNat numel}) return;\n"
  lines := lines ++ s!"  uint out_byte = gid * {renderNat elemSize};\n"
  lines := lines ++ "  uint tmp = gid;\n"

  for i in [:outRank] do
    let stride := outStrides.getD i 0
    if stride == 0 then
      lines := lines ++ s!"  uint o{i} = 0u;\n"
    else
      lines := lines ++ s!"  uint o{i} = tmp / {renderNat stride};\n"
      lines := lines ++ s!"  tmp = tmp % {renderNat stride};\n"

  lines := lines ++ "  bool valid = true;\n"
  lines := lines ++ s!"  long idx_off = {renderInt64 plan.idxView.offset};\n"

  for j in [:maskRank] do
    let coord :=
      if j < axis then
        s!"o{j}"
      else if j == axis then
        "0u"
      else
        s!"o{j - 1}"
    let start := plan.idxView.maskStart.getD j 0
    let stop := plan.idxView.maskEnd.getD j 0
    let stride := plan.idxView.strides.getD j 0
    lines := lines ++ s!"  uint idx{j} = {coord};\n"
    lines := lines ++ s!"  if (idx{j} < {renderNat start} || idx{j} >= {renderNat stop}) valid = false;\n"
    lines := lines ++ s!"  idx_off += {renderInt64 stride} * (long)idx{j};\n"

  let zeroBlock := renderZeroStore elemSize "    "
  lines := lines ++ s!"  if (!valid || idx_off < 0 || idx_off >= {renderInt64 idxNumel64}) " ++ "{\n" ++ zeroBlock ++ "    return;\n  }\n"
  lines := lines ++ s!"  long idx_val = (long)idx[(uint)idx_off];\n"
  lines := lines ++ s!"  if (idx_val < 0 || idx_val >= (long){renderNat classDim}) " ++ "{\n" ++ zeroBlock ++ "    return;\n  }\n"

  lines := lines ++ s!"  long x_off = {renderInt64 plan.xView.offset};\n"
  for j in [:maskRank] do
    let coord :=
      if j < axis then
        s!"o{j}"
      else if j == axis then
        "(uint)idx_val"
      else
        s!"o{j - 1}"
    let start := plan.xView.maskStart.getD j 0
    let stop := plan.xView.maskEnd.getD j 0
    let stride := plan.xView.strides.getD j 0
    lines := lines ++ s!"  uint x{j} = {coord};\n"
    lines := lines ++ s!"  if (x{j} < {renderNat start} || x{j} >= {renderNat stop}) valid = false;\n"
    lines := lines ++ s!"  x_off += {renderInt64 stride} * (long)x{j};\n"

  lines := lines ++ s!"  if (!valid || x_off < 0 || x_off >= {renderInt64 xNumel64}) " ++ "{\n" ++ zeroBlock ++ "    return;\n  }\n"
  lines := lines ++ renderCopyStore elemSize "  "

  s!"#include <metal_stdlib>\nusing namespace metal;\n\nkernel void {name}(\n" ++
    s!"  device const uchar* x [[buffer(0)]],\n" ++
    s!"  device const {idxType}* idx [[buffer(1)]],\n" ++
    "  device uchar* out [[buffer(2)]],\n" ++
    "  uint gid [[thread_position_in_grid]]\n" ++
    ") " ++ "{\n" ++ lines ++ "}"

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

  let xBuf ← metalAllocBytes xBytes
  metalCopyInBytes xBuf x.data
  let idxBuf ← metalAllocBytes idxBytes
  metalCopyInBytes idxBuf idx.data

  let outBuf ← metalAllocBytes outBytes

  let prog ← getOrCompile name shader
  let threadsPerGroup : Nat := 256
  let totalThreads := numel
  metalLaunch prog #[xBuf, idxBuf, outBuf] totalThreads 1 1 threadsPerGroup 1 1
  metalSync

  let outBytes' ← metalCopyOutBytes outBuf outBytes

  metalFree xBuf
  metalFree idxBuf
  metalFree outBuf

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

  let shader := renderGatherKernelFromPlan name plan outShape xNumel idxNumel elemSize idxElemSize
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
  let available ← Metal.isAvailable
  if available then
    try
      runFusedGather plan x idx outShape dtype
    catch _ =>
      return runGatherCPU plan x idx outShape dtype
  else
    return runGatherCPU plan x idx outShape dtype

end TinyGrad4.Backend.MetalGather
