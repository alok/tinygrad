import TinyGrad4.Backend.FusedGather
import TinyGrad4.Shape

/-!
# Gather Kernel Codegen

Shared codegen for gather kernels across GPU backends.
The backend-specific wrappers provide only the signature and launch details.
-/

namespace TinyGrad4.Backend.GatherKernel

open TinyGrad4
open TinyGrad4.Backend

inductive Target where
  | metal
  | cuda
  deriving Repr, DecidableEq

private def renderInt64 (v : Int64) : String :=
  s!"{v}l"

private def renderNat (v : Nat) : String :=
  s!"{v}u"

private def renderU32Type (target : Target) : String :=
  match target with
  | .metal => "uint"
  | .cuda => "unsigned int"

private def renderZeroStore (elemSize : Nat) (indent : String) : String := Id.run do
  let mut out := ""
  if elemSize == 1 then
    out := out ++ s!"{indent}out[gid] = 0;\n"
  else
    for i in [:elemSize] do
      out := out ++ s!"{indent}out[out_byte + {renderNat i}] = 0;\n"
  out

private def renderCopyStore (elemSize : Nat) (indent u32Type : String) : String := Id.run do
  let mut out := ""
  if elemSize == 1 then
    out := out ++ s!"{indent}out[gid] = x[({u32Type})x_off];\n"
  else
    out := out ++ s!"{indent}{u32Type} x_byte = ({u32Type})x_off * {renderNat elemSize};\n"
    for i in [:elemSize] do
      out := out ++ s!"{indent}out[out_byte + {renderNat i}] = x[x_byte + {renderNat i}];\n"
  out

private def renderIdxType (target : Target) (idxElemSize : Nat) : String :=
  match target, idxElemSize with
  | .metal, 1 => "char"
  | .metal, 2 => "short"
  | .metal, 4 => "int"
  | .metal, 8 => "long"
  | .cuda, 1 => "char"
  | .cuda, 2 => "short"
  | .cuda, 4 => "int"
  | .cuda, 8 => "long long"
  | _, _ => "int"

private def renderHeader (target : Target) (name idxType : String) : String :=
  let u32Type := renderU32Type target
  match target with
  | .metal =>
      s!"#include <metal_stdlib>\nusing namespace metal;\n\nkernel void {name}(\n" ++
      "  device const uchar* x [[buffer(0)]],\n" ++
      s!"  device const {idxType}* idx [[buffer(1)]],\n" ++
      "  device uchar* out [[buffer(2)]],\n" ++
      s!"  {u32Type} gid [[thread_position_in_grid]]\n" ++
      ") {\n"
  | .cuda =>
      s!"extern \"C\" __global__ void {name}(\n" ++
      "  const unsigned char* x,\n" ++
      s!"  const {idxType}* idx,\n" ++
      "  unsigned char* out\n" ++
      ") {\n" ++
      s!"  {u32Type} gid = ({u32Type})(blockIdx.x * blockDim.x + threadIdx.x);\n"

private def renderFooter : String :=
  "}\n"

def renderGatherKernel (target : Target) (name : String) (plan : FusedGather.Plan)
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
  let idxType := renderIdxType target idxElemSize
  let u32Type := renderU32Type target

  let mut lines := ""
  lines := lines ++ s!"  if (gid >= {renderNat numel}) return;\n"
  lines := lines ++ s!"  {u32Type} out_byte = gid * {renderNat elemSize};\n"
  lines := lines ++ s!"  {u32Type} tmp = gid;\n"

  for i in [:outRank] do
    let stride := outStrides.getD i 0
    if stride == 0 then
      lines := lines ++ s!"  {u32Type} o{i} = 0u;\n"
    else
      lines := lines ++ s!"  {u32Type} o{i} = tmp / {renderNat stride};\n"
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
    lines := lines ++ s!"  {u32Type} idx{j} = {coord};\n"
    lines := lines ++ s!"  if (idx{j} < {renderNat start} || idx{j} >= {renderNat stop}) valid = false;\n"
    lines := lines ++ s!"  idx_off += {renderInt64 stride} * (long)idx{j};\n"

  let zeroBlock := renderZeroStore elemSize "    "
  lines := lines ++ s!"  if (!valid || idx_off < 0 || idx_off >= {renderInt64 idxNumel64}) " ++ "{\n" ++
    zeroBlock ++ "    return;\n  }\n"
  lines := lines ++ s!"  long idx_val = (long)idx[({u32Type})idx_off];\n"
  lines := lines ++ s!"  if (idx_val < 0 || idx_val >= {renderInt64 (Int64.ofNat classDim)}) " ++ "{\n" ++
    zeroBlock ++ "    return;\n  }\n"

  lines := lines ++ s!"  long x_off = {renderInt64 plan.xView.offset};\n"
  for j in [:maskRank] do
    let coord :=
      if j < axis then
        s!"o{j}"
      else if j == axis then
        s!"({u32Type})idx_val"
      else
        s!"o{j - 1}"
    let start := plan.xView.maskStart.getD j 0
    let stop := plan.xView.maskEnd.getD j 0
    let stride := plan.xView.strides.getD j 0
    lines := lines ++ s!"  {u32Type} x{j} = {coord};\n"
    lines := lines ++ s!"  if (x{j} < {renderNat start} || x{j} >= {renderNat stop}) valid = false;\n"
    lines := lines ++ s!"  x_off += {renderInt64 stride} * (long)x{j};\n"

  lines := lines ++ s!"  if (!valid || x_off < 0 || x_off >= {renderInt64 xNumel64}) " ++ "{\n" ++
    zeroBlock ++ "    return;\n  }\n"
  lines := lines ++ renderCopyStore elemSize "  " u32Type

  renderHeader target name idxType ++ lines ++ renderFooter

end TinyGrad4.Backend.GatherKernel
