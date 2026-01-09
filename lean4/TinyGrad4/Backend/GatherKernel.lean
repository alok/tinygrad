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

private def renderU64Type (target : Target) : String :=
  match target with
  | .metal => "ulong"
  | .cuda => "unsigned long long"

private def renderZeroStore (elemSize : Nat) (indent : String) : String := Id.run do
  let mut out := ""
  if elemSize == 1 then
    out := out ++ s!"{indent}out[gid] = 0;\n"
  else
    for i in [:elemSize] do
      out := out ++ s!"{indent}out[out_byte + {renderNat i}] = 0;\n"
  out

private def renderCopyStore (elemSize : Nat) (indent u64Type : String) : String := Id.run do
  let mut out := ""
  if elemSize == 1 then
    out := out ++ s!"{indent}out[gid] = x[({u64Type})x_off];\n"
  else
    out := out ++ s!"{indent}{u64Type} x_byte = ({u64Type})x_off * {renderNat elemSize};\n"
    for i in [:elemSize] do
      out := out ++ s!"{indent}out[out_byte + {renderNat i}] = x[x_byte + {renderNat i}];\n"
  out

private def renderIdxType (target : Target) (idxElemSize : Nat) (idxSigned : Bool) : String :=
  match target, idxElemSize, idxSigned with
  | .metal, 1, true => "char"
  | .metal, 1, false => "uchar"
  | .metal, 2, true => "short"
  | .metal, 2, false => "ushort"
  | .metal, 4, true => "int"
  | .metal, 4, false => "uint"
  | .metal, 8, true => "long"
  | .metal, 8, false => "ulong"
  | .cuda, 1, true => "char"
  | .cuda, 1, false => "unsigned char"
  | .cuda, 2, true => "short"
  | .cuda, 2, false => "unsigned short"
  | .cuda, 4, true => "int"
  | .cuda, 4, false => "unsigned int"
  | .cuda, 8, true => "long long"
  | .cuda, 8, false => "unsigned long long"
  | _, _, _ => if idxSigned then "int" else "unsigned int"

private def renderHeader (target : Target) (name idxType : String) : String :=
  match target with
  | .metal =>
      s!"#include <metal_stdlib>\nusing namespace metal;\n\nkernel void {name}(\n" ++
      "  device const uchar* x [[buffer(0)]],\n" ++
      s!"  device const {idxType}* idx [[buffer(1)]],\n" ++
      "  device uchar* out [[buffer(2)]],\n" ++
      "  uint gid_in [[thread_position_in_grid]]\n" ++
      ") {\n"
  | .cuda =>
      s!"extern \"C\" __global__ void {name}(\n" ++
      "  const unsigned char* x,\n" ++
      s!"  const {idxType}* idx,\n" ++
      "  unsigned char* out\n" ++
      ") {\n"

private def renderFooter : String :=
  "}\n"

def renderGatherKernel (target : Target) (name : String) (plan : FusedGather.Plan)
    (outShape : Shape) (xNumel idxNumel : Nat) (elemSize idxElemSize : Nat) (idxSigned : Bool) : String := Id.run do
  let outRank := outShape.length
  let maskRank := plan.maskShape.size
  let numel := listProd outShape
  let axis := plan.reduceAxis
  let classDim := plan.maskShape.getD axis 0
  let idxNumel64 := Int64.ofNat idxNumel
  let xNumel64 := Int64.ofNat xNumel
  let outStrides :=
    (Shape.unitStrides outShape).map (fun i => Int.toNat i) |>.toArray
  let idxType := renderIdxType target idxElemSize idxSigned
  let u64Type := renderU64Type target
  let numelConst := s!"({u64Type}){renderNat numel}"
  let bodyIndent := match target with | .cuda => "    " | .metal => "  "

  let mut body := ""
  body := body ++ s!"{bodyIndent}{u64Type} out_byte = gid * {renderNat elemSize};\n"
  body := body ++ s!"{bodyIndent}{u64Type} tmp = gid;\n"

  for i in [:outRank] do
    let stride := outStrides.getD i 0
    if stride == 0 then
      body := body ++ s!"{bodyIndent}{u64Type} o{i} = 0u;\n"
    else
      body := body ++ s!"{bodyIndent}{u64Type} o{i} = tmp / {renderNat stride};\n"
      body := body ++ s!"{bodyIndent}tmp = tmp % {renderNat stride};\n"

  body := body ++ s!"{bodyIndent}bool valid = true;\n"
  body := body ++ s!"{bodyIndent}long idx_off = {renderInt64 plan.idxView.offset};\n"

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
    body := body ++ s!"{bodyIndent}{u64Type} idx{j} = {coord};\n"
    body := body ++
      s!"{bodyIndent}if (idx{j} < ({u64Type}){renderNat start} || idx{j} >= ({u64Type}){renderNat stop}) " ++
      "valid = false;\n"
    body := body ++ s!"{bodyIndent}idx_off += {renderInt64 stride} * (long)idx{j};\n"

  let blockIndent := bodyIndent ++ "  "
  let zeroBlock := renderZeroStore elemSize blockIndent
  body := body ++ s!"{bodyIndent}if (!valid || idx_off < 0 || idx_off >= {renderInt64 idxNumel64}) " ++ "{\n" ++
    zeroBlock ++ s!"{blockIndent}return;\n{bodyIndent}}\n"
  if idxSigned then
    body := body ++ s!"{bodyIndent}long idx_val = (long)idx[({u64Type})idx_off];\n"
    body := body ++ s!"{bodyIndent}if (idx_val < 0 || idx_val >= {renderInt64 (Int64.ofNat classDim)}) " ++ "{\n" ++
      zeroBlock ++ s!"{blockIndent}return;\n{bodyIndent}}\n"
  else
    body := body ++ s!"{bodyIndent}{u64Type} idx_val_u = ({u64Type})idx[({u64Type})idx_off];\n"
    body := body ++ s!"{bodyIndent}if (idx_val_u >= ({u64Type}){renderNat classDim}) " ++ "{\n" ++
      zeroBlock ++ s!"{blockIndent}return;\n{bodyIndent}}\n"
    body := body ++ s!"{bodyIndent}long idx_val = (long)idx_val_u;\n"

  body := body ++ s!"{bodyIndent}long x_off = {renderInt64 plan.xView.offset};\n"
  for j in [:maskRank] do
    let coord :=
      if j < axis then
        s!"o{j}"
      else if j == axis then
        s!"({u64Type})idx_val"
      else
        s!"o{j - 1}"
    let start := plan.xView.maskStart.getD j 0
    let stop := plan.xView.maskEnd.getD j 0
    let stride := plan.xView.strides.getD j 0
    body := body ++ s!"{bodyIndent}{u64Type} x{j} = {coord};\n"
    body := body ++
      s!"{bodyIndent}if (x{j} < ({u64Type}){renderNat start} || x{j} >= ({u64Type}){renderNat stop}) " ++
      "valid = false;\n"
    body := body ++ s!"{bodyIndent}x_off += {renderInt64 stride} * (long)x{j};\n"

  body := body ++ s!"{bodyIndent}if (!valid || x_off < 0 || x_off >= {renderInt64 xNumel64}) " ++ "{\n" ++
    zeroBlock ++ s!"{blockIndent}return;\n{bodyIndent}}\n"
  body := body ++ renderCopyStore elemSize bodyIndent u64Type

  let mut lines := ""
  match target with
  | .cuda =>
      lines := lines ++ s!"  {u64Type} stride = ({u64Type})blockDim.x * ({u64Type})gridDim.x;\n"
      lines := lines ++ s!"  for ({u64Type} gid = ({u64Type})blockIdx.x * ({u64Type})blockDim.x + ({u64Type})threadIdx.x; " ++
        s!"gid < {numelConst}; gid += stride) " ++ "{\n"
      lines := lines ++ body
      lines := lines ++ "  }\n"
  | .metal =>
      lines := lines ++ s!"  {u64Type} gid = ({u64Type})gid_in;\n"
      lines := lines ++ s!"  if (gid >= {numelConst}) return;\n"
      lines := lines ++ body

  renderHeader target name idxType ++ lines ++ renderFooter

end TinyGrad4.Backend.GatherKernel
