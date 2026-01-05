import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import TinyGrad4.Backend.View
import TinyGrad4.Backend.ShapeTracker
import Std.Data.HashMap

/-!
Fused elementwise kernel compiler.

Compiles elementwise UOp expressions into execution plans and selects a native kernel.
Pattern detection happens in Lean via {lit}`detectKernel`, and {lit}`Interpreter.evalFusedEwise`
dispatches to a native kernel based on {lit}`plan.kernel`.

Bytecode instructions are 64-bit: {lit}`(imm << 8) | opcode`, evaluated with a stack machine.
-/

namespace TinyGrad4.Backend.FusedEwise

open Std

/-- Specialized kernel variants detected at compile time.
    Pattern matching happens here in Lean, not in C. -/
inductive Kernel where
  | bytecode            -- General bytecode interpreter (slow path)
  | negContiguous       -- neg on contiguous f32
  | sqrtContiguous      -- sqrt on contiguous f32
  | recipContiguous     -- 1/x on contiguous f32
  | exp2Contiguous      -- exp2 on contiguous f32
  | log2Contiguous      -- log2 on contiguous f32
  | sinContiguous       -- sin on contiguous f32
  | cosContiguous       -- cos on contiguous f32
  | tanContiguous       -- tan on contiguous f32
  | addContiguous       -- a + b on contiguous f32
  | subContiguous       -- a - b on contiguous f32
  | mulContiguous       -- a * b on contiguous f32
  | divContiguous       -- a / b on contiguous f32
  | maxContiguous       -- max(a, b) on contiguous f32
  | powContiguous       -- pow(a, b) on contiguous f32
  deriving Repr, BEq

structure Plan where
  root : UOpId
  cover : UOpIdSet
  leafBases : Array UOpId
  leafDtypes : Array Nat
  leafShapes : Array (Array Nat)
  leafStrides : Array (Array Int64)
  leafOffsets : Array Int64
  leafMaskStarts : Array (Array Nat)
  leafMaskEnds : Array (Array Nat)
  leafStackShapes : Array (Array (Array Nat))
  leafStackStrides : Array (Array (Array Int64))
  leafStackOffsets : Array (Array Int64)
  leafStackMaskStarts : Array (Array (Array Nat))
  leafStackMaskEnds : Array (Array (Array Nat))
  prog : Array UInt64
  fast : Bool
  needsStack : Bool
  kernel : Kernel := .bytecode  -- Specialized kernel variant (default: bytecode)
  deriving Repr

namespace Plan

def mapIds (p : Plan) (f : UOpId → UOpId) : Plan :=
  { p with
    root := f p.root
    cover := UOpIdSet.map p.cover f
    leafBases := p.leafBases.map f }

end Plan

/-! ## Public instruction builders for tests -/

/-- Create LOAD instruction: push {lit}`leaf[idx]` onto stack. -/
def instLoad (idx : Nat) : UInt64 := UInt64.ofNat idx <<< 8

/-- Create binary op instruction (ADD=0, SUB=1, MUL=2, DIV=3, MAX=4) -/
def instBinary (op : Nat) : UInt64 :=
  match op with
  | 0 => 7   -- ADD
  | 1 => 8   -- SUB
  | 2 => 9   -- MUL
  | 3 => 10  -- DIV
  | 4 => 11  -- MAX
  | _ => 7   -- default to ADD

/-- Create WHERE instruction (ternary: cond, x, y → result) -/
def instWhere : UInt64 := 12

/-- Create CONST f32 instruction with bits -/
def instConstF32Bits (bits : UInt32) : UInt64 :=
  -- Encoding: const opcode doesn't exist in standard opcodes
  -- We'll use push + special marker for constant embedding
  -- Actually the program format expects leaves to be buffers, not inline consts
  -- This is a placeholder - real constants should be passed as input buffers
  0

private def opCodePush : UInt64 := 0
private def opCodeNeg : UInt64 := 1
private def opCodeSqrt : UInt64 := 2
private def opCodeRecip : UInt64 := 3
private def opCodeExp2 : UInt64 := 4
private def opCodeLog2 : UInt64 := 5
private def opCodeSin : UInt64 := 6
private def opCodeCos : UInt64 := 14
private def opCodeTan : UInt64 := 15
private def opCodeAdd : UInt64 := 7
private def opCodeSub : UInt64 := 8
private def opCodeMul : UInt64 := 9
private def opCodeDiv : UInt64 := 10
private def opCodeMax : UInt64 := 11
private def opCodeWhere : UInt64 := 12
private def opCodeMulAcc : UInt64 := 13

private def mkInstr (op : UInt64) (imm : Nat := 0) : UInt64 :=
  (UInt64.ofNat imm <<< 8) ||| op

private def opCodeOf (op : Ops) : Option UInt64 :=
  match op with
  | .NEG => some opCodeNeg
  | .SQRT => some opCodeSqrt
  | .RECIPROCAL => some opCodeRecip
  | .EXP2 => some opCodeExp2
  | .LOG2 => some opCodeLog2
  | .SIN => some opCodeSin
  | .COS => some opCodeCos
  | .TAN => some opCodeTan
  | .ADD => some opCodeAdd
  | .SUB => some opCodeSub
  | .MUL => some opCodeMul
  | .FDIV => some opCodeDiv
  | .MAX => some opCodeMax
  | .WHERE => some opCodeWhere
  | .MULACC => some opCodeMulAcc
  | _ => none

private def allFloat32 (xs : List UOp) : Bool :=
  xs.all (fun u => u.dtype == .float32)

private def condOk (u : UOp) : Bool :=
  u.dtype == .bool || u.dtype == .float32

private def promoteView (v : View) (targetRank : Nat) : View :=
  let rank := v.kernelShape.size
  if rank >= targetRank then
    v
  else
    let pad := targetRank - rank
    let padShape := Array.replicate pad 1
    let padStride : Array Int64 := Array.replicate pad 0
    let padMaskStart := Array.replicate pad 0
    let padMaskEnd := Array.replicate pad 1
    { kernelShape := padShape.append v.kernelShape
      strides := padStride.append v.strides
      offset := v.offset
      maskStart := padMaskStart.append v.maskStart
      maskEnd := padMaskEnd.append v.maskEnd }

private def promoteTop (st : Backend.ShapeTracker) (targetRank : Nat) : Backend.ShapeTracker :=
  let v := promoteView st.top targetRank
  Backend.ShapeTracker.replaceTop st v

private structure BuildState where
  leafMap : HashMap UOpId Nat := ∅
  leafBases : Array UOpId := #[]
  leafDtypes : Array Nat := #[]
  leafShapes : Array (Array Nat) := #[]
  leafStrides : Array (Array Int64) := #[]
  leafOffsets : Array Int64 := #[]
  leafMaskStarts : Array (Array Nat) := #[]
  leafMaskEnds : Array (Array Nat) := #[]
  leafStackShapes : Array (Array (Array Nat)) := #[]
  leafStackStrides : Array (Array (Array Int64)) := #[]
  leafStackOffsets : Array (Array Int64) := #[]
  leafStackMaskStarts : Array (Array (Array Nat)) := #[]
  leafStackMaskEnds : Array (Array (Array Nat)) := #[]
  prog : Array UInt64 := #[]
  cover : UOpIdSet := UOpIdSet.mkEmpty
  fast : Bool := true
  needsStack : Bool := false
  deriving Repr

private abbrev BuildM := StateM BuildState

private def addLeaf (u : UOp) (outShape : Shape) : BuildM (Option Nat) := do
  let st ← get
  match st.leafMap.get? u.uid with
  | some idx => return some idx
  | none =>
    let dtCode :=
      if u.dtype == .bool then
        some 1
      else if u.dtype == .float32 then
        some 0
      else if u.dtype == .uint8 then
        some 2
      else
        none
    match dtCode with
    | none => return none
    | some code =>
      let outRank := outShape.length
      let result? : Option (UOpId × View × Backend.StackInfo × Bool) ←
        match View.ofUOp? u with
        | some (base, v0) =>
          let v1 := promoteView v0 outRank
          match v1.expand outShape with
          | some v =>
            let st := Backend.ShapeTracker.ofViews #[v]
            let info := Backend.ShapeTracker.stackInfo st
            pure (some (base, v, info, Backend.ShapeTracker.needsStack st))
          | none => pure none
        | none =>
          match Backend.ShapeTracker.ofUOp? u with
          | some (base, st0) =>
            let st1 := promoteTop st0 outRank
            match Backend.ShapeTracker.expand st1 outShape with
            | some st =>
              let info := Backend.ShapeTracker.stackInfo st
              pure (some (base, st.top, info, Backend.ShapeTracker.needsStack st))
            | none => pure none
          | none => pure none
      let (base, topView, stackInfo, needsStack) ← match result? with
        | some r => pure r
        | none => return none
      let idx := st.leafBases.size
      let st' :=
        { st with
          leafMap := st.leafMap.insert u.uid idx
          leafBases := st.leafBases.push base
          leafDtypes := st.leafDtypes.push code
          leafShapes := st.leafShapes.push u.shape.toArray
          leafStrides := st.leafStrides.push topView.strides
          leafOffsets := st.leafOffsets.push topView.offset
          leafMaskStarts := st.leafMaskStarts.push topView.maskStart
          leafMaskEnds := st.leafMaskEnds.push topView.maskEnd
          leafStackShapes := st.leafStackShapes.push stackInfo.shapes
          leafStackStrides := st.leafStackStrides.push stackInfo.strides
          leafStackOffsets := st.leafStackOffsets.push stackInfo.offsets
          leafStackMaskStarts := st.leafStackMaskStarts.push stackInfo.maskStarts
          leafStackMaskEnds := st.leafStackMaskEnds.push stackInfo.maskEnds
          fast := st.fast && (base == u.uid)
          needsStack := st.needsStack || needsStack }
      set st'
      return some idx

private def pushProg (instr : UInt64) : BuildM Unit := do
  modify fun st => { st with prog := st.prog.push instr }

private def addCover (uid : UOpId) : BuildM Unit := do
  modify fun st => { st with cover := UOpIdSet.add st.cover uid }

private def canFuseNode (u : UOp) : Bool :=
  if u.dtype != .float32 then
    false
  else
    match u.op with
    | .CAST =>
        u.src.length == 1 && u.src[0]!.dtype == .uint8
    | .NEG | .SQRT | .RECIPROCAL | .EXP2 | .LOG2 | .SIN | .COS | .TAN =>
        u.src.length == 1 && allFloat32 u.src
    | .ADD | .SUB | .MUL | .FDIV | .MAX =>
        u.src.length == 2 && allFloat32 u.src
    | .WHERE =>
        u.src.length == 3 &&
          condOk u.src[0]! &&
          u.src[1]!.dtype == .float32 &&
          u.src[2]!.dtype == .float32
    | .MULACC =>
        u.src.length == 3 && allFloat32 u.src
    | _ => false

private partial def emitExpr (rootId : UOpId) (u : UOp) (outShape : Shape) (keep : UOpIdSet)
    (refCnt : HashMap UOpId Nat) (allowRootShared : Bool) : BuildM Bool := do
  if u.op == .CAST then
    match u.src with
    | [s] =>
        if u.dtype == .float32 && s.dtype == .uint8 then
          if (← emitExpr rootId s outShape keep refCnt allowRootShared) then
            addCover u.uid
            return true
          else
            return false
        else
          return false
    | _ => return false
  let fusable := canFuseNode u
  let shouldFuse :=
    if !fusable then
      false
    else if u.uid == rootId then
      allowRootShared || (!UOpIdSet.member keep u.uid && refCnt.getD u.uid 0 == 1)
    else
      !UOpIdSet.member keep u.uid && refCnt.getD u.uid 0 == 1
  if !shouldFuse then
    match (← addLeaf u outShape) with
    | some idx =>
      pushProg (mkInstr opCodePush idx)
      return true
    | none => return false
  else
    for s in u.src do
      let ok ← emitExpr rootId s outShape keep refCnt allowRootShared
      if !ok then
        return false
    addCover u.uid
    match opCodeOf u.op with
    | some code =>
      pushProg (mkInstr code)
      return true
    | none => return false

/-- Check if all leaf shapes are equal (required for contiguous fast paths). -/
private def allShapesEqual (shapes : Array (Array Nat)) : Bool :=
  if shapes.size ≤ 1 then true
  else
    let first := shapes[0]!
    shapes.all (· == first)

/-- Detect specialized kernel from bytecode pattern.
    Pattern matching happens here in Lean, not in C.
    Contiguous kernels require all leaf shapes to be identical. -/
private def detectKernel (prog : Array UInt64) (fast : Bool) (nLeaves : Nat)
    (leafShapes : Array (Array Nat)) : Kernel :=
  if !fast then .bytecode  -- Need contiguous tensors for fast paths
  else
    let getOp (instr : UInt64) : UInt64 := instr &&& 0xFF
    let getImm (instr : UInt64) : UInt64 := instr >>> 8
    -- Unary pattern: [PUSH 0, OP]
    if prog.size == 2 && nLeaves == 1 then
      let instr0 := prog[0]!
      let instr1 := prog[1]!
      if getOp instr0 == opCodePush && getImm instr0 == 0 then
        match getOp instr1 with
        | 1 => .negContiguous      -- opCodeNeg
        | 2 => .sqrtContiguous     -- opCodeSqrt
        | 3 => .recipContiguous    -- opCodeRecip
        | 4 => .exp2Contiguous     -- opCodeExp2
        | 5 => .log2Contiguous     -- opCodeLog2
        | 6 => .sinContiguous      -- opCodeSin
        | 14 => .cosContiguous     -- opCodeCos
        | 15 => .tanContiguous     -- opCodeTan
        | _ => .bytecode
      else .bytecode
    -- Binary pattern: [PUSH 0, PUSH 1, OP]
    -- IMPORTANT: Contiguous binary ops require both inputs to have the SAME shape.
    -- If shapes differ (e.g., tensor / scalar), we must use bytecode with broadcasting.
    else if prog.size == 3 && nLeaves == 2 && allShapesEqual leafShapes then
      let instr0 := prog[0]!
      let instr1 := prog[1]!
      let instr2 := prog[2]!
      if getOp instr0 == opCodePush && getImm instr0 == 0 &&
         getOp instr1 == opCodePush && getImm instr1 == 1 then
        match getOp instr2 with
        | 7 => .addContiguous      -- opCodeAdd
        | 8 => .subContiguous      -- opCodeSub
        | 9 => .mulContiguous      -- opCodeMul
        | 10 => .divContiguous     -- opCodeDiv
        | 11 => .maxContiguous     -- opCodeMax
        | 16 => .powContiguous     -- opCodePow
        | _ => .bytecode
      else .bytecode
    else .bytecode

private def compileWith (u : UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat)
    (allowRootShared : Bool) (requireRootFusable : Bool) : Option Plan := Id.run do
  if requireRootFusable && !canFuseNode u then
    return none
  let init : BuildState := {}
  let (ok, st) := (emitExpr u.uid u u.shape keep refCnt allowRootShared).run init
  if !ok then
    return none
  if st.prog.isEmpty then
    return none
  let kernel :=
    if st.leafDtypes.all (· == 0) then
      detectKernel st.prog st.fast st.leafBases.size st.leafShapes
    else
      .bytecode
  return some
    { root := u.uid
      cover := st.cover
      leafBases := st.leafBases
      leafDtypes := st.leafDtypes
      leafShapes := st.leafShapes
      leafStrides := st.leafStrides
      leafOffsets := st.leafOffsets
      leafMaskStarts := st.leafMaskStarts
      leafMaskEnds := st.leafMaskEnds
      leafStackShapes := st.leafStackShapes
      leafStackStrides := st.leafStackStrides
      leafStackOffsets := st.leafStackOffsets
      leafStackMaskStarts := st.leafStackMaskStarts
      leafStackMaskEnds := st.leafStackMaskEnds
      prog := st.prog
      fast := st.fast
      needsStack := st.needsStack
      kernel := kernel }

def compile (u : UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : Option Plan :=
  compileWith u keep refCnt true true

def compileForReduce (u : UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : Option Plan :=
  compileWith u keep refCnt false false

def compileVariants (u : UOp) (keep : UOpIdSet) (refCnt : HashMap UOpId Nat) : Array Plan :=
  match compile u keep refCnt with
  | some p => #[p]
  | none => #[]

end TinyGrad4.Backend.FusedEwise
