import Float64
import TinyGrad4.Tensor.Tensor
import TinyGrad4.Tensor.Movement

set_option maxHeartbeats 800000

namespace TinyGrad4

namespace StaticTensor

private def shapeEq {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t1 t2 : StaticTensor s d device) : t1.uop.shape = t2.uop.shape := by
  simp [t1.h_shape, t2.h_shape]

private def dtypeEq {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t1 t2 : StaticTensor s d device) : t1.uop.dtype = t2.uop.dtype := by
  simp [t1.h_dtype, t2.h_dtype]

private def liftBroadcast {s1 s2 : List Nat} {d : DType} {device : Backend.DeviceType}
    (t1 : StaticTensor s1 d device) (t2 : StaticTensor s2 d device)
    (h : Shape.broadcastable s1 s2 = true) : Shape.broadcastable t1.uop.shape t2.uop.shape = true := by
  simpa [t1.h_shape, t2.h_shape] using h

private def build {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (u : UOp) (requiresGrad : Bool := false) : StaticTensor s d device :=
  StaticTensor.ofUOp u (requiresGrad := requiresGrad)

private theorem concatValidAxis {s1 s2 : Shape} {axis : Nat}
    (h : Shape.concatValid s1 s2 axis = true) : axis < s1.length := by
  unfold Shape.concatValid at h
  have h' : s1.length = s2.length ∧ axis < s1.length ∧
      listAll (fun i => if i == axis then true else Shape.dim s1 i == Shape.dim s2 i) (listRange s1.length) = true := by
    simpa [Bool.and_assoc, decide_eq_true_eq] using h
  exact h'.2.1

private theorem concatValidToListValid {s1 s2 : Shape} {axis : Nat}
    (h : Shape.concatValid s1 s2 axis = true) : Shape.concatListValid [s1, s2] axis = true := by
  unfold Shape.concatListValid
  have hAxis : axis < s1.length := concatValidAxis h
  have hAxisB : decide (axis < s1.length) = true := decide_eq_true hAxis
  simpa [listAll, h, decide_eq_true_eq] using hAxisB

private theorem toUOpsShapeList {d : DType} {device : Backend.DeviceType} {shapes : List Shape}
    (ts : TensorList d device shapes) : (TensorList.toUOps ts).map (fun u => u.shape) = shapes := by
  induction ts with
  | nil => simp [TensorList.toUOps]
  | cons t rest ih => simp [TensorList.toUOps, t.h_shape, ih]

def add {s : List Nat} {d : DType} {device : Backend.DeviceType} (t1 t2 : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let hShape := shapeEq t1 t2
  let hType := dtypeEq t1 t2
  let result ← UOp.binaryOpSame .ADD t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def addBroadcast {s1 s2 : List Nat} {d : DType} {device : Backend.DeviceType}
    (t1 : StaticTensor s1 d device) (t2 : StaticTensor s2 d device)
    (h : Shape.broadcastable s1 s2 = true)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) d device) := do
  let hShape := liftBroadcast t1 t2 h
  let hType : t1.uop.dtype = t2.uop.dtype := by
    simp [t1.h_dtype, t2.h_dtype]
  let result ← UOp.binaryOpBroadcastSame .ADD t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def mul {s : List Nat} {d : DType} {device : Backend.DeviceType} (t1 t2 : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let hShape := shapeEq t1 t2
  let hType := dtypeEq t1 t2
  let result ← UOp.binaryOpSame .MUL t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def mulBroadcast {s1 s2 : List Nat} {d : DType} {device : Backend.DeviceType}
    (t1 : StaticTensor s1 d device) (t2 : StaticTensor s2 d device)
    (h : Shape.broadcastable s1 s2 = true)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) d device) := do
  let hShape := liftBroadcast t1 t2 h
  let hType : t1.uop.dtype = t2.uop.dtype := by
    simp [t1.h_dtype, t2.h_dtype]
  let result ← UOp.binaryOpBroadcastSame .MUL t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def sub {s : List Nat} {d : DType} {device : Backend.DeviceType} (t1 t2 : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let hShape := shapeEq t1 t2
  let hType := dtypeEq t1 t2
  let result ← UOp.binaryOpSame .SUB t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def subBroadcast {s1 s2 : List Nat} {d : DType} {device : Backend.DeviceType}
    (t1 : StaticTensor s1 d device) (t2 : StaticTensor s2 d device)
    (h : Shape.broadcastable s1 s2 = true)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) d device) := do
  let hShape := liftBroadcast t1 t2 h
  let hType : t1.uop.dtype = t2.uop.dtype := by
    simp [t1.h_dtype, t2.h_dtype]
  let result ← UOp.binaryOpBroadcastSame .SUB t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def div {s : List Nat} {d : DType} {device : Backend.DeviceType} (t1 t2 : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let hShape := shapeEq t1 t2
  let hType := dtypeEq t1 t2
  let result ← UOp.binaryOpSame .FDIV t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def divBroadcast {s1 s2 : List Nat} {d : DType} {device : Backend.DeviceType}
    (t1 : StaticTensor s1 d device) (t2 : StaticTensor s2 d device)
    (h : Shape.broadcastable s1 s2 = true)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) d device) := do
  let hShape := liftBroadcast t1 t2 h
  let hType : t1.uop.dtype = t2.uop.dtype := by
    simp [t1.h_dtype, t2.h_dtype]
  let result ← UOp.binaryOpBroadcastSame .FDIV t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def pow {s : List Nat} {d : DType} {device : Backend.DeviceType} (t1 t2 : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let hShape := shapeEq t1 t2
  let hType := dtypeEq t1 t2
  let result ← UOp.binaryOpSame .POW t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def powBroadcast {s1 s2 : List Nat} {d : DType} {device : Backend.DeviceType}
    (t1 : StaticTensor s1 d device) (t2 : StaticTensor s2 d device)
    (h : Shape.broadcastable s1 s2 = true)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) d device) := do
  let hShape := liftBroadcast t1 t2 h
  let hType : t1.uop.dtype = t2.uop.dtype := by
    simp [t1.h_dtype, t2.h_dtype]
  let result ← UOp.binaryOpBroadcastSame .POW t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def cmplt {s : List Nat} {d : DType} {device : Backend.DeviceType} (t1 t2 : StaticTensor s d device) : TensorM (StaticTensor s .bool device) := do
  let hShape := shapeEq t1 t2
  let hType := dtypeEq t1 t2
  let result ← UOp.binaryOpSame .CMPLT t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def cmpltBroadcast {s1 s2 : List Nat} {d : DType} {device : Backend.DeviceType}
    (t1 : StaticTensor s1 d device) (t2 : StaticTensor s2 d device)
    (h : Shape.broadcastable s1 s2 = true)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool device) := do
  let hShape := liftBroadcast t1 t2 h
  let hType : t1.uop.dtype = t2.uop.dtype := by
    simp [t1.h_dtype, t2.h_dtype]
  let result ← UOp.binaryOpBroadcastSame .CMPLT t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def cmpgt {s : List Nat} {d : DType} {device : Backend.DeviceType} (t1 t2 : StaticTensor s d device) : TensorM (StaticTensor s .bool device) := do
  cmplt t2 t1

def cmpgtBroadcast {s1 s2 : List Nat} {d : DType} {device : Backend.DeviceType}
    (t1 : StaticTensor s1 d device) (t2 : StaticTensor s2 d device)
    (h : Shape.broadcastable s1 s2 = true)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool device) := do
  let hSwap : Shape.broadcastable s2 s1 = true := by
    simpa [Shape.broadcastable_comm s1 s2] using h
  let hShape : Shape.broadcastable t2.uop.shape t1.uop.shape = true := by
    simpa [t2.h_shape, t1.h_shape] using hSwap
  let hType : t2.uop.dtype = t1.uop.dtype := by
    simp [t2.h_dtype, t1.h_dtype]
  let result ← UOp.binaryOpBroadcastSame .CMPLT t2.uop t1.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def cmpeq {s : List Nat} {d : DType} {device : Backend.DeviceType} (t1 t2 : StaticTensor s d device) : TensorM (StaticTensor s .bool device) := do
  let hShape := shapeEq t1 t2
  let hType := dtypeEq t1 t2
  let result ← UOp.binaryOpSame .CMPEQ t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def cmpeqBroadcast {s1 s2 : List Nat} {d : DType} {device : Backend.DeviceType}
    (t1 : StaticTensor s1 d device) (t2 : StaticTensor s2 d device)
    (h : Shape.broadcastable s1 s2 = true)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool device) := do
  let hShape := liftBroadcast t1 t2 h
  let hType : t1.uop.dtype = t2.uop.dtype := by
    simp [t1.h_dtype, t2.h_dtype]
  let result ← UOp.binaryOpBroadcastSame .CMPEQ t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def cmpne {s : List Nat} {d : DType} {device : Backend.DeviceType} (t1 t2 : StaticTensor s d device) : TensorM (StaticTensor s .bool device) := do
  let hShape := shapeEq t1 t2
  let hType := dtypeEq t1 t2
  let result ← UOp.binaryOpSame .CMPNE t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def cmpneBroadcast {s1 s2 : List Nat} {d : DType} {device : Backend.DeviceType}
    (t1 : StaticTensor s1 d device) (t2 : StaticTensor s2 d device)
    (h : Shape.broadcastable s1 s2 = true)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool device) := do
  let hShape := liftBroadcast t1 t2 h
  let hType : t1.uop.dtype = t2.uop.dtype := by
    simp [t1.h_dtype, t2.h_dtype]
  let result ← UOp.binaryOpBroadcastSame .CMPNE t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def cat {s1 s2 : List Nat} {d : DType} {device : Backend.DeviceType} (t1 : StaticTensor s1 d device) (t2 : StaticTensor s2 d device)
    (axis : Nat) (h : Shape.concatValid s1 s2 axis = true) : TensorM (StaticTensor (Shape.concatOut s1 s2 axis) d device) := do
  let hList : Shape.concatListValid [s1, s2] axis = true := concatValidToListValid h
  let hList' : Shape.concatListValid [t1.uop.shape, t2.uop.shape] axis = true := by
    simpa [t1.h_shape, t2.h_shape] using hList
  let out ← UOp.catValid [t1.uop, t2.uop] axis d hList'
  pure (build out (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def catList {d : DType} {device : Backend.DeviceType} {shapes : List Shape} (ts : TensorList d device shapes) (axis : Nat)
    (h : Shape.concatListValid shapes axis = true)
    : TensorM (StaticTensor (Shape.concatOutList shapes axis) d device) := do
  let uops := TensorList.toUOps ts
  have hShapes : uops.map (fun u => u.shape) = shapes := by
    simpa [uops] using toUOpsShapeList ts
  let h' : Shape.concatListValid (uops.map (fun u => u.shape)) axis = true := by
    simpa [hShapes] using h
  let out ← UOp.catValid uops axis d h'
  let reqGrad := TensorList.anyRequiresGrad ts
  pure (build out (requiresGrad := reqGrad))

def bitand {s : List Nat} {device : Backend.DeviceType} (t1 t2 : StaticTensor s .bool device)
    : TensorM (StaticTensor s .bool device) := do
  let hShape := shapeEq t1 t2
  let hType := dtypeEq t1 t2
  let result ← UOp.binaryOpSame .AND t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def bitandBroadcast {s1 s2 : List Nat} {device : Backend.DeviceType}
    (t1 : StaticTensor s1 .bool device) (t2 : StaticTensor s2 .bool device)
    (h : Shape.broadcastable s1 s2 = true)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool device) := do
  let hShape := liftBroadcast t1 t2 h
  let hType : t1.uop.dtype = t2.uop.dtype := by
    simp [t1.h_dtype, t2.h_dtype]
  let result ← UOp.binaryOpBroadcastSame .AND t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def bitor {s : List Nat} {device : Backend.DeviceType} (t1 t2 : StaticTensor s .bool device)
    : TensorM (StaticTensor s .bool device) := do
  let hShape := shapeEq t1 t2
  let hType := dtypeEq t1 t2
  let result ← UOp.binaryOpSame .OR t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def bitorBroadcast {s1 s2 : List Nat} {device : Backend.DeviceType}
    (t1 : StaticTensor s1 .bool device) (t2 : StaticTensor s2 .bool device)
    (h : Shape.broadcastable s1 s2 = true)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool device) := do
  let hShape := liftBroadcast t1 t2 h
  let hType : t1.uop.dtype = t2.uop.dtype := by
    simp [t1.h_dtype, t2.h_dtype]
  let result ← UOp.binaryOpBroadcastSame .OR t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def bitxor {s : List Nat} {device : Backend.DeviceType} (t1 t2 : StaticTensor s .bool device)
    : TensorM (StaticTensor s .bool device) := do
  let hShape := shapeEq t1 t2
  let hType := dtypeEq t1 t2
  let result ← UOp.binaryOpSame .XOR t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def bitxorBroadcast {s1 s2 : List Nat} {device : Backend.DeviceType}
    (t1 : StaticTensor s1 .bool device) (t2 : StaticTensor s2 .bool device)
    (h : Shape.broadcastable s1 s2 = true)
    : TensorM (StaticTensor (Shape.broadcastOut s1 s2) .bool device) := do
  let hShape := liftBroadcast t1 t2 h
  let hType : t1.uop.dtype = t2.uop.dtype := by
    simp [t1.h_dtype, t2.h_dtype]
  let result ← UOp.binaryOpBroadcastSame .XOR t1.uop t2.uop hShape hType
  pure (build result (requiresGrad := t1.requiresGrad || t2.requiresGrad))

def cast {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (dtype : DType)
    : TensorM (StaticTensor s dtype device) := do
  let result ← UOp.cast t.uop dtype
  pure (build result (requiresGrad := t.requiresGrad))

def bitcast {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (dtype : DType)
    (hBits : d.itemsize = dtype.itemsize)
    : TensorM (StaticTensor s dtype device) := do
  let hBits' : t.uop.dtype.itemsize = dtype.itemsize := by
    simpa [t.h_dtype] using hBits
  let result ← UOp.bitcastValid t.uop dtype hBits'
  pure (build result (requiresGrad := t.requiresGrad))

def to {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (dtype : DType)
    : TensorM (StaticTensor s dtype device) :=
  cast t dtype

def select {s1 s2 s3 : List Nat} {d : DType} {device : Backend.DeviceType}
    (cond : StaticTensor s1 .bool device) (x : StaticTensor s2 d device) (y : StaticTensor s3 d device)
    (hXY : Shape.broadcastable s2 s3 = true)
    (hCond : Shape.broadcastable s1 (Shape.broadcastOut s2 s3) = true)
    : TensorM (StaticTensor (Shape.broadcastOut s1 (Shape.broadcastOut s2 s3)) d device) := do
  let hXY' : Shape.broadcastable x.uop.shape y.uop.shape = true := by
    simpa [x.h_shape, y.h_shape] using hXY
  let hCond' : Shape.broadcastable cond.uop.shape (Shape.broadcastOut x.uop.shape y.uop.shape) = true := by
    simpa [cond.h_shape, x.h_shape, y.h_shape] using hCond
  let out ← UOp.whereBroadcast cond.uop x.uop y.uop hXY' hCond'
  pure (build out (requiresGrad := x.requiresGrad || y.requiresGrad))

private def whereSame {s : Shape} {d : DType} {device : Backend.DeviceType}
    (cond : StaticTensor s .bool device) (x y : StaticTensor s d device)
    : TensorM (StaticTensor s d device) := do
  let hCondType : cond.uop.dtype = .bool := by
    simp [cond.h_dtype]
  let hXY : Shape.broadcastable x.uop.shape y.uop.shape = true := by
    simpa [x.h_shape, y.h_shape] using Shape.broadcastable_refl s
  let hCond : Shape.broadcastable cond.uop.shape (Shape.broadcastOut x.uop.shape y.uop.shape) = true := by
    simpa [cond.h_shape, x.h_shape, y.h_shape, Shape.broadcastOut_refl] using Shape.broadcastable_out_refl s
  let hType : x.uop.dtype = y.uop.dtype := by
    simp [x.h_dtype, y.h_dtype]
  let out ← UOp.whereBroadcastSame cond.uop x.uop y.uop hCondType hXY hCond hType
  pure (StaticTensor.ofUOp out (requiresGrad := x.requiresGrad || y.requiresGrad))

def addRow {batch dim : Nat} {d : DType} {device : Backend.DeviceType}
    (x : Matrix batch dim d device) (row : StaticTensor [1, dim] d device)
    : TensorM (Matrix batch dim d device) := do
  let rowExpanded ← expand row [batch, dim] (by simp [Shape.expandValid, listAll])
  add x rowExpanded

def addVector {batch dim : Nat} {d : DType} {device : Backend.DeviceType}
    (x : Matrix batch dim d device) (v : Vector dim d device)
    : TensorM (Matrix batch dim d device) := do
  let vRow ← reshape v [1, dim] (by simp [Shape.reshapeValid, Shape.numel, listProd])
  let vExpanded ← expand vRow [batch, dim] (by simp [Shape.expandValid, listAll])
  add x vExpanded

def subColumn {batch dim : Nat} {d : DType} {device : Backend.DeviceType}
    (x : Matrix batch dim d device) (col : Matrix batch 1 d device)
    : TensorM (Matrix batch dim d device) := do
  let colExpanded ← expand col [batch, dim] (by simp [Shape.expandValid, listAll])
  sub x colExpanded

def mulColumn {batch dim : Nat} {d : DType} {device : Backend.DeviceType}
    (x : Matrix batch dim d device) (col : Matrix batch 1 d device)
    : TensorM (Matrix batch dim d device) := do
  let colExpanded ← expand col [batch, dim] (by simp [Shape.expandValid, listAll])
  mul x colExpanded

def mulVector {batch dim : Nat} {d : DType} {device : Backend.DeviceType}
    (x : Matrix batch dim d device) (v : Vector dim d device)
    : TensorM (Matrix batch dim d device) := do
  let vRow ← reshape v [1, dim] (by simp [Shape.reshapeValid, Shape.numel, listProd])
  let vExpanded ← expand vRow [batch, dim] (by simp [Shape.expandValid, listAll])
  mul x vExpanded

def addChannelNCHW {batch channels height width : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, channels, height, width] d device)
    (ch : StaticTensor [1, channels, 1, 1] d device)
    : TensorM (StaticTensor [batch, channels, height, width] d device) := do
  let chExpanded ← expand ch [batch, channels, height, width] (by simp [Shape.expandValid, listAll])
  add x chExpanded

def subChannelNCHW {batch channels height width : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, channels, height, width] d device)
    (ch : StaticTensor [1, channels, 1, 1] d device)
    : TensorM (StaticTensor [batch, channels, height, width] d device) := do
  let chExpanded ← expand ch [batch, channels, height, width] (by simp [Shape.expandValid, listAll])
  sub x chExpanded

def mulChannelNCHW {batch channels height width : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, channels, height, width] d device)
    (ch : StaticTensor [1, channels, 1, 1] d device)
    : TensorM (StaticTensor [batch, channels, height, width] d device) := do
  let chExpanded ← expand ch [batch, channels, height, width] (by simp [Shape.expandValid, listAll])
  mul x chExpanded

def channelVectorNCHW {channels : Nat} {d : DType} {device : Backend.DeviceType}
    (v : Vector channels d device)
    : TensorM (StaticTensor [1, channels, 1, 1] d device) :=
  reshape v [1, channels, 1, 1] (by simp [Shape.reshapeValid, Shape.numel, listProd])

def addChannelVectorNCHW {batch channels height width : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, channels, height, width] d device)
    (v : Vector channels d device)
    : TensorM (StaticTensor [batch, channels, height, width] d device) := do
  let ch ← channelVectorNCHW v
  addChannelNCHW x ch

def subChannelVectorNCHW {batch channels height width : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, channels, height, width] d device)
    (v : Vector channels d device)
    : TensorM (StaticTensor [batch, channels, height, width] d device) := do
  let ch ← channelVectorNCHW v
  subChannelNCHW x ch

def mulChannelVectorNCHW {batch channels height width : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, channels, height, width] d device)
    (v : Vector channels d device)
    : TensorM (StaticTensor [batch, channels, height, width] d device) := do
  let ch ← channelVectorNCHW v
  mulChannelNCHW x ch

def addChannelNC {batch channels : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, channels] d device) (ch : StaticTensor [1, channels] d device)
    : TensorM (StaticTensor [batch, channels] d device) := do
  let chExpanded ← expand ch [batch, channels] (by simp [Shape.expandValid, listAll])
  add x chExpanded

def subChannelNC {batch channels : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, channels] d device) (ch : StaticTensor [1, channels] d device)
    : TensorM (StaticTensor [batch, channels] d device) := do
  let chExpanded ← expand ch [batch, channels] (by simp [Shape.expandValid, listAll])
  sub x chExpanded

def mulChannelNC {batch channels : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, channels] d device) (ch : StaticTensor [1, channels] d device)
    : TensorM (StaticTensor [batch, channels] d device) := do
  let chExpanded ← expand ch [batch, channels] (by simp [Shape.expandValid, listAll])
  mul x chExpanded

def channelVectorNC {channels : Nat} {d : DType} {device : Backend.DeviceType}
    (v : Vector channels d device)
    : TensorM (StaticTensor [1, channels] d device) :=
  reshape v [1, channels] (by simp [Shape.reshapeValid, Shape.numel, listProd])

def addChannelVectorNC {batch channels : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, channels] d device)
    (v : Vector channels d device)
    : TensorM (StaticTensor [batch, channels] d device) := do
  let ch ← channelVectorNC v
  addChannelNC x ch

def subChannelVectorNC {batch channels : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, channels] d device)
    (v : Vector channels d device)
    : TensorM (StaticTensor [batch, channels] d device) := do
  let ch ← channelVectorNC v
  subChannelNC x ch

def mulChannelVectorNC {batch channels : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, channels] d device)
    (v : Vector channels d device)
    : TensorM (StaticTensor [batch, channels] d device) := do
  let ch ← channelVectorNC v
  mulChannelNC x ch

infixl:65 " +. " => addBroadcast
infixl:65 " -. " => subBroadcast
infixl:70 " *. " => mulBroadcast
infixl:70 " /. " => divBroadcast

def neg {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let result ← UOp.neg t.uop
  pure (build result (requiresGrad := t.requiresGrad))

def trunc {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let result ← UOp.trunc t.uop
  pure (build result (requiresGrad := t.requiresGrad))

def floor {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let truncT ← trunc t
  let isNeg ← UOp.cmplt t.uop truncT.uop
  let one ← UOp.const d 1.0
  let truncMinusOne ← UOp.sub truncT.uop one
  let isNegT : StaticTensor s .bool device := StaticTensor.ofUOp isNeg
  let truncMinusOneT : StaticTensor s d device := StaticTensor.ofUOp truncMinusOne (requiresGrad := t.requiresGrad)
  whereSame isNegT truncMinusOneT truncT

def ceil {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let truncT ← trunc t
  let isPos ← UOp.cmplt truncT.uop t.uop
  let one ← UOp.const d 1.0
  let truncPlusOne ← UOp.add truncT.uop one
  let isPosT : StaticTensor s .bool device := StaticTensor.ofUOp isPos
  let truncPlusOneT : StaticTensor s d device := StaticTensor.ofUOp truncPlusOne (requiresGrad := t.requiresGrad)
  whereSame isPosT truncPlusOneT truncT

def sqrt {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let result ← UOp.sqrt t.uop
  pure (build result (requiresGrad := t.requiresGrad))

def rsqrt {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let sqrtT ← sqrt t
  let result ← UOp.recip sqrtT.uop
  pure (build result (requiresGrad := t.requiresGrad))

def exp2 {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let result ← UOp.exp2 t.uop
  pure (build result (requiresGrad := t.requiresGrad))

def log2 {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let result ← UOp.log2 t.uop
  pure (build result (requiresGrad := t.requiresGrad))

/-- Sine function -/
def sin {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let result ← UOp.sin t.uop
  pure (build result (requiresGrad := t.requiresGrad))

/-- Cosine function -/
def cos {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let result ← UOp.cos t.uop
  pure (build result (requiresGrad := t.requiresGrad))

/-- Tangent function -/
def tan {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let result ← UOp.tan t.uop
  pure (build result (requiresGrad := t.requiresGrad))

/-- Reciprocal (1/x) -/
def recip {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let result ← UOp.recip t.uop
  pure (build result (requiresGrad := t.requiresGrad))

def sum {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (Scalar d device) := do
  let axes := listRange s.length
  let result ← UOp.sum t.uop axes false
  pure (StaticTensor.ofUOp result)

def max {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (Scalar d device) := do
  let axes := listRange s.length
  let result ← UOp.max_ t.uop axes false
  pure (StaticTensor.ofUOp result)

def min {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (Scalar d device) := do
  let negT ← neg t
  let maxNeg ← max negT
  neg maxNeg

def mean {shape : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor shape d device) : TensorM (Scalar d device) := do
  let sumT ← sum t
  let n := listProd shape
  let nConst ← UOp.const d n.toFloat32
  let result ← UOp.div sumT.uop nConst
  pure (StaticTensor.ofUOp result)

-- Constants for exp/log conversion
-- ln(2) ≈ 0.693147
-- log2(e) ≈ 1.442695
def ln2 : Float64 := 0.6931471805599453
def log2e : Float64 := 1.4426950408889634

-- NOTE: We use Float32 for const construction so float32 graphs can stay in Float32/ByteArray land.
def ln2f32 : Float32 := 0.6931471805599453
def log2ef32 : Float32 := 1.4426950408889634

/-- Natural exponential: e^x = 2^(x * log2(e)) -/
def exp {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let log2eConst ← UOp.const d log2ef32
  let scaled ← UOp.mul t.uop log2eConst
  let result ← UOp.exp2 scaled
  pure (build result (requiresGrad := t.requiresGrad))

/-- Natural logarithm: ln(x) = log2(x) * ln(2) -/
def log {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let log2Result ← UOp.log2 t.uop
  let ln2Const ← UOp.const d ln2f32
  let result ← UOp.mul log2Result ln2Const
  pure (StaticTensor.ofUOp result (requiresGrad := t.requiresGrad))

/-- ReLU: max(0, x) -/
def relu {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let zero ← UOp.const d 0.0
  let result ← UOp.maxBinary t.uop zero
  pure (StaticTensor.ofUOp result (requiresGrad := t.requiresGrad))

/-- ReLU6: min(max(x, 0), 6). -/
def relu6 {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let reluT ← relu t
  let six ← UOp.const d 6.0
  let minus ← UOp.sub t.uop six
  let minusT : StaticTensor s d device  := StaticTensor.ofUOp minus (requiresGrad := t.requiresGrad)
  let reluMinus ← relu minusT
  let out ← UOp.sub reluT.uop reluMinus.uop
  pure (StaticTensor.ofUOp out (requiresGrad := t.requiresGrad))

/-- Hardsigmoid: relu(alpha*x + beta) - relu(alpha*x + beta - 1). -/
def hardsigmoid {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (alpha : Float32 := 0.16666667)
    (beta : Float32 := 0.5) : TensorM (StaticTensor s d device) := do
  let alphaConst ← UOp.const d alpha
  let betaConst ← UOp.const d beta
  let scaled ← UOp.mul t.uop alphaConst
  let shifted ← UOp.add scaled betaConst
  let shiftedT : StaticTensor s d device  := StaticTensor.ofUOp shifted (requiresGrad := t.requiresGrad)
  let reluShifted ← relu shiftedT
  let one ← UOp.const d 1.0
  let shiftedMinusOne ← UOp.sub shifted one
  let shiftedMinusOneT : StaticTensor s d device  := StaticTensor.ofUOp shiftedMinusOne (requiresGrad := t.requiresGrad)
  let reluShiftedMinusOne ← relu shiftedMinusOneT
  let out ← UOp.sub reluShifted.uop reluShiftedMinusOne.uop
  pure (StaticTensor.ofUOp out (requiresGrad := t.requiresGrad))

/-- Sigmoid: 1 / (1 + exp(-x)) -/
def sigmoid {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let negT ← neg t
  let expNeg ← exp negT
  let one ← UOp.const d 1.0
  let denom ← UOp.add expNeg.uop one
  let result ← UOp.div one denom
  pure (StaticTensor.ofUOp result (requiresGrad := t.requiresGrad))

/-- Tanh via exp: (e^x - e^-x) / (e^x + e^-x) -/
def tanh {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let negT ← neg t
  let expPos ← exp t
  let expNeg ← exp negT
  let num ← UOp.sub expPos.uop expNeg.uop
  let denom ← UOp.add expPos.uop expNeg.uop
  let result ← UOp.div num denom
  pure (StaticTensor.ofUOp result (requiresGrad := t.requiresGrad))

/-- Softplus: log(1 + exp(x)) -/
def softplus {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let expT ← exp t
  let one ← UOp.const d 1.0
  let onePlus ← UOp.add expT.uop one
  let onePlusT : StaticTensor s d device  := StaticTensor.ofUOp onePlus (requiresGrad := t.requiresGrad)
  log onePlusT

/-- GELU (tanh approximation). -/
def gelu {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let x2 ← UOp.mul t.uop t.uop
  let x3 ← UOp.mul x2 t.uop
  let c0 ← UOp.const d 0.044715
  let x3Scaled ← UOp.mul x3 c0
  let inner ← UOp.add t.uop x3Scaled
  let c1 ← UOp.const d 0.7978845608
  let scaled ← UOp.mul inner c1
  let scaledT : StaticTensor s d device  := StaticTensor.ofUOp scaled (requiresGrad := t.requiresGrad)
  let tanhScaled ← tanh scaledT
  let one ← UOp.const d 1.0
  let onePlus ← UOp.add tanhScaled.uop one
  let half ← UOp.const d 0.5
  let halfOnePlus ← UOp.mul onePlus half
  let result ← UOp.mul t.uop halfOnePlus
  pure (StaticTensor.ofUOp result (requiresGrad := t.requiresGrad))

/-- Abs: |x| -/
def abs {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let zero ← UOp.const d 0.0
  let negT ← UOp.neg t.uop
  let isNeg ← UOp.cmplt t.uop zero
  let isNegT : StaticTensor s .bool device := StaticTensor.ofUOp isNeg
  let negOutT : StaticTensor s d device := StaticTensor.ofUOp negT (requiresGrad := t.requiresGrad)
  whereSame isNegT negOutT t

/-- Square: x * x. -/
def square {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  mul t t

/-- SiLU / Swish: x * sigmoid(x) -/
def silu {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let sig ← sigmoid t
  mul t sig

/-- Swish alias for SiLU. -/
def swish {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) :=
  silu t

/-- Hardswish: x * relu6(x+3) / 6. -/
def hardswish {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let three ← UOp.const d 3.0
  let tPlusThree ← UOp.add t.uop three
  let tPlusThreeT : StaticTensor s d device  := StaticTensor.ofUOp tPlusThree (requiresGrad := t.requiresGrad)
  let relu6T ← relu6 tPlusThreeT
  let mul1 ← UOp.mul t.uop relu6T.uop
  let oneSixth ← UOp.const d 0.16666667
  let out ← UOp.mul mul1 oneSixth
  pure (StaticTensor.ofUOp out (requiresGrad := t.requiresGrad))

/-- Leaky ReLU: x if x >= 0, alpha * x otherwise. -/
def leakyRelu {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (alpha : Float32 := 0.01)
    : TensorM (StaticTensor s d device) := do
  let zero ← UOp.const d 0.0
  let alphaUop ← UOp.const d alpha
  let isNeg ← UOp.cmplt t.uop zero
  let negOut ← UOp.mul t.uop alphaUop
  let isNegT : StaticTensor s .bool device := StaticTensor.ofUOp isNeg
  let negOutT : StaticTensor s d device := StaticTensor.ofUOp negOut (requiresGrad := t.requiresGrad)
  whereSame isNegT negOutT t

/-- ELU: x if x >= 0, alpha * (exp(x) - 1) otherwise. -/
def elu {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (alpha : Float32 := 1.0)
    : TensorM (StaticTensor s d device) := do
  let zero ← UOp.const d 0.0
  let alphaUop ← UOp.const d alpha
  let isNeg ← UOp.cmplt t.uop zero
  let expT ← exp t
  let one ← UOp.const d 1.0
  let expm1 ← UOp.sub expT.uop one
  let negOut ← UOp.mul expm1 alphaUop
  let isNegT : StaticTensor s .bool device := StaticTensor.ofUOp isNeg
  let negOutT : StaticTensor s d device := StaticTensor.ofUOp negOut (requiresGrad := t.requiresGrad)
  whereSame isNegT negOutT t

/-- Log-sigmoid: log(sigmoid(x)) -/
def logSigmoid {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  let sig ← sigmoid t
  log sig

/-- Clamp values to [lo, hi]. -/
def clamp {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (lo hi : Float32) : TensorM (StaticTensor s d device) := do
  let loConst ← UOp.const d lo
  let hiConst ← UOp.const d hi
  let below ← UOp.cmplt t.uop loConst
  let above ← UOp.cmplt hiConst t.uop
  let clippedLo ← UOp.where_ below loConst t.uop
  let clipped ← UOp.where_ above hiConst clippedLo
  pure (StaticTensor.ofUOp clipped (requiresGrad := t.requiresGrad))

/-- Clip values to [lo, hi]. Alias for clamp. -/
def clip {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (lo hi : Float32) : TensorM (StaticTensor s d device) :=
  clamp t lo hi

/-- Hardtanh clamps values to [minVal, maxVal]. -/
def hardtanh {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (minVal : Float32 := -1.0) (maxVal : Float32 := 1.0)
    : TensorM (StaticTensor s d device) := do
  clamp t minVal maxVal

/-- Max along axis with keepdim -/
def maxAxis {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d device) := do
  let result ← UOp.max_ t.uop [axis] keepdim
  pure (StaticTensor.ofUOp result (requiresGrad := t.requiresGrad))

def minAxis {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d device) := do
  let negT ← neg t
  let maxNeg ← maxAxis negT axis keepdim
  neg maxNeg

/-- Max along axis with keepdim (statically checked axis). -/
def maxAxisF {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (axis : Fin s.length) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis.val] keepdim) d device) := do
  let hAxes : [axis.val].all (fun ax => ax < t.uop.shape.length) = true := by
    simpa [t.h_shape] using axis.isLt
  let result ← UOp.reduceValid t.uop .MAX [axis.val] keepdim hAxes
  pure (StaticTensor.ofUOp result (requiresGrad := t.requiresGrad))

def minAxisF {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (axis : Fin s.length) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis.val] keepdim) d device) := do
  let negT ← neg t
  let maxNeg ← maxAxisF negT axis keepdim
  neg maxNeg

/-- Sum along axis with keepdim -/
def sumAxis {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d device) := do
  let result ← UOp.sum t.uop [axis] keepdim
  pure (StaticTensor.ofUOp result (requiresGrad := t.requiresGrad))

/-- Sum along axis with keepdim (statically checked axis). -/
def sumAxisF {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (axis : Fin s.length) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis.val] keepdim) d device) := do
  let hAxes : [axis.val].all (fun ax => ax < t.uop.shape.length) = true := by
    simpa [t.h_shape] using axis.isLt
  let result ← UOp.reduceValid t.uop .ADD [axis.val] keepdim hAxes
  pure (StaticTensor.ofUOp result (requiresGrad := t.requiresGrad))

/-- Mean along axis with keepdim -/
def meanAxis {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d device) := do
  let sumT ← sumAxis t axis keepdim
  let n := listGetD s axis 1
  let nConst ← UOp.const d (Float64.ofNat n).toFloat32
  let result ← UOp.div sumT.uop nConst
  pure (StaticTensor.ofUOp result (requiresGrad := t.requiresGrad))

/-- Mean over matrix feature axis: `[B, D] -> [B, 1]`. -/
def meanMatrixFeatures {batch dim : Nat} {d : DType} {device : Backend.DeviceType}
    (x : Matrix batch dim d device) : TensorM (Matrix batch 1 d device) := do
  let out ← meanAxis x 1 true
  have out' : Matrix batch 1 d device := by
    simpa [Shape.reduce, listEnum, listRange] using out
  pure out'

/-- Mean over NCHW width axis: `[N, C, H, W] -> [N, C, H, 1]`. -/
def meanNCHWWidth {batch channels height width : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, channels, height, width] d device)
    : TensorM (StaticTensor [batch, channels, height, 1] d device) := do
  let out ← meanAxis x 3 true
  have out' : StaticTensor [batch, channels, height, 1] d device := by
    simpa [Shape.reduce, listEnum, listRange] using out
  pure out'

/-- Mean over NCHW height axis (for inputs shaped `[N, C, H, 1]`): `[N, C, H, 1] -> [N, C, 1, 1]`. -/
def meanNCHWHeight {batch channels height : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, channels, height, 1] d device)
    : TensorM (StaticTensor [batch, channels, 1, 1] d device) := do
  let out ← meanAxis x 2 true
  have out' : StaticTensor [batch, channels, 1, 1] d device := by
    simpa [Shape.reduce, listEnum, listRange] using out
  pure out'

/-- Mean over NCHW batch axis (for inputs shaped `[N, C, 1, 1]`): `[N, C, 1, 1] -> [1, C, 1, 1]`. -/
def meanNCHWBatch {batch channels : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, channels, 1, 1] d device)
    : TensorM (StaticTensor [1, channels, 1, 1] d device) := do
  let out ← meanAxis x 0 true
  have out' : StaticTensor [1, channels, 1, 1] d device := by
    simpa [Shape.reduce, listEnum, listRange] using out
  pure out'

/-- Mean over NC batch axis: `[N, C] -> [1, C]`. -/
def meanNCBatch {batch channels : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, channels] d device)
    : TensorM (StaticTensor [1, channels] d device) := do
  let out ← meanAxis x 0 true
  have out' : StaticTensor [1, channels] d device := by
    simpa [Shape.reduce, listEnum, listRange] using out
  pure out'

private def assertAxisInRange (axis rank : Nat) : Unit :=
  if axis < rank then () else panic! s!"axis {axis} out of range for rank {rank}"

private def replaceDimNow (shape : Shape) (axis newDim : Nat) : Shape :=
  (listEnum shape).map fun p =>
    if p.1 == axis then newDim else p.2

private def axisSliceBoundsNow (shape : Shape) (axis start stop : Nat) : List (Nat × Nat) :=
  (listEnum shape).map fun p =>
    if p.1 == axis then (start, stop) else (0, p.2)

/-- Variance along axis with keepdim -/
def varAxis {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d device) := do
  let meanT ← meanAxis t axis true
  let centered ← UOp.sub t.uop meanT.uop
  let centeredT : StaticTensor s d device  := StaticTensor.ofUOp centered (requiresGrad := t.requiresGrad)
  let sq ← mul centeredT centeredT
  meanAxis sq axis keepdim

/-- Product along an axis with keepdim. -/
def prodAxis {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d device) := do
  let _ := assertAxisInRange axis s.length
  let dim := listGetD s axis 0
  let keepShape := Shape.reduce s [axis] true
  let mut acc : StaticTensor keepShape d device ← Tensor.ones (device := device) keepShape d
  for i in [:dim] do
    let bounds := axisSliceBoundsNow s axis i (i + 1)
    let sliceU ← UOp.shrink t.uop bounds
    let slice : StaticTensor keepShape d device := StaticTensor.ofUOp sliceU (requiresGrad := t.requiresGrad)
    acc ← mul acc slice
  match keepdim with
  | true =>
    pure (StaticTensor.ofUOp acc.uop (requiresGrad := t.requiresGrad))
  | false =>
    reshape acc (Shape.reduce s [axis] false) (by
      simpa [Shape.reshapeValid, Shape.numel] using Shape.reduce_single_numel_eq s axis)

/-- Product over all elements. -/
def prod {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) : TensorM (Scalar d device) := do
  let flat ← flatten t
  let out ← prodAxis flat 0 false
  pure out

/-- Population variance over all elements. -/
def var {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) : TensorM (Scalar d device) := do
  let meanT ← mean t
  let centered ← UOp.sub t.uop meanT.uop
  let centeredT : StaticTensor s d device := StaticTensor.ofUOp centered (requiresGrad := t.requiresGrad)
  let sq ← mul centeredT centeredT
  mean sq

/-- Standard deviation along an axis with keepdim. -/
def stdAxis {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d device) := do
  let v ← varAxis t axis keepdim
  sqrt v

/-- Population standard deviation over all elements. -/
def std {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) : TensorM (Scalar d device) := do
  let v ← var t
  sqrt v

/-- Population variance and mean over all elements. -/
def varMean {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) : TensorM (Scalar d device × Scalar d device) := do
  let v ← var t
  let m ← mean t
  pure (v, m)

/-- Population standard deviation and mean over all elements. -/
def stdMean {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) : TensorM (Scalar d device × Scalar d device) := do
  let sdev ← std t
  let m ← mean t
  pure (sdev, m)

/-- Cumulative sum along an axis. -/
def cumsumAxis {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) (axis : Nat) : TensorM (StaticTensor s d device) := do
  let _ := assertAxisInRange axis s.length
  let dim := listGetD s axis 0
  if dim == 0 then
    pure t
  else
    let keepShape := Shape.reduce s [axis] true
    let mut parts : List UOp := []
    for i in [:dim] do
      let prefBounds := axisSliceBoundsNow s axis 0 (i + 1)
      let prefShape := replaceDimNow s axis (i + 1)
      let prefU ← UOp.shrink t.uop prefBounds
      let prefTensor : StaticTensor prefShape d device := StaticTensor.ofUOp prefU (requiresGrad := t.requiresGrad)
      let partRed ← sumAxis prefTensor axis true
      let part : StaticTensor keepShape d device := StaticTensor.ofUOp partRed.uop (requiresGrad := t.requiresGrad)
      parts := parts ++ [part.uop]
    let out ← UOp.cat parts axis
    pure (StaticTensor.ofUOp out (requiresGrad := t.requiresGrad))

/-- Cumulative product along an axis. -/
def cumprodAxis {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) (axis : Nat) : TensorM (StaticTensor s d device) := do
  let _ := assertAxisInRange axis s.length
  let dim := listGetD s axis 0
  if dim == 0 then
    pure t
  else
    let keepShape := Shape.reduce s [axis] true
    let mut parts : List UOp := []
    for i in [:dim] do
      let prefBounds := axisSliceBoundsNow s axis 0 (i + 1)
      let prefShape := replaceDimNow s axis (i + 1)
      let prefU ← UOp.shrink t.uop prefBounds
      let prefTensor : StaticTensor prefShape d device := StaticTensor.ofUOp prefU (requiresGrad := t.requiresGrad)
      let partRed ← prodAxis prefTensor axis true
      let part : StaticTensor keepShape d device := StaticTensor.ofUOp partRed.uop (requiresGrad := t.requiresGrad)
      parts := parts ++ [part.uop]
    let out ← UOp.cat parts axis
    pure (StaticTensor.ofUOp out (requiresGrad := t.requiresGrad))

/-- Cumulative max along an axis. -/
def cummaxAxis {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) (axis : Nat) : TensorM (StaticTensor s d device) := do
  let _ := assertAxisInRange axis s.length
  let dim := listGetD s axis 0
  if dim == 0 then
    pure t
  else
    let keepShape := Shape.reduce s [axis] true
    let mut parts : List UOp := []
    for i in [:dim] do
      let prefBounds := axisSliceBoundsNow s axis 0 (i + 1)
      let prefShape := replaceDimNow s axis (i + 1)
      let prefU ← UOp.shrink t.uop prefBounds
      let prefTensor : StaticTensor prefShape d device := StaticTensor.ofUOp prefU (requiresGrad := t.requiresGrad)
      let partRed ← maxAxis prefTensor axis true
      let part : StaticTensor keepShape d device := StaticTensor.ofUOp partRed.uop (requiresGrad := t.requiresGrad)
      parts := parts ++ [part.uop]
    let out ← UOp.cat parts axis
    pure (StaticTensor.ofUOp out (requiresGrad := t.requiresGrad))

/-- Cumulative sum along the last axis. -/
def cumsum {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) : TensorM (StaticTensor s d device) :=
  cumsumAxis t (s.length - 1)

/-- Cumulative product along the last axis. -/
def cumprod {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) : TensorM (StaticTensor s d device) :=
  cumprodAxis t (s.length - 1)

/-- Cumulative max along the last axis. -/
def cummax {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) : TensorM (StaticTensor s d device) :=
  cummaxAxis t (s.length - 1)

/-- Layer norm over an axis (last axis by default). -/
def layerNorm {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (axis : Nat := s.length - 1)
    (eps : Float32 := 1.0e-5) : TensorM (StaticTensor s d device) := do
  let meanT ← meanAxis t axis true
  let centered ← UOp.sub t.uop meanT.uop
  let centeredT : StaticTensor s d device  := StaticTensor.ofUOp centered (requiresGrad := t.requiresGrad)
  let sq ← mul centeredT centeredT
  let varT ← meanAxis sq axis true
  let epsConst ← UOp.const d eps
  let varEps ← UOp.add varT.uop epsConst
  let std ← UOp.sqrt varEps
  let invStd ← UOp.recip std
  let out ← UOp.mul centered invStd
  pure (StaticTensor.ofUOp out (requiresGrad := t.requiresGrad))

/-- RMS norm over an axis (last axis by default). -/
def rmsNorm {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (axis : Nat := s.length - 1)
    (eps : Float32 := 1.0e-5) : TensorM (StaticTensor s d device) := do
  let sq ← mul t t
  let meanSq ← meanAxis sq axis true
  let epsConst ← UOp.const d eps
  let varEps ← UOp.add meanSq.uop epsConst
  let rms ← UOp.sqrt varEps
  let invRms ← UOp.recip rms
  let out ← UOp.mul t.uop invRms
  pure (StaticTensor.ofUOp out (requiresGrad := t.requiresGrad))

private def classRangeF32 (n : Nat) : Array Float32 := Id.run do
  let mut out := Array.emptyWithCapacity n
  for i in [:n] do
    out := out.push (Float64.ofNat i).toFloat32
  return out

private def resolveDim (dim rank : Nat) : Nat :=
  if dim < rank then dim else
    panic! s!"dim {dim} out of range for rank {rank}"

private def replaceDim (shape : Shape) (axis newDim : Nat) : Shape :=
  (listEnum shape).map fun p =>
    if p.1 == axis then newDim else p.2

private def axisSliceBounds (shape : Shape) (axis start stop : Nat) : List (Nat × Nat) :=
  (listEnum shape).map fun p =>
    if p.1 == axis then (start, stop) else (0, p.2)

private def swapLastPerm (rank dim : Nat) : List Nat :=
  let last := rank - 1
  (listRange rank).map fun i =>
    if i == dim then last else if i == last then dim else i

private def replaceLast (s : Shape) (n : Nat) : Shape :=
  if s.isEmpty then [n] else s.take (s.length - 1) ++ [n]

private def gatherShapeOk (shape idxShape : Shape) (dim : Nat) : Bool :=
  shape.length == idxShape.length &&
  listAll (fun i => if i == dim then true else listGetD shape i 0 >= listGetD idxShape i 0) (listRange shape.length)

private def oneHotLastF32 {s : Shape} {device : Backend.DeviceType}
    (idx : StaticTensor s .float32 device) (numClasses : Nat)
    : TensorM (StaticTensor (replaceLast s numClasses) .bool device) := do
  let classUop ← UOp.vconstF32 (classRangeF32 numClasses)
  let classes : StaticTensor [numClasses] .float32 device  := StaticTensor.ofUOp classUop
  let eq ← UOp.cmpeq idx.uop classes.uop
  pure (StaticTensor.ofUOp eq)

private def lastSliceShape (s : Shape) (_i : Nat) : Shape :=
  if s.isEmpty then [1] else s.take (s.length - 1) ++ [1]

/-- One-hot encoding for class indices (float32). -/
private def oneHotF32 {batch numClasses : Nat} {device : Backend.DeviceType}
    (targets : StaticTensor [batch] .float32 device)
    : TensorM (StaticTensor [batch, numClasses] .float32 device) := do
  let classUop ← UOp.vconstF32 (classRangeF32 numClasses)
  let classes : StaticTensor [numClasses] .float32 device  := StaticTensor.ofUOp classUop
  let targets2 ← reshape targets [batch, 1] (by simp [Shape.reshapeValid, Shape.numel, listProd])
  let classes2 ← reshape classes [1, numClasses] (by simp [Shape.reshapeValid, Shape.numel, listProd])
  let cmp ← UOp.cmpeq targets2.uop classes2.uop
  let one ← UOp.const .float32 1.0
  let zero ← UOp.const .float32 0.0
  let out ← UOp.where_ cmp one zero
  pure (StaticTensor.ofUOp out)

/-- Gather along an axis using index values (float32 indices). -/
private def gatherF32 {s idxShape : Shape} {device : Backend.DeviceType}
    (t : StaticTensor s .float32 device) (dim : Nat)
    (index : StaticTensor idxShape .float32 device)
    : TensorM (StaticTensor idxShape .float32 device) := do
  let tShape := s
  let idxShape' := idxShape
  let dim' := resolveDim dim tShape.length
  if !gatherShapeOk tShape idxShape' dim' then
    panic! s!"gather: invalid shapes {tShape} {idxShape'} for dim {dim'}"
  let bounds := (listRange tShape.length).map fun i =>
    if i == dim' then (0, listGetD tShape i 0) else (0, listGetD idxShape' i 0)
  let tShrunk ← shrinkUnsafe t bounds
  let tUnsq ← unsqueezeUnsafe tShrunk tShape.length
  let tPerm ← permuteUnsafe tUnsq (swapLastPerm (tShape.length + 1) dim')
  let idxUnsq ← unsqueezeUnsafe index idxShape'.length
  let numClasses := listGetD tShape dim' 0
  let oneHot ← oneHotLastF32 idxUnsq numClasses
  let zero ← UOp.const .float32 0.0
  let masked ← UOp.where_ oneHot.uop tPerm.uop zero
  let reduced ← UOp.sum masked [idxShape'.length] false
  pure (StaticTensor.ofUOp reduced (requiresGrad := t.requiresGrad))

/-- Gather along an axis using int32 indices. -/
def gather {s idxShape : Shape} {device : Backend.DeviceType}
    (t : StaticTensor s .float32 device) (dim : Nat)
    (index : StaticTensor idxShape .int32 device)
    : TensorM (StaticTensor idxShape .float32 device) := do
  let indexF ← cast index .float32
  gatherF32 t dim indexF

/-- Axis-typed gather: axis bounds are checked by the type system. -/
def gatherAxis {s idxShape : Shape} {device : Backend.DeviceType}
    (t : StaticTensor s .float32 device) (dim : Fin s.length)
    (index : StaticTensor idxShape .int32 device)
    : TensorM (StaticTensor idxShape .float32 device) :=
  gather t dim.1 index

/-- Gather along the last axis using class indices (float32). -/
private def gatherLastF32 {batch numClasses : Nat} {device : Backend.DeviceType}
    (x : StaticTensor [batch, numClasses] .float32 device)
    (targets : StaticTensor [batch] .float32 device)
    : TensorM (StaticTensor [batch] .float32 device) := do
  let targets2 ← reshape targets [batch, 1] (by simp [Shape.reshapeValid, Shape.numel, listProd])
  let gathered ← gatherF32 x 1 targets2
  reshape gathered [batch] (by simp [Shape.reshapeValid, Shape.numel, listProd])

def gatherLast {batch numClasses : Nat} {device : Backend.DeviceType}
    (x : StaticTensor [batch, numClasses] .float32 device)
    (targets : StaticTensor [batch] .int32 device)
    : TensorM (StaticTensor [batch] .float32 device) := do
  let targetsF ← cast targets .float32
  gatherLastF32 x targetsF

private def scatterLastF32 {batch numClasses : Nat} {device : Backend.DeviceType}
    (values : StaticTensor [batch] .float32 device)
    (targets : StaticTensor [batch] .float32 device)
    : TensorM (StaticTensor [batch, numClasses] .float32 device) := do
  let oneHot ← oneHotF32 targets
  let values2 ← reshape values [batch, 1] (by simp [Shape.reshapeValid, Shape.numel, listProd])
  let valuesB ← expand values2 [batch, numClasses] (by simp [Shape.expandValid, listAll])
  let out ← mul oneHot valuesB
  pure out

def scatterLast {batch numClasses : Nat} {device : Backend.DeviceType}
    (values : StaticTensor [batch] .float32 device)
    (targets : StaticTensor [batch] .int32 device)
    : TensorM (StaticTensor [batch, numClasses] .float32 device) := do
  let targetsF ← cast targets .float32
  scatterLastF32 values targetsF

inductive ScatterReduce where
  | sum
  | mean
  | amax
  | amin
  deriving Repr, DecidableEq

private def preScatterF32 {s idxShape srcShape : Shape} {device : Backend.DeviceType}
    (_self : StaticTensor s .float32 device) (dim : Nat)
    (index : StaticTensor idxShape .float32 device) (src : StaticTensor srcShape .float32 device)
    : TensorM
        (StaticTensor (s ++ [listGetD s dim 0]) .float32 device ×
         StaticTensor (s ++ [listGetD s dim 0]) .bool device) := do
  let selfShape := s
  let idxShape' := idxShape
  let srcShape' := srcShape
  let dim' := resolveDim dim selfShape.length
  if selfShape.length != idxShape'.length || selfShape.length != srcShape'.length then
    panic! s!"scatter: rank mismatch {selfShape} {idxShape'} {srcShape'}"
  let ok := listAll (fun i =>
    if i == dim' then true
    else (listGetD selfShape i 0 >= listGetD idxShape' i 0) &&
         (listGetD srcShape' i 0 >= listGetD idxShape' i 0)) (listRange selfShape.length)
  if !ok then
    panic! s!"scatter: invalid shapes {selfShape} {idxShape'} {srcShape'} for dim {dim'}"
  let srcBounds := (listRange srcShape'.length).map fun i => (0, listGetD idxShape' i 0)
  let srcShrunk ← shrinkUnsafe src srcBounds
  let srcUnsq ← unsqueezeUnsafe srcShrunk srcShape'.length
  let numClasses := listGetD selfShape dim' 0
  let srcExpandedShape := idxShape' ++ [numClasses]
  let srcExpanded ← expandUnsafe srcUnsq srcExpandedShape
  let perm := swapLastPerm (selfShape.length + 1) dim'
  let srcTShape := Shape.permute srcExpandedShape perm
  let srcT ← permuteUnsafe srcExpanded perm
  let idxUnsq ← unsqueezeUnsafe index idxShape'.length
  let maskT ← oneHotLastF32 idxUnsq numClasses
  let maskP ← permuteUnsafe maskT perm
  let padSpec := (listRange selfShape.length).map fun i =>
    if i == dim' then (0, 0)
    else
      let need := listGetD selfShape i 0
      let haveDim := listGetD srcTShape i 0
      if need < haveDim then (0, 0) else (0, need - haveDim)
  let padSpec := padSpec ++ [(0, 0)]
  let srcP ← padUnsafe srcT padSpec
  let maskP ← padUnsafe maskP padSpec
  let srcOut : StaticTensor (s ++ [listGetD s dim 0]) .float32 device  := StaticTensor.ofUOp srcP.uop
  let maskOut : StaticTensor (s ++ [listGetD s dim 0]) .bool device  := StaticTensor.ofUOp maskP.uop
  pure (srcOut, maskOut)

private def sliceLast {s : Shape} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) (i : Nat)
    : TensorM (StaticTensor (lastSliceShape s i) d device) := do
  let shape := s
  if shape.isEmpty then
    panic! "sliceLast: empty shape"
  let last := shape.length - 1
  let bounds := (listRange last).map fun j => (0, listGetD shape j 0)
  let bounds := bounds ++ [(i, i + 1)]
  let sliced ← shrinkUnsafe t bounds
  pure (StaticTensor.ofUOp sliced.uop (requiresGrad := t.requiresGrad))

private def squeezeLast {s : Shape} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) : TensorM (StaticTensor (s.take (s.length - 1)) d device) := do
  let shape := s
  if shape.isEmpty then
    panic! "squeezeLast: empty shape"
  let newShape := shape.take (shape.length - 1)
  let reshaped ← UOp.reshape t.uop newShape
  pure (StaticTensor.ofUOp reshaped (requiresGrad := t.requiresGrad))

private def maskedSetitemLast {s vShape : Shape} {device : Backend.DeviceType}
    (target : StaticTensor s .float32 device)
    (values : StaticTensor vShape .float32 device)
    (mask : StaticTensor vShape .bool device)
    : TensorM (StaticTensor s .float32 device) := do
  let shape := vShape
  if shape.isEmpty then
    panic! "maskedSetitemLast: empty shape"
  let lastDim := listGetD shape (shape.length - 1) 0
  if lastDim == 0 then
    pure target
  else
    let accVal0 ← sliceLast values 0
    let accMask0 ← sliceLast mask 0
    let mut accVal := accVal0
    let mut accMask := accMask0
    if lastDim > 1 then
      for i in [:lastDim] do
        if i == 0 then
          pure ()
        else
          let vi ← sliceLast values i
          let mi ← sliceLast mask i
          let accVal' ← whereSame mi vi accVal
          let accMask' ← bitor accMask mi
          accVal := accVal'
          accMask := accMask'
    let accValS ← squeezeLast accVal
    let accMaskS ← squeezeLast accMask
    let accValOut : StaticTensor s .float32 device  := StaticTensor.ofUOp accValS.uop
    let accMaskOut : StaticTensor s .bool device  := StaticTensor.ofUOp accMaskS.uop
    whereSame accMaskOut accValOut target

private def scatterReduceF32 {s idxShape srcShape : Shape} {device : Backend.DeviceType}
    (self : StaticTensor s .float32 device) (dim : Nat)
    (index : StaticTensor idxShape .float32 device) (src : StaticTensor srcShape .float32 device)
    (reduce : ScatterReduce) (includeSelf : Bool := true)
    : TensorM (StaticTensor s .float32 device) := do
  let (srcP, maskP) ← preScatterF32 self dim index src
  let lastAxis := s.length
  let zero ← UOp.const .float32 0.0
  let one ← UOp.const .float32 1.0
  let maskOnes ← UOp.where_ maskP.uop one zero
  let countU ← UOp.sum maskOnes [lastAxis] false
  let noUpdateU ← UOp.cmpeq countU zero
  let noUpdate : StaticTensor s .bool device  := StaticTensor.ofUOp noUpdateU
  let maskedU ← UOp.where_ maskP.uop srcP.uop zero
  let sumU ← UOp.sum maskedU [lastAxis] false
  let sumT : StaticTensor s .float32 device  := StaticTensor.ofUOp sumU
  match reduce with
  | .sum =>
    if includeSelf then
      add sumT self
    else
      let invU ← UOp.where_ noUpdate.uop self.uop zero
      let invT : StaticTensor s .float32 device  := StaticTensor.ofUOp invU
      add sumT invT
  | .mean =>
    let baseNum ←
      if includeSelf then
        add sumT self
      else
        let invU ← UOp.where_ noUpdate.uop self.uop zero
        let invT : StaticTensor s .float32 device  := StaticTensor.ofUOp invU
        add sumT invT
    let countAddU ←
      if includeSelf then
        UOp.add countU one
      else
        let addOneU ← UOp.where_ noUpdate.uop one zero
        UOp.add countU addOneU
    let countT : StaticTensor s .float32 device  := StaticTensor.ofUOp countAddU
    div baseNum countT
  | .amax =>
    let negInf ← UOp.const .float32 (-1.0e38)
    let maskedMax ← UOp.where_ maskP.uop srcP.uop negInf
    let maxU ← UOp.max_ maskedMax [lastAxis] false
    let maxT : StaticTensor s .float32 device  := StaticTensor.ofUOp maxU
    if includeSelf then
      let outU ← UOp.maxBinary maxT.uop self.uop
      pure (StaticTensor.ofUOp outU)
    else
      let outU ← UOp.where_ noUpdate.uop self.uop maxT.uop
      pure (StaticTensor.ofUOp outU)
  | .amin =>
    let negInf ← UOp.const .float32 (-1.0e38)
    let negSrc ← neg srcP
    let negSelf ← neg self
    let maskedNeg ← UOp.where_ maskP.uop negSrc.uop negInf
    let maxNegU ← UOp.max_ maskedNeg [lastAxis] false
    let maxNegT : StaticTensor s .float32 device  := StaticTensor.ofUOp maxNegU
    let mergedNegU ←
      if includeSelf then
        UOp.maxBinary maxNegT.uop negSelf.uop
      else
        UOp.where_ noUpdate.uop negSelf.uop maxNegT.uop
    let mergedNegT : StaticTensor s .float32 device  := StaticTensor.ofUOp mergedNegU
    neg mergedNegT

def scatterReduce {s idxShape srcShape : Shape} {device : Backend.DeviceType}
    (self : StaticTensor s .float32 device) (dim : Nat)
    (index : StaticTensor idxShape .int32 device) (src : StaticTensor srcShape .float32 device)
    (reduce : ScatterReduce) (includeSelf : Bool := true)
    : TensorM (StaticTensor s .float32 device) := do
  let indexF ← cast index .float32
  scatterReduceF32 self dim indexF src reduce includeSelf

/-- Axis-typed scatterReduce: axis bounds are checked by the type system. -/
def scatterReduceAxis {s idxShape srcShape : Shape} {device : Backend.DeviceType}
    (self : StaticTensor s .float32 device) (dim : Fin s.length)
    (index : StaticTensor idxShape .int32 device) (src : StaticTensor srcShape .float32 device)
    (reduce : ScatterReduce) (includeSelf : Bool := true)
    : TensorM (StaticTensor s .float32 device) :=
  scatterReduce self dim.1 index src reduce includeSelf

private def scatterF32 {s idxShape srcShape : Shape} {device : Backend.DeviceType}
    (self : StaticTensor s .float32 device) (dim : Nat)
    (index : StaticTensor idxShape .float32 device) (src : StaticTensor srcShape .float32 device)
    : TensorM (StaticTensor s .float32 device) := do
  let (srcP, maskP) ← preScatterF32 self dim index src
  maskedSetitemLast self srcP maskP

def scatter {s idxShape srcShape : Shape} {device : Backend.DeviceType}
    (self : StaticTensor s .float32 device) (dim : Nat)
    (index : StaticTensor idxShape .int32 device) (src : StaticTensor srcShape .float32 device)
    : TensorM (StaticTensor s .float32 device) := do
  let indexF ← cast index .float32
  scatterF32 self dim indexF src

/-- Axis-typed scatter: axis bounds are checked by the type system. -/
def scatterAxis {s idxShape srcShape : Shape} {device : Backend.DeviceType}
    (self : StaticTensor s .float32 device) (dim : Fin s.length)
    (index : StaticTensor idxShape .int32 device) (src : StaticTensor srcShape .float32 device)
    : TensorM (StaticTensor s .float32 device) :=
  scatter self dim.1 index src

/-- Log-sum-exp along axis (numerically stable). -/
def logsumexpAxis {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (axis : Nat) (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis] keepdim) d device) := do
  let maxVal ← UOp.max_ t.uop [axis] true
  let shifted ← UOp.sub t.uop maxVal
  let shiftedT : StaticTensor s d device  := StaticTensor.ofUOp shifted (requiresGrad := t.requiresGrad)
  let expShifted ← exp shiftedT
  let sumExp ← UOp.sum expShifted.uop [axis] true
  let sumExpT : StaticTensor (Shape.reduce s [axis] true) d device  := StaticTensor.ofUOp sumExp (requiresGrad := t.requiresGrad)
  let logSum ← log sumExpT
  let outKeep ← UOp.add logSum.uop maxVal
  match keepdim with
  | true =>
    pure (StaticTensor.ofUOp outKeep (requiresGrad := t.requiresGrad))
  | false =>
    let outKeepT : StaticTensor (Shape.reduce s [axis] true) d device  := StaticTensor.ofUOp outKeep (requiresGrad := t.requiresGrad)
    reshape outKeepT (Shape.reduce s [axis] false) (by
      simpa [Shape.reshapeValid, Shape.numel] using Shape.reduce_single_numel_eq s axis)

/-- Log-sum-exp along axis (numerically stable, statically checked axis). -/
def logsumexpAxisF {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (axis : Fin s.length)
    (keepdim : Bool := true)
    : TensorM (StaticTensor (Shape.reduce s [axis.val] keepdim) d device) := do
  let hAxes : [axis.val].all (fun ax => ax < t.uop.shape.length) = true := by
    simpa [t.h_shape] using axis.isLt
  let maxVal ← UOp.reduceValid t.uop .MAX [axis.val] true hAxes
  let shifted ← UOp.sub t.uop maxVal
  let shiftedT : StaticTensor s d device := StaticTensor.ofUOp shifted (requiresGrad := t.requiresGrad)
  let expShifted ← exp shiftedT
  let hAxesExp : [axis.val].all (fun ax => ax < expShifted.uop.shape.length) = true := by
    simpa [expShifted.h_shape] using axis.isLt
  let sumExp ← UOp.reduceValid expShifted.uop .ADD [axis.val] true hAxesExp
  let sumExpT : StaticTensor (Shape.reduce s [axis.val] true) d device := StaticTensor.ofUOp sumExp (requiresGrad := t.requiresGrad)
  let logSum ← log sumExpT
  let outKeep ← UOp.add logSum.uop maxVal
  match keepdim with
  | true =>
    pure (StaticTensor.ofUOp outKeep (requiresGrad := t.requiresGrad))
  | false =>
    let outKeepT : StaticTensor (Shape.reduce s [axis.val] true) d device := StaticTensor.ofUOp outKeep (requiresGrad := t.requiresGrad)
    reshape outKeepT (Shape.reduce s [axis.val] false) (by
      simpa [Shape.reshapeValid, Shape.numel] using Shape.reduce_single_numel_eq s axis.val)

/-- Cumulative log-sum-exp along an axis. -/
def logcumsumexpAxis {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) (axis : Nat) : TensorM (StaticTensor s d device) := do
  let _ := assertAxisInRange axis s.length
  let dim := listGetD s axis 0
  if dim == 0 then
    pure t
  else
    let keepShape := Shape.reduce s [axis] true
    let mut parts : List UOp := []
    for i in [:dim] do
      let prefBounds := axisSliceBoundsNow s axis 0 (i + 1)
      let prefShape := replaceDimNow s axis (i + 1)
      let prefU ← UOp.shrink t.uop prefBounds
      let prefTensor : StaticTensor prefShape d device := StaticTensor.ofUOp prefU (requiresGrad := t.requiresGrad)
      let partRed ← logsumexpAxis prefTensor axis true
      let part : StaticTensor keepShape d device := StaticTensor.ofUOp partRed.uop (requiresGrad := t.requiresGrad)
      parts := parts ++ [part.uop]
    let out ← UOp.cat parts axis
    pure (StaticTensor.ofUOp out (requiresGrad := t.requiresGrad))

/-- Cumulative log-sum-exp along the last axis. -/
def logcumsumexp {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) : TensorM (StaticTensor s d device) :=
  logcumsumexpAxis t (s.length - 1)

/-- Log-sum-exp over all elements. -/
def logsumexp {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (t : StaticTensor s d device) : TensorM (Scalar d device) := do
  let flat ← flatten t
  let out ← logsumexpAxis flat 0 false
  pure out

/-- Log-softmax along an axis (stable). -/
def logSoftmaxAxis {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (axis : Nat) : TensorM (StaticTensor s d device) := do
  let logSum ← logsumexpAxis t axis true
  let result ← UOp.sub t.uop logSum.uop
  pure (StaticTensor.ofUOp result (requiresGrad := t.requiresGrad))

/-- Log-softmax along an axis (stable, statically checked axis). -/
def logSoftmaxAxisF {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (axis : Fin s.length)
    : TensorM (StaticTensor s d device) := do
  let logSum ← logsumexpAxisF t axis true
  let result ← UOp.sub t.uop logSum.uop
  pure (StaticTensor.ofUOp result (requiresGrad := t.requiresGrad))

/-- Softmax along an axis (stable). -/
def softmaxAxis {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (axis : Nat) : TensorM (StaticTensor s d device) := do
  let maxVal ← UOp.max_ t.uop [axis] true
  let shifted ← UOp.sub t.uop maxVal
  let shiftedT : StaticTensor s d device  := StaticTensor.ofUOp shifted (requiresGrad := t.requiresGrad)
  let expShifted ← exp shiftedT
  let sumExp ← UOp.sum expShifted.uop [axis] true
  let out ← UOp.div expShifted.uop sumExp
  pure (StaticTensor.ofUOp out (requiresGrad := t.requiresGrad))

/-- Softmax along an axis (stable, statically checked axis). -/
def softmaxAxisF {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (axis : Fin s.length)
    : TensorM (StaticTensor s d device) := do
  let hAxes : [axis.val].all (fun ax => ax < t.uop.shape.length) = true := by
    simpa [t.h_shape] using axis.isLt
  let maxVal ← UOp.reduceValid t.uop .MAX [axis.val] true hAxes
  let shifted ← UOp.sub t.uop maxVal
  let shiftedT : StaticTensor s d device := StaticTensor.ofUOp shifted (requiresGrad := t.requiresGrad)
  let expShifted ← exp shiftedT
  let hAxesExp : [axis.val].all (fun ax => ax < expShifted.uop.shape.length) = true := by
    simpa [expShifted.h_shape] using axis.isLt
  let sumExp ← UOp.reduceValid expShifted.uop .ADD [axis.val] true hAxesExp
  let out ← UOp.div expShifted.uop sumExp
  pure (StaticTensor.ofUOp out (requiresGrad := t.requiresGrad))

/-- Softmax along last axis: exp(x - max(x)) / sum(exp(x - max(x))) -/
def softmax {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  softmaxAxis t (s.length - 1)

/-- Log-softmax along last axis: x - max(x) - log(sum(exp(x - max(x)))) -/
def logSoftmax {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) : TensorM (StaticTensor s d device) := do
  logSoftmaxAxis t (s.length - 1)

private def argmaxF32 {batch n : Nat} {device : Backend.DeviceType} (t : StaticTensor [batch, n] .float32 device)
    : TensorM (StaticTensor [batch] .int32 device) := do
  let maxVal ← maxAxisF t ⟨1, by simp⟩ true
  let eq ← UOp.cmpeq t.uop maxVal.uop
  let eqT : StaticTensor [batch, n] .bool device  := StaticTensor.ofUOp eq
  let eqF ← cast eqT .float32
  let classesUop ← UOp.vconstF32 (classRangeF32 n)
  let classes : StaticTensor [n] .float32 device  := StaticTensor.ofUOp classesUop
  let classes2 ← reshape classes [1, n] (by simp [Shape.reshapeValid, Shape.numel, listProd])
  let classesB ← expand classes2 [batch, n] (by simp [Shape.expandValid, listAll])
  let prod ← mul eqF classesB
  let sumC ← sumAxisF prod ⟨1, by simp⟩ false
  cast sumC .int32

/-- Argmax along last axis - returns indices (non-differentiable). -/
def argmax {batch n : Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor [batch, n] d device)
    : TensorM (StaticTensor [batch] .int32 device) := do
  let tF ← cast t .float32
  argmaxF32 tF

/-- Argmin along last axis - returns indices (non-differentiable). -/
def argmin {batch n : Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor [batch, n] d device)
    : TensorM (StaticTensor [batch] .int32 device) := do
  let tF ← cast t .float32
  let negT ← neg tF
  argmaxF32 negT

/-- Scalar multiplication: t * scalar -/
def scale {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (scalar : Float32)
    : TensorM (StaticTensor s d device) := do
  let scalarUop ← UOp.const d scalar
  let result ← UOp.mul t.uop scalarUop
  pure (StaticTensor.ofUOp result (requiresGrad := t.requiresGrad))

/-- Add scalar: t + scalar -/
def addScalar {s : List Nat} {d : DType} {device : Backend.DeviceType} (t : StaticTensor s d device) (scalar : Float32)
    : TensorM (StaticTensor s d device) := do
  let scalarUop ← UOp.const d scalar
  let result ← UOp.add t.uop scalarUop
  pure (StaticTensor.ofUOp result (requiresGrad := t.requiresGrad))

/-- Cross-entropy loss with log-softmax
    logits: [batch, numClasses], targets: [batch] (class indices)
    Returns scalar loss -/
def crossEntropyLoss {batch numClasses : Nat} {device : Backend.DeviceType}
    (logits : StaticTensor [batch, numClasses] .float32 device)
    (targets : StaticTensor [batch] .int32 device)
    : TensorM (Scalar .float32 device) := do
  let logProbs ← logSoftmax logits
  let picked ← gatherLast logProbs targets
  let negPicked ← neg picked
  mean negPicked

/-- Cross-entropy loss with one-hot targets.
    logits: [batch, numClasses], targets: [batch, numClasses] (one-hot)
    Returns scalar loss: mean over batch of -sum(target * log_softmax(logits)) -/
def crossEntropyOneHot {batch numClasses : Nat} {d : DType} {device : Backend.DeviceType}
    (logits : StaticTensor [batch, numClasses] d device)
    (targets : StaticTensor [batch, numClasses] d device)
    : TensorM (Scalar d device) := do
  let logProbs ← logSoftmax logits
  let prod ← mul logProbs targets
  let sumC ← sumAxisF prod ⟨1, by simp⟩ true
  let negSum ← neg sumC
  mean negSum

/-- Negative log likelihood loss (assumes log_softmax input)
    log_probs: [batch, numClasses], already log-softmax'd
    target_indices: [batch] containing class indices
    For MVP: averages all log probs (placeholder until gather/index support) -/
def nllLoss {batch numClasses : Nat} {device : Backend.DeviceType}
    (logProbs : StaticTensor [batch, numClasses] .float32 device)
    (targets : StaticTensor [batch] .int32 device)
    : TensorM (Scalar .float32 device) := do
  let picked ← gatherLast logProbs targets
  let negPicked ← neg picked
  mean negPicked

/-- Smooth L1 (Huber) loss with beta (default 1.0). -/
def smoothL1Loss {s : List Nat} {device : Backend.DeviceType} (pred target : StaticTensor s .float32 device) (beta : Float32 := 1.0)
    : TensorM (Scalar .float32 device) := do
  let diff ← sub pred target
  let absDiff ← abs diff
  let betaConst ← UOp.const .float32 beta
  let half ← UOp.const .float32 0.5
  let isSmall ← UOp.cmplt absDiff.uop betaConst
  let sq ← mul diff diff
  let sqHalf ← UOp.mul sq.uop half
  let sqScaled ← UOp.div sqHalf betaConst
  let betaHalf ← UOp.mul betaConst half
  let linTerm ← UOp.sub absDiff.uop betaHalf
  let isSmallT : StaticTensor s .bool device := StaticTensor.ofUOp isSmall
  let sqScaledT : StaticTensor s .float32 device := StaticTensor.ofUOp sqScaled (requiresGrad := pred.requiresGrad || target.requiresGrad)
  let linTermT : StaticTensor s .float32 device := StaticTensor.ofUOp linTerm (requiresGrad := pred.requiresGrad || target.requiresGrad)
  let outT ← whereSame isSmallT sqScaledT linTermT
  mean outT

/-- Binary cross-entropy loss (expects probabilities in [0, 1]). -/
def binaryCrossEntropy {s : List Nat} {device : Backend.DeviceType}
    (pred target : StaticTensor s .float32 device) (eps : Float32 := 1.0e-7)
    : TensorM (Scalar .float32 device) := do
  let predClamped ← clamp pred eps (1.0 - eps)
  let logPred ← log predClamped
  let one ← UOp.const .float32 1.0
  let oneMinusPred ← UOp.sub one predClamped.uop
  let oneMinusPredT : StaticTensor s .float32 device  := StaticTensor.ofUOp oneMinusPred (requiresGrad := pred.requiresGrad)
  let logOneMinusPred ← log oneMinusPredT
  let oneMinusTarget ← UOp.sub one target.uop
  let oneMinusTargetT : StaticTensor s .float32 device  := StaticTensor.ofUOp oneMinusTarget (requiresGrad := target.requiresGrad)
  let term1 ← mul target logPred
  let term2 ← mul oneMinusTargetT logOneMinusPred
  let sumTerms ← add term1 term2
  let negSum ← neg sumTerms
  mean negSum

/-- Binary cross-entropy with logits (numerically stable). -/
def binaryCrossEntropyWithLogits {s : List Nat} {device : Backend.DeviceType}
    (logits target : StaticTensor s .float32 device) : TensorM (Scalar .float32 device) := do
  let zero ← UOp.const .float32 0.0
  let maxZero ← UOp.maxBinary logits.uop zero
  let absLogits ← abs logits
  let negAbs ← neg absLogits
  let expNegAbs ← exp negAbs
  let one ← UOp.const .float32 1.0
  let onePlus ← UOp.add expNegAbs.uop one
  let onePlusT : StaticTensor s .float32 device  := StaticTensor.ofUOp onePlus (requiresGrad := logits.requiresGrad)
  let logOnePlus ← log onePlusT
  let prod ← UOp.mul logits.uop target.uop
  let tmp ← UOp.sub maxZero prod
  let lossUop ← UOp.add tmp logOnePlus.uop
  let lossT : StaticTensor s .float32 device  := StaticTensor.ofUOp lossUop (requiresGrad := logits.requiresGrad)
  mean lossT

/-- Mean squared error loss. -/
def mseLoss {s : List Nat} {d : DType} {device : Backend.DeviceType}
    (pred target : StaticTensor s d device) : TensorM (Scalar d device) := do
  let diff ← sub pred target
  let sq ← mul diff diff
  mean sq

/-- Matrix multiplication: [m, k] @ [k, n] -> [m, n] -/
def matmul {m k n : Nat} {d : DType} {device : Backend.DeviceType}
    (a : Matrix m k d device) (b : Matrix k n d device)
    : TensorM (Matrix m n d device) := do
  let hMatmul0 : Shape.matmulShape [m, k] [k, n] = some [m, n] := by
    have hk : listGetD [m, k] 1 0 = listGetD [k, n] 0 0 := by simp [listGetD]
    have hm : listGetD [m, k] 0 0 = m := by simp [listGetD]
    have hn : listGetD [k, n] 1 0 = n := by simp [listGetD]
    simp [Shape.matmulShape, Shape.broadcast, Shape.broadcastable, listAll, hk, hm, hn]
  let hMatmul : Shape.matmulShape a.uop.shape b.uop.shape = some [m, n] := by
    simpa [a.h_shape, b.h_shape] using hMatmul0
  let outUop ← UOp.contract2DValid a.uop b.uop [m, n] hMatmul
  pure (StaticTensor.ofUOp outUop (requiresGrad := a.requiresGrad || b.requiresGrad))

/-- Fully-connected (linear) layer: X @ W -> [batch, out]. -/
def linear {batch inDim outDim : Nat} {d : DType} {device : Backend.DeviceType}
    (x : Matrix batch inDim d device) (w : Matrix inDim outDim d device)
    : TensorM (Matrix batch outDim d device) := do
  matmul x w

/-- Fully-connected layer with optional bias: X @ W (+ b) -/
def linearOpt {batch inDim outDim : Nat} {d : DType} {device : Backend.DeviceType}
    (x : Matrix batch inDim d device) (w : Matrix inDim outDim d device)
    (bias : Option (Vector outDim d device) := none)
    : TensorM (Matrix batch outDim d device) := do
  let y ← matmul x w
  match bias with
  | none => pure y
  | some b => addVector y b

/-- Fully-connected layer with bias: X @ W + b (broadcasted over batch). -/
def linearBias {batch inDim outDim : Nat} {d : DType} {device : Backend.DeviceType}
    (x : Matrix batch inDim d device) (w : Matrix inDim outDim d device) (b : Vector outDim d device)
    : TensorM (Matrix batch outDim d device) := do
  linearOpt x w (some b)

/-- Batched matrix multiplication with broadcast on the batch dim:
    [b1, m, k] @ [b2, k, n] -> [max b1 b2, m, n]. -/
def bmatmul {b1 b2 m k n : Nat} {d : DType} {device : Backend.DeviceType}
    (a : BMatrix b1 m k d device) (b : BMatrix b2 k n d device)
    (hBatch : Shape.broadcastable [b1] [b2] = true)
    : TensorM (BMatrix (Nat.max b1 b2) m n d device) := do
  let hMatmul0 : Shape.matmulShape [b1, m, k] [b2, k, n] = some [Nat.max b1 b2, m, n] := by
    have hk : listGetD [b1, m, k] 2 0 = listGetD [b2, k, n] 1 0 := by simp [listGetD]
    have hm : listGetD [b1, m, k] 1 0 = m := by simp [listGetD]
    have hn : listGetD [b2, k, n] 2 0 = n := by simp [listGetD]
    have hBroad : Shape.broadcast [b1] [b2] = some [Nat.max b1 b2] := by
      simp [Shape.broadcast, hBatch]
    simp [Shape.matmulShape, hk, hm, hn, hBroad]
  let hMatmul : Shape.matmulShape a.uop.shape b.uop.shape = some [Nat.max b1 b2, m, n] := by
    simpa [a.h_shape, b.h_shape] using hMatmul0
  let outUop ← UOp.contract2DValid a.uop b.uop [Nat.max b1 b2, m, n] hMatmul
  pure (StaticTensor.ofUOp outUop (requiresGrad := a.requiresGrad || b.requiresGrad))

-- ============================================================================
-- Initialization
-- ============================================================================

/-- Generate uniform random tensor in [low, high) range.
    Uses: rand * (high - low) + low -/
def uniformInit (device : Backend.DeviceType := .CPU) (shape : Shape) (dt : DType) (low high : Float32) (seed : Nat)
    : TensorM (StaticTensor shape dt device) := do
  -- rand produces [0, 1)
  let r ← Tensor.rand (device := device) shape dt seed
  -- Scale to [low, high): r * (high - low) + low
  let range := high - low
  let rangeT ← Tensor.full (device := device) shape dt range
  let lowT ← Tensor.full (device := device) shape dt low
  let scaled ← mul r rangeT
  add scaled lowT

-- ============================================================================
-- Convolution Operations (pool/im2col + matmul)
-- ============================================================================

/-- Placeholder conv2d - returns correctly shaped output tensor.
    Full implementation requires UOp-level pool operation. -/
def conv2dPlaceholder {batch cin cout h w kH kW hOut wOut : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, cin, h, w] d device)
    (weight : StaticTensor [cout, cin, kH, kW] d device)
    (bias : Option (StaticTensor [cout] d device) := none)
    (_padding : Nat := 0)
    (_stride : Nat := 1)
    (_dilation : Nat := 1)
    : TensorM (StaticTensor [batch, cout, hOut, wOut] d device) := do
  -- Create output buffer with correct shape
  let outShape := [batch, cout, hOut, wOut]
  let out ← UOp.buffer d outShape
  let biasGrad := match bias with | none => false | some b => b.requiresGrad
  let reqGrad := x.requiresGrad || weight.requiresGrad || biasGrad
  pure (StaticTensor.ofUOp out (requiresGrad := reqGrad))

/-- Placeholder maxPool2d - returns correctly shaped output tensor. -/
def maxPool2dPlaceholder {batch cin h w hOut wOut : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, cin, h, w] d device)
    (_kernelSize : Nat)
    (_stride : Nat := 0)
    : TensorM (StaticTensor [batch, cin, hOut, wOut] d device) := do
  let outShape := [batch, cin, hOut, wOut]
  let out ← UOp.buffer d outShape
  pure (StaticTensor.ofUOp out (requiresGrad := x.requiresGrad))

/-- Placeholder avgPool2d - returns correctly shaped output tensor. -/
def avgPool2dPlaceholder {batch cin h w hOut wOut : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, cin, h, w] d device)
    (_kernelSize : Nat)
    (_stride : Nat := 0)
    : TensorM (StaticTensor [batch, cin, hOut, wOut] d device) := do
  let outShape := [batch, cin, hOut, wOut]
  let out ← UOp.buffer d outShape
  pure (StaticTensor.ofUOp out (requiresGrad := x.requiresGrad))

/-- Pad 1D tensor with symmetric padding on W dimension.
    Input:  [batch, channels, width]
    Output: [batch, channels, width + padW + padW] -/
def pad1d {batch cin w : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, cin, w] d device)
    (padW : Nat)
    : TensorM (StaticTensor [batch, cin, w + padW + padW] d device) := do
  let result ← pad x [(0, 0), (0, 0), (padW, padW)] (by simp [Shape.padValid])
  have out : StaticTensor [batch, cin, w + padW + padW] d device := by
    simpa [Shape.pad, listZipWith, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using result
  pure out

/-- Pad 2D tensor with symmetric padding on H and W dimensions.
    Input:  [batch, channels, height, width]
    Output: [batch, channels, height + padH + padH, width + padW + padW] -/
def pad2d {batch cin h w : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, cin, h, w] d device)
    (padH padW : Nat)
    : TensorM (StaticTensor [batch, cin, h + padH + padH, w + padW + padW] d device) := do
  let result ← pad x [(0, 0), (0, 0), (padH, padH), (padW, padW)] (by simp [Shape.padValid])
  have out : StaticTensor [batch, cin, h + padH + padH, w + padW + padW] d device := by
    simpa [Shape.pad, listZipWith, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using result
  pure out

/-- Max pooling 2D operation using pool/im2col + reduce.
    Input:  [batch, channels, height, width]
    Output: [batch, channels, outHeight, outWidth] -/
def maxPool2d {batch cin h w : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, cin, h, w] d device)
    (kernelSize : Nat)
    (stride : Nat)
    (padding : Nat := 0)
    : TensorM (StaticTensor (Shape.pool2dShape [batch, cin, h, w] kernelSize padding stride) d device) := do
  -- Step 1: Pad if needed
  let xPadded : StaticTensor [batch, cin, h + padding + padding, w + padding + padding] d device ←
    if hpad : padding > 0 then
      pad2d x padding padding
    else
      have hzero : padding = 0 := Nat.eq_zero_of_not_pos hpad
      have hx : StaticTensor [batch, cin, h + padding + padding, w + padding + padding] d device := by
        simpa [hzero, Nat.add_assoc] using x
      pure hx

  -- Step 2: Apply pool/im2col to get patches
  -- Result shape: [batch, cin, hOut, wOut, kH, kW]
  let patches ← pool xPadded [kernelSize, kernelSize] [stride, stride] [1, 1]

  -- Step 3: Take max over the kernel dimensions (last 2 dims)
  -- Reduce over axis -1 (kW) then axis -1 (kH)
  let patchShape := Shape.poolOut [batch, cin, h, w] [kernelSize, kernelSize] [stride, stride] [1, 1]
  let axis1 := patchShape.length - 1  -- kW axis
  let reduced1 ← UOp.max_ patches.uop [axis1] false
  let axis2 := patchShape.length - 2  -- kH axis (now shifted)
  let reduced2 ← UOp.max_ reduced1 [axis2] false

  pure (StaticTensor.ofUOp reduced2 (requiresGrad := x.requiresGrad))

/-- Average pooling 2D operation using pool/im2col + reduce.
    Input:  [batch, channels, height, width]
    Output: [batch, channels, outHeight, outWidth] -/
def avgPool2d {batch cin h w : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, cin, h, w] d device)
    (kernelSize : Nat)
    (stride : Nat)
    (padding : Nat := 0)
    : TensorM (StaticTensor (Shape.pool2dShape [batch, cin, h, w] kernelSize padding stride) d device) := do
  -- Step 1: Pad if needed
  let xPadded : StaticTensor [batch, cin, h + padding + padding, w + padding + padding] d device ←
    if hpad : padding > 0 then
      pad2d x padding padding
    else
      have hzero : padding = 0 := Nat.eq_zero_of_not_pos hpad
      have hx : StaticTensor [batch, cin, h + padding + padding, w + padding + padding] d device := by
        simpa [hzero, Nat.add_assoc] using x
      pure hx

  -- Step 2: Apply pool/im2col to get patches
  -- Result shape: [batch, cin, hOut, wOut, kH, kW]
  let patches ← pool xPadded [kernelSize, kernelSize] [stride, stride] [1, 1]

  -- Step 3: Take mean over the kernel dimensions (last 2 dims)
  let patchShape := Shape.poolOut [batch, cin, h, w] [kernelSize, kernelSize] [stride, stride] [1, 1]
  let axis1 := patchShape.length - 1  -- kW axis
  let sum1 ← UOp.sum patches.uop [axis1] false
  let axis2 := patchShape.length - 2  -- kH axis (now shifted)
  let sum2 ← UOp.sum sum1 [axis2] false

  -- Divide by kernel area for mean
  let kernelArea := (kernelSize * kernelSize : Nat)
  let divisor ← UOp.const d (Float64.ofNat kernelArea).toFloat32
  let result ← UOp.div sum2 divisor

  pure (StaticTensor.ofUOp result (requiresGrad := x.requiresGrad))

/-- Conv1d operation using pool/im2col + matmul.
    Input:  [batch, inChannels, width]
    Weight: [outChannels, inChannels, kernelW]
    Output: [batch, outChannels, outWidth] -/
def conv1d {batch cin cout w kW : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, cin, w] d device)
    (weight : StaticTensor [cout, cin, kW] d device)
    (bias : Option (StaticTensor [cout] d device) := none)
    (padding : Nat := 0)
    (stride : Nat := 1)
    (dilation : Nat := 1)
    : TensorM (StaticTensor (Shape.conv1dOut [batch, cin, w]
                                              [cout, cin, kW]
                                              padding stride dilation) d device) := do
  let wOut := Shape.convOutDim w padding dilation kW stride

  -- Step 1: Pad input if needed
  let xPadded : StaticTensor [batch, cin, w + padding + padding] d device ←
    if hpad : padding > 0 then
      pad1d x padding
    else
      have hzero : padding = 0 := Nat.eq_zero_of_not_pos hpad
      have hx : StaticTensor [batch, cin, w + padding + padding] d device := by
        simpa [hzero, Nat.add_assoc] using x
      pure hx

  -- Step 2: Apply pool/im2col to get patches
  -- Input: [batch, cin, wPadded]
  -- Output: [batch, cin, wOut, kW]
  let patches ← pool xPadded [kW] [stride] [dilation]

  -- Step 3: Reshape patches for matmul
  -- [batch, cin, wOut, kW] -> [batch * wOut, cin * kW]
  let patchFlat := batch * wOut
  let kernelFlat := cin * kW
  let patchesReshaped ← reshape patches [patchFlat, kernelFlat] (by
    have hpad2 : padding + padding = padding * 2 := by
      calc
        padding + padding = 2 * padding := by simp [Nat.two_mul]
        _ = padding * 2 := by simp [Nat.mul_comm]
    have hwpad2 : w + padding + padding = w + padding * 2 := by
      simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using congrArg (fun t => w + t) hpad2
    simp [Shape.reshapeValid, Shape.numel, Shape.poolOut, Shape.convOutDim, listProd, listZipWith5,
      patchFlat, kernelFlat, wOut, hwpad2, Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm])

  -- Step 4: Reshape weight
  -- [cout, cin, kW] -> [cout, cin * kW]
  let weightReshaped ← reshape weight [cout, kernelFlat] (by simp [Shape.reshapeValid, Shape.numel, listProd, kernelFlat])

  -- Step 5: Transpose weight for matmul: [cout, kernelFlat] -> [kernelFlat, cout]
  let weightT ← T weightReshaped

  -- Step 6: Matmul: [patchFlat, kernelFlat] @ [kernelFlat, cout] = [patchFlat, cout]
  let mm ← matmul patchesReshaped weightT

  -- Step 7: Reshape to [batch, wOut, cout]
  let mmReshaped ← reshape mm [batch, wOut, cout] (by
    simp [Shape.reshapeValid, Shape.numel, listProd, patchFlat, Nat.mul_assoc])

  -- Step 8: Permute to [batch, cout, wOut]
  let result ← permute mmReshaped [0, 2, 1] (by simp [Shape.permuteValid, listAll, listRange])

  -- Step 9: Add bias if present
  let finalUop ← match bias with
  | none => pure result.uop
  | some b =>
    let biasReshaped ← reshape b [1, cout, 1] (by simp [Shape.reshapeValid, Shape.numel, listProd])
    UOp.add result.uop biasReshaped.uop

  let biasGrad := match bias with | none => false | some b => b.requiresGrad
  let reqGrad := x.requiresGrad || weight.requiresGrad || biasGrad
  pure (StaticTensor.ofUOp finalUop (requiresGrad := reqGrad))

/-- Conv2d operation using pool/im2col + matmul.
    Input:  [batch, inChannels, height, width]
    Weight: [outChannels, inChannels, kernelH, kernelW]
    Output: [batch, outChannels, outHeight, outWidth]

    Algorithm:
    1. Pad input if needed
    2. Apply pool/im2col to get patches: [batch, cin, hOut, wOut, kH, kW]
    3. Reshape for matmul: patches -> [batch*hOut*wOut, cin*kH*kW]
    4. Reshape weight: [cout, cin*kH*kW]
    5. Matmul: [batch*hOut*wOut, cin*kH*kW] @ [cin*kH*kW, cout]^T = [batch*hOut*wOut, cout]
    6. Reshape to [batch, hOut, wOut, cout]
    7. Permute to [batch, cout, hOut, wOut]
    8. Add bias if present -/
def conv2d {batch cin cout h w kH kW : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, cin, h, w] d device)
    (weight : StaticTensor [cout, cin, kH, kW] d device)
    (bias : Option (StaticTensor [cout] d device) := none)
    (padding : Nat := 0)
    (stride : Nat := 1)
    (dilation : Nat := 1)
    : TensorM (StaticTensor (Shape.conv2dOut [batch, cin, h, w]
                                              [cout, cin, kH, kW]
                                              padding stride dilation) d device) := do
  -- Compute output dimensions
  let hOut := Shape.convOutDim h padding dilation kH stride
  let wOut := Shape.convOutDim w padding dilation kW stride

  -- Step 1: Pad input if needed
  let xPadded : StaticTensor [batch, cin, h + padding + padding, w + padding + padding] d device ←
    if hpad : padding > 0 then
      pad2d x padding padding
    else
      have hzero : padding = 0 := Nat.eq_zero_of_not_pos hpad
      have hx : StaticTensor [batch, cin, h + padding + padding, w + padding + padding] d device := by
        simpa [hzero, Nat.add_assoc] using x
      pure hx

  -- Step 2: Apply pool/im2col to get patches
  -- Input to pool: [batch, cin, hPadded, wPadded]
  -- Output: [batch, cin, hOut, wOut, kH, kW]
  let patches ← pool xPadded [kH, kW] [stride, stride] [dilation, dilation]

  -- Step 3: Reshape patches for matmul
  -- [batch, cin, hOut, wOut, kH, kW] -> [batch * hOut * wOut, cin * kH * kW]
  let patchFlat := batch * hOut * wOut
  let kernelFlat := cin * kH * kW
  let patchesReshaped ← reshape patches [patchFlat, kernelFlat] (by
    have hpad2 : padding + padding = padding * 2 := by
      calc
        padding + padding = 2 * padding := by simp [Nat.two_mul]
        _ = padding * 2 := by simp [Nat.mul_comm]
    have hhpad2 : h + padding + padding = h + padding * 2 := by
      simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using congrArg (fun t => h + t) hpad2
    have hwpad2 : w + padding + padding = w + padding * 2 := by
      simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using congrArg (fun t => w + t) hpad2
    simp [Shape.reshapeValid, Shape.numel, Shape.poolOut, Shape.convOutDim, listProd, listZipWith5,
      patchFlat, kernelFlat, hOut, wOut, hhpad2, hwpad2, Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm])

  -- Step 4: Reshape weight
  -- [cout, cin, kH, kW] -> [cout, cin * kH * kW]
  let weightReshaped ← reshape weight [cout, kernelFlat] (by
    simp [Shape.reshapeValid, Shape.numel, listProd, kernelFlat, Nat.mul_assoc])

  -- Step 5: Transpose weight for matmul: [cout, kernelFlat] -> [kernelFlat, cout]
  let weightT ← T weightReshaped

  -- Step 6: Matmul: [patchFlat, kernelFlat] @ [kernelFlat, cout] = [patchFlat, cout]
  let mm ← matmul patchesReshaped weightT

  -- Step 7: Reshape to [batch, hOut, wOut, cout]
  let mmReshaped ← reshape mm [batch, hOut, wOut, cout] (by
    simp [Shape.reshapeValid, Shape.numel, listProd, patchFlat, Nat.mul_assoc])

  -- Step 8: Permute to [batch, cout, hOut, wOut]
  let result ← permute mmReshaped [0, 3, 1, 2] (by simp [Shape.permuteValid, listAll, listRange])

  -- Step 9: Add bias if present
  let finalUop ← match bias with
  | none => pure result.uop
  | some b =>
    -- Reshape bias [cout] -> [1, cout, 1, 1] for broadcasting
    let biasReshaped ← reshape b [1, cout, 1, 1] (by simp [Shape.reshapeValid, Shape.numel, listProd])
    UOp.add result.uop biasReshaped.uop

  let biasGrad := match bias with | none => false | some b => b.requiresGrad
  let reqGrad := x.requiresGrad || weight.requiresGrad || biasGrad
  pure (StaticTensor.ofUOp finalUop (requiresGrad := reqGrad))

/-- Depthwise 2D convolution: each input channel is convolved with its own filter.
    This is a specialized case of grouped convolution where groups = cin = cout.

    Weight shape: [cin, 1, kH, kW] (each channel has one 1×kH×kW filter)

    Implementation using batched matmul (fast, like Python tinygrad):
    1. Pool to get patches: [batch, cin, hOut, wOut, kH, kW]
    2. Reshape patches: [batch, cin, hOut*wOut, kH*kW]
    3. Reshape weight: [cin, 1, kH, kW] -> [1, cin, kH*kW, 1]
    4. Batched matmul: [batch, cin, hOut*wOut, kH*kW] @ [1, cin, kH*kW, 1]
       -> broadcasts batch dims, performs matmul for each (batch, cin)
       -> result: [batch, cin, hOut*wOut, 1]
    5. Squeeze and reshapeUnsafe to [batch, cin, hOut, wOut]

    This uses a single batched CONTRACT operation instead of expandUnsafe+multiply+sum.
-/
def depthwiseConv2d {batch cin h w kH kW : Nat} {d : DType} {device : Backend.DeviceType}
    (x : StaticTensor [batch, cin, h, w] d device)
    (weight : StaticTensor [cin, 1, kH, kW] d device)
    (bias : Option (StaticTensor [cin] d device) := none)
    (padding : Nat := 0)
    (stride : Nat := 1)
    (dilation : Nat := 1)
    : TensorM (StaticTensor (Shape.conv2dOut [batch, cin, h, w]
                                              [cin, 1, kH, kW]
                                              padding stride dilation) d device) := do
  let hOut := Shape.convOutDim h padding dilation kH stride
  let wOut := Shape.convOutDim w padding dilation kW stride
  let spatialOut := hOut * wOut
  let kernelFlat := kH * kW

  -- Step 1: Pad if needed
  let xPadded : StaticTensor [batch, cin, h + padding + padding, w + padding + padding] d device ←
    if hpad : padding > 0 then
      pad2d x padding padding
    else
      have hzero : padding = 0 := Nat.eq_zero_of_not_pos hpad
      have hx : StaticTensor [batch, cin, h + padding + padding, w + padding + padding] d device := by
        simpa [hzero, Nat.add_assoc] using x
      pure hx

  -- Step 2: Pool to get patches: [batch, cin, hOut, wOut, kH, kW]
  let patches ← pool xPadded [kH, kW] [stride, stride] [dilation, dilation]

  -- Step 3: Reshape patches for batched matmul: [batch, cin, hOut*wOut, kH*kW]
  let patchesReshaped ← UOp.reshape patches.uop [batch, cin, spatialOut, kernelFlat]

  -- Step 4: Reshape weight for batched matmul: [cin, 1, kH, kW] -> [1, cin, kH*kW, 1]
  -- This allows broadcasting with batch dimension
  let weightReshaped ← UOp.reshape weight.uop [1, cin, kernelFlat, 1]

  -- Step 5: Batched matmul using UOp.contract2D
  -- [batch, cin, hOut*wOut, kH*kW] @ [1, cin, kH*kW, 1]
  -- Batch dims [batch, cin] and [1, cin] broadcast to [batch, cin]
  -- Result: [batch, cin, hOut*wOut, 1]
  let mmResult ← UOp.contract2D patchesReshaped weightReshaped

  -- Step 6: Squeeze and reshapeUnsafe to [batch, cin, hOut, wOut]
  let result ← UOp.reshape mmResult [batch, cin, hOut, wOut]

  -- Step 7: Add bias if present
  let finalUop ← match bias with
  | none => pure result
  | some b =>
    let biasReshaped ← UOp.reshape b.uop [1, cin, 1, 1]
    UOp.add result biasReshaped

  let biasGrad := match bias with | none => false | some b => b.requiresGrad
  let reqGrad := x.requiresGrad || weight.requiresGrad || biasGrad
  pure (StaticTensor.ofUOp finalUop (requiresGrad := reqGrad))

end StaticTensor

end TinyGrad4
