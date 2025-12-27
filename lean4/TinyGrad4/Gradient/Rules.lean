import TinyGrad4.UOp.UOp
import TinyGrad4.UOp.Graph
import TinyGrad4.Gradient.Adjoint

namespace TinyGrad4

namespace Gradient

structure GradResult where
  srcGrads : List (Option UOp)
  adjoint : AdjointSpec
  deriving Repr

private def mkResult (u : UOp) (srcGrads : List (Option UOp)) : GradResult :=
  { srcGrads, adjoint := adjointSpec u }

private def isZeroConst (u : UOp) : Bool :=
  u.op == .CONST && u.shape == [] && u.arg.getFloat.getD 1.0 == 0.0

def gradOp (u : UOp) (grad : UOp) : UOpM GradResult := do
  match u.op with
  | .NEG => do
    let dx ← UOp.neg grad
    pure (mkResult u [some dx])

  | .RECIPROCAL => do
    let x := u.src[0]!
    let x2 ← UOp.mul x x
    let neg_grad ← UOp.neg grad
    let dx ← UOp.div neg_grad x2
    pure (mkResult u [some dx])

  | .SQRT => do
    let two ← UOp.const u.dtype 2.0
    let two_sqrt ← UOp.mul two u
    let dx ← UOp.div grad two_sqrt
    pure (mkResult u [some dx])

  | .EXP2 => do
    let ln2 ← UOp.const u.dtype 0.6931471805599453
    let scaled ← UOp.mul u ln2
    let dx ← UOp.mul grad scaled
    pure (mkResult u [some dx])

  | .LOG2 => do
    let x := u.src[0]!
    let ln2 ← UOp.const u.dtype 0.6931471805599453
    let x_ln2 ← UOp.mul x ln2
    let dx ← UOp.div grad x_ln2
    pure (mkResult u [some dx])

  | .SIN => do
    -- d/dx sin(x) = cos(x) * grad
    -- cos(x) = sin(pi/2 - x)
    let x := u.src[0]!
    let pi_half ← UOp.const u.dtype 1.5707963267948966
    let shifted ← UOp.sub pi_half x
    let cos_x ← UOp.unaryOp .SIN shifted
    let dx ← UOp.mul cos_x grad
    pure (mkResult u [some dx])

  | .COS => do
    -- d/dx cos(x) = -sin(x) * grad
    let x := u.src[0]!
    let sin_x ← UOp.unaryOp .SIN x
    let neg_sin ← UOp.neg sin_x
    let dx ← UOp.mul neg_sin grad
    pure (mkResult u [some dx])

  | .TAN => do
    -- d/dx tan(x) = sec^2(x) * grad = (1 + tan^2(x)) * grad
    let one ← UOp.const u.dtype 1.0
    let tan2 ← UOp.mul u u
    let sec2 ← UOp.add one tan2
    let dx ← UOp.mul sec2 grad
    pure (mkResult u [some dx])

  | .POW => do
    -- d/db b^e = e * b^(e-1) * grad
    -- d/de b^e = b^e * ln(b) * grad = ret * log2(b) * ln(2) * grad
    let b := u.src[0]!
    let e := u.src[1]!
    let one ← UOp.const u.dtype 1.0
    let ln2 ← UOp.const u.dtype 0.6931471805599453
    -- d/db = e * b^(e-1) * grad
    let e_minus_1 ← UOp.sub e one
    let b_pow_em1 ← UOp.binaryOp .POW b e_minus_1
    let e_times_bpow ← UOp.mul e b_pow_em1
    let db ← UOp.mul e_times_bpow grad
    -- d/de = ret * log2(b) * ln(2) * grad
    let log2_b ← UOp.unaryOp .LOG2 b
    let log2_b_ln2 ← UOp.mul log2_b ln2
    let ret_log ← UOp.mul u log2_b_ln2
    let de ← UOp.mul ret_log grad
    pure (mkResult u [some db, some de])

  | .ADD => pure (mkResult u [some grad, some grad])

  | .MUL => do
    let x := u.src[0]!
    let y := u.src[1]!
    let dx ← UOp.mul grad y
    let dy ← UOp.mul grad x
    pure (mkResult u [some dx, some dy])

  | .SUB => do
    let neg_grad ← UOp.neg grad
    pure (mkResult u [some grad, some neg_grad])

  | .FDIV => do
    let x := u.src[0]!
    let y := u.src[1]!
    let dx ← UOp.div grad y
    let y2 ← UOp.mul y y
    let neg_grad ← UOp.neg grad
    let neg_grad_x ← UOp.mul neg_grad x
    let dy ← UOp.div neg_grad_x y2
    pure (mkResult u [some dx, some dy])

  | .CONTRACT => do
    let a := u.src[0]!
    let b := u.src[1]!
    let rA := a.rank
    let rB := b.rank
    if rA < 2 || rB < 2 then
      pure (mkResult u [none, none])
    else
      let permA := (listRange rA).take (rA - 2) ++ [rA - 1, rA - 2]
      let permB := (listRange rB).take (rB - 2) ++ [rB - 1, rB - 2]
      let aT ← UOp.permute a permA
      let bT ← UOp.permute b permB
      -- dL/da = grad @ bᵀ, dL/db = aᵀ @ grad
      let dA ← UOp.contract2D grad bT
      let dB ← UOp.contract2D aT grad
      pure (mkResult u [some dA, some dB])

  | .RESHAPE => do
    let x := u.src[0]!
    let dx ← UOp.reshape grad x.shape
    pure (mkResult u [some dx])

  | .EXPAND => do
    let x := u.src[0]!
    -- `EXPAND` follows broadcast semantics (right-aligned), so handle rank mismatches by left-padding `x.shape` with 1s.
    let pad := u.shape.length - x.shape.length
    let xShapePadded : Shape := List.replicate pad 1 ++ x.shape
    let expandedAxes := (xShapePadded.zip u.shape) |> listEnum |>.filterMap fun (i, (oldDim, newDim)) =>
      if oldDim == 1 && newDim > 1 then some i else none
    -- If no axes were expanded, just pass gradient through (possibly reshape)
    if expandedAxes.isEmpty then
      let dx ← if grad.shape == x.shape then pure grad else UOp.reshape grad x.shape
      pure (mkResult u [some dx])
    else
      -- Sum over expanded axes, then reshape if needed
      let dx ← UOp.reduce grad .ADD expandedAxes true
      let dx' ← if dx.shape == x.shape then pure dx else UOp.reshape dx x.shape
      pure (mkResult u [some dx'])

  | .PERMUTE => do
    let perm := u.arg.getPermutation.getD []
    let invPerm := listRange perm.length |>.map fun i => listIndexOf perm i
    let dx ← UOp.permute grad invPerm
    pure (mkResult u [some dx])

  | .SHRINK => do
    let x := u.src[0]!
    match u.arg with
    | .bounds bounds =>
      let padding := listZipWith (fun (s, e) dim => (s, dim - e)) bounds x.shape
      let dx ← UOp.pad grad padding
      pure (mkResult u [some dx])
    | _ => pure (mkResult u [some grad])

  | .PAD => do
    -- PAD gradient is SHRINK - remove the padding from gradient
    let x := u.src[0]!
    match u.arg with
    | .bounds padding =>
      -- shrink bounds are (start, end) where start=pad_before, end=pad_before+original_dim
      let shrinkBounds := listZipWith (fun (padBefore, _padAfter) origDim => (padBefore, padBefore + origDim)) padding x.shape
      let dx ← UOp.shrink grad shrinkBounds
      pure (mkResult u [some dx])
    | _ => pure (mkResult u [some grad])

  | .FLIP => do
    let axes := u.arg.getAxes.getD []
    let dx ← UOp.flip grad axes
    pure (mkResult u [some dx])

  | .REDUCE_AXIS => do
    let x := u.src[0]!
    let dx ← UOp.expand grad x.shape
    pure (mkResult u [some dx])

  | .CAT => do
    match u.arg with
    | .axes [axis] =>
      if axis >= u.shape.length then
        pure (mkResult u (u.src.map (fun _ => none)))
      else
        let rec loop (srcs : List UOp) (offset : Nat) : UOpM (List (Option UOp)) := do
          match srcs with
          | [] => pure []
          | s :: ss =>
            let dim := listGetD s.shape axis 0
            let start := offset
            let stop := offset + dim
            let bounds := (listRange u.shape.length).map fun i =>
              if i == axis then (start, stop) else (0, listGetD u.shape i 0)
            let g ← UOp.shrink grad bounds
            let rest ← loop ss stop
            pure (some g :: rest)
        let grads ← loop u.src 0
        pure (mkResult u grads)
    | _ =>
      pure (mkResult u (u.src.map (fun _ => none)))

  | .CONTIGUOUS | .DETACH | .CAST | .BITCAST =>
    pure (mkResult u [some grad])

  | .CONTIGUOUS_BACKWARD => do
    -- CONTIGUOUS_BACKWARD: apply contiguous to gradient
    let dx ← UOp.unaryOp .CONTIGUOUS grad
    pure (mkResult u [some dx])

  | .CONST | .BUFFER => pure (mkResult u [])

  | .CMPLT | .CMPNE | .CMPEQ => pure (mkResult u [none, none])

  -- Binary MAX: grad flows to the larger input
  -- d/dx max(x,y) = 1 if x >= y, else 0
  -- d/dy max(x,y) = 1 if y > x, else 0
  | .MAX => do
    let spec := adjointSpec u
    let x := u.src[0]!
    let y := u.src[1]!
    match spec.tape with
    | .saveOutput => do
      let zero ← UOp.const u.dtype 0.0
      let one ← UOp.const u.dtype 1.0
      let gt0 ← UOp.cmplt zero u
      let mask ← UOp.where_ gt0 one zero
      let dx ← UOp.mul grad mask
      let xZero := isZeroConst x
      let yZero := isZeroConst y
      if xZero && !yZero then
        pure { srcGrads := [none, some dx], adjoint := spec }
      else if yZero && !xZero then
        pure { srcGrads := [some dx, none], adjoint := spec }
      else
        pure { srcGrads := [some dx, some dx], adjoint := spec }
    | _ => do
      -- mask_x = (x >= y) = !(x < y)
      let x_lt_y ← UOp.cmplt x y
      let one ← UOp.const u.dtype 1.0
      let zero ← UOp.const u.dtype 0.0
      -- mask_x = where(x < y, 0, 1)
      let mask_x ← UOp.where_ x_lt_y zero one
      -- mask_y = where(x < y, 1, 0)
      let mask_y ← UOp.where_ x_lt_y one zero
      let dx ← UOp.mul grad mask_x
      let dy ← UOp.mul grad mask_y
      pure { srcGrads := [some dx, some dy], adjoint := spec }

  -- WHERE: cond ? x : y
  -- d/dx = grad * cond, d/dy = grad * (1 - cond)
  | .WHERE => do
    let cond := u.src[0]!
    let one ← UOp.const u.dtype 1.0
    let zero ← UOp.const u.dtype 0.0
    -- Turn the (bool) condition into float masks to avoid bool arithmetic.
    let mask_x ← UOp.where_ cond one zero
    let mask_y ← UOp.where_ cond zero one
    let dx ← UOp.mul grad mask_x
    let dy ← UOp.mul grad mask_y
    -- cond has no gradient (it's discrete)
    pure (mkResult u [none, some dx, some dy])

  | _ => pure (mkResult u (u.src.map (fun _ => none)))

end Gradient

end TinyGrad4
