namespace TinyGrad4.Backend.Native

/-!
# Native kernels (portable)

This module is the bridge between the TinyGrad4 IR/interpreter and low-level kernels.

Today: a portable C matmul on `FloatArray` (Lean `Float` = C `double`).
Tomorrow: optional BLAS backends can be swapped in under the same API.
-/

@[extern "tg4_full_f32"]
opaque fullF32 (n : @& Nat) (v : @& Float) : ByteArray

@[extern "tg4_full_f32_bits"]
opaque fullF32Bits (n : @& Nat) (bits : @& UInt32) : ByteArray

@[extern "tg4_expand_scalar_f32"]
opaque expandScalarF32 (scalar : @& ByteArray) (n : @& Nat) : ByteArray

@[extern "tg4_expand_bcast_f32"]
opaque expandBcastF32 (a : @& ByteArray) (aShape outShape : @& Array Nat) : ByteArray

@[extern "tg4_expand_bcast_u8"]
opaque expandBcastU8 (a : @& ByteArray) (aShape outShape : @& Array Nat) : ByteArray

@[extern "tg4_neg_f32"]
opaque negF32 (a : @& ByteArray) : ByteArray

@[extern "tg4_sqrt_f32"]
opaque sqrtF32 (a : @& ByteArray) : ByteArray

@[extern "tg4_reciprocal_f32"]
opaque reciprocalF32 (a : @& ByteArray) : ByteArray

@[extern "tg4_exp2_f32"]
opaque exp2F32 (a : @& ByteArray) : ByteArray

@[extern "tg4_log2_f32"]
opaque log2F32 (a : @& ByteArray) : ByteArray

@[extern "tg4_sin_f32"]
opaque sinF32 (a : @& ByteArray) : ByteArray

@[extern "tg4_cos_f32"]
opaque cosF32 (a : @& ByteArray) : ByteArray

@[extern "tg4_tan_f32"]
opaque tanF32 (a : @& ByteArray) : ByteArray

@[extern "tg4_relu_f32"]
opaque reluF32 (a : @& ByteArray) : ByteArray

@[extern "tg4_add_f32"]
opaque addF32 (a b : @& ByteArray) : ByteArray

@[extern "tg4_add_bcast_f32"]
opaque addBcastF32 (a b : @& ByteArray) (aShape bShape outShape : @& Array Nat) : ByteArray

@[extern "tg4_sub_f32"]
opaque subF32 (a b : @& ByteArray) : ByteArray

@[extern "tg4_sub_bcast_f32"]
opaque subBcastF32 (a b : @& ByteArray) (aShape bShape outShape : @& Array Nat) : ByteArray

@[extern "tg4_mul_f32"]
opaque mulF32 (a b : @& ByteArray) : ByteArray

@[extern "tg4_sgd_update_f32"]
opaque sgdUpdateF32 (w grad : @& ByteArray) (lr : @& Float) : ByteArray

@[extern "tg4_mul_bcast_f32"]
opaque mulBcastF32 (a b : @& ByteArray) (aShape bShape outShape : @& Array Nat) : ByteArray

@[extern "tg4_div_f32"]
opaque divF32 (a b : @& ByteArray) : ByteArray

@[extern "tg4_div_bcast_f32"]
opaque divBcastF32 (a b : @& ByteArray) (aShape bShape outShape : @& Array Nat) : ByteArray

@[extern "tg4_max_f32"]
opaque maxF32 (a b : @& ByteArray) : ByteArray

@[extern "tg4_max_bcast_f32"]
opaque maxBcastF32 (a b : @& ByteArray) (aShape bShape outShape : @& Array Nat) : ByteArray

@[extern "tg4_pow_f32"]
opaque powF32 (a b : @& ByteArray) : ByteArray

@[extern "tg4_pow_bcast_f32"]
opaque powBcastF32 (a b : @& ByteArray) (aShape bShape outShape : @& Array Nat) : ByteArray

@[extern "tg4_cmplt_f32"]
opaque cmpltF32 (a b : @& ByteArray) : ByteArray

@[extern "tg4_cmplt_bcast_f32"]
opaque cmpltBcastF32 (a b : @& ByteArray) (aShape bShape outShape : @& Array Nat) : ByteArray

@[extern "tg4_where_f32"]
opaque whereF32 (cond x y : @& ByteArray) (n : @& Nat) : ByteArray

@[extern "tg4_where_bcast_f32"]
opaque whereBcastF32 (cond x y : @& ByteArray) (condShape xShape yShape outShape : @& Array Nat) : ByteArray

@[extern "tg4_fused_ewise_f32"]
opaque fusedEwiseF32 (inputs : @& Array ByteArray) (inputShapes : @& Array (Array Nat))
    (inputDtypes : @& Array Nat) (outShape : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_ewise_view_f32"]
opaque fusedEwiseViewF32 (inputs : @& Array ByteArray) (inputStrides : @& Array (Array Int64))
    (inputOffsets : @& Array Int64) (inputMaskStarts inputMaskEnds : @& Array (Array Nat))
    (inputDtypes : @& Array Nat) (outShape : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_ewise_view_stack_f32"]
opaque fusedEwiseViewStackF32 (inputs : @& Array ByteArray)
    (stackShapes : @& Array (Array (Array Nat))) (stackStrides : @& Array (Array (Array Int64)))
    (stackOffsets : @& Array (Array Int64)) (stackMaskStarts stackMaskEnds : @& Array (Array (Array Nat)))
    (inputDtypes : @& Array Nat) (outShape : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_sum_last_f32"]
opaque fusedReduceSumLastF32 (inputs : @& Array ByteArray) (inputShapes : @& Array (Array Nat))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_sum_last_view_f32"]
opaque fusedReduceSumLastViewF32 (inputs : @& Array ByteArray) (inputStrides : @& Array (Array Int64))
    (inputOffsets : @& Array Int64) (inputMaskStarts inputMaskEnds : @& Array (Array Nat))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_sum_last_view_stack_f32"]
opaque fusedReduceSumLastViewStackF32 (inputs : @& Array ByteArray)
    (stackShapes : @& Array (Array (Array Nat))) (stackStrides : @& Array (Array (Array Int64)))
    (stackOffsets : @& Array (Array Int64)) (stackMaskStarts stackMaskEnds : @& Array (Array (Array Nat)))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_max_last_f32"]
opaque fusedReduceMaxLastF32 (inputs : @& Array ByteArray) (inputShapes : @& Array (Array Nat))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_max_last_view_f32"]
opaque fusedReduceMaxLastViewF32 (inputs : @& Array ByteArray) (inputStrides : @& Array (Array Int64))
    (inputOffsets : @& Array Int64) (inputMaskStarts inputMaskEnds : @& Array (Array Nat))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_max_last_view_stack_f32"]
opaque fusedReduceMaxLastViewStackF32 (inputs : @& Array ByteArray)
    (stackShapes : @& Array (Array (Array Nat))) (stackStrides : @& Array (Array (Array Int64)))
    (stackOffsets : @& Array (Array Int64)) (stackMaskStarts stackMaskEnds : @& Array (Array (Array Nat)))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_sum_axis_f32"]
opaque fusedReduceSumAxisF32 (inputs : @& Array ByteArray) (inputShapes : @& Array (Array Nat))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (axis : @& Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_sum_axis_view_f32"]
opaque fusedReduceSumAxisViewF32 (inputs : @& Array ByteArray) (inputStrides : @& Array (Array Int64))
    (inputOffsets : @& Array Int64) (inputMaskStarts inputMaskEnds : @& Array (Array Nat))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (axis : @& Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_sum_axis_view_stack_f32"]
opaque fusedReduceSumAxisViewStackF32 (inputs : @& Array ByteArray)
    (stackShapes : @& Array (Array (Array Nat))) (stackStrides : @& Array (Array (Array Int64)))
    (stackOffsets : @& Array (Array Int64)) (stackMaskStarts stackMaskEnds : @& Array (Array (Array Nat)))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (axis : @& Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_max_axis_f32"]
opaque fusedReduceMaxAxisF32 (inputs : @& Array ByteArray) (inputShapes : @& Array (Array Nat))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (axis : @& Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_max_axis_view_f32"]
opaque fusedReduceMaxAxisViewF32 (inputs : @& Array ByteArray) (inputStrides : @& Array (Array Int64))
    (inputOffsets : @& Array Int64) (inputMaskStarts inputMaskEnds : @& Array (Array Nat))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (axis : @& Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_max_axis_view_stack_f32"]
opaque fusedReduceMaxAxisViewStackF32 (inputs : @& Array ByteArray)
    (stackShapes : @& Array (Array (Array Nat))) (stackStrides : @& Array (Array (Array Int64)))
    (stackOffsets : @& Array (Array Int64)) (stackMaskStarts stackMaskEnds : @& Array (Array (Array Nat)))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (axis : @& Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_sum_axes_f32"]
opaque fusedReduceSumAxesF32 (inputs : @& Array ByteArray) (inputShapes : @& Array (Array Nat))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (outShape : @& Array Nat)
    (axes : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_sum_axes_view_f32"]
opaque fusedReduceSumAxesViewF32 (inputs : @& Array ByteArray) (inputStrides : @& Array (Array Int64))
    (inputOffsets : @& Array Int64) (inputMaskStarts inputMaskEnds : @& Array (Array Nat))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (outShape : @& Array Nat)
    (axes : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_sum_axes_view_stack_f32"]
opaque fusedReduceSumAxesViewStackF32 (inputs : @& Array ByteArray)
    (stackShapes : @& Array (Array (Array Nat))) (stackStrides : @& Array (Array (Array Int64)))
    (stackOffsets : @& Array (Array Int64)) (stackMaskStarts stackMaskEnds : @& Array (Array (Array Nat)))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (outShape : @& Array Nat)
    (axes : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_max_axes_f32"]
opaque fusedReduceMaxAxesF32 (inputs : @& Array ByteArray) (inputShapes : @& Array (Array Nat))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (outShape : @& Array Nat)
    (axes : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_max_axes_view_f32"]
opaque fusedReduceMaxAxesViewF32 (inputs : @& Array ByteArray) (inputStrides : @& Array (Array Int64))
    (inputOffsets : @& Array Int64) (inputMaskStarts inputMaskEnds : @& Array (Array Nat))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (outShape : @& Array Nat)
    (axes : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_max_axes_view_stack_f32"]
opaque fusedReduceMaxAxesViewStackF32 (inputs : @& Array ByteArray)
    (stackShapes : @& Array (Array (Array Nat))) (stackStrides : @& Array (Array (Array Int64)))
    (stackOffsets : @& Array (Array Int64)) (stackMaskStarts stackMaskEnds : @& Array (Array (Array Nat)))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (outShape : @& Array Nat)
    (axes : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_sum_all_f32"]
opaque fusedReduceSumAllF32 (inputs : @& Array ByteArray) (inputShapes : @& Array (Array Nat))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_sum_all_view_f32"]
opaque fusedReduceSumAllViewF32 (inputs : @& Array ByteArray) (inputStrides : @& Array (Array Int64))
    (inputOffsets : @& Array Int64) (inputMaskStarts inputMaskEnds : @& Array (Array Nat))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_sum_all_view_stack_f32"]
opaque fusedReduceSumAllViewStackF32 (inputs : @& Array ByteArray)
    (stackShapes : @& Array (Array (Array Nat))) (stackStrides : @& Array (Array (Array Int64)))
    (stackOffsets : @& Array (Array Int64)) (stackMaskStarts stackMaskEnds : @& Array (Array (Array Nat)))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_max_all_f32"]
opaque fusedReduceMaxAllF32 (inputs : @& Array ByteArray) (inputShapes : @& Array (Array Nat))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_max_all_view_f32"]
opaque fusedReduceMaxAllViewF32 (inputs : @& Array ByteArray) (inputStrides : @& Array (Array Int64))
    (inputOffsets : @& Array Int64) (inputMaskStarts inputMaskEnds : @& Array (Array Nat))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_fused_reduce_max_all_view_stack_f32"]
opaque fusedReduceMaxAllViewStackF32 (inputs : @& Array ByteArray)
    (stackShapes : @& Array (Array (Array Nat))) (stackStrides : @& Array (Array (Array Int64)))
    (stackOffsets : @& Array (Array Int64)) (stackMaskStarts stackMaskEnds : @& Array (Array (Array Nat)))
    (inputDtypes : @& Array Nat) (fullShape : @& Array Nat) (prog : @& Array UInt64) : ByteArray

@[extern "tg4_softmax_last_f32"]
opaque softmaxLastF32 (a : @& ByteArray) (outer inner : @& Nat) (scaleBits : @& UInt32) : ByteArray

@[extern "tg4_logsoftmax_last_f32"]
opaque logSoftmaxLastF32 (a : @& ByteArray) (outer inner : @& Nat) (scaleBits ln2Bits : @& UInt32) : ByteArray

@[extern "tg4_sum_all_f32"]
opaque sumAllF32 (a : @& ByteArray) : ByteArray

@[extern "tg4_reduce_sum_last_f32"]
opaque reduceSumLastF32 (a : @& ByteArray) (outer inner : @& Nat) : ByteArray

@[extern "tg4_reduce_max_last_f32"]
opaque reduceMaxLastF32 (a : @& ByteArray) (outer inner : @& Nat) : ByteArray

@[extern "tg4_reduce_sum_axis_f32"]
opaque reduceSumAxisF32 (a : @& ByteArray) (outer reduce inner : @& Nat) : ByteArray

@[extern "tg4_reduce_max_axis_f32"]
opaque reduceMaxAxisF32 (a : @& ByteArray) (outer reduce inner : @& Nat) : ByteArray

@[extern "tg4_transpose2d_f32"]
opaque transpose2dF32 (a : @& ByteArray) (m n : @& Nat) : ByteArray

@[extern "tg4_permute_f32"]
opaque permuteF32 (a : @& ByteArray) (shape perm : @& Array Nat) : ByteArray

@[extern "tg4_permute_u8"]
opaque permuteU8 (a : @& ByteArray) (shape perm : @& Array Nat) : ByteArray

@[extern "tg4_pad_f32"]
opaque padF32 (a : @& ByteArray) (shape padLeft padRight : @& Array Nat) : ByteArray

@[extern "tg4_pad_u8"]
opaque padU8 (a : @& ByteArray) (shape padLeft padRight : @& Array Nat) : ByteArray

@[extern "tg4_shrink_f32"]
opaque shrinkF32 (a : @& ByteArray) (shape starts stops : @& Array Nat) : ByteArray

@[extern "tg4_shrink_u8"]
opaque shrinkU8 (a : @& ByteArray) (shape starts stops : @& Array Nat) : ByteArray

@[extern "tg4_cat_f32"]
opaque catF32 (inputs : @& Array ByteArray) (inputShapes : @& Array (Array Nat))
    (axis : @& Nat) (outShape : @& Array Nat) : ByteArray

@[extern "tg4_cat_u8"]
opaque catU8 (inputs : @& Array ByteArray) (inputShapes : @& Array (Array Nat))
    (axis : @& Nat) (outShape : @& Array Nat) : ByteArray

@[extern "tg4_cat_bytes"]
opaque catBytes (inputs : @& Array ByteArray) (inputShapes : @& Array (Array Nat))
    (axis : @& Nat) (outShape : @& Array Nat) (elemSize : @& Nat) : ByteArray

@[extern "tg4_flip_f32"]
opaque flipF32 (a : @& ByteArray) (shape axes : @& Array Nat) : ByteArray

@[extern "tg4_flip_u8"]
opaque flipU8 (a : @& ByteArray) (shape axes : @& Array Nat) : ByteArray

@[extern "tg4_pack_f32_from_f64"]
opaque packF32FromF64 (a : @& FloatArray) : ByteArray

@[extern "tg4_unpack_f64_from_f32"]
opaque unpackF64FromF32 (a : @& ByteArray) : FloatArray

@[extern "tg4_matmul_f32"]
opaque matmulF32 (a b : @& ByteArray) (m k n : @& Nat) : ByteArray

@[extern "tg4_matmul_view_f32"]
opaque matmulViewF32 (a b : @& ByteArray)
    (aStrides : @& Array Int64) (aOffset : @& Int64) (aMaskStarts aMaskEnds : @& Array Nat)
    (bStrides : @& Array Int64) (bOffset : @& Int64) (bMaskStarts bMaskEnds : @& Array Nat)
    (outShape : @& Array Nat) (k : @& Nat) : ByteArray

@[extern "tg4_matmul_view_stack_f32"]
opaque matmulViewStackF32 (a b : @& ByteArray)
    (aStackShapes : @& Array (Array Nat)) (aStackStrides : @& Array (Array Int64))
    (aStackOffsets : @& Array Int64) (aStackMaskStarts aStackMaskEnds : @& Array (Array Nat))
    (bStackShapes : @& Array (Array Nat)) (bStackStrides : @& Array (Array Int64))
    (bStackOffsets : @& Array Int64) (bStackMaskStarts bStackMaskEnds : @& Array (Array Nat))
    (outShape : @& Array Nat) (k : @& Nat) : ByteArray

@[extern "tg4_matmul_bias_f32"]
opaque matmulBiasF32 (a b bias : @& ByteArray) (biasShape : @& Array Nat) (m k n : @& Nat) : ByteArray

@[extern "tg4_matmul_bias_scale_f32"]
opaque matmulBiasScaleF32 (a b bias : @& ByteArray) (biasShape : @& Array Nat) (m k n : @& Nat) (scaleBits : @& UInt32) : ByteArray

@[extern "tg4_matmul_bias_relu_f32"]
opaque matmulBiasReluF32 (a b bias : @& ByteArray) (biasShape : @& Array Nat) (m k n : @& Nat) : ByteArray

@[extern "tg4_matmul_bias_scale_relu_f32"]
opaque matmulBiasScaleReluF32 (a b bias : @& ByteArray) (biasShape : @& Array Nat) (m k n : @& Nat) (scaleBits : @& UInt32) : ByteArray

@[extern "tg4_matmul_bias2_f32"]
opaque matmulBias2F32 (a b bias0 : @& ByteArray) (bias0Shape : @& Array Nat) (bias1 : @& ByteArray) (bias1Shape : @& Array Nat)
    (m k n : @& Nat) : ByteArray

@[extern "tg4_matmul_bias2_scale_f32"]
opaque matmulBias2ScaleF32 (a b bias0 : @& ByteArray) (bias0Shape : @& Array Nat) (bias1 : @& ByteArray) (bias1Shape : @& Array Nat)
    (m k n : @& Nat) (scaleBits : @& UInt32) : ByteArray

@[extern "tg4_matmul_bias2_relu_f32"]
opaque matmulBias2ReluF32 (a b bias0 : @& ByteArray) (bias0Shape : @& Array Nat) (bias1 : @& ByteArray) (bias1Shape : @& Array Nat)
    (m k n : @& Nat) : ByteArray

@[extern "tg4_matmul_bias2_scale_relu_f32"]
opaque matmulBias2ScaleReluF32 (a b bias0 : @& ByteArray) (bias0Shape : @& Array Nat) (bias1 : @& ByteArray) (bias1Shape : @& Array Nat)
    (m k n : @& Nat) (scaleBits : @& UInt32) : ByteArray

@[extern "tg4_matmul_view_bias_f32"]
opaque matmulViewBiasF32 (a b bias : @& ByteArray)
    (aStrides : @& Array Int64) (aOffset : @& Int64) (aMaskStarts aMaskEnds : @& Array Nat)
    (bStrides : @& Array Int64) (bOffset : @& Int64) (bMaskStarts bMaskEnds : @& Array Nat)
    (biasStrides : @& Array Int64) (biasOffset : @& Int64) (biasMaskStarts biasMaskEnds : @& Array Nat)
    (outShape : @& Array Nat) (k : @& Nat) : ByteArray

@[extern "tg4_matmul_view_bias_scale_f32"]
opaque matmulViewBiasScaleF32 (a b bias : @& ByteArray)
    (aStrides : @& Array Int64) (aOffset : @& Int64) (aMaskStarts aMaskEnds : @& Array Nat)
    (bStrides : @& Array Int64) (bOffset : @& Int64) (bMaskStarts bMaskEnds : @& Array Nat)
    (biasStrides : @& Array Int64) (biasOffset : @& Int64) (biasMaskStarts biasMaskEnds : @& Array Nat)
    (outShape : @& Array Nat) (k : @& Nat) (scaleBits : @& UInt32) : ByteArray

@[extern "tg4_matmul_view_bias_relu_f32"]
opaque matmulViewBiasReluF32 (a b bias : @& ByteArray)
    (aStrides : @& Array Int64) (aOffset : @& Int64) (aMaskStarts aMaskEnds : @& Array Nat)
    (bStrides : @& Array Int64) (bOffset : @& Int64) (bMaskStarts bMaskEnds : @& Array Nat)
    (biasStrides : @& Array Int64) (biasOffset : @& Int64) (biasMaskStarts biasMaskEnds : @& Array Nat)
    (outShape : @& Array Nat) (k : @& Nat) : ByteArray

@[extern "tg4_matmul_view_bias_scale_relu_f32"]
opaque matmulViewBiasScaleReluF32 (a b bias : @& ByteArray)
    (aStrides : @& Array Int64) (aOffset : @& Int64) (aMaskStarts aMaskEnds : @& Array Nat)
    (bStrides : @& Array Int64) (bOffset : @& Int64) (bMaskStarts bMaskEnds : @& Array Nat)
    (biasStrides : @& Array Int64) (biasOffset : @& Int64) (biasMaskStarts biasMaskEnds : @& Array Nat)
    (outShape : @& Array Nat) (k : @& Nat) (scaleBits : @& UInt32) : ByteArray

@[extern "tg4_matmul_view_bias2_f32"]
opaque matmulViewBias2F32 (a b bias0 bias1 : @& ByteArray)
    (aStrides : @& Array Int64) (aOffset : @& Int64) (aMaskStarts aMaskEnds : @& Array Nat)
    (bStrides : @& Array Int64) (bOffset : @& Int64) (bMaskStarts bMaskEnds : @& Array Nat)
    (bias0Strides : @& Array Int64) (bias0Offset : @& Int64) (bias0MaskStarts bias0MaskEnds : @& Array Nat)
    (bias1Strides : @& Array Int64) (bias1Offset : @& Int64) (bias1MaskStarts bias1MaskEnds : @& Array Nat)
    (outShape : @& Array Nat) (k : @& Nat) : ByteArray

@[extern "tg4_matmul_view_bias2_scale_f32"]
opaque matmulViewBias2ScaleF32 (a b bias0 bias1 : @& ByteArray)
    (aStrides : @& Array Int64) (aOffset : @& Int64) (aMaskStarts aMaskEnds : @& Array Nat)
    (bStrides : @& Array Int64) (bOffset : @& Int64) (bMaskStarts bMaskEnds : @& Array Nat)
    (bias0Strides : @& Array Int64) (bias0Offset : @& Int64) (bias0MaskStarts bias0MaskEnds : @& Array Nat)
    (bias1Strides : @& Array Int64) (bias1Offset : @& Int64) (bias1MaskStarts bias1MaskEnds : @& Array Nat)
    (outShape : @& Array Nat) (k : @& Nat) (scaleBits : @& UInt32) : ByteArray

@[extern "tg4_matmul_view_bias2_relu_f32"]
opaque matmulViewBias2ReluF32 (a b bias0 bias1 : @& ByteArray)
    (aStrides : @& Array Int64) (aOffset : @& Int64) (aMaskStarts aMaskEnds : @& Array Nat)
    (bStrides : @& Array Int64) (bOffset : @& Int64) (bMaskStarts bMaskEnds : @& Array Nat)
    (bias0Strides : @& Array Int64) (bias0Offset : @& Int64) (bias0MaskStarts bias0MaskEnds : @& Array Nat)
    (bias1Strides : @& Array Int64) (bias1Offset : @& Int64) (bias1MaskStarts bias1MaskEnds : @& Array Nat)
    (outShape : @& Array Nat) (k : @& Nat) : ByteArray

@[extern "tg4_matmul_view_bias2_scale_relu_f32"]
opaque matmulViewBias2ScaleReluF32 (a b bias0 bias1 : @& ByteArray)
    (aStrides : @& Array Int64) (aOffset : @& Int64) (aMaskStarts aMaskEnds : @& Array Nat)
    (bStrides : @& Array Int64) (bOffset : @& Int64) (bMaskStarts bMaskEnds : @& Array Nat)
    (bias0Strides : @& Array Int64) (bias0Offset : @& Int64) (bias0MaskStarts bias0MaskEnds : @& Array Nat)
    (bias1Strides : @& Array Int64) (bias1Offset : @& Int64) (bias1MaskStarts bias1MaskEnds : @& Array Nat)
    (outShape : @& Array Nat) (k : @& Nat) (scaleBits : @& UInt32) : ByteArray

@[extern "tg4_matmul_batched_f32"]
opaque matmulBatchedF32 (a b : @& ByteArray) (aStarts bStarts : @& Array Nat) (m k n : @& Nat) : ByteArray

@[extern "tg4_matmul_batched_bias_f32"]
opaque matmulBatchedBiasF32 (a b bias : @& ByteArray) (biasShape : @& Array Nat)
    (aStarts bStarts biasStarts : @& Array Nat) (m k n : @& Nat) : ByteArray

@[extern "tg4_matmul_batched_bias_scale_f32"]
opaque matmulBatchedBiasScaleF32 (a b bias : @& ByteArray) (biasShape : @& Array Nat)
    (aStarts bStarts biasStarts : @& Array Nat) (m k n : @& Nat) (scaleBits : @& UInt32) : ByteArray

@[extern "tg4_matmul_batched_bias_relu_f32"]
opaque matmulBatchedBiasReluF32 (a b bias : @& ByteArray) (biasShape : @& Array Nat)
    (aStarts bStarts biasStarts : @& Array Nat) (m k n : @& Nat) : ByteArray

@[extern "tg4_matmul_batched_bias_scale_relu_f32"]
opaque matmulBatchedBiasScaleReluF32 (a b bias : @& ByteArray) (biasShape : @& Array Nat)
    (aStarts bStarts biasStarts : @& Array Nat) (m k n : @& Nat) (scaleBits : @& UInt32) : ByteArray

@[extern "tg4_matmul_batched_bias2_f32"]
opaque matmulBatchedBias2F32 (a b bias0 : @& ByteArray) (bias0Shape : @& Array Nat) (bias1 : @& ByteArray) (bias1Shape : @& Array Nat)
    (aStarts bStarts bias0Starts bias1Starts : @& Array Nat) (m k n : @& Nat) : ByteArray

@[extern "tg4_matmul_batched_bias2_scale_f32"]
opaque matmulBatchedBias2ScaleF32 (a b bias0 : @& ByteArray) (bias0Shape : @& Array Nat) (bias1 : @& ByteArray) (bias1Shape : @& Array Nat)
    (aStarts bStarts bias0Starts bias1Starts : @& Array Nat) (m k n : @& Nat) (scaleBits : @& UInt32) : ByteArray

@[extern "tg4_matmul_batched_bias2_relu_f32"]
opaque matmulBatchedBias2ReluF32 (a b bias0 : @& ByteArray) (bias0Shape : @& Array Nat) (bias1 : @& ByteArray) (bias1Shape : @& Array Nat)
    (aStarts bStarts bias0Starts bias1Starts : @& Array Nat) (m k n : @& Nat) : ByteArray

@[extern "tg4_matmul_batched_bias2_scale_relu_f32"]
opaque matmulBatchedBias2ScaleReluF32 (a b bias0 : @& ByteArray) (bias0Shape : @& Array Nat) (bias1 : @& ByteArray) (bias1Shape : @& Array Nat)
    (aStarts bStarts bias0Starts bias1Starts : @& Array Nat) (m k n : @& Nat) (scaleBits : @& UInt32) : ByteArray

@[extern "tg4_matmul_f64"]
opaque matmulF64 (a b : @& FloatArray) (m k n : @& Nat) : FloatArray

end TinyGrad4.Backend.Native
