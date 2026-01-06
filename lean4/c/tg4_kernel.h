#pragma once

// TinyGrad4 kernel boundary (portable C ABI)
//
// Goal: keep the trusted/perf-critical surface *small*.
// - Everything above this boundary can be proven correct in Lean (rewrites, fusion, scheduling).
// - Everything below this boundary can be implemented in C/CUDA/Metal/etc.
//
// This header intentionally does not mention Lean. A Lean backend can wrap these APIs via FFI.
//
// Minimal useful set:
// - `copy`            : materialize a view (layout transform / contiguous)
// - `ewise_*`         : elementwise unary/binary/ternary on strided views (broadcast = stride 0)
// - `reduce`          : sum/max reductions (prefer keepdim=true; dropping dims is a view op)
// - `gemm`            : matmul/contract
//
// Less minimal, faster set:
// - `fused_ewise`     : execute a small scalar program per output element
// - `map_reduce`      : fused map+reduce (common in softmax/layernorm)
// - `gemm_epilogue`   : gemm + bias + activation (+ optional residual)
//
// Note: for maximum portability, all shapes/strides are provided as host-side int64 arrays.
// Strides are in *bytes*. Broadcast dims are represented by stride = 0.
// Negative strides are allowed (e.g. flip), with `offset_bytes` adjusted accordingly.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum tg4_dtype {
  TG4_DTYPE_INVALID = 0,
  TG4_DTYPE_BOOL = 1,
  TG4_DTYPE_F32 = 2,
} tg4_dtype;

typedef struct tg4_buf {
  void *data;       // host ptr, device ptr, or backend-specific pointer
  size_t nbytes;    // total bytes accessible from `data`
  tg4_dtype dtype;  // element dtype (buffer is conceptually a flat array of this dtype)
} tg4_buf;

typedef struct tg4_view {
  tg4_buf buf;
  int32_t rank;
  const int64_t *shape;        // length = rank, in elements
  const int64_t *strides;      // length = rank, in bytes
  int64_t offset_bytes;        // byte offset into buf.data
} tg4_view;

typedef enum tg4_unary_op {
  TG4_UNARY_NEG = 0,
  TG4_UNARY_SQRT = 1,
  TG4_UNARY_RECIPROCAL = 2,
  TG4_UNARY_EXP2 = 3,
  TG4_UNARY_LOG2 = 4,
  TG4_UNARY_SIN = 5,
} tg4_unary_op;

typedef enum tg4_binary_op {
  TG4_BINARY_ADD = 0,
  TG4_BINARY_SUB = 1,
  TG4_BINARY_MUL = 2,
  TG4_BINARY_DIV = 3,
  TG4_BINARY_MAX = 4,
  TG4_BINARY_CMPLT = 5, // outputs bool
} tg4_binary_op;

typedef enum tg4_ternary_op {
  TG4_TERNARY_WHERE = 0, // cond ? x : y
} tg4_ternary_op;

typedef enum tg4_reduce_op {
  TG4_REDUCE_SUM = 0,
  TG4_REDUCE_MAX = 1,
} tg4_reduce_op;

typedef struct tg4_gemm_params {
  // Shapes follow (..., M, K) @ (..., K, N) -> (..., M, N).
  // Most backends will assume the trailing 2D matrices are contiguous in row-major order.
  // Batch dims can be broadcast via strides (including 0-stride).
  int64_t m;
  int64_t k;
  int64_t n;
} tg4_gemm_params;

typedef enum tg4_status {
  TG4_STATUS_OK = 0,
  TG4_STATUS_INVALID = 1,
  TG4_STATUS_OOM = 2,
  TG4_STATUS_UNSUPPORTED = 3,
} tg4_status;

// Backend context (CPU, CUDA stream/device, Metal queue, etc).
typedef struct tg4_ctx tg4_ctx;

// ---- Minimal primitives ----

tg4_status tg4_copy(tg4_ctx *ctx, tg4_view dst, tg4_view src);

tg4_status tg4_ewise_unary(tg4_ctx *ctx, tg4_view dst, tg4_view x, tg4_unary_op op);
tg4_status tg4_ewise_binary(tg4_ctx *ctx, tg4_view dst, tg4_view x, tg4_view y, tg4_binary_op op);
tg4_status tg4_ewise_ternary(tg4_ctx *ctx, tg4_view dst, tg4_view a, tg4_view b, tg4_view c, tg4_ternary_op op);

tg4_status tg4_reduce(
    tg4_ctx *ctx,
    tg4_view dst,
    tg4_view src,
    tg4_reduce_op op,
    const int32_t *axes,
    int32_t axes_len);

tg4_status tg4_gemm(tg4_ctx *ctx, tg4_view dst, tg4_view a, tg4_view b, tg4_gemm_params p);

// ---- Extensions for speed (optional) ----

typedef enum tg4_inst_kind {
  TG4_INST_CONST_F32_BITS = 0, // push u32 bits as float32
  TG4_INST_LOAD = 1,           // push input[input_idx] at current output index
  TG4_INST_UNARY = 2,
  TG4_INST_BINARY = 3,
  TG4_INST_TERNARY = 4,
} tg4_inst_kind;

typedef struct tg4_inst {
  uint8_t kind; // tg4_inst_kind
  uint8_t op;   // op code (unary/binary/ternary depending on kind)
  uint16_t arg; // e.g. input index, or unused
  uint32_t imm; // e.g. f32 bits for const
} tg4_inst;

typedef struct tg4_program {
  const tg4_inst *insts;
  int32_t insts_len;
  int32_t n_inputs;
} tg4_program;

tg4_status tg4_fused_ewise(tg4_ctx *ctx, tg4_view dst, const tg4_view *inputs, int32_t n_inputs, tg4_program prog);

tg4_status tg4_map_reduce(
    tg4_ctx *ctx,
    tg4_view dst,
    const tg4_view *inputs,
    int32_t n_inputs,
    tg4_program prog,
    tg4_reduce_op rop,
    const int32_t *axes,
    int32_t axes_len);

#ifdef __cplusplus
} // extern "C"
#endif

