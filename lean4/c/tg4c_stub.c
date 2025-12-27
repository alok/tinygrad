#include <lean/lean.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* View information for strided access with optional masking */
struct view_info {
  size_t rank;
  int64_t* strides;
  size_t* mask_start;
  size_t* mask_end;
  int64_t offset;
};
typedef struct view_info view_info;

/* Stack of views for multi-level strided access */
struct view_stack {
  size_t depth;
  view_info* views;
  size_t** shapes;
  size_t* numels;
  size_t max_rank;
};
typedef struct view_stack view_stack;

static inline lean_object* mk_byte_array(size_t size) {
  return lean_alloc_sarray(1, size, size);
}

static inline uint8_t* byte_array_cptr(lean_object* a) {
  return (uint8_t*)lean_sarray_cptr(a);
}

static inline size_t byte_array_size(b_lean_obj_arg a) {
  return lean_sarray_size(a);
}

static inline size_t f32_numel(b_lean_obj_arg a) {
  return byte_array_size(a) / 4;
}

static inline lean_object* mk_float_array(size_t size) {
  return lean_alloc_sarray(sizeof(double), size, size);
}

static inline double* float_array_cptr(lean_object* a) {
  return (double*)lean_sarray_cptr(a);
}

static inline size_t float_array_size(b_lean_obj_arg a) {
  return lean_sarray_size(a);
}

static inline size_t nat_to_size(b_lean_obj_arg n) {
  return (size_t)lean_uint64_of_nat(n);
}

static inline size_t array_size(b_lean_obj_arg a) {
  return lean_array_size(a);
}

static inline size_t nat_array_get(b_lean_obj_arg a, size_t i) {
  return nat_to_size(lean_array_get_core(a, i));
}

static inline int64_t int64_array_get(b_lean_obj_arg a, size_t i) {
  lean_object* v = lean_array_get_core(a, i);
  if (lean_is_scalar(v)) {
    return (int64_t)lean_unbox(v);
  }
  return (int64_t)lean_ctor_get_uint64(v, 0);
}

static inline uint64_t uint64_array_get(b_lean_obj_arg a, size_t i) {
  lean_object* v = lean_array_get_core(a, i);
  if (lean_is_scalar(v)) {
    return (uint64_t)lean_unbox(v);
  }
  return (uint64_t)lean_ctor_get_uint64(v, 0);
}

static inline uint32_t unbox_u32(b_lean_obj_arg o) {
  return lean_unbox_uint32(o);
}

static inline int64_t unbox_int64(b_lean_obj_arg o) {
  if (lean_is_scalar(o)) {
    return (int64_t)lean_unbox(o);
  }
  return (int64_t)lean_ctor_get_uint64(o, 0);
}

static inline float f32_from_bits(uint32_t bits) {
  union { uint32_t u; float f; } v;
  v.u = bits;
  return v.f;
}

static inline uint32_t f32_to_bits(float f) {
  union { uint32_t u; float f; } v;
  v.f = f;
  return v.u;
}

static inline float read_f32(const uint8_t* data, size_t idx) {
  float v;
  memcpy(&v, data + idx * 4, 4);
  return v;
}

static inline void write_f32(uint8_t* data, size_t idx, float v) {
  memcpy(data + idx * 4, &v, 4);
}

static size_t shape_numel(b_lean_obj_arg shape) {
  size_t rank = array_size(shape);
  size_t prod = 1;
  for (size_t i = 0; i < rank; ++i) {
    prod *= nat_array_get(shape, i);
  }
  return prod;
}

static void unflatten_index(size_t flat, const size_t* shape, size_t rank, size_t* out) {
  for (size_t i = rank; i-- > 0;) {
    size_t dim = shape[i];
    size_t v = dim == 0 ? 0 : (flat % dim);
    out[i] = v;
    flat = dim == 0 ? 0 : (flat / dim);
  }
}

static void make_strides(const size_t* shape, size_t rank, size_t* out) {
  size_t acc = 1;
  for (size_t i = rank; i-- > 0;) {
    out[i] = acc;
    acc *= shape[i];
  }
}

static size_t broadcast_flat_index(const size_t* out_idx, size_t out_rank,
    const size_t* in_shape, size_t in_rank, const size_t* in_strides) {
  size_t offset = out_rank >= in_rank ? (out_rank - in_rank) : 0;
  size_t flat = 0;
  for (size_t i = 0; i < in_rank; ++i) {
    size_t dim = in_shape[i];
    size_t idx = dim == 1 ? 0 : out_idx[offset + i];
    flat += idx * in_strides[i];
  }
  return flat;
}

static void view_info_free(view_info* v);
static int view_info_init(view_info* v, b_lean_obj_arg strides, int64_t offset,
  b_lean_obj_arg mask_start, b_lean_obj_arg mask_end);
static float view_read_value(const uint8_t* data, const view_info* v, const size_t* idx,
  int is_bool);
static float view_read_f32(const uint8_t* data, const view_info* v, const size_t* idx);
static void view_stack_free(view_stack* v);
static int view_stack_init(view_stack* v, b_lean_obj_arg shapes, b_lean_obj_arg strides,
  b_lean_obj_arg offsets, b_lean_obj_arg mask_start, b_lean_obj_arg mask_end);
static float view_stack_read_value(const uint8_t* data, const view_stack* v, const size_t* idx,
  size_t* idx_buf, size_t* tmp_buf, int is_bool);
static float view_stack_read_f32(const uint8_t* data, const view_stack* v, const size_t* idx,
  size_t* idx_buf, size_t* tmp_buf);

static lean_object* bcast_unary_f32(b_lean_obj_arg a, b_lean_obj_arg aShape, b_lean_obj_arg outShape) {
  size_t out_rank = array_size(outShape);
  size_t out_numel = shape_numel(outShape);
  size_t in_rank = array_size(aShape);

  size_t* out_dims = malloc(out_rank * sizeof(size_t));
  size_t* in_dims = malloc(in_rank * sizeof(size_t));
  size_t* in_strides = malloc(in_rank * sizeof(size_t));
  size_t* out_idx = malloc(out_rank * sizeof(size_t));

  for (size_t i = 0; i < out_rank; ++i) out_dims[i] = nat_array_get(outShape, i);
  for (size_t i = 0; i < in_rank; ++i) in_dims[i] = nat_array_get(aShape, i);
  make_strides(in_dims, in_rank, in_strides);

  lean_object* out = mk_byte_array(out_numel * 4);
  uint8_t* out_data = byte_array_cptr(out);
  const uint8_t* a_data = byte_array_cptr((lean_object*)a);

  for (size_t i = 0; i < out_numel; ++i) {
    unflatten_index(i, out_dims, out_rank, out_idx);
    size_t a_flat = broadcast_flat_index(out_idx, out_rank, in_dims, in_rank, in_strides);
    write_f32(out_data, i, read_f32(a_data, a_flat));
  }

  free(out_dims);
  free(in_dims);
  free(in_strides);
  free(out_idx);
  return out;
}

static lean_object* bcast_unary_u8(b_lean_obj_arg a, b_lean_obj_arg aShape, b_lean_obj_arg outShape) {
  size_t out_rank = array_size(outShape);
  size_t out_numel = shape_numel(outShape);
  size_t in_rank = array_size(aShape);

  size_t* out_dims = malloc(out_rank * sizeof(size_t));
  size_t* in_dims = malloc(in_rank * sizeof(size_t));
  size_t* in_strides = malloc(in_rank * sizeof(size_t));
  size_t* out_idx = malloc(out_rank * sizeof(size_t));

  for (size_t i = 0; i < out_rank; ++i) out_dims[i] = nat_array_get(outShape, i);
  for (size_t i = 0; i < in_rank; ++i) in_dims[i] = nat_array_get(aShape, i);
  make_strides(in_dims, in_rank, in_strides);

  lean_object* out = mk_byte_array(out_numel);
  uint8_t* out_data = byte_array_cptr(out);
  const uint8_t* a_data = byte_array_cptr((lean_object*)a);

  for (size_t i = 0; i < out_numel; ++i) {
    unflatten_index(i, out_dims, out_rank, out_idx);
    size_t a_flat = broadcast_flat_index(out_idx, out_rank, in_dims, in_rank, in_strides);
    out_data[i] = a_data[a_flat];
  }

  free(out_dims);
  free(in_dims);
  free(in_strides);
  free(out_idx);
  return out;
}

static lean_object* bcast_binary_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg aShape, b_lean_obj_arg bShape, b_lean_obj_arg outShape,
    float (*op)(float, float)) {
  size_t out_rank = array_size(outShape);
  size_t out_numel = shape_numel(outShape);
  size_t a_rank = array_size(aShape);
  size_t b_rank = array_size(bShape);

  size_t* out_dims = malloc(out_rank * sizeof(size_t));
  size_t* a_dims = malloc(a_rank * sizeof(size_t));
  size_t* b_dims = malloc(b_rank * sizeof(size_t));
  size_t* a_strides = malloc(a_rank * sizeof(size_t));
  size_t* b_strides = malloc(b_rank * sizeof(size_t));
  size_t* out_idx = malloc(out_rank * sizeof(size_t));

  for (size_t i = 0; i < out_rank; ++i) out_dims[i] = nat_array_get(outShape, i);
  for (size_t i = 0; i < a_rank; ++i) a_dims[i] = nat_array_get(aShape, i);
  for (size_t i = 0; i < b_rank; ++i) b_dims[i] = nat_array_get(bShape, i);
  make_strides(a_dims, a_rank, a_strides);
  make_strides(b_dims, b_rank, b_strides);

  lean_object* out = mk_byte_array(out_numel * 4);
  uint8_t* out_data = byte_array_cptr(out);
  const uint8_t* a_data = byte_array_cptr((lean_object*)a);
  const uint8_t* b_data = byte_array_cptr((lean_object*)b);

  for (size_t i = 0; i < out_numel; ++i) {
    unflatten_index(i, out_dims, out_rank, out_idx);
    size_t a_flat = broadcast_flat_index(out_idx, out_rank, a_dims, a_rank, a_strides);
    size_t b_flat = broadcast_flat_index(out_idx, out_rank, b_dims, b_rank, b_strides);
    float av = read_f32(a_data, a_flat);
    float bv = read_f32(b_data, b_flat);
    write_f32(out_data, i, op(av, bv));
  }

  free(out_dims);
  free(a_dims);
  free(b_dims);
  free(a_strides);
  free(b_strides);
  free(out_idx);
  return out;
}

static lean_object* bcast_binary_u8(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg aShape, b_lean_obj_arg bShape, b_lean_obj_arg outShape,
    uint8_t (*op)(uint8_t, uint8_t)) {
  size_t out_rank = array_size(outShape);
  size_t out_numel = shape_numel(outShape);
  size_t a_rank = array_size(aShape);
  size_t b_rank = array_size(bShape);

  size_t* out_dims = malloc(out_rank * sizeof(size_t));
  size_t* a_dims = malloc(a_rank * sizeof(size_t));
  size_t* b_dims = malloc(b_rank * sizeof(size_t));
  size_t* a_strides = malloc(a_rank * sizeof(size_t));
  size_t* b_strides = malloc(b_rank * sizeof(size_t));
  size_t* out_idx = malloc(out_rank * sizeof(size_t));

  for (size_t i = 0; i < out_rank; ++i) out_dims[i] = nat_array_get(outShape, i);
  for (size_t i = 0; i < a_rank; ++i) a_dims[i] = nat_array_get(aShape, i);
  for (size_t i = 0; i < b_rank; ++i) b_dims[i] = nat_array_get(bShape, i);
  make_strides(a_dims, a_rank, a_strides);
  make_strides(b_dims, b_rank, b_strides);

  lean_object* out = mk_byte_array(out_numel);
  uint8_t* out_data = byte_array_cptr(out);
  const uint8_t* a_data = byte_array_cptr((lean_object*)a);
  const uint8_t* b_data = byte_array_cptr((lean_object*)b);

  for (size_t i = 0; i < out_numel; ++i) {
    unflatten_index(i, out_dims, out_rank, out_idx);
    size_t a_flat = broadcast_flat_index(out_idx, out_rank, a_dims, a_rank, a_strides);
    size_t b_flat = broadcast_flat_index(out_idx, out_rank, b_dims, b_rank, b_strides);
    out_data[i] = op(a_data[a_flat], b_data[b_flat]);
  }

  free(out_dims);
  free(a_dims);
  free(b_dims);
  free(a_strides);
  free(b_strides);
  free(out_idx);
  return out;
}

static float op_add(float a, float b) { return a + b; }
static float op_sub(float a, float b) { return a - b; }
static float op_mul(float a, float b) { return a * b; }
static float op_div(float a, float b) { return a / b; }
static float op_max(float a, float b) { return fmaxf(a, b); }
static float op_pow(float a, float b) { return powf(a, b); }
static uint8_t op_cmplt_u8(uint8_t a, uint8_t b) { return a < b ? 1 : 0; }

LEAN_EXPORT lean_obj_res tg4_full_f32(b_lean_obj_arg n, b_lean_obj_arg v) {
  size_t count = nat_to_size(n);
  float fv = (float)lean_unbox_float(v);
  lean_object* out = mk_byte_array(count * 4);
  uint8_t* data = byte_array_cptr(out);
  for (size_t i = 0; i < count; ++i) {
    write_f32(data, i, fv);
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_full_f32_bits(b_lean_obj_arg n, b_lean_obj_arg bits) {
  size_t count = nat_to_size(n);
  float fv = f32_from_bits(unbox_u32(bits));
  lean_object* out = mk_byte_array(count * 4);
  uint8_t* data = byte_array_cptr(out);
  for (size_t i = 0; i < count; ++i) {
    write_f32(data, i, fv);
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_expand_scalar_f32(b_lean_obj_arg scalar, b_lean_obj_arg n) {
  size_t count = nat_to_size(n);
  const uint8_t* sdata = byte_array_cptr((lean_object*)scalar);
  float v = read_f32(sdata, 0);
  lean_object* out = mk_byte_array(count * 4);
  uint8_t* data = byte_array_cptr(out);
  for (size_t i = 0; i < count; ++i) {
    write_f32(data, i, v);
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_expand_bcast_f32(b_lean_obj_arg a, b_lean_obj_arg aShape,
    b_lean_obj_arg outShape) {
  return bcast_unary_f32(a, aShape, outShape);
}

LEAN_EXPORT lean_obj_res tg4_expand_bcast_u8(b_lean_obj_arg a, b_lean_obj_arg aShape,
    b_lean_obj_arg outShape) {
  return bcast_unary_u8(a, aShape, outShape);
}

LEAN_EXPORT lean_obj_res tg4_neg_f32(b_lean_obj_arg a) {
  size_t n = f32_numel(a);
  lean_object* out = mk_byte_array(n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  for (size_t i = 0; i < n; ++i) {
    write_f32(o, i, -read_f32(x, i));
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_sqrt_f32(b_lean_obj_arg a) {
  size_t n = f32_numel(a);
  lean_object* out = mk_byte_array(n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  for (size_t i = 0; i < n; ++i) {
    write_f32(o, i, sqrtf(read_f32(x, i)));
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_reciprocal_f32(b_lean_obj_arg a) {
  size_t n = f32_numel(a);
  lean_object* out = mk_byte_array(n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  for (size_t i = 0; i < n; ++i) {
    float v = read_f32(x, i);
    write_f32(o, i, v == 0.0f ? 0.0f : 1.0f / v);
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_exp2_f32(b_lean_obj_arg a) {
  size_t n = f32_numel(a);
  lean_object* out = mk_byte_array(n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  for (size_t i = 0; i < n; ++i) {
    write_f32(o, i, exp2f(read_f32(x, i)));
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_log2_f32(b_lean_obj_arg a) {
  size_t n = f32_numel(a);
  lean_object* out = mk_byte_array(n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  for (size_t i = 0; i < n; ++i) {
    write_f32(o, i, log2f(read_f32(x, i)));
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_sin_f32(b_lean_obj_arg a) {
  size_t n = f32_numel(a);
  lean_object* out = mk_byte_array(n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  for (size_t i = 0; i < n; ++i) {
    write_f32(o, i, sinf(read_f32(x, i)));
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_cos_f32(b_lean_obj_arg a) {
  size_t n = f32_numel(a);
  lean_object* out = mk_byte_array(n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  for (size_t i = 0; i < n; ++i) {
    write_f32(o, i, cosf(read_f32(x, i)));
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_tan_f32(b_lean_obj_arg a) {
  size_t n = f32_numel(a);
  lean_object* out = mk_byte_array(n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  for (size_t i = 0; i < n; ++i) {
    write_f32(o, i, tanf(read_f32(x, i)));
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_relu_f32(b_lean_obj_arg a) {
  size_t n = f32_numel(a);
  lean_object* out = mk_byte_array(n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  for (size_t i = 0; i < n; ++i) {
    float v = read_f32(x, i);
    write_f32(o, i, v > 0.0f ? v : 0.0f);
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_add_f32(b_lean_obj_arg a, b_lean_obj_arg b) {
  size_t n = f32_numel(a);
  size_t nb = f32_numel(b);
  if (nb < n) n = nb;
  lean_object* out = mk_byte_array(n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  const uint8_t* y = byte_array_cptr((lean_object*)b);
  for (size_t i = 0; i < n; ++i) {
    write_f32(o, i, read_f32(x, i) + read_f32(y, i));
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_add_bcast_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg aShape, b_lean_obj_arg bShape, b_lean_obj_arg outShape) {
  return bcast_binary_f32(a, b, aShape, bShape, outShape, op_add);
}

LEAN_EXPORT lean_obj_res tg4_sub_f32(b_lean_obj_arg a, b_lean_obj_arg b) {
  size_t n = f32_numel(a);
  size_t nb = f32_numel(b);
  if (nb < n) n = nb;
  lean_object* out = mk_byte_array(n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  const uint8_t* y = byte_array_cptr((lean_object*)b);
  for (size_t i = 0; i < n; ++i) {
    write_f32(o, i, read_f32(x, i) - read_f32(y, i));
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_sub_bcast_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg aShape, b_lean_obj_arg bShape, b_lean_obj_arg outShape) {
  return bcast_binary_f32(a, b, aShape, bShape, outShape, op_sub);
}

LEAN_EXPORT lean_obj_res tg4_mul_f32(b_lean_obj_arg a, b_lean_obj_arg b) {
  size_t n = f32_numel(a);
  size_t nb = f32_numel(b);
  if (nb < n) n = nb;
  lean_object* out = mk_byte_array(n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  const uint8_t* y = byte_array_cptr((lean_object*)b);
  for (size_t i = 0; i < n; ++i) {
    write_f32(o, i, read_f32(x, i) * read_f32(y, i));
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_sgd_update_f32(b_lean_obj_arg w, b_lean_obj_arg grad, b_lean_obj_arg lr) {
  size_t n = f32_numel(w);
  size_t ng = f32_numel(grad);
  if (ng < n) n = ng;
  float lr_v = (float)lean_unbox_float(lr);
  lean_object* out = mk_byte_array(n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)w);
  const uint8_t* g = byte_array_cptr((lean_object*)grad);
  for (size_t i = 0; i < n; ++i) {
    float v = read_f32(x, i) - lr_v * read_f32(g, i);
    write_f32(o, i, v);
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_mul_bcast_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg aShape, b_lean_obj_arg bShape, b_lean_obj_arg outShape) {
  return bcast_binary_f32(a, b, aShape, bShape, outShape, op_mul);
}

LEAN_EXPORT lean_obj_res tg4_div_f32(b_lean_obj_arg a, b_lean_obj_arg b) {
  size_t n = f32_numel(a);
  size_t nb = f32_numel(b);
  if (nb < n) n = nb;
  lean_object* out = mk_byte_array(n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  const uint8_t* y = byte_array_cptr((lean_object*)b);
  for (size_t i = 0; i < n; ++i) {
    write_f32(o, i, read_f32(x, i) / read_f32(y, i));
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_div_bcast_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg aShape, b_lean_obj_arg bShape, b_lean_obj_arg outShape) {
  return bcast_binary_f32(a, b, aShape, bShape, outShape, op_div);
}

LEAN_EXPORT lean_obj_res tg4_max_f32(b_lean_obj_arg a, b_lean_obj_arg b) {
  size_t n = f32_numel(a);
  size_t nb = f32_numel(b);
  if (nb < n) n = nb;
  lean_object* out = mk_byte_array(n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  const uint8_t* y = byte_array_cptr((lean_object*)b);
  for (size_t i = 0; i < n; ++i) {
    write_f32(o, i, fmaxf(read_f32(x, i), read_f32(y, i)));
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_max_bcast_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg aShape, b_lean_obj_arg bShape, b_lean_obj_arg outShape) {
  return bcast_binary_f32(a, b, aShape, bShape, outShape, op_max);
}

LEAN_EXPORT lean_obj_res tg4_pow_f32(b_lean_obj_arg a, b_lean_obj_arg b) {
  size_t n = f32_numel(a);
  size_t nb = f32_numel(b);
  if (nb < n) n = nb;
  lean_object* out = mk_byte_array(n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  const uint8_t* y = byte_array_cptr((lean_object*)b);
  for (size_t i = 0; i < n; ++i) {
    write_f32(o, i, powf(read_f32(x, i), read_f32(y, i)));
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_pow_bcast_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg aShape, b_lean_obj_arg bShape, b_lean_obj_arg outShape) {
  return bcast_binary_f32(a, b, aShape, bShape, outShape, op_pow);
}

LEAN_EXPORT lean_obj_res tg4_cmplt_f32(b_lean_obj_arg a, b_lean_obj_arg b) {
  size_t n = f32_numel(a);
  size_t nb = f32_numel(b);
  if (nb < n) n = nb;
  lean_object* out = mk_byte_array(n);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  const uint8_t* y = byte_array_cptr((lean_object*)b);
  for (size_t i = 0; i < n; ++i) {
    o[i] = read_f32(x, i) < read_f32(y, i) ? 1 : 0;
  }
  return out;
}

static uint8_t op_cmplt(uint8_t a, uint8_t b) { return a < b ? 1 : 0; }

LEAN_EXPORT lean_obj_res tg4_cmplt_bcast_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg aShape, b_lean_obj_arg bShape, b_lean_obj_arg outShape) {
  size_t out_rank = array_size(outShape);
  size_t out_numel = shape_numel(outShape);
  size_t a_rank = array_size(aShape);
  size_t b_rank = array_size(bShape);

  size_t* out_dims = malloc(out_rank * sizeof(size_t));
  size_t* a_dims = malloc(a_rank * sizeof(size_t));
  size_t* b_dims = malloc(b_rank * sizeof(size_t));
  size_t* a_strides = malloc(a_rank * sizeof(size_t));
  size_t* b_strides = malloc(b_rank * sizeof(size_t));
  size_t* out_idx = malloc(out_rank * sizeof(size_t));

  for (size_t i = 0; i < out_rank; ++i) out_dims[i] = nat_array_get(outShape, i);
  for (size_t i = 0; i < a_rank; ++i) a_dims[i] = nat_array_get(aShape, i);
  for (size_t i = 0; i < b_rank; ++i) b_dims[i] = nat_array_get(bShape, i);
  make_strides(a_dims, a_rank, a_strides);
  make_strides(b_dims, b_rank, b_strides);

  lean_object* out = mk_byte_array(out_numel);
  uint8_t* out_data = byte_array_cptr(out);
  const uint8_t* a_data = byte_array_cptr((lean_object*)a);
  const uint8_t* b_data = byte_array_cptr((lean_object*)b);

  for (size_t i = 0; i < out_numel; ++i) {
    unflatten_index(i, out_dims, out_rank, out_idx);
    size_t a_flat = broadcast_flat_index(out_idx, out_rank, a_dims, a_rank, a_strides);
    size_t b_flat = broadcast_flat_index(out_idx, out_rank, b_dims, b_rank, b_strides);
    float av = read_f32(a_data, a_flat);
    float bv = read_f32(b_data, b_flat);
    out_data[i] = av < bv ? 1 : 0;
  }

  free(out_dims);
  free(a_dims);
  free(b_dims);
  free(a_strides);
  free(b_strides);
  free(out_idx);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_where_f32(b_lean_obj_arg cond, b_lean_obj_arg x, b_lean_obj_arg y,
    b_lean_obj_arg n) {
  size_t count = nat_to_size(n);
  lean_object* out = mk_byte_array(count * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* c = byte_array_cptr((lean_object*)cond);
  const uint8_t* xv = byte_array_cptr((lean_object*)x);
  const uint8_t* yv = byte_array_cptr((lean_object*)y);
  for (size_t i = 0; i < count; ++i) {
    float v = c[i] ? read_f32(xv, i) : read_f32(yv, i);
    write_f32(o, i, v);
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_where_bcast_f32(b_lean_obj_arg cond, b_lean_obj_arg x, b_lean_obj_arg y,
    b_lean_obj_arg cShape, b_lean_obj_arg xShape, b_lean_obj_arg yShape, b_lean_obj_arg outShape) {
  size_t out_rank = array_size(outShape);
  size_t out_numel = shape_numel(outShape);

  size_t c_rank = array_size(cShape);
  size_t x_rank = array_size(xShape);
  size_t y_rank = array_size(yShape);

  size_t* out_dims = malloc(out_rank * sizeof(size_t));
  size_t* c_dims = malloc(c_rank * sizeof(size_t));
  size_t* x_dims = malloc(x_rank * sizeof(size_t));
  size_t* y_dims = malloc(y_rank * sizeof(size_t));
  size_t* c_strides = malloc(c_rank * sizeof(size_t));
  size_t* x_strides = malloc(x_rank * sizeof(size_t));
  size_t* y_strides = malloc(y_rank * sizeof(size_t));
  size_t* out_idx = malloc(out_rank * sizeof(size_t));

  for (size_t i = 0; i < out_rank; ++i) out_dims[i] = nat_array_get(outShape, i);
  for (size_t i = 0; i < c_rank; ++i) c_dims[i] = nat_array_get(cShape, i);
  for (size_t i = 0; i < x_rank; ++i) x_dims[i] = nat_array_get(xShape, i);
  for (size_t i = 0; i < y_rank; ++i) y_dims[i] = nat_array_get(yShape, i);
  make_strides(c_dims, c_rank, c_strides);
  make_strides(x_dims, x_rank, x_strides);
  make_strides(y_dims, y_rank, y_strides);

  lean_object* out = mk_byte_array(out_numel * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* c = byte_array_cptr((lean_object*)cond);
  const uint8_t* xv = byte_array_cptr((lean_object*)x);
  const uint8_t* yv = byte_array_cptr((lean_object*)y);

  for (size_t i = 0; i < out_numel; ++i) {
    unflatten_index(i, out_dims, out_rank, out_idx);
    size_t c_flat = broadcast_flat_index(out_idx, out_rank, c_dims, c_rank, c_strides);
    size_t x_flat = broadcast_flat_index(out_idx, out_rank, x_dims, x_rank, x_strides);
    size_t y_flat = broadcast_flat_index(out_idx, out_rank, y_dims, y_rank, y_strides);
    float v = c[c_flat] ? read_f32(xv, x_flat) : read_f32(yv, y_flat);
    write_f32(o, i, v);
  }

  free(out_dims);
  free(c_dims);
  free(x_dims);
  free(y_dims);
  free(c_strides);
  free(x_strides);
  free(y_strides);
  free(out_idx);
  return out;
}

enum {
  TG4_EWISE_PUSH = 0,
  TG4_EWISE_NEG = 1,
  TG4_EWISE_SQRT = 2,
  TG4_EWISE_RECIP = 3,
  TG4_EWISE_EXP2 = 4,
  TG4_EWISE_LOG2 = 5,
  TG4_EWISE_SIN = 6,
  TG4_EWISE_ADD = 7,
  TG4_EWISE_SUB = 8,
  TG4_EWISE_MUL = 9,
  TG4_EWISE_DIV = 10,
  TG4_EWISE_MAX = 11,
  TG4_EWISE_WHERE = 12,
  TG4_EWISE_MULACC = 13,
  TG4_EWISE_COS = 14,
  TG4_EWISE_TAN = 15,
  TG4_EWISE_POW = 16,
};

typedef struct {
  size_t n_inputs;
  const uint8_t** inputs;
  size_t* in_ranks;
  size_t** in_dims;
  size_t** in_strides;
  uint8_t* in_is_bool;
  size_t out_rank;
  size_t* out_dims;
  uint64_t* prog;
  size_t prog_len;
} fused_ewise_ctx;

static void fused_ewise_free(fused_ewise_ctx* ctx) {
  if (!ctx) return;
  if (ctx->in_dims && ctx->in_strides) {
    for (size_t i = 0; i < ctx->n_inputs; ++i) {
      free(ctx->in_dims[i]);
      free(ctx->in_strides[i]);
    }
  }
  free(ctx->inputs);
  free(ctx->in_ranks);
  free(ctx->in_dims);
  free(ctx->in_strides);
  free(ctx->in_is_bool);
  free(ctx->out_dims);
  free(ctx->prog);
}

static int fused_ewise_init(fused_ewise_ctx* ctx, b_lean_obj_arg inputs, b_lean_obj_arg inputShapes,
    b_lean_obj_arg inputDtypes, b_lean_obj_arg outShape, b_lean_obj_arg prog) {
  memset(ctx, 0, sizeof(*ctx));
  ctx->n_inputs = array_size(inputs);
  ctx->out_rank = array_size(outShape);
  ctx->prog_len = array_size(prog);
  ctx->inputs = malloc(ctx->n_inputs * sizeof(uint8_t*));
  ctx->in_ranks = malloc(ctx->n_inputs * sizeof(size_t));
  ctx->in_dims = malloc(ctx->n_inputs * sizeof(size_t*));
  ctx->in_strides = malloc(ctx->n_inputs * sizeof(size_t*));
  ctx->in_is_bool = malloc(ctx->n_inputs * sizeof(uint8_t));
  ctx->out_dims = ctx->out_rank == 0 ? NULL : malloc(ctx->out_rank * sizeof(size_t));
  ctx->prog = ctx->prog_len == 0 ? NULL : malloc(ctx->prog_len * sizeof(uint64_t));
  if ((!ctx->inputs && ctx->n_inputs) || (!ctx->in_ranks && ctx->n_inputs) ||
      (!ctx->in_dims && ctx->n_inputs) || (!ctx->in_strides && ctx->n_inputs) ||
      (!ctx->in_is_bool && ctx->n_inputs) || (!ctx->out_dims && ctx->out_rank) ||
      (!ctx->prog && ctx->prog_len)) {
    fused_ewise_free(ctx);
    return 0;
  }

  for (size_t i = 0; i < ctx->out_rank; ++i) {
    ctx->out_dims[i] = nat_array_get(outShape, i);
  }

  for (size_t i = 0; i < ctx->n_inputs; ++i) {
    lean_object* inp = lean_array_get_core(inputs, i);
    lean_object* sh = lean_array_get_core(inputShapes, i);
    ctx->inputs[i] = byte_array_cptr(inp);
    ctx->in_is_bool[i] = nat_array_get(inputDtypes, i) == 1 ? 1 : 0;
    size_t rank = array_size(sh);
    ctx->in_ranks[i] = rank;
    if (rank == 0) {
      ctx->in_dims[i] = NULL;
      ctx->in_strides[i] = NULL;
    } else {
      ctx->in_dims[i] = malloc(rank * sizeof(size_t));
      ctx->in_strides[i] = malloc(rank * sizeof(size_t));
      if (!ctx->in_dims[i] || !ctx->in_strides[i]) {
        fused_ewise_free(ctx);
        return 0;
      }
      for (size_t j = 0; j < rank; ++j) {
        ctx->in_dims[i][j] = nat_array_get(sh, j);
      }
      make_strides(ctx->in_dims[i], rank, ctx->in_strides[i]);
    }
  }

  for (size_t i = 0; i < ctx->prog_len; ++i) {
    ctx->prog[i] = uint64_array_get(prog, i);
  }
  return 1;
}

static float fused_ewise_eval_at(const fused_ewise_ctx* ctx, const size_t* out_idx, float* stack) {
  size_t sp = 0;
  for (size_t i = 0; i < ctx->prog_len; ++i) {
    uint64_t instr = ctx->prog[i];
    uint64_t op = instr & 0xFF;
    uint64_t imm = instr >> 8;
    switch (op) {
      case TG4_EWISE_PUSH: {
        size_t idx = (size_t)imm;
        size_t flat = broadcast_flat_index(out_idx, ctx->out_rank, ctx->in_dims[idx],
          ctx->in_ranks[idx], ctx->in_strides[idx]);
        float v = ctx->in_is_bool[idx] ? (ctx->inputs[idx][flat] ? 1.0f : 0.0f)
                                       : read_f32(ctx->inputs[idx], flat);
        stack[sp++] = v;
        break;
      }
      case TG4_EWISE_NEG:
        stack[sp - 1] = -stack[sp - 1];
        break;
      case TG4_EWISE_SQRT:
        stack[sp - 1] = sqrtf(stack[sp - 1]);
        break;
      case TG4_EWISE_RECIP:
        stack[sp - 1] = 1.0f / stack[sp - 1];
        break;
      case TG4_EWISE_EXP2:
        stack[sp - 1] = exp2f(stack[sp - 1]);
        break;
      case TG4_EWISE_LOG2:
        stack[sp - 1] = log2f(stack[sp - 1]);
        break;
      case TG4_EWISE_SIN:
        stack[sp - 1] = sinf(stack[sp - 1]);
        break;
      case TG4_EWISE_COS:
        stack[sp - 1] = cosf(stack[sp - 1]);
        break;
      case TG4_EWISE_TAN:
        stack[sp - 1] = tanf(stack[sp - 1]);
        break;
      case TG4_EWISE_ADD: {
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = a + b;
        break;
      }
      case TG4_EWISE_SUB: {
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = a - b;
        break;
      }
      case TG4_EWISE_MUL: {
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = a * b;
        break;
      }
      case TG4_EWISE_DIV: {
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = a / b;
        break;
      }
      case TG4_EWISE_MAX: {
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = a > b ? a : b;
        break;
      }
      case TG4_EWISE_WHERE: {
        float y = stack[--sp];
        float x = stack[--sp];
        float c = stack[--sp];
        stack[sp++] = c != 0.0f ? x : y;
        break;
      }
      case TG4_EWISE_POW: {
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = powf(a, b);
        break;
      }
      case TG4_EWISE_MULACC: {
        float c = stack[--sp];
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = a * b + c;
        break;
      }
      default:
        return 0.0f;
    }
  }
  return sp ? stack[sp - 1] : 0.0f;
}

typedef struct {
  size_t n_inputs;
  const uint8_t** inputs;
  view_info* views;
  uint8_t* in_is_bool;
  size_t out_rank;
  size_t* out_dims;
  uint64_t* prog;
  size_t prog_len;
} fused_ewise_view_ctx;

typedef struct {
  size_t n_inputs;
  const uint8_t** inputs;
  view_stack* stacks;
  uint8_t* in_is_bool;
  size_t out_rank;
  size_t* out_dims;
  uint64_t* prog;
  size_t prog_len;
  size_t max_rank;
} fused_ewise_stack_ctx;

static void fused_ewise_view_free(fused_ewise_view_ctx* ctx) {
  if (!ctx) return;
  if (ctx->views) {
    for (size_t i = 0; i < ctx->n_inputs; ++i) {
      view_info_free(&ctx->views[i]);
    }
  }
  free(ctx->inputs);
  free(ctx->views);
  free(ctx->in_is_bool);
  free(ctx->out_dims);
  free(ctx->prog);
}

static void fused_ewise_stack_free(fused_ewise_stack_ctx* ctx) {
  if (!ctx) return;
  if (ctx->stacks) {
    for (size_t i = 0; i < ctx->n_inputs; ++i) {
      view_stack_free(&ctx->stacks[i]);
    }
  }
  free(ctx->inputs);
  free(ctx->stacks);
  free(ctx->in_is_bool);
  free(ctx->out_dims);
  free(ctx->prog);
}

static int fused_ewise_stack_init(fused_ewise_stack_ctx* ctx, b_lean_obj_arg inputs,
    b_lean_obj_arg stackShapes, b_lean_obj_arg stackStrides, b_lean_obj_arg stackOffsets,
    b_lean_obj_arg stackMaskStarts, b_lean_obj_arg stackMaskEnds, b_lean_obj_arg inputDtypes,
    b_lean_obj_arg outShape, b_lean_obj_arg prog) {
  memset(ctx, 0, sizeof(*ctx));
  ctx->n_inputs = array_size(inputs);
  ctx->out_rank = array_size(outShape);
  ctx->prog_len = array_size(prog);
  ctx->inputs = malloc(ctx->n_inputs * sizeof(uint8_t*));
  ctx->stacks = ctx->n_inputs == 0 ? NULL : calloc(ctx->n_inputs, sizeof(view_stack));
  ctx->in_is_bool = malloc(ctx->n_inputs * sizeof(uint8_t));
  ctx->out_dims = ctx->out_rank == 0 ? NULL : malloc(ctx->out_rank * sizeof(size_t));
  ctx->prog = ctx->prog_len == 0 ? NULL : malloc(ctx->prog_len * sizeof(uint64_t));
  if ((!ctx->inputs && ctx->n_inputs) || (!ctx->stacks && ctx->n_inputs) ||
      (!ctx->in_is_bool && ctx->n_inputs) || (!ctx->out_dims && ctx->out_rank) ||
      (!ctx->prog && ctx->prog_len)) {
    fused_ewise_stack_free(ctx);
    return 0;
  }

  for (size_t i = 0; i < ctx->out_rank; ++i) {
    ctx->out_dims[i] = nat_array_get(outShape, i);
  }

  ctx->max_rank = 0;
  for (size_t i = 0; i < ctx->n_inputs; ++i) {
    lean_object* inp = lean_array_get_core(inputs, i);
    lean_object* shapesArr = lean_array_get_core(stackShapes, i);
    lean_object* stridesArr = lean_array_get_core(stackStrides, i);
    lean_object* offsetsArr = lean_array_get_core(stackOffsets, i);
    lean_object* maskStartArr = lean_array_get_core(stackMaskStarts, i);
    lean_object* maskEndArr = lean_array_get_core(stackMaskEnds, i);
    ctx->inputs[i] = byte_array_cptr(inp);
    ctx->in_is_bool[i] = nat_array_get(inputDtypes, i) == 1 ? 1 : 0;
    if (!view_stack_init(&ctx->stacks[i], shapesArr, stridesArr, offsetsArr, maskStartArr,
        maskEndArr)) {
      fused_ewise_stack_free(ctx);
      return 0;
    }
    size_t top_rank = ctx->stacks[i].views[ctx->stacks[i].depth - 1].rank;
    if (top_rank != ctx->out_rank) {
      fused_ewise_stack_free(ctx);
      return 0;
    }
    if (ctx->stacks[i].max_rank > ctx->max_rank) ctx->max_rank = ctx->stacks[i].max_rank;
  }

  for (size_t i = 0; i < ctx->prog_len; ++i) {
    ctx->prog[i] = uint64_array_get(prog, i);
  }
  return 1;
}

static int fused_ewise_view_init(fused_ewise_view_ctx* ctx, b_lean_obj_arg inputs,
    b_lean_obj_arg inputStrides, b_lean_obj_arg inputOffsets, b_lean_obj_arg inputMaskStarts,
    b_lean_obj_arg inputMaskEnds, b_lean_obj_arg inputDtypes, b_lean_obj_arg outShape,
    b_lean_obj_arg prog) {
  memset(ctx, 0, sizeof(*ctx));
  ctx->n_inputs = array_size(inputs);
  ctx->out_rank = array_size(outShape);
  ctx->prog_len = array_size(prog);
  ctx->inputs = malloc(ctx->n_inputs * sizeof(uint8_t*));
  ctx->views = ctx->n_inputs == 0 ? NULL : calloc(ctx->n_inputs, sizeof(view_info));
  ctx->in_is_bool = malloc(ctx->n_inputs * sizeof(uint8_t));
  ctx->out_dims = ctx->out_rank == 0 ? NULL : malloc(ctx->out_rank * sizeof(size_t));
  ctx->prog = ctx->prog_len == 0 ? NULL : malloc(ctx->prog_len * sizeof(uint64_t));
  if ((!ctx->inputs && ctx->n_inputs) || (!ctx->views && ctx->n_inputs) ||
      (!ctx->in_is_bool && ctx->n_inputs) || (!ctx->out_dims && ctx->out_rank) ||
      (!ctx->prog && ctx->prog_len)) {
    fused_ewise_view_free(ctx);
    return 0;
  }

  for (size_t i = 0; i < ctx->out_rank; ++i) {
    ctx->out_dims[i] = nat_array_get(outShape, i);
  }

  for (size_t i = 0; i < ctx->n_inputs; ++i) {
    lean_object* inp = lean_array_get_core(inputs, i);
    lean_object* strides = lean_array_get_core(inputStrides, i);
    lean_object* maskStart = lean_array_get_core(inputMaskStarts, i);
    lean_object* maskEnd = lean_array_get_core(inputMaskEnds, i);
    ctx->inputs[i] = byte_array_cptr(inp);
    ctx->in_is_bool[i] = nat_array_get(inputDtypes, i) == 1 ? 1 : 0;
    size_t rank = array_size(strides);
    if (rank != ctx->out_rank) {
      fused_ewise_view_free(ctx);
      return 0;
    }
    int64_t off = int64_array_get(inputOffsets, i);
    if (!view_info_init(&ctx->views[i], strides, off, maskStart, maskEnd)) {
      fused_ewise_view_free(ctx);
      return 0;
    }
  }

  for (size_t i = 0; i < ctx->prog_len; ++i) {
    ctx->prog[i] = uint64_array_get(prog, i);
  }
  return 1;
}

static int fused_ewise_view_init_stack(fused_ewise_view_ctx* ctx, b_lean_obj_arg inputs,
    b_lean_obj_arg stackStrides, b_lean_obj_arg stackOffsets, b_lean_obj_arg stackMaskStarts,
    b_lean_obj_arg stackMaskEnds, b_lean_obj_arg inputDtypes, b_lean_obj_arg outShape,
    b_lean_obj_arg prog) {
  memset(ctx, 0, sizeof(*ctx));
  ctx->n_inputs = array_size(inputs);
  ctx->out_rank = array_size(outShape);
  ctx->prog_len = array_size(prog);
  ctx->inputs = malloc(ctx->n_inputs * sizeof(uint8_t*));
  ctx->views = ctx->n_inputs == 0 ? NULL : calloc(ctx->n_inputs, sizeof(view_info));
  ctx->in_is_bool = malloc(ctx->n_inputs * sizeof(uint8_t));
  ctx->out_dims = ctx->out_rank == 0 ? NULL : malloc(ctx->out_rank * sizeof(size_t));
  ctx->prog = ctx->prog_len == 0 ? NULL : malloc(ctx->prog_len * sizeof(uint64_t));
  if ((!ctx->inputs && ctx->n_inputs) || (!ctx->views && ctx->n_inputs) ||
      (!ctx->in_is_bool && ctx->n_inputs) || (!ctx->out_dims && ctx->out_rank) ||
      (!ctx->prog && ctx->prog_len)) {
    fused_ewise_view_free(ctx);
    return 0;
  }

  for (size_t i = 0; i < ctx->out_rank; ++i) {
    ctx->out_dims[i] = nat_array_get(outShape, i);
  }

  for (size_t i = 0; i < ctx->n_inputs; ++i) {
    lean_object* inp = lean_array_get_core(inputs, i);
    lean_object* stridesArr = lean_array_get_core(stackStrides, i);
    lean_object* offsetsArr = lean_array_get_core(stackOffsets, i);
    lean_object* maskStartArr = lean_array_get_core(stackMaskStarts, i);
    lean_object* maskEndArr = lean_array_get_core(stackMaskEnds, i);
    if (array_size(stridesArr) == 0 || array_size(offsetsArr) == 0 ||
        array_size(maskStartArr) == 0 || array_size(maskEndArr) == 0) {
      fused_ewise_view_free(ctx);
      return 0;
    }
    lean_object* strides = lean_array_get_core(stridesArr, 0);
    lean_object* maskStart = lean_array_get_core(maskStartArr, 0);
    lean_object* maskEnd = lean_array_get_core(maskEndArr, 0);
    int64_t off = int64_array_get(offsetsArr, 0);
    ctx->inputs[i] = byte_array_cptr(inp);
    ctx->in_is_bool[i] = nat_array_get(inputDtypes, i) == 1 ? 1 : 0;
    size_t rank = array_size(strides);
    if (rank != ctx->out_rank) {
      fused_ewise_view_free(ctx);
      return 0;
    }
    if (!view_info_init(&ctx->views[i], strides, off, maskStart, maskEnd)) {
      fused_ewise_view_free(ctx);
      return 0;
    }
  }

  for (size_t i = 0; i < ctx->prog_len; ++i) {
    ctx->prog[i] = uint64_array_get(prog, i);
  }
  return 1;
}

static float fused_ewise_view_eval_at(const fused_ewise_view_ctx* ctx, const size_t* out_idx,
    float* stack) {
  size_t sp = 0;
  for (size_t i = 0; i < ctx->prog_len; ++i) {
    uint64_t instr = ctx->prog[i];
    uint64_t op = instr & 0xFF;
    uint64_t imm = instr >> 8;
    switch (op) {
      case TG4_EWISE_PUSH: {
        size_t idx = (size_t)imm;
        float v = view_read_value(ctx->inputs[idx], &ctx->views[idx], out_idx,
          ctx->in_is_bool[idx]);
        stack[sp++] = v;
        break;
      }
      case TG4_EWISE_NEG:
        stack[sp - 1] = -stack[sp - 1];
        break;
      case TG4_EWISE_SQRT:
        stack[sp - 1] = sqrtf(stack[sp - 1]);
        break;
      case TG4_EWISE_RECIP:
        stack[sp - 1] = 1.0f / stack[sp - 1];
        break;
      case TG4_EWISE_EXP2:
        stack[sp - 1] = exp2f(stack[sp - 1]);
        break;
      case TG4_EWISE_LOG2:
        stack[sp - 1] = log2f(stack[sp - 1]);
        break;
      case TG4_EWISE_SIN:
        stack[sp - 1] = sinf(stack[sp - 1]);
        break;
      case TG4_EWISE_ADD: {
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = a + b;
        break;
      }
      case TG4_EWISE_SUB: {
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = a - b;
        break;
      }
      case TG4_EWISE_MUL: {
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = a * b;
        break;
      }
      case TG4_EWISE_DIV: {
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = a / b;
        break;
      }
      case TG4_EWISE_MAX: {
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = a > b ? a : b;
        break;
      }
      case TG4_EWISE_WHERE: {
        float y = stack[--sp];
        float x = stack[--sp];
        float c = stack[--sp];
        stack[sp++] = c != 0.0f ? x : y;
        break;
      }
      case TG4_EWISE_MULACC: {
        float c = stack[--sp];
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = a * b + c;
        break;
      }
      default:
        return 0.0f;
    }
  }
  return sp ? stack[sp - 1] : 0.0f;
}

static float fused_ewise_stack_eval_at(const fused_ewise_stack_ctx* ctx, const size_t* out_idx,
    float* stack, size_t* idx_buf, size_t* tmp_buf) {
  size_t sp = 0;
  for (size_t i = 0; i < ctx->prog_len; ++i) {
    uint64_t instr = ctx->prog[i];
    uint64_t op = instr & 0xFF;
    uint64_t imm = instr >> 8;
    switch (op) {
      case TG4_EWISE_PUSH: {
        size_t idx = (size_t)imm;
        float v = view_stack_read_value(ctx->inputs[idx], &ctx->stacks[idx], out_idx,
          idx_buf, tmp_buf, ctx->in_is_bool[idx]);
        stack[sp++] = v;
        break;
      }
      case TG4_EWISE_NEG:
        stack[sp - 1] = -stack[sp - 1];
        break;
      case TG4_EWISE_SQRT:
        stack[sp - 1] = sqrtf(stack[sp - 1]);
        break;
      case TG4_EWISE_RECIP:
        stack[sp - 1] = 1.0f / stack[sp - 1];
        break;
      case TG4_EWISE_EXP2:
        stack[sp - 1] = exp2f(stack[sp - 1]);
        break;
      case TG4_EWISE_LOG2:
        stack[sp - 1] = log2f(stack[sp - 1]);
        break;
      case TG4_EWISE_SIN:
        stack[sp - 1] = sinf(stack[sp - 1]);
        break;
      case TG4_EWISE_ADD: {
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = a + b;
        break;
      }
      case TG4_EWISE_SUB: {
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = a - b;
        break;
      }
      case TG4_EWISE_MUL: {
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = a * b;
        break;
      }
      case TG4_EWISE_DIV: {
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = a / b;
        break;
      }
      case TG4_EWISE_MAX: {
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = a > b ? a : b;
        break;
      }
      case TG4_EWISE_WHERE: {
        float y = stack[--sp];
        float x = stack[--sp];
        float c = stack[--sp];
        stack[sp++] = c != 0.0f ? x : y;
        break;
      }
      case TG4_EWISE_MULACC: {
        float c = stack[--sp];
        float b = stack[--sp];
        float a = stack[--sp];
        stack[sp++] = a * b + c;
        break;
      }
      default:
        return 0.0f;
    }
  }
  return sp ? stack[sp - 1] : 0.0f;
}

LEAN_EXPORT lean_obj_res tg4_fused_ewise_f32(lean_obj_arg inputs, lean_obj_arg inputShapes,
    lean_obj_arg inputDtypes, lean_obj_arg outShape, lean_obj_arg prog) {
  /* NOTE: Fast path detection is now done in Lean (FusedEwise.Kernel).
     This function is only called for bytecode fallback. */
  fused_ewise_ctx ctx;
  if (!fused_ewise_init(&ctx, inputs, inputShapes, inputDtypes, outShape, prog)) {
    return mk_byte_array(0);
  }
  size_t out_numel = shape_numel(outShape);
  lean_object* out = mk_byte_array(out_numel * 4);
  uint8_t* o = byte_array_cptr(out);
  float* stack = ctx.prog_len == 0 ? NULL : malloc(ctx.prog_len * sizeof(float));
  size_t* out_idx = ctx.out_rank == 0 ? NULL : malloc(ctx.out_rank * sizeof(size_t));
  if ((ctx.prog_len && !stack) || (ctx.out_rank && !out_idx)) {
    free(stack);
    free(out_idx);
    fused_ewise_free(&ctx);
    return mk_byte_array(0);
  }
  for (size_t i = 0; i < out_numel; ++i) {
    unflatten_index(i, ctx.out_dims, ctx.out_rank, out_idx);
    float v = fused_ewise_eval_at(&ctx, out_idx, stack);
    write_f32(o, i, v);
  }
  free(stack);
  free(out_idx);
  fused_ewise_free(&ctx);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_ewise_view_f32(lean_obj_arg inputs, lean_obj_arg inputStrides,
    lean_obj_arg inputOffsets, lean_obj_arg inputMaskStarts, lean_obj_arg inputMaskEnds,
    lean_obj_arg inputDtypes, lean_obj_arg outShape, lean_obj_arg prog) {
  fused_ewise_view_ctx ctx;
  if (!fused_ewise_view_init(&ctx, inputs, inputStrides, inputOffsets, inputMaskStarts,
      inputMaskEnds, inputDtypes, outShape, prog)) {
    return mk_byte_array(0);
  }
  size_t out_numel = shape_numel(outShape);
  lean_object* out = mk_byte_array(out_numel * 4);
  uint8_t* o = byte_array_cptr(out);
  float* stack = ctx.prog_len == 0 ? NULL : malloc(ctx.prog_len * sizeof(float));
  size_t* out_idx = ctx.out_rank == 0 ? NULL : malloc(ctx.out_rank * sizeof(size_t));
  if ((ctx.prog_len && !stack) || (ctx.out_rank && !out_idx)) {
    free(stack);
    free(out_idx);
    fused_ewise_view_free(&ctx);
    return mk_byte_array(0);
  }
  for (size_t i = 0; i < out_numel; ++i) {
    unflatten_index(i, ctx.out_dims, ctx.out_rank, out_idx);
    float v = fused_ewise_view_eval_at(&ctx, out_idx, stack);
    write_f32(o, i, v);
  }
  free(stack);
  free(out_idx);
  fused_ewise_view_free(&ctx);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_ewise_view_stack_f32(lean_obj_arg inputs,
    lean_obj_arg stackShapes, lean_obj_arg stackStrides, lean_obj_arg stackOffsets,
    lean_obj_arg stackMaskStarts, lean_obj_arg stackMaskEnds, lean_obj_arg inputDtypes,
    lean_obj_arg outShape, lean_obj_arg prog) {
  fused_ewise_stack_ctx ctx;
  if (!fused_ewise_stack_init(&ctx, inputs, stackShapes, stackStrides, stackOffsets,
      stackMaskStarts, stackMaskEnds, inputDtypes, outShape, prog)) {
    return mk_byte_array(0);
  }
  size_t out_numel = shape_numel(outShape);
  lean_object* out = mk_byte_array(out_numel * 4);
  uint8_t* o = byte_array_cptr(out);
  float* stack = ctx.prog_len == 0 ? NULL : malloc(ctx.prog_len * sizeof(float));
  size_t* out_idx = ctx.out_rank == 0 ? NULL : malloc(ctx.out_rank * sizeof(size_t));
  size_t* idx_buf = ctx.max_rank == 0 ? NULL : malloc(ctx.max_rank * sizeof(size_t));
  size_t* tmp_buf = ctx.max_rank == 0 ? NULL : malloc(ctx.max_rank * sizeof(size_t));
  if ((ctx.prog_len && !stack) || (ctx.out_rank && !out_idx) ||
      (ctx.max_rank && !idx_buf) || (ctx.max_rank && !tmp_buf)) {
    free(stack);
    free(out_idx);
    free(idx_buf);
    free(tmp_buf);
    fused_ewise_stack_free(&ctx);
    return mk_byte_array(0);
  }
  for (size_t i = 0; i < out_numel; ++i) {
    unflatten_index(i, ctx.out_dims, ctx.out_rank, out_idx);
    float v = fused_ewise_stack_eval_at(&ctx, out_idx, stack, idx_buf, tmp_buf);
    write_f32(o, i, v);
  }
  free(stack);
  free(out_idx);
  free(idx_buf);
  free(tmp_buf);
  fused_ewise_stack_free(&ctx);
  return out;
}

static lean_object* fused_reduce_all(lean_obj_arg inputs, lean_obj_arg inputShapes,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg prog, int is_max) {
  fused_ewise_ctx ctx;
  if (!fused_ewise_init(&ctx, inputs, inputShapes, inputDtypes, fullShape, prog)) {
    return mk_byte_array(0);
  }
  size_t full_numel = shape_numel(fullShape);
  float* stack = ctx.prog_len == 0 ? NULL : malloc(ctx.prog_len * sizeof(float));
  size_t* idx = ctx.out_rank == 0 ? NULL : malloc(ctx.out_rank * sizeof(size_t));
  if ((ctx.prog_len && !stack) || (ctx.out_rank && !idx)) {
    free(stack);
    free(idx);
    fused_ewise_free(&ctx);
    return mk_byte_array(0);
  }
  float acc = is_max ? -INFINITY : 0.0f;
  for (size_t i = 0; i < full_numel; ++i) {
    unflatten_index(i, ctx.out_dims, ctx.out_rank, idx);
    float v = fused_ewise_eval_at(&ctx, idx, stack);
    if (is_max) {
      acc = v > acc ? v : acc;
    } else {
      acc += v;
    }
  }
  lean_object* out = mk_byte_array(4);
  write_f32(byte_array_cptr(out), 0, acc);
  free(stack);
  free(idx);
  fused_ewise_free(&ctx);
  return out;
}

static lean_object* fused_reduce_axis(lean_obj_arg inputs, lean_obj_arg inputShapes,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg prog, size_t axis, int is_max) {
  fused_ewise_ctx ctx;
  if (!fused_ewise_init(&ctx, inputs, inputShapes, inputDtypes, fullShape, prog)) {
    return mk_byte_array(0);
  }
  size_t full_rank = ctx.out_rank;
  if (axis >= full_rank) {
    fused_ewise_free(&ctx);
    return mk_byte_array(0);
  }
  size_t out_rank = full_rank > 0 ? (full_rank - 1) : 0;
  size_t* out_dims = out_rank == 0 ? NULL : malloc(out_rank * sizeof(size_t));
  size_t* out_strides = out_rank == 0 ? NULL : malloc(out_rank * sizeof(size_t));
  if ((out_rank && !out_dims) || (out_rank && !out_strides)) {
    free(out_dims);
    free(out_strides);
    fused_ewise_free(&ctx);
    return mk_byte_array(0);
  }
  for (size_t i = 0, j = 0; i < full_rank; ++i) {
    if (i == axis) continue;
    out_dims[j++] = ctx.out_dims[i];
  }
  make_strides(out_dims, out_rank, out_strides);
  size_t out_numel = 1;
  for (size_t i = 0; i < out_rank; ++i) out_numel *= out_dims[i];

  lean_object* out = mk_byte_array(out_numel * 4);
  uint8_t* o = byte_array_cptr(out);
  for (size_t i = 0; i < out_numel; ++i) {
    write_f32(o, i, is_max ? -INFINITY : 0.0f);
  }

  size_t full_numel = shape_numel(fullShape);
  float* stack = ctx.prog_len == 0 ? NULL : malloc(ctx.prog_len * sizeof(float));
  size_t* full_idx = full_rank == 0 ? NULL : malloc(full_rank * sizeof(size_t));
  size_t* out_idx = out_rank == 0 ? NULL : malloc(out_rank * sizeof(size_t));
  if ((ctx.prog_len && !stack) || (full_rank && !full_idx) || (out_rank && !out_idx)) {
    free(stack);
    free(full_idx);
    free(out_idx);
    free(out_dims);
    free(out_strides);
    fused_ewise_free(&ctx);
    return mk_byte_array(0);
  }

  for (size_t flat = 0; flat < full_numel; ++flat) {
    unflatten_index(flat, ctx.out_dims, full_rank, full_idx);
    for (size_t i = 0, j = 0; i < full_rank; ++i) {
      if (i == axis) continue;
      out_idx[j++] = full_idx[i];
    }
    size_t out_flat = 0;
    for (size_t i = 0; i < out_rank; ++i) {
      out_flat += out_idx[i] * out_strides[i];
    }
    float v = fused_ewise_eval_at(&ctx, full_idx, stack);
    float cur = read_f32(o, out_flat);
    if (is_max) {
      write_f32(o, out_flat, v > cur ? v : cur);
    } else {
      write_f32(o, out_flat, cur + v);
    }
  }

  free(stack);
  free(full_idx);
  free(out_idx);
  free(out_dims);
  free(out_strides);
  fused_ewise_free(&ctx);
  return out;
}

typedef struct {
  size_t full_rank;
  size_t out_rank;
  size_t axes_n;
  size_t* full_dims;
  size_t* out_dims;
  uint8_t* reduce_mask;
  size_t* reduce_axes;
  size_t* reduce_dims;
  size_t out_numel;
  size_t reduce_numel;
} reduce_axes_info;

static void reduce_axes_info_free(reduce_axes_info* info) {
  if (!info) return;
  free(info->full_dims);
  free(info->out_dims);
  free(info->reduce_mask);
  free(info->reduce_axes);
  free(info->reduce_dims);
}

static int reduce_axes_info_init(reduce_axes_info* info, b_lean_obj_arg fullShape,
    b_lean_obj_arg outShape, b_lean_obj_arg axes) {
  memset(info, 0, sizeof(*info));
  info->full_rank = array_size(fullShape);
  info->out_rank = array_size(outShape);
  size_t axes_in = array_size(axes);
  if (axes_in == 0) return 0;
  info->full_dims = info->full_rank == 0 ? NULL : malloc(info->full_rank * sizeof(size_t));
  info->out_dims = info->out_rank == 0 ? NULL : malloc(info->out_rank * sizeof(size_t));
  info->reduce_mask = info->full_rank == 0 ? NULL : calloc(info->full_rank, sizeof(uint8_t));
  if ((info->full_rank && !info->full_dims) || (info->out_rank && !info->out_dims) ||
      (info->full_rank && !info->reduce_mask)) {
    reduce_axes_info_free(info);
    return 0;
  }
  for (size_t i = 0; i < info->full_rank; ++i) {
    info->full_dims[i] = nat_array_get(fullShape, i);
  }
  for (size_t i = 0; i < info->out_rank; ++i) {
    info->out_dims[i] = nat_array_get(outShape, i);
  }
  for (size_t i = 0; i < axes_in; ++i) {
    size_t ax = nat_array_get(axes, i);
    if (ax >= info->full_rank) {
      reduce_axes_info_free(info);
      return 0;
    }
    info->reduce_mask[ax] = 1;
  }
  size_t uniq_axes = 0;
  for (size_t i = 0; i < info->full_rank; ++i) {
    if (info->reduce_mask[i]) uniq_axes++;
  }
  if (uniq_axes == 0) {
    reduce_axes_info_free(info);
    return 0;
  }
  size_t expected_out_rank = info->full_rank - uniq_axes;
  if (info->out_rank != info->full_rank && info->out_rank != expected_out_rank) {
    reduce_axes_info_free(info);
    return 0;
  }
  if (info->out_rank == info->full_rank) {
    for (size_t i = 0; i < info->full_rank; ++i) {
      if (info->reduce_mask[i] && info->out_dims[i] != 1) {
        reduce_axes_info_free(info);
        return 0;
      }
    }
  }

  info->reduce_axes = malloc(uniq_axes * sizeof(size_t));
  info->reduce_dims = malloc(uniq_axes * sizeof(size_t));
  if (!info->reduce_axes || !info->reduce_dims) {
    reduce_axes_info_free(info);
    return 0;
  }
  size_t ridx = 0;
  for (size_t i = 0; i < info->full_rank; ++i) {
    if (!info->reduce_mask[i]) continue;
    info->reduce_axes[ridx] = i;
    info->reduce_dims[ridx] = info->full_dims[i];
    ridx++;
  }
  info->axes_n = uniq_axes;
  info->out_numel = 1;
  for (size_t i = 0; i < info->out_rank; ++i) info->out_numel *= info->out_dims[i];
  info->reduce_numel = 1;
  for (size_t i = 0; i < info->axes_n; ++i) info->reduce_numel *= info->reduce_dims[i];
  return 1;
}

static lean_object* fused_reduce_all_view_ctx(fused_ewise_view_ctx* ctx,
    b_lean_obj_arg fullShape, int is_max) {
  size_t full_numel = shape_numel(fullShape);
  float* stack = ctx->prog_len == 0 ? NULL : malloc(ctx->prog_len * sizeof(float));
  size_t* idx = ctx->out_rank == 0 ? NULL : malloc(ctx->out_rank * sizeof(size_t));
  if ((ctx->prog_len && !stack) || (ctx->out_rank && !idx)) {
    free(stack);
    free(idx);
    return mk_byte_array(0);
  }
  float acc = is_max ? -INFINITY : 0.0f;
  for (size_t i = 0; i < full_numel; ++i) {
    unflatten_index(i, ctx->out_dims, ctx->out_rank, idx);
    float v = fused_ewise_view_eval_at(ctx, idx, stack);
    if (is_max) {
      acc = v > acc ? v : acc;
    } else {
      acc += v;
    }
  }
  lean_object* out = mk_byte_array(4);
  write_f32(byte_array_cptr(out), 0, acc);
  free(stack);
  free(idx);
  return out;
}

static lean_object* fused_reduce_axis_view_ctx(fused_ewise_view_ctx* ctx,
    b_lean_obj_arg fullShape, size_t axis, int is_max) {
  size_t full_rank = ctx->out_rank;
  if (axis >= full_rank) {
    return mk_byte_array(0);
  }
  size_t out_rank = full_rank > 0 ? (full_rank - 1) : 0;
  size_t* out_dims = out_rank == 0 ? NULL : malloc(out_rank * sizeof(size_t));
  size_t* out_strides = out_rank == 0 ? NULL : malloc(out_rank * sizeof(size_t));
  if ((out_rank && !out_dims) || (out_rank && !out_strides)) {
    free(out_dims);
    free(out_strides);
    return mk_byte_array(0);
  }
  for (size_t i = 0, j = 0; i < full_rank; ++i) {
    if (i == axis) continue;
    out_dims[j++] = ctx->out_dims[i];
  }
  make_strides(out_dims, out_rank, out_strides);
  size_t out_numel = 1;
  for (size_t i = 0; i < out_rank; ++i) out_numel *= out_dims[i];

  lean_object* out = mk_byte_array(out_numel * 4);
  uint8_t* o = byte_array_cptr(out);
  for (size_t i = 0; i < out_numel; ++i) {
    write_f32(o, i, is_max ? -INFINITY : 0.0f);
  }

  size_t full_numel = shape_numel(fullShape);
  float* stack = ctx->prog_len == 0 ? NULL : malloc(ctx->prog_len * sizeof(float));
  size_t* full_idx = full_rank == 0 ? NULL : malloc(full_rank * sizeof(size_t));
  size_t* out_idx = out_rank == 0 ? NULL : malloc(out_rank * sizeof(size_t));
  if ((ctx->prog_len && !stack) || (full_rank && !full_idx) || (out_rank && !out_idx)) {
    free(stack);
    free(full_idx);
    free(out_idx);
    free(out_dims);
    free(out_strides);
    return mk_byte_array(0);
  }

  for (size_t flat = 0; flat < full_numel; ++flat) {
    unflatten_index(flat, ctx->out_dims, full_rank, full_idx);
    for (size_t i = 0, j = 0; i < full_rank; ++i) {
      if (i == axis) continue;
      out_idx[j++] = full_idx[i];
    }
    size_t out_flat = 0;
    for (size_t i = 0; i < out_rank; ++i) {
      out_flat += out_idx[i] * out_strides[i];
    }
    float v = fused_ewise_view_eval_at(ctx, full_idx, stack);
    float cur = read_f32(o, out_flat);
    if (is_max) {
      write_f32(o, out_flat, v > cur ? v : cur);
    } else {
      write_f32(o, out_flat, cur + v);
    }
  }

  free(stack);
  free(full_idx);
  free(out_idx);
  free(out_dims);
  free(out_strides);
  return out;
}

static lean_object* fused_reduce_all_stack_ctx(fused_ewise_stack_ctx* ctx,
    b_lean_obj_arg fullShape, int is_max) {
  size_t full_numel = shape_numel(fullShape);
  float* stack = ctx->prog_len == 0 ? NULL : malloc(ctx->prog_len * sizeof(float));
  size_t* idx = ctx->out_rank == 0 ? NULL : malloc(ctx->out_rank * sizeof(size_t));
  size_t* idx_buf = ctx->max_rank == 0 ? NULL : malloc(ctx->max_rank * sizeof(size_t));
  size_t* tmp_buf = ctx->max_rank == 0 ? NULL : malloc(ctx->max_rank * sizeof(size_t));
  if ((ctx->prog_len && !stack) || (ctx->out_rank && !idx) ||
      (ctx->max_rank && !idx_buf) || (ctx->max_rank && !tmp_buf)) {
    free(stack);
    free(idx);
    free(idx_buf);
    free(tmp_buf);
    return mk_byte_array(0);
  }
  float acc = is_max ? -INFINITY : 0.0f;
  for (size_t i = 0; i < full_numel; ++i) {
    unflatten_index(i, ctx->out_dims, ctx->out_rank, idx);
    float v = fused_ewise_stack_eval_at(ctx, idx, stack, idx_buf, tmp_buf);
    if (is_max) {
      acc = v > acc ? v : acc;
    } else {
      acc += v;
    }
  }
  lean_object* out = mk_byte_array(4);
  write_f32(byte_array_cptr(out), 0, acc);
  free(stack);
  free(idx);
  free(idx_buf);
  free(tmp_buf);
  return out;
}

static lean_object* fused_reduce_axis_stack_ctx(fused_ewise_stack_ctx* ctx,
    b_lean_obj_arg fullShape, size_t axis, int is_max) {
  size_t full_rank = ctx->out_rank;
  if (axis >= full_rank) {
    return mk_byte_array(0);
  }
  size_t out_rank = full_rank > 0 ? (full_rank - 1) : 0;
  size_t* out_dims = out_rank == 0 ? NULL : malloc(out_rank * sizeof(size_t));
  size_t* out_strides = out_rank == 0 ? NULL : malloc(out_rank * sizeof(size_t));
  if ((out_rank && !out_dims) || (out_rank && !out_strides)) {
    free(out_dims);
    free(out_strides);
    return mk_byte_array(0);
  }
  for (size_t i = 0, j = 0; i < full_rank; ++i) {
    if (i == axis) continue;
    out_dims[j++] = ctx->out_dims[i];
  }
  make_strides(out_dims, out_rank, out_strides);
  size_t out_numel = 1;
  for (size_t i = 0; i < out_rank; ++i) out_numel *= out_dims[i];

  lean_object* out = mk_byte_array(out_numel * 4);
  uint8_t* o = byte_array_cptr(out);
  for (size_t i = 0; i < out_numel; ++i) {
    write_f32(o, i, is_max ? -INFINITY : 0.0f);
  }

  size_t full_numel = shape_numel(fullShape);
  float* stack = ctx->prog_len == 0 ? NULL : malloc(ctx->prog_len * sizeof(float));
  size_t* full_idx = full_rank == 0 ? NULL : malloc(full_rank * sizeof(size_t));
  size_t* out_idx = out_rank == 0 ? NULL : malloc(out_rank * sizeof(size_t));
  size_t* idx_buf = ctx->max_rank == 0 ? NULL : malloc(ctx->max_rank * sizeof(size_t));
  size_t* tmp_buf = ctx->max_rank == 0 ? NULL : malloc(ctx->max_rank * sizeof(size_t));
  if ((ctx->prog_len && !stack) || (full_rank && !full_idx) || (out_rank && !out_idx) ||
      (ctx->max_rank && !idx_buf) || (ctx->max_rank && !tmp_buf)) {
    free(stack);
    free(full_idx);
    free(out_idx);
    free(idx_buf);
    free(tmp_buf);
    free(out_dims);
    free(out_strides);
    return mk_byte_array(0);
  }

  for (size_t flat = 0; flat < full_numel; ++flat) {
    unflatten_index(flat, ctx->out_dims, full_rank, full_idx);
    for (size_t i = 0, j = 0; i < full_rank; ++i) {
      if (i == axis) continue;
      out_idx[j++] = full_idx[i];
    }
    size_t out_flat = 0;
    for (size_t i = 0; i < out_rank; ++i) {
      out_flat += out_idx[i] * out_strides[i];
    }
    float v = fused_ewise_stack_eval_at(ctx, full_idx, stack, idx_buf, tmp_buf);
    float cur = read_f32(o, out_flat);
    if (is_max) {
      write_f32(o, out_flat, v > cur ? v : cur);
    } else {
      write_f32(o, out_flat, cur + v);
    }
  }

  free(stack);
  free(full_idx);
  free(out_idx);
  free(idx_buf);
  free(tmp_buf);
  free(out_dims);
  free(out_strides);
  return out;
}

static lean_object* fused_reduce_all_view(lean_obj_arg inputs, lean_obj_arg inputStrides,
    lean_obj_arg inputOffsets, lean_obj_arg inputMaskStarts, lean_obj_arg inputMaskEnds,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg prog, int is_max) {
  fused_ewise_view_ctx ctx;
  if (!fused_ewise_view_init(&ctx, inputs, inputStrides, inputOffsets, inputMaskStarts,
      inputMaskEnds, inputDtypes, fullShape, prog)) {
    return mk_byte_array(0);
  }
  lean_object* out = fused_reduce_all_view_ctx(&ctx, fullShape, is_max);
  fused_ewise_view_free(&ctx);
  return out;
}

static lean_object* fused_reduce_axis_view(lean_obj_arg inputs, lean_obj_arg inputStrides,
    lean_obj_arg inputOffsets, lean_obj_arg inputMaskStarts, lean_obj_arg inputMaskEnds,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg prog, size_t axis, int is_max) {
  fused_ewise_view_ctx ctx;
  if (!fused_ewise_view_init(&ctx, inputs, inputStrides, inputOffsets, inputMaskStarts,
      inputMaskEnds, inputDtypes, fullShape, prog)) {
    return mk_byte_array(0);
  }
  lean_object* out = fused_reduce_axis_view_ctx(&ctx, fullShape, axis, is_max);
  fused_ewise_view_free(&ctx);
  return out;
}

static lean_object* fused_reduce_all_view_stack(lean_obj_arg inputs, lean_obj_arg stackShapes,
    lean_obj_arg stackStrides, lean_obj_arg stackOffsets, lean_obj_arg stackMaskStarts,
    lean_obj_arg stackMaskEnds,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg prog, int is_max) {
  fused_ewise_stack_ctx ctx;
  if (!fused_ewise_stack_init(&ctx, inputs, stackShapes, stackStrides, stackOffsets,
      stackMaskStarts, stackMaskEnds, inputDtypes, fullShape, prog)) {
    return mk_byte_array(0);
  }
  lean_object* out = fused_reduce_all_stack_ctx(&ctx, fullShape, is_max);
  fused_ewise_stack_free(&ctx);
  return out;
}

static lean_object* fused_reduce_axis_view_stack(lean_obj_arg inputs, lean_obj_arg stackShapes,
    lean_obj_arg stackStrides, lean_obj_arg stackOffsets, lean_obj_arg stackMaskStarts,
    lean_obj_arg stackMaskEnds,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg prog, size_t axis, int is_max) {
  fused_ewise_stack_ctx ctx;
  if (!fused_ewise_stack_init(&ctx, inputs, stackShapes, stackStrides, stackOffsets,
      stackMaskStarts, stackMaskEnds, inputDtypes, fullShape, prog)) {
    return mk_byte_array(0);
  }
  lean_object* out = fused_reduce_axis_stack_ctx(&ctx, fullShape, axis, is_max);
  fused_ewise_stack_free(&ctx);
  return out;
}

static lean_object* fused_reduce_axes(lean_obj_arg inputs, lean_obj_arg inputShapes,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg outShape, lean_obj_arg axes,
    lean_obj_arg prog, int is_max) {
  fused_ewise_ctx ctx;
  if (!fused_ewise_init(&ctx, inputs, inputShapes, inputDtypes, fullShape, prog)) {
    return mk_byte_array(0);
  }
  reduce_axes_info info;
  if (!reduce_axes_info_init(&info, fullShape, outShape, axes)) {
    fused_ewise_free(&ctx);
    return mk_byte_array(0);
  }
  lean_object* out = mk_byte_array(info.out_numel * 4);
  uint8_t* o = byte_array_cptr(out);
  float* stack = ctx.prog_len == 0 ? NULL : malloc(ctx.prog_len * sizeof(float));
  size_t* out_idx = info.out_rank == 0 ? NULL : malloc(info.out_rank * sizeof(size_t));
  size_t* full_idx = info.full_rank == 0 ? NULL : malloc(info.full_rank * sizeof(size_t));
  size_t* reduce_idx = info.axes_n == 0 ? NULL : malloc(info.axes_n * sizeof(size_t));
  if ((ctx.prog_len && !stack) || (info.out_rank && !out_idx) || (info.full_rank && !full_idx) ||
      (info.axes_n && !reduce_idx)) {
    free(stack);
    free(out_idx);
    free(full_idx);
    free(reduce_idx);
    reduce_axes_info_free(&info);
    fused_ewise_free(&ctx);
    return mk_byte_array(0);
  }

  for (size_t out_flat = 0; out_flat < info.out_numel; ++out_flat) {
    if (info.out_rank) {
      unflatten_index(out_flat, info.out_dims, info.out_rank, out_idx);
    }
    if (info.out_rank == info.full_rank) {
      for (size_t i = 0; i < info.full_rank; ++i) {
        full_idx[i] = info.reduce_mask[i] ? 0 : out_idx[i];
      }
    } else {
      size_t pos = 0;
      for (size_t i = 0; i < info.full_rank; ++i) {
        if (info.reduce_mask[i]) {
          full_idx[i] = 0;
        } else {
          full_idx[i] = out_idx[pos++];
        }
      }
    }

    float acc = is_max ? -INFINITY : 0.0f;
    for (size_t red_flat = 0; red_flat < info.reduce_numel; ++red_flat) {
      if (info.axes_n) {
        unflatten_index(red_flat, info.reduce_dims, info.axes_n, reduce_idx);
        for (size_t r = 0; r < info.axes_n; ++r) {
          full_idx[info.reduce_axes[r]] = reduce_idx[r];
        }
      }
      float v = fused_ewise_eval_at(&ctx, full_idx, stack);
      if (is_max) {
        acc = v > acc ? v : acc;
      } else {
        acc += v;
      }
    }
    write_f32(o, out_flat, acc);
  }

  free(stack);
  free(out_idx);
  free(full_idx);
  free(reduce_idx);
  reduce_axes_info_free(&info);
  fused_ewise_free(&ctx);
  return out;
}

static lean_object* fused_reduce_axes_view(lean_obj_arg inputs, lean_obj_arg inputStrides,
    lean_obj_arg inputOffsets, lean_obj_arg inputMaskStarts, lean_obj_arg inputMaskEnds,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg outShape, lean_obj_arg axes,
    lean_obj_arg prog, int is_max) {
  fused_ewise_view_ctx ctx;
  if (!fused_ewise_view_init(&ctx, inputs, inputStrides, inputOffsets, inputMaskStarts,
      inputMaskEnds, inputDtypes, fullShape, prog)) {
    return mk_byte_array(0);
  }
  reduce_axes_info info;
  if (!reduce_axes_info_init(&info, fullShape, outShape, axes)) {
    fused_ewise_view_free(&ctx);
    return mk_byte_array(0);
  }
  lean_object* out = mk_byte_array(info.out_numel * 4);
  uint8_t* o = byte_array_cptr(out);
  float* stack = ctx.prog_len == 0 ? NULL : malloc(ctx.prog_len * sizeof(float));
  size_t* out_idx = info.out_rank == 0 ? NULL : malloc(info.out_rank * sizeof(size_t));
  size_t* full_idx = info.full_rank == 0 ? NULL : malloc(info.full_rank * sizeof(size_t));
  size_t* reduce_idx = info.axes_n == 0 ? NULL : malloc(info.axes_n * sizeof(size_t));
  if ((ctx.prog_len && !stack) || (info.out_rank && !out_idx) || (info.full_rank && !full_idx) ||
      (info.axes_n && !reduce_idx)) {
    free(stack);
    free(out_idx);
    free(full_idx);
    free(reduce_idx);
    reduce_axes_info_free(&info);
    fused_ewise_view_free(&ctx);
    return mk_byte_array(0);
  }

  for (size_t out_flat = 0; out_flat < info.out_numel; ++out_flat) {
    if (info.out_rank) {
      unflatten_index(out_flat, info.out_dims, info.out_rank, out_idx);
    }
    if (info.out_rank == info.full_rank) {
      for (size_t i = 0; i < info.full_rank; ++i) {
        full_idx[i] = info.reduce_mask[i] ? 0 : out_idx[i];
      }
    } else {
      size_t pos = 0;
      for (size_t i = 0; i < info.full_rank; ++i) {
        if (info.reduce_mask[i]) {
          full_idx[i] = 0;
        } else {
          full_idx[i] = out_idx[pos++];
        }
      }
    }

    float acc = is_max ? -INFINITY : 0.0f;
    for (size_t red_flat = 0; red_flat < info.reduce_numel; ++red_flat) {
      if (info.axes_n) {
        unflatten_index(red_flat, info.reduce_dims, info.axes_n, reduce_idx);
        for (size_t r = 0; r < info.axes_n; ++r) {
          full_idx[info.reduce_axes[r]] = reduce_idx[r];
        }
      }
      float v = fused_ewise_view_eval_at(&ctx, full_idx, stack);
      if (is_max) {
        acc = v > acc ? v : acc;
      } else {
        acc += v;
      }
    }
    write_f32(o, out_flat, acc);
  }

  free(stack);
  free(out_idx);
  free(full_idx);
  free(reduce_idx);
  reduce_axes_info_free(&info);
  fused_ewise_view_free(&ctx);
  return out;
}

static lean_object* fused_reduce_axes_view_stack(lean_obj_arg inputs, lean_obj_arg stackShapes,
    lean_obj_arg stackStrides, lean_obj_arg stackOffsets, lean_obj_arg stackMaskStarts,
    lean_obj_arg stackMaskEnds, lean_obj_arg inputDtypes, lean_obj_arg fullShape,
    lean_obj_arg outShape, lean_obj_arg axes, lean_obj_arg prog, int is_max) {
  fused_ewise_stack_ctx ctx;
  if (!fused_ewise_stack_init(&ctx, inputs, stackShapes, stackStrides, stackOffsets,
      stackMaskStarts, stackMaskEnds, inputDtypes, fullShape, prog)) {
    return mk_byte_array(0);
  }
  reduce_axes_info info;
  if (!reduce_axes_info_init(&info, fullShape, outShape, axes)) {
    fused_ewise_stack_free(&ctx);
    return mk_byte_array(0);
  }
  lean_object* out = mk_byte_array(info.out_numel * 4);
  uint8_t* o = byte_array_cptr(out);
  float* stack = ctx.prog_len == 0 ? NULL : malloc(ctx.prog_len * sizeof(float));
  size_t* out_idx = info.out_rank == 0 ? NULL : malloc(info.out_rank * sizeof(size_t));
  size_t* full_idx = info.full_rank == 0 ? NULL : malloc(info.full_rank * sizeof(size_t));
  size_t* reduce_idx = info.axes_n == 0 ? NULL : malloc(info.axes_n * sizeof(size_t));
  size_t* idx_buf = ctx.max_rank == 0 ? NULL : malloc(ctx.max_rank * sizeof(size_t));
  size_t* tmp_buf = ctx.max_rank == 0 ? NULL : malloc(ctx.max_rank * sizeof(size_t));
  if ((ctx.prog_len && !stack) || (info.out_rank && !out_idx) || (info.full_rank && !full_idx) ||
      (info.axes_n && !reduce_idx) || (ctx.max_rank && !idx_buf) || (ctx.max_rank && !tmp_buf)) {
    free(stack);
    free(out_idx);
    free(full_idx);
    free(reduce_idx);
    free(idx_buf);
    free(tmp_buf);
    reduce_axes_info_free(&info);
    fused_ewise_stack_free(&ctx);
    return mk_byte_array(0);
  }

  for (size_t out_flat = 0; out_flat < info.out_numel; ++out_flat) {
    if (info.out_rank) {
      unflatten_index(out_flat, info.out_dims, info.out_rank, out_idx);
    }
    if (info.out_rank == info.full_rank) {
      for (size_t i = 0; i < info.full_rank; ++i) {
        full_idx[i] = info.reduce_mask[i] ? 0 : out_idx[i];
      }
    } else {
      size_t pos = 0;
      for (size_t i = 0; i < info.full_rank; ++i) {
        if (info.reduce_mask[i]) {
          full_idx[i] = 0;
        } else {
          full_idx[i] = out_idx[pos++];
        }
      }
    }

    float acc = is_max ? -INFINITY : 0.0f;
    for (size_t red_flat = 0; red_flat < info.reduce_numel; ++red_flat) {
      if (info.axes_n) {
        unflatten_index(red_flat, info.reduce_dims, info.axes_n, reduce_idx);
        for (size_t r = 0; r < info.axes_n; ++r) {
          full_idx[info.reduce_axes[r]] = reduce_idx[r];
        }
      }
      float v = fused_ewise_stack_eval_at(&ctx, full_idx, stack, idx_buf, tmp_buf);
      if (is_max) {
        acc = v > acc ? v : acc;
      } else {
        acc += v;
      }
    }
    write_f32(o, out_flat, acc);
  }

  free(stack);
  free(out_idx);
  free(full_idx);
  free(reduce_idx);
  free(idx_buf);
  free(tmp_buf);
  reduce_axes_info_free(&info);
  fused_ewise_stack_free(&ctx);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_last_f32(lean_obj_arg inputs, lean_obj_arg inputShapes,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg prog) {
  size_t rank = array_size(fullShape);
  if (rank == 0) {
    return fused_reduce_all(inputs, inputShapes, inputDtypes, fullShape, prog, 0);
  }
  return fused_reduce_axis(inputs, inputShapes, inputDtypes, fullShape, prog, rank - 1, 0);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_last_view_f32(lean_obj_arg inputs, lean_obj_arg inputStrides,
    lean_obj_arg inputOffsets, lean_obj_arg inputMaskStarts, lean_obj_arg inputMaskEnds,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg prog) {
  size_t rank = array_size(fullShape);
  if (rank == 0) {
    return fused_reduce_all_view(inputs, inputStrides, inputOffsets, inputMaskStarts, inputMaskEnds,
      inputDtypes, fullShape, prog, 0);
  }
  return fused_reduce_axis_view(inputs, inputStrides, inputOffsets, inputMaskStarts, inputMaskEnds,
    inputDtypes, fullShape, prog, rank - 1, 0);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_last_view_stack_f32(lean_obj_arg inputs,
    lean_obj_arg stackShapes, lean_obj_arg stackStrides, lean_obj_arg stackOffsets,
    lean_obj_arg stackMaskStarts, lean_obj_arg stackMaskEnds, lean_obj_arg inputDtypes,
    lean_obj_arg fullShape, lean_obj_arg prog) {
  size_t rank = array_size(fullShape);
  if (rank == 0) {
    return fused_reduce_all_view_stack(inputs, stackShapes, stackStrides, stackOffsets,
      stackMaskStarts, stackMaskEnds, inputDtypes, fullShape, prog, 0);
  }
  return fused_reduce_axis_view_stack(inputs, stackShapes, stackStrides, stackOffsets,
    stackMaskStarts, stackMaskEnds, inputDtypes, fullShape, prog, rank - 1, 0);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_last_f32(lean_obj_arg inputs, lean_obj_arg inputShapes,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg prog) {
  size_t rank = array_size(fullShape);
  if (rank == 0) {
    return fused_reduce_all(inputs, inputShapes, inputDtypes, fullShape, prog, 1);
  }
  return fused_reduce_axis(inputs, inputShapes, inputDtypes, fullShape, prog, rank - 1, 1);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_last_view_f32(lean_obj_arg inputs, lean_obj_arg inputStrides,
    lean_obj_arg inputOffsets, lean_obj_arg inputMaskStarts, lean_obj_arg inputMaskEnds,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg prog) {
  size_t rank = array_size(fullShape);
  if (rank == 0) {
    return fused_reduce_all_view(inputs, inputStrides, inputOffsets, inputMaskStarts, inputMaskEnds,
      inputDtypes, fullShape, prog, 1);
  }
  return fused_reduce_axis_view(inputs, inputStrides, inputOffsets, inputMaskStarts, inputMaskEnds,
    inputDtypes, fullShape, prog, rank - 1, 1);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_last_view_stack_f32(lean_obj_arg inputs,
    lean_obj_arg stackShapes, lean_obj_arg stackStrides, lean_obj_arg stackOffsets,
    lean_obj_arg stackMaskStarts, lean_obj_arg stackMaskEnds, lean_obj_arg inputDtypes,
    lean_obj_arg fullShape, lean_obj_arg prog) {
  size_t rank = array_size(fullShape);
  if (rank == 0) {
    return fused_reduce_all_view_stack(inputs, stackShapes, stackStrides, stackOffsets,
      stackMaskStarts, stackMaskEnds, inputDtypes, fullShape, prog, 1);
  }
  return fused_reduce_axis_view_stack(inputs, stackShapes, stackStrides, stackOffsets,
    stackMaskStarts, stackMaskEnds, inputDtypes, fullShape, prog, rank - 1, 1);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_axis_f32(lean_obj_arg inputs, lean_obj_arg inputShapes,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg axis, lean_obj_arg prog) {
  return fused_reduce_axis(inputs, inputShapes, inputDtypes, fullShape, prog, nat_to_size(axis), 0);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_axis_view_f32(lean_obj_arg inputs, lean_obj_arg inputStrides,
    lean_obj_arg inputOffsets, lean_obj_arg inputMaskStarts, lean_obj_arg inputMaskEnds,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg axis, lean_obj_arg prog) {
  return fused_reduce_axis_view(inputs, inputStrides, inputOffsets, inputMaskStarts, inputMaskEnds,
    inputDtypes, fullShape, prog, nat_to_size(axis), 0);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_axis_view_stack_f32(lean_obj_arg inputs,
    lean_obj_arg stackShapes, lean_obj_arg stackStrides, lean_obj_arg stackOffsets,
    lean_obj_arg stackMaskStarts, lean_obj_arg stackMaskEnds, lean_obj_arg inputDtypes,
    lean_obj_arg fullShape, lean_obj_arg axis, lean_obj_arg prog) {
  return fused_reduce_axis_view_stack(inputs, stackShapes, stackStrides, stackOffsets,
    stackMaskStarts, stackMaskEnds, inputDtypes, fullShape, prog, nat_to_size(axis), 0);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_axis_f32(lean_obj_arg inputs, lean_obj_arg inputShapes,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg axis, lean_obj_arg prog) {
  return fused_reduce_axis(inputs, inputShapes, inputDtypes, fullShape, prog, nat_to_size(axis), 1);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_axis_view_f32(lean_obj_arg inputs, lean_obj_arg inputStrides,
    lean_obj_arg inputOffsets, lean_obj_arg inputMaskStarts, lean_obj_arg inputMaskEnds,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg axis, lean_obj_arg prog) {
  return fused_reduce_axis_view(inputs, inputStrides, inputOffsets, inputMaskStarts, inputMaskEnds,
    inputDtypes, fullShape, prog, nat_to_size(axis), 1);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_axis_view_stack_f32(lean_obj_arg inputs,
    lean_obj_arg stackShapes, lean_obj_arg stackStrides, lean_obj_arg stackOffsets,
    lean_obj_arg stackMaskStarts, lean_obj_arg stackMaskEnds, lean_obj_arg inputDtypes,
    lean_obj_arg fullShape, lean_obj_arg axis, lean_obj_arg prog) {
  return fused_reduce_axis_view_stack(inputs, stackShapes, stackStrides, stackOffsets,
    stackMaskStarts, stackMaskEnds, inputDtypes, fullShape, prog, nat_to_size(axis), 1);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_axes_f32(lean_obj_arg inputs, lean_obj_arg inputShapes,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg outShape, lean_obj_arg axes,
    lean_obj_arg prog) {
  return fused_reduce_axes(inputs, inputShapes, inputDtypes, fullShape, outShape, axes, prog, 0);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_axes_view_f32(lean_obj_arg inputs,
    lean_obj_arg inputStrides, lean_obj_arg inputOffsets, lean_obj_arg inputMaskStarts,
    lean_obj_arg inputMaskEnds, lean_obj_arg inputDtypes, lean_obj_arg fullShape,
    lean_obj_arg outShape, lean_obj_arg axes, lean_obj_arg prog) {
  return fused_reduce_axes_view(inputs, inputStrides, inputOffsets, inputMaskStarts, inputMaskEnds,
    inputDtypes, fullShape, outShape, axes, prog, 0);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_axes_view_stack_f32(lean_obj_arg inputs,
    lean_obj_arg stackShapes, lean_obj_arg stackStrides, lean_obj_arg stackOffsets,
    lean_obj_arg stackMaskStarts, lean_obj_arg stackMaskEnds, lean_obj_arg inputDtypes,
    lean_obj_arg fullShape, lean_obj_arg outShape, lean_obj_arg axes, lean_obj_arg prog) {
  return fused_reduce_axes_view_stack(inputs, stackShapes, stackStrides, stackOffsets,
    stackMaskStarts, stackMaskEnds, inputDtypes, fullShape, outShape, axes, prog, 0);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_axes_f32(lean_obj_arg inputs, lean_obj_arg inputShapes,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg outShape, lean_obj_arg axes,
    lean_obj_arg prog) {
  return fused_reduce_axes(inputs, inputShapes, inputDtypes, fullShape, outShape, axes, prog, 1);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_axes_view_f32(lean_obj_arg inputs,
    lean_obj_arg inputStrides, lean_obj_arg inputOffsets, lean_obj_arg inputMaskStarts,
    lean_obj_arg inputMaskEnds, lean_obj_arg inputDtypes, lean_obj_arg fullShape,
    lean_obj_arg outShape, lean_obj_arg axes, lean_obj_arg prog) {
  return fused_reduce_axes_view(inputs, inputStrides, inputOffsets, inputMaskStarts, inputMaskEnds,
    inputDtypes, fullShape, outShape, axes, prog, 1);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_axes_view_stack_f32(lean_obj_arg inputs,
    lean_obj_arg stackShapes, lean_obj_arg stackStrides, lean_obj_arg stackOffsets,
    lean_obj_arg stackMaskStarts, lean_obj_arg stackMaskEnds, lean_obj_arg inputDtypes,
    lean_obj_arg fullShape, lean_obj_arg outShape, lean_obj_arg axes, lean_obj_arg prog) {
  return fused_reduce_axes_view_stack(inputs, stackShapes, stackStrides, stackOffsets,
    stackMaskStarts, stackMaskEnds, inputDtypes, fullShape, outShape, axes, prog, 1);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_all_f32(lean_obj_arg inputs, lean_obj_arg inputShapes,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg prog) {
  return fused_reduce_all(inputs, inputShapes, inputDtypes, fullShape, prog, 0);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_all_view_f32(lean_obj_arg inputs, lean_obj_arg inputStrides,
    lean_obj_arg inputOffsets, lean_obj_arg inputMaskStarts, lean_obj_arg inputMaskEnds,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg prog) {
  return fused_reduce_all_view(inputs, inputStrides, inputOffsets, inputMaskStarts, inputMaskEnds,
    inputDtypes, fullShape, prog, 0);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_all_view_stack_f32(lean_obj_arg inputs,
    lean_obj_arg stackShapes, lean_obj_arg stackStrides, lean_obj_arg stackOffsets,
    lean_obj_arg stackMaskStarts, lean_obj_arg stackMaskEnds, lean_obj_arg inputDtypes,
    lean_obj_arg fullShape, lean_obj_arg prog) {
  return fused_reduce_all_view_stack(inputs, stackShapes, stackStrides, stackOffsets,
    stackMaskStarts, stackMaskEnds, inputDtypes, fullShape, prog, 0);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_all_f32(lean_obj_arg inputs, lean_obj_arg inputShapes,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg prog) {
  return fused_reduce_all(inputs, inputShapes, inputDtypes, fullShape, prog, 1);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_all_view_f32(lean_obj_arg inputs, lean_obj_arg inputStrides,
    lean_obj_arg inputOffsets, lean_obj_arg inputMaskStarts, lean_obj_arg inputMaskEnds,
    lean_obj_arg inputDtypes, lean_obj_arg fullShape, lean_obj_arg prog) {
  return fused_reduce_all_view(inputs, inputStrides, inputOffsets, inputMaskStarts, inputMaskEnds,
    inputDtypes, fullShape, prog, 1);
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_all_view_stack_f32(lean_obj_arg inputs,
    lean_obj_arg stackShapes, lean_obj_arg stackStrides, lean_obj_arg stackOffsets,
    lean_obj_arg stackMaskStarts, lean_obj_arg stackMaskEnds, lean_obj_arg inputDtypes,
    lean_obj_arg fullShape, lean_obj_arg prog) {
  return fused_reduce_all_view_stack(inputs, stackShapes, stackStrides, stackOffsets,
    stackMaskStarts, stackMaskEnds, inputDtypes, fullShape, prog, 1);
}

LEAN_EXPORT lean_obj_res tg4_softmax_last_f32(b_lean_obj_arg a, b_lean_obj_arg outer,
    b_lean_obj_arg inner, b_lean_obj_arg scaleBits) {
  size_t outer_n = nat_to_size(outer);
  size_t inner_n = nat_to_size(inner);
  float scale = f32_from_bits(unbox_u32(scaleBits));
  size_t total = outer_n * inner_n;
  lean_object* out = mk_byte_array(total * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  for (size_t r = 0; r < outer_n; ++r) {
    float maxv = -INFINITY;
    for (size_t c = 0; c < inner_n; ++c) {
      float v = read_f32(x, r * inner_n + c) * scale;
      if (v > maxv) maxv = v;
    }
    float sum = 0.0f;
    for (size_t c = 0; c < inner_n; ++c) {
      float v = expf(read_f32(x, r * inner_n + c) * scale - maxv);
      sum += v;
      write_f32(o, r * inner_n + c, v);
    }
    float inv = sum == 0.0f ? 0.0f : 1.0f / sum;
    for (size_t c = 0; c < inner_n; ++c) {
      float v = read_f32(o, r * inner_n + c) * inv;
      write_f32(o, r * inner_n + c, v);
    }
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_logsoftmax_last_f32(b_lean_obj_arg a, b_lean_obj_arg outer,
    b_lean_obj_arg inner, b_lean_obj_arg scaleBits, b_lean_obj_arg ln2Bits) {
  size_t outer_n = nat_to_size(outer);
  size_t inner_n = nat_to_size(inner);
  float scale = f32_from_bits(unbox_u32(scaleBits));
  float ln2 = f32_from_bits(unbox_u32(ln2Bits));
  size_t total = outer_n * inner_n;
  lean_object* out = mk_byte_array(total * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  for (size_t r = 0; r < outer_n; ++r) {
    float maxv = -INFINITY;
    for (size_t c = 0; c < inner_n; ++c) {
      float v = read_f32(x, r * inner_n + c) * scale;
      if (v > maxv) maxv = v;
    }
    float sum = 0.0f;
    for (size_t c = 0; c < inner_n; ++c) {
      float v = expf(read_f32(x, r * inner_n + c) * scale - maxv);
      sum += v;
    }
    float logsum = sum == 0.0f ? -INFINITY : logf(sum) / ln2 + maxv;
    for (size_t c = 0; c < inner_n; ++c) {
      float v = read_f32(x, r * inner_n + c) * scale - logsum;
      write_f32(o, r * inner_n + c, v);
    }
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_sum_all_f32(b_lean_obj_arg a) {
  size_t n = f32_numel(a);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) sum += read_f32(x, i);
  lean_object* out = mk_byte_array(4);
  write_f32(byte_array_cptr(out), 0, sum);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_reduce_sum_last_f32(b_lean_obj_arg a, b_lean_obj_arg outer,
    b_lean_obj_arg inner) {
  size_t outer_n = nat_to_size(outer);
  size_t inner_n = nat_to_size(inner);
  lean_object* out = mk_byte_array(outer_n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  for (size_t r = 0; r < outer_n; ++r) {
    float sum = 0.0f;
    for (size_t c = 0; c < inner_n; ++c) {
      sum += read_f32(x, r * inner_n + c);
    }
    write_f32(o, r, sum);
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_reduce_max_last_f32(b_lean_obj_arg a, b_lean_obj_arg outer,
    b_lean_obj_arg inner) {
  size_t outer_n = nat_to_size(outer);
  size_t inner_n = nat_to_size(inner);
  lean_object* out = mk_byte_array(outer_n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  for (size_t r = 0; r < outer_n; ++r) {
    float maxv = -INFINITY;
    for (size_t c = 0; c < inner_n; ++c) {
      float v = read_f32(x, r * inner_n + c);
      if (v > maxv) maxv = v;
    }
    write_f32(o, r, maxv);
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_reduce_sum_axis_f32(b_lean_obj_arg a, b_lean_obj_arg outer,
    b_lean_obj_arg reduce, b_lean_obj_arg inner) {
  size_t outer_n = nat_to_size(outer);
  size_t reduce_n = nat_to_size(reduce);
  size_t inner_n = nat_to_size(inner);
  size_t out_numel = outer_n * inner_n;
  lean_object* out = mk_byte_array(out_numel * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  for (size_t oidx = 0; oidx < outer_n; ++oidx) {
    for (size_t iidx = 0; iidx < inner_n; ++iidx) {
      float sum = 0.0f;
      for (size_t r = 0; r < reduce_n; ++r) {
        size_t idx = (oidx * reduce_n + r) * inner_n + iidx;
        sum += read_f32(x, idx);
      }
      write_f32(o, oidx * inner_n + iidx, sum);
    }
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_reduce_max_axis_f32(b_lean_obj_arg a, b_lean_obj_arg outer,
    b_lean_obj_arg reduce, b_lean_obj_arg inner) {
  size_t outer_n = nat_to_size(outer);
  size_t reduce_n = nat_to_size(reduce);
  size_t inner_n = nat_to_size(inner);
  size_t out_numel = outer_n * inner_n;
  lean_object* out = mk_byte_array(out_numel * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  for (size_t oidx = 0; oidx < outer_n; ++oidx) {
    for (size_t iidx = 0; iidx < inner_n; ++iidx) {
      float maxv = -INFINITY;
      for (size_t r = 0; r < reduce_n; ++r) {
        size_t idx = (oidx * reduce_n + r) * inner_n + iidx;
        float v = read_f32(x, idx);
        if (v > maxv) maxv = v;
      }
      write_f32(o, oidx * inner_n + iidx, maxv);
    }
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_transpose2d_f32(b_lean_obj_arg a, b_lean_obj_arg m,
    b_lean_obj_arg n) {
  size_t m_n = nat_to_size(m);
  size_t n_n = nat_to_size(n);
  lean_object* out = mk_byte_array(m_n * n_n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  for (size_t i = 0; i < m_n; ++i) {
    for (size_t j = 0; j < n_n; ++j) {
      float v = read_f32(x, i * n_n + j);
      write_f32(o, j * m_n + i, v);
    }
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_permute_f32(b_lean_obj_arg a, b_lean_obj_arg shape,
    b_lean_obj_arg perm) {
  size_t rank = array_size(shape);
  size_t* dims = malloc(rank * sizeof(size_t));
  size_t* strides = malloc(rank * sizeof(size_t));
  size_t* out_dims = malloc(rank * sizeof(size_t));
  size_t* out_idx = malloc(rank * sizeof(size_t));
  size_t* in_idx = malloc(rank * sizeof(size_t));

  for (size_t i = 0; i < rank; ++i) {
    dims[i] = nat_array_get(shape, i);
    size_t p = nat_array_get(perm, i);
    out_dims[i] = p < rank ? dims[p] : 0;
  }
  make_strides(dims, rank, strides);
  size_t out_numel = 1;
  for (size_t i = 0; i < rank; ++i) out_numel *= out_dims[i];

  lean_object* out = mk_byte_array(out_numel * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);

  for (size_t i = 0; i < out_numel; ++i) {
    unflatten_index(i, out_dims, rank, out_idx);
    for (size_t j = 0; j < rank; ++j) in_idx[j] = 0;
    for (size_t j = 0; j < rank; ++j) {
      size_t p = nat_array_get(perm, j);
      if (p < rank) in_idx[p] = out_idx[j];
    }
    size_t in_flat = 0;
    for (size_t j = 0; j < rank; ++j) in_flat += in_idx[j] * strides[j];
    write_f32(o, i, read_f32(x, in_flat));
  }

  free(dims);
  free(strides);
  free(out_dims);
  free(out_idx);
  free(in_idx);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_permute_u8(b_lean_obj_arg a, b_lean_obj_arg shape,
    b_lean_obj_arg perm) {
  size_t rank = array_size(shape);
  size_t* dims = malloc(rank * sizeof(size_t));
  size_t* strides = malloc(rank * sizeof(size_t));
  size_t* out_dims = malloc(rank * sizeof(size_t));
  size_t* out_idx = malloc(rank * sizeof(size_t));
  size_t* in_idx = malloc(rank * sizeof(size_t));

  for (size_t i = 0; i < rank; ++i) {
    dims[i] = nat_array_get(shape, i);
    size_t p = nat_array_get(perm, i);
    out_dims[i] = p < rank ? dims[p] : 0;
  }
  make_strides(dims, rank, strides);
  size_t out_numel = 1;
  for (size_t i = 0; i < rank; ++i) out_numel *= out_dims[i];

  lean_object* out = mk_byte_array(out_numel);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);

  for (size_t i = 0; i < out_numel; ++i) {
    unflatten_index(i, out_dims, rank, out_idx);
    for (size_t j = 0; j < rank; ++j) in_idx[j] = 0;
    for (size_t j = 0; j < rank; ++j) {
      size_t p = nat_array_get(perm, j);
      if (p < rank) in_idx[p] = out_idx[j];
    }
    size_t in_flat = 0;
    for (size_t j = 0; j < rank; ++j) in_flat += in_idx[j] * strides[j];
    o[i] = x[in_flat];
  }

  free(dims);
  free(strides);
  free(out_dims);
  free(out_idx);
  free(in_idx);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_pad_f32(b_lean_obj_arg a, b_lean_obj_arg shape,
    b_lean_obj_arg padLeft, b_lean_obj_arg padRight) {
  size_t rank = array_size(shape);
  size_t* dims = malloc(rank * sizeof(size_t));
  size_t* out_dims = malloc(rank * sizeof(size_t));
  size_t* out_idx = malloc(rank * sizeof(size_t));

  for (size_t i = 0; i < rank; ++i) {
    size_t dim = nat_array_get(shape, i);
    size_t l = nat_array_get(padLeft, i);
    size_t r = nat_array_get(padRight, i);
    dims[i] = dim;
    out_dims[i] = dim + l + r;
  }

  size_t out_numel = 1;
  for (size_t i = 0; i < rank; ++i) out_numel *= out_dims[i];

  lean_object* out = mk_byte_array(out_numel * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);

  size_t* in_idx = malloc(rank * sizeof(size_t));
  size_t* in_strides = malloc(rank * sizeof(size_t));
  make_strides(dims, rank, in_strides);

  for (size_t i = 0; i < out_numel; ++i) {
    unflatten_index(i, out_dims, rank, out_idx);
    int inside = 1;
    for (size_t j = 0; j < rank; ++j) {
      size_t l = nat_array_get(padLeft, j);
      size_t dim = dims[j];
      size_t idx = out_idx[j];
      if (idx < l || idx >= l + dim) {
        inside = 0;
        break;
      }
      in_idx[j] = idx - l;
    }
    if (!inside) {
      write_f32(o, i, 0.0f);
    } else {
      size_t in_flat = 0;
      for (size_t j = 0; j < rank; ++j) in_flat += in_idx[j] * in_strides[j];
      write_f32(o, i, read_f32(x, in_flat));
    }
  }

  free(dims);
  free(out_dims);
  free(out_idx);
  free(in_idx);
  free(in_strides);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_pad_u8(b_lean_obj_arg a, b_lean_obj_arg shape,
    b_lean_obj_arg padLeft, b_lean_obj_arg padRight) {
  size_t rank = array_size(shape);
  size_t* dims = malloc(rank * sizeof(size_t));
  size_t* out_dims = malloc(rank * sizeof(size_t));
  size_t* out_idx = malloc(rank * sizeof(size_t));

  for (size_t i = 0; i < rank; ++i) {
    size_t dim = nat_array_get(shape, i);
    size_t l = nat_array_get(padLeft, i);
    size_t r = nat_array_get(padRight, i);
    dims[i] = dim;
    out_dims[i] = dim + l + r;
  }

  size_t out_numel = 1;
  for (size_t i = 0; i < rank; ++i) out_numel *= out_dims[i];

  lean_object* out = mk_byte_array(out_numel);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);

  size_t* in_idx = malloc(rank * sizeof(size_t));
  size_t* in_strides = malloc(rank * sizeof(size_t));
  make_strides(dims, rank, in_strides);

  for (size_t i = 0; i < out_numel; ++i) {
    unflatten_index(i, out_dims, rank, out_idx);
    int inside = 1;
    for (size_t j = 0; j < rank; ++j) {
      size_t l = nat_array_get(padLeft, j);
      size_t dim = dims[j];
      size_t idx = out_idx[j];
      if (idx < l || idx >= l + dim) {
        inside = 0;
        break;
      }
      in_idx[j] = idx - l;
    }
    if (!inside) {
      o[i] = 0;
    } else {
      size_t in_flat = 0;
      for (size_t j = 0; j < rank; ++j) in_flat += in_idx[j] * in_strides[j];
      o[i] = x[in_flat];
    }
  }

  free(dims);
  free(out_dims);
  free(out_idx);
  free(in_idx);
  free(in_strides);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_shrink_f32(b_lean_obj_arg a, b_lean_obj_arg shape,
    b_lean_obj_arg starts, b_lean_obj_arg stops) {
  size_t rank = array_size(shape);
  size_t* dims = malloc(rank * sizeof(size_t));
  size_t* out_dims = malloc(rank * sizeof(size_t));
  size_t* out_idx = malloc(rank * sizeof(size_t));
  size_t* in_idx = malloc(rank * sizeof(size_t));

  for (size_t i = 0; i < rank; ++i) {
    size_t start = nat_array_get(starts, i);
    size_t stop = nat_array_get(stops, i);
    dims[i] = nat_array_get(shape, i);
    out_dims[i] = stop > start ? (stop - start) : 0;
  }

  size_t out_numel = 1;
  for (size_t i = 0; i < rank; ++i) out_numel *= out_dims[i];

  lean_object* out = mk_byte_array(out_numel * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);

  size_t* in_strides = malloc(rank * sizeof(size_t));
  make_strides(dims, rank, in_strides);

  for (size_t i = 0; i < out_numel; ++i) {
    unflatten_index(i, out_dims, rank, out_idx);
    for (size_t j = 0; j < rank; ++j) {
      size_t start = nat_array_get(starts, j);
      in_idx[j] = out_idx[j] + start;
    }
    size_t in_flat = 0;
    for (size_t j = 0; j < rank; ++j) in_flat += in_idx[j] * in_strides[j];
    write_f32(o, i, read_f32(x, in_flat));
  }

  free(dims);
  free(out_dims);
  free(out_idx);
  free(in_idx);
  free(in_strides);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_shrink_u8(b_lean_obj_arg a, b_lean_obj_arg shape,
    b_lean_obj_arg starts, b_lean_obj_arg stops) {
  size_t rank = array_size(shape);
  size_t* dims = malloc(rank * sizeof(size_t));
  size_t* out_dims = malloc(rank * sizeof(size_t));
  size_t* out_idx = malloc(rank * sizeof(size_t));
  size_t* in_idx = malloc(rank * sizeof(size_t));

  for (size_t i = 0; i < rank; ++i) {
    size_t start = nat_array_get(starts, i);
    size_t stop = nat_array_get(stops, i);
    dims[i] = nat_array_get(shape, i);
    out_dims[i] = stop > start ? (stop - start) : 0;
  }

  size_t out_numel = 1;
  for (size_t i = 0; i < rank; ++i) out_numel *= out_dims[i];

  lean_object* out = mk_byte_array(out_numel);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);

  size_t* in_strides = malloc(rank * sizeof(size_t));
  make_strides(dims, rank, in_strides);

  for (size_t i = 0; i < out_numel; ++i) {
    unflatten_index(i, out_dims, rank, out_idx);
    for (size_t j = 0; j < rank; ++j) {
      size_t start = nat_array_get(starts, j);
      in_idx[j] = out_idx[j] + start;
    }
    size_t in_flat = 0;
    for (size_t j = 0; j < rank; ++j) in_flat += in_idx[j] * in_strides[j];
    o[i] = x[in_flat];
  }

  free(dims);
  free(out_dims);
  free(out_idx);
  free(in_idx);
  free(in_strides);
  return out;
}

static void cat_copy_bytes(uint8_t* out, size_t out_off, const uint8_t* in, size_t in_off,
    size_t count) {
  memcpy(out + out_off, in + in_off, count);
}

LEAN_EXPORT lean_obj_res tg4_cat_f32(b_lean_obj_arg inputs, b_lean_obj_arg inputShapes,
    b_lean_obj_arg axis, b_lean_obj_arg outShape) {
  size_t axis_n = nat_to_size(axis);
  size_t rank = array_size(outShape);
  size_t* out_dims = malloc(rank * sizeof(size_t));
  for (size_t i = 0; i < rank; ++i) out_dims[i] = nat_array_get(outShape, i);

  size_t inner = 1;
  for (size_t i = axis_n + 1; i < rank; ++i) inner *= out_dims[i];
  size_t outer = 1;
  for (size_t i = 0; i < axis_n; ++i) outer *= out_dims[i];

  size_t out_numel = shape_numel(outShape);
  lean_object* out = mk_byte_array(out_numel * 4);
  uint8_t* o = byte_array_cptr(out);

  size_t num_inputs = array_size(inputs);

  for (size_t outer_i = 0; outer_i < outer; ++outer_i) {
    size_t out_pos = outer_i * out_dims[axis_n] * inner;
    size_t out_byte = out_pos * 4;
    for (size_t t = 0; t < num_inputs; ++t) {
      lean_object* in_ba = lean_array_get_core(inputs, t);
      lean_object* in_shape = lean_array_get_core(inputShapes, t);
      size_t axis_dim = nat_array_get(in_shape, axis_n);
      size_t chunk = axis_dim * inner;
      size_t in_pos = outer_i * axis_dim * inner;
      cat_copy_bytes(o, out_byte, byte_array_cptr(in_ba), in_pos * 4, chunk * 4);
      out_byte += chunk * 4;
    }
  }

  free(out_dims);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_cat_u8(b_lean_obj_arg inputs, b_lean_obj_arg inputShapes,
    b_lean_obj_arg axis, b_lean_obj_arg outShape) {
  size_t axis_n = nat_to_size(axis);
  size_t rank = array_size(outShape);
  size_t* out_dims = malloc(rank * sizeof(size_t));
  for (size_t i = 0; i < rank; ++i) out_dims[i] = nat_array_get(outShape, i);

  size_t inner = 1;
  for (size_t i = axis_n + 1; i < rank; ++i) inner *= out_dims[i];
  size_t outer = 1;
  for (size_t i = 0; i < axis_n; ++i) outer *= out_dims[i];

  size_t out_numel = shape_numel(outShape);
  lean_object* out = mk_byte_array(out_numel);
  uint8_t* o = byte_array_cptr(out);

  size_t num_inputs = array_size(inputs);

  for (size_t outer_i = 0; outer_i < outer; ++outer_i) {
    size_t out_pos = outer_i * out_dims[axis_n] * inner;
    size_t out_byte = out_pos;
    for (size_t t = 0; t < num_inputs; ++t) {
      lean_object* in_ba = lean_array_get_core(inputs, t);
      lean_object* in_shape = lean_array_get_core(inputShapes, t);
      size_t axis_dim = nat_array_get(in_shape, axis_n);
      size_t chunk = axis_dim * inner;
      size_t in_pos = outer_i * axis_dim * inner;
      cat_copy_bytes(o, out_byte, byte_array_cptr(in_ba), in_pos, chunk);
      out_byte += chunk;
    }
  }

  free(out_dims);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_cat_bytes(b_lean_obj_arg inputs, b_lean_obj_arg inputShapes,
    b_lean_obj_arg axis, b_lean_obj_arg outShape, b_lean_obj_arg elemSize) {
  size_t axis_n = nat_to_size(axis);
  size_t elem = nat_to_size(elemSize);
  size_t rank = array_size(outShape);
  size_t* out_dims = malloc(rank * sizeof(size_t));
  for (size_t i = 0; i < rank; ++i) out_dims[i] = nat_array_get(outShape, i);

  size_t inner = 1;
  for (size_t i = axis_n + 1; i < rank; ++i) inner *= out_dims[i];
  size_t outer = 1;
  for (size_t i = 0; i < axis_n; ++i) outer *= out_dims[i];

  size_t out_numel = shape_numel(outShape);
  lean_object* out = mk_byte_array(out_numel * elem);
  uint8_t* o = byte_array_cptr(out);

  size_t num_inputs = array_size(inputs);

  for (size_t outer_i = 0; outer_i < outer; ++outer_i) {
    size_t out_pos = outer_i * out_dims[axis_n] * inner;
    size_t out_byte = out_pos * elem;
    for (size_t t = 0; t < num_inputs; ++t) {
      lean_object* in_ba = lean_array_get_core(inputs, t);
      lean_object* in_shape = lean_array_get_core(inputShapes, t);
      size_t axis_dim = nat_array_get(in_shape, axis_n);
      size_t chunk = axis_dim * inner;
      size_t in_pos = outer_i * axis_dim * inner;
      cat_copy_bytes(o, out_byte, byte_array_cptr(in_ba), in_pos * elem, chunk * elem);
      out_byte += chunk * elem;
    }
  }

  free(out_dims);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_flip_f32(b_lean_obj_arg a, b_lean_obj_arg shape, b_lean_obj_arg axes) {
  size_t rank = array_size(shape);
  size_t* dims = malloc(rank * sizeof(size_t));
  size_t* strides = malloc(rank * sizeof(size_t));
  size_t* out_idx = malloc(rank * sizeof(size_t));
  int* flip = malloc(rank * sizeof(int));

  for (size_t i = 0; i < rank; ++i) {
    dims[i] = nat_array_get(shape, i);
    flip[i] = 0;
  }
  size_t axes_n = array_size(axes);
  for (size_t i = 0; i < axes_n; ++i) {
    size_t ax = nat_array_get(axes, i);
    if (ax < rank) flip[ax] = 1;
  }
  make_strides(dims, rank, strides);
  size_t total = shape_numel(shape);

  lean_object* out = mk_byte_array(total * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);

  for (size_t i = 0; i < total; ++i) {
    unflatten_index(i, dims, rank, out_idx);
    for (size_t j = 0; j < rank; ++j) {
      if (flip[j]) out_idx[j] = dims[j] - 1 - out_idx[j];
    }
    size_t in_flat = 0;
    for (size_t j = 0; j < rank; ++j) in_flat += out_idx[j] * strides[j];
    write_f32(o, i, read_f32(x, in_flat));
  }

  free(dims);
  free(strides);
  free(out_idx);
  free(flip);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_flip_u8(b_lean_obj_arg a, b_lean_obj_arg shape, b_lean_obj_arg axes) {
  size_t rank = array_size(shape);
  size_t* dims = malloc(rank * sizeof(size_t));
  size_t* strides = malloc(rank * sizeof(size_t));
  size_t* out_idx = malloc(rank * sizeof(size_t));
  int* flip = malloc(rank * sizeof(int));

  for (size_t i = 0; i < rank; ++i) {
    dims[i] = nat_array_get(shape, i);
    flip[i] = 0;
  }
  size_t axes_n = array_size(axes);
  for (size_t i = 0; i < axes_n; ++i) {
    size_t ax = nat_array_get(axes, i);
    if (ax < rank) flip[ax] = 1;
  }
  make_strides(dims, rank, strides);
  size_t total = shape_numel(shape);

  lean_object* out = mk_byte_array(total);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);

  for (size_t i = 0; i < total; ++i) {
    unflatten_index(i, dims, rank, out_idx);
    for (size_t j = 0; j < rank; ++j) {
      if (flip[j]) out_idx[j] = dims[j] - 1 - out_idx[j];
    }
    size_t in_flat = 0;
    for (size_t j = 0; j < rank; ++j) in_flat += out_idx[j] * strides[j];
    o[i] = x[in_flat];
  }

  free(dims);
  free(strides);
  free(out_idx);
  free(flip);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_pack_f32_from_f64(b_lean_obj_arg arr) {
  size_t n = float_array_size(arr);
  lean_object* out = mk_byte_array(n * 4);
  uint8_t* o = byte_array_cptr(out);
  const double* x = float_array_cptr((lean_object*)arr);
  for (size_t i = 0; i < n; ++i) {
    float v = (float)x[i];
    write_f32(o, i, v);
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_unpack_f64_from_f32(b_lean_obj_arg a) {
  size_t n = f32_numel(a);
  lean_object* out = mk_float_array(n);
  double* o = float_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  for (size_t i = 0; i < n; ++i) {
    o[i] = (double)read_f32(x, i);
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg m, b_lean_obj_arg k, b_lean_obj_arg n) {
  size_t m_n = nat_to_size(m);
  size_t k_n = nat_to_size(k);
  size_t n_n = nat_to_size(n);
  lean_object* out = mk_byte_array(m_n * n_n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  const uint8_t* y = byte_array_cptr((lean_object*)b);
  for (size_t i = 0; i < m_n; ++i) {
    for (size_t j = 0; j < n_n; ++j) {
      float sum = 0.0f;
      for (size_t t = 0; t < k_n; ++t) {
        float av = read_f32(x, i * k_n + t);
        float bv = read_f32(y, t * n_n + j);
        sum += av * bv;
      }
      write_f32(o, i * n_n + j, sum);
    }
  }
  return out;
}

/* view_info and view_stack structs defined at top of file */

static void view_info_free(view_info* v) {
  if (!v) return;
  free(v->strides);
  free(v->mask_start);
  free(v->mask_end);
}

static int view_info_init(view_info* v, b_lean_obj_arg strides, int64_t offset,
    b_lean_obj_arg mask_start, b_lean_obj_arg mask_end) {
  v->rank = array_size(strides);
  v->offset = offset;
  v->strides = v->rank == 0 ? NULL : malloc(v->rank * sizeof(int64_t));
  v->mask_start = v->rank == 0 ? NULL : malloc(v->rank * sizeof(size_t));
  v->mask_end = v->rank == 0 ? NULL : malloc(v->rank * sizeof(size_t));
  if ((v->rank && !v->strides) || (v->rank && !v->mask_start) || (v->rank && !v->mask_end)) {
    view_info_free(v);
    return 0;
  }
  for (size_t i = 0; i < v->rank; ++i) {
    v->strides[i] = int64_array_get(strides, i);
    v->mask_start[i] = nat_array_get(mask_start, i);
    v->mask_end[i] = nat_array_get(mask_end, i);
  }
  return 1;
}

static float view_read_value(const uint8_t* data, const view_info* v, const size_t* idx,
    int is_bool) {
  int64_t off = v->offset;
  for (size_t i = 0; i < v->rank; ++i) {
    size_t ix = idx[i];
    if (ix < v->mask_start[i] || ix >= v->mask_end[i]) return 0.0f;
    off += (int64_t)ix * v->strides[i];
  }
  if (off < 0) return 0.0f;
  if (is_bool) return data[(size_t)off] ? 1.0f : 0.0f;
  return read_f32(data, (size_t)off);
}

static float view_read_f32(const uint8_t* data, const view_info* v, const size_t* idx) {
  return view_read_value(data, v, idx, 0);
}

static void view_stack_free(view_stack* v) {
  if (!v) return;
  if (v->views) {
    for (size_t i = 0; i < v->depth; ++i) {
      view_info_free(&v->views[i]);
    }
  }
  if (v->shapes) {
    for (size_t i = 0; i < v->depth; ++i) {
      free(v->shapes[i]);
    }
  }
  free(v->views);
  free(v->shapes);
  free(v->numels);
}

static int view_stack_init(view_stack* v, b_lean_obj_arg shapes, b_lean_obj_arg strides,
    b_lean_obj_arg offsets, b_lean_obj_arg mask_start, b_lean_obj_arg mask_end) {
  memset(v, 0, sizeof(*v));
  v->depth = array_size(strides);
  if (v->depth == 0) return 0;
  if (array_size(shapes) != v->depth || array_size(offsets) != v->depth ||
      array_size(mask_start) != v->depth || array_size(mask_end) != v->depth) {
    return 0;
  }
  v->views = calloc(v->depth, sizeof(view_info));
  v->shapes = calloc(v->depth, sizeof(size_t*));
  v->numels = calloc(v->depth, sizeof(size_t));
  if (!v->views || !v->shapes || !v->numels) {
    view_stack_free(v);
    return 0;
  }
  v->max_rank = 0;
  for (size_t i = 0; i < v->depth; ++i) {
    lean_object* stridesArr = lean_array_get_core(strides, i);
    lean_object* shapeArr = lean_array_get_core(shapes, i);
    lean_object* maskStartArr = lean_array_get_core(mask_start, i);
    lean_object* maskEndArr = lean_array_get_core(mask_end, i);
    int64_t off = int64_array_get(offsets, i);
    if (!view_info_init(&v->views[i], stridesArr, off, maskStartArr, maskEndArr)) {
      view_stack_free(v);
      return 0;
    }
    size_t rank = array_size(shapeArr);
    if (rank != v->views[i].rank) {
      view_stack_free(v);
      return 0;
    }
    v->shapes[i] = rank == 0 ? NULL : malloc(rank * sizeof(size_t));
    if (rank && !v->shapes[i]) {
      view_stack_free(v);
      return 0;
    }
    size_t numel = 1;
    for (size_t j = 0; j < rank; ++j) {
      size_t dim = nat_array_get(shapeArr, j);
      v->shapes[i][j] = dim;
      numel *= dim;
    }
    v->numels[i] = numel;
    if (rank > v->max_rank) v->max_rank = rank;
  }
  return 1;
}

static float view_stack_read_value(const uint8_t* data, const view_stack* v, const size_t* idx,
    size_t* idx_buf, size_t* tmp_buf, int is_bool) {
  if (!v || v->depth == 0) return 0.0f;
  size_t top = v->depth - 1;
  size_t top_rank = v->views[top].rank;
  for (size_t i = 0; i < top_rank; ++i) {
    idx_buf[i] = idx[i];
  }
  for (size_t level = v->depth; level-- > 0;) {
    const view_info* cur = &v->views[level];
    size_t rank = cur->rank;
    int64_t off = cur->offset;
    for (size_t i = 0; i < rank; ++i) {
      size_t ix = idx_buf[i];
      if (ix < cur->mask_start[i] || ix >= cur->mask_end[i]) return 0.0f;
      off += (int64_t)ix * cur->strides[i];
    }
    if (off < 0) return 0.0f;
    if (level == 0) {
      size_t flat = (size_t)off;
      if (is_bool) return data[flat] ? 1.0f : 0.0f;
      return read_f32(data, flat);
    }
    size_t flat = (size_t)off;
    if (flat >= v->numels[level - 1]) return 0.0f;
    unflatten_index(flat, v->shapes[level - 1], v->views[level - 1].rank, tmp_buf);
    for (size_t i = 0; i < v->views[level - 1].rank; ++i) {
      idx_buf[i] = tmp_buf[i];
    }
  }
  return 0.0f;
}

static float view_stack_read_f32(const uint8_t* data, const view_stack* v, const size_t* idx,
    size_t* idx_buf, size_t* tmp_buf) {
  return view_stack_read_value(data, v, idx, idx_buf, tmp_buf, 0);
}

static lean_object* matmul_view_core(const uint8_t* a, const view_info* aView,
    const uint8_t* b, const view_info* bView, const uint8_t* bias0, const view_info* bias0View,
    const uint8_t* bias1, const view_info* bias1View, b_lean_obj_arg outShape, size_t k,
    int do_relu, int do_scale, uint32_t scaleBits) {
  size_t out_rank = array_size(outShape);
  if (out_rank < 2) return mk_byte_array(0);
  size_t m = nat_array_get(outShape, out_rank - 2);
  size_t n = nat_array_get(outShape, out_rank - 1);
  size_t batch_rank = out_rank - 2;
  size_t batch_numel = 1;
  size_t* batch_dims = batch_rank == 0 ? NULL : malloc(batch_rank * sizeof(size_t));
  for (size_t i = 0; i < batch_rank; ++i) {
    size_t dim = nat_array_get(outShape, i);
    if (batch_dims) batch_dims[i] = dim;
    batch_numel *= dim;
  }
  size_t out_numel = batch_numel * m * n;
  lean_object* out = mk_byte_array(out_numel * 4);
  uint8_t* o = byte_array_cptr(out);
  size_t* batch_idx = batch_rank == 0 ? NULL : malloc(batch_rank * sizeof(size_t));
  size_t* a_idx = malloc(out_rank * sizeof(size_t));
  size_t* b_idx = malloc(out_rank * sizeof(size_t));
  size_t* out_idx = malloc(out_rank * sizeof(size_t));
  if ((batch_rank && !batch_dims) || (batch_rank && !batch_idx) || !a_idx || !b_idx || !out_idx) {
    free(batch_dims);
    free(batch_idx);
    free(a_idx);
    free(b_idx);
    free(out_idx);
    return mk_byte_array(0);
  }
  float scale = do_scale ? f32_from_bits(scaleBits) : 1.0f;
  for (size_t bflat = 0; bflat < batch_numel; ++bflat) {
    if (batch_rank) unflatten_index(bflat, batch_dims, batch_rank, batch_idx);
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (size_t d = 0; d < batch_rank; ++d) {
          a_idx[d] = batch_idx[d];
          b_idx[d] = batch_idx[d];
          out_idx[d] = batch_idx[d];
        }
        out_idx[batch_rank] = i;
        out_idx[batch_rank + 1] = j;
        for (size_t t = 0; t < k; ++t) {
          a_idx[batch_rank] = i;
          a_idx[batch_rank + 1] = t;
          b_idx[batch_rank] = t;
          b_idx[batch_rank + 1] = j;
          float av = view_read_f32(a, aView, a_idx);
          float bv = view_read_f32(b, bView, b_idx);
          sum += av * bv;
        }
        sum *= scale;
        if (bias0 && bias0View) sum += view_read_f32(bias0, bias0View, out_idx);
        if (bias1 && bias1View) sum += view_read_f32(bias1, bias1View, out_idx);
        if (do_relu && sum < 0.0f) sum = 0.0f;
        size_t out_flat = (bflat * m + i) * n + j;
        write_f32(o, out_flat, sum);
      }
    }
  }
  free(batch_dims);
  free(batch_idx);
  free(a_idx);
  free(b_idx);
  free(out_idx);
  return out;
}

static lean_object* matmul_view_stack_core(const uint8_t* a, const view_stack* aView,
    const uint8_t* b, const view_stack* bView, b_lean_obj_arg outShape, size_t k) {
  size_t out_rank = array_size(outShape);
  if (out_rank < 2) return mk_byte_array(0);
  size_t m = nat_array_get(outShape, out_rank - 2);
  size_t n = nat_array_get(outShape, out_rank - 1);
  size_t batch_rank = out_rank - 2;
  size_t batch_numel = 1;
  size_t* batch_dims = batch_rank == 0 ? NULL : malloc(batch_rank * sizeof(size_t));
  for (size_t i = 0; i < batch_rank; ++i) {
    size_t dim = nat_array_get(outShape, i);
    if (batch_dims) batch_dims[i] = dim;
    batch_numel *= dim;
  }
  size_t out_numel = batch_numel * m * n;
  lean_object* out = mk_byte_array(out_numel * 4);
  uint8_t* o = byte_array_cptr(out);
  size_t* batch_idx = batch_rank == 0 ? NULL : malloc(batch_rank * sizeof(size_t));
  size_t* a_idx = malloc(out_rank * sizeof(size_t));
  size_t* b_idx = malloc(out_rank * sizeof(size_t));
  size_t* out_idx = malloc(out_rank * sizeof(size_t));
  size_t max_rank = aView->max_rank > bView->max_rank ? aView->max_rank : bView->max_rank;
  size_t* idx_buf = max_rank == 0 ? NULL : malloc(max_rank * sizeof(size_t));
  size_t* tmp_buf = max_rank == 0 ? NULL : malloc(max_rank * sizeof(size_t));
  if ((batch_rank && !batch_dims) || (batch_rank && !batch_idx) || !a_idx || !b_idx || !out_idx ||
      (max_rank && !idx_buf) || (max_rank && !tmp_buf)) {
    free(batch_dims);
    free(batch_idx);
    free(a_idx);
    free(b_idx);
    free(out_idx);
    free(idx_buf);
    free(tmp_buf);
    return mk_byte_array(0);
  }
  for (size_t bflat = 0; bflat < batch_numel; ++bflat) {
    if (batch_rank) unflatten_index(bflat, batch_dims, batch_rank, batch_idx);
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (size_t d = 0; d < batch_rank; ++d) {
          a_idx[d] = batch_idx[d];
          b_idx[d] = batch_idx[d];
          out_idx[d] = batch_idx[d];
        }
        out_idx[batch_rank] = i;
        out_idx[batch_rank + 1] = j;
        for (size_t t = 0; t < k; ++t) {
          a_idx[batch_rank] = i;
          a_idx[batch_rank + 1] = t;
          b_idx[batch_rank] = t;
          b_idx[batch_rank + 1] = j;
          float av = view_stack_read_f32(a, aView, a_idx, idx_buf, tmp_buf);
          float bv = view_stack_read_f32(b, bView, b_idx, idx_buf, tmp_buf);
          sum += av * bv;
        }
        size_t out_flat = (bflat * m + i) * n + j;
        write_f32(o, out_flat, sum);
      }
    }
  }
  free(batch_dims);
  free(batch_idx);
  free(a_idx);
  free(b_idx);
  free(out_idx);
  free(idx_buf);
  free(tmp_buf);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_f32(lean_obj_arg a, lean_obj_arg b,
    lean_obj_arg aStrides, lean_obj_arg aOffset, lean_obj_arg aMaskStarts, lean_obj_arg aMaskEnds,
    lean_obj_arg bStrides, lean_obj_arg bOffset, lean_obj_arg bMaskStarts, lean_obj_arg bMaskEnds,
    lean_obj_arg outShape, lean_obj_arg k) {
  view_info aView = {0}, bView = {0};
  if (!view_info_init(&aView, aStrides, unbox_int64(aOffset), aMaskStarts, aMaskEnds) ||
      !view_info_init(&bView, bStrides, unbox_int64(bOffset), bMaskStarts, bMaskEnds)) {
    view_info_free(&aView);
    view_info_free(&bView);
    return mk_byte_array(0);
  }
  lean_object* out = matmul_view_core(byte_array_cptr((lean_object*)a), &aView,
    byte_array_cptr((lean_object*)b), &bView, NULL, NULL, NULL, NULL, outShape,
    nat_to_size(k), 0, 0, 0);
  view_info_free(&aView);
  view_info_free(&bView);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_stack_f32(lean_obj_arg a, lean_obj_arg b,
    lean_obj_arg aStackShapes, lean_obj_arg aStackStrides, lean_obj_arg aStackOffsets,
    lean_obj_arg aStackMaskStarts, lean_obj_arg aStackMaskEnds,
    lean_obj_arg bStackShapes, lean_obj_arg bStackStrides, lean_obj_arg bStackOffsets,
    lean_obj_arg bStackMaskStarts, lean_obj_arg bStackMaskEnds,
    lean_obj_arg outShape, lean_obj_arg k) {
  view_stack aView = {0}, bView = {0};
  if (!view_stack_init(&aView, aStackShapes, aStackStrides, aStackOffsets, aStackMaskStarts,
      aStackMaskEnds) ||
      !view_stack_init(&bView, bStackShapes, bStackStrides, bStackOffsets, bStackMaskStarts,
        bStackMaskEnds)) {
    view_stack_free(&aView);
    view_stack_free(&bView);
    return mk_byte_array(0);
  }
  size_t out_rank = array_size(outShape);
  if (aView.views[aView.depth - 1].rank != out_rank || bView.views[bView.depth - 1].rank != out_rank) {
    view_stack_free(&aView);
    view_stack_free(&bView);
    return mk_byte_array(0);
  }
  lean_object* out = matmul_view_stack_core(byte_array_cptr((lean_object*)a), &aView,
    byte_array_cptr((lean_object*)b), &bView, outShape, nat_to_size(k));
  view_stack_free(&aView);
  view_stack_free(&bView);
  return out;
}

static lean_object* matmul_view_bias_core(b_lean_obj_arg a, b_lean_obj_arg b, b_lean_obj_arg bias0,
    b_lean_obj_arg bias1, b_lean_obj_arg aStrides, b_lean_obj_arg aOffset,
    b_lean_obj_arg aMaskStarts, b_lean_obj_arg aMaskEnds, b_lean_obj_arg bStrides,
    b_lean_obj_arg bOffset, b_lean_obj_arg bMaskStarts, b_lean_obj_arg bMaskEnds,
    b_lean_obj_arg bias0Strides, b_lean_obj_arg bias0Offset, b_lean_obj_arg bias0MaskStarts,
    b_lean_obj_arg bias0MaskEnds, b_lean_obj_arg bias1Strides, b_lean_obj_arg bias1Offset,
    b_lean_obj_arg bias1MaskStarts, b_lean_obj_arg bias1MaskEnds, b_lean_obj_arg outShape,
    b_lean_obj_arg k, int do_relu, int do_scale, uint32_t scaleBits, int has_bias1) {
  view_info aView = {0}, bView = {0}, b0View = {0}, b1View = {0};
  int ok = view_info_init(&aView, aStrides, unbox_int64(aOffset), aMaskStarts, aMaskEnds) &&
           view_info_init(&bView, bStrides, unbox_int64(bOffset), bMaskStarts, bMaskEnds) &&
           view_info_init(&b0View, bias0Strides, unbox_int64(bias0Offset), bias0MaskStarts,
             bias0MaskEnds);
  if (!ok) {
    view_info_free(&aView);
    view_info_free(&bView);
    view_info_free(&b0View);
    return mk_byte_array(0);
  }
  if (has_bias1) {
    if (!view_info_init(&b1View, bias1Strides, unbox_int64(bias1Offset), bias1MaskStarts,
        bias1MaskEnds)) {
      view_info_free(&aView);
      view_info_free(&bView);
      view_info_free(&b0View);
      view_info_free(&b1View);
      return mk_byte_array(0);
    }
  }
  lean_object* out = matmul_view_core(byte_array_cptr((lean_object*)a), &aView,
    byte_array_cptr((lean_object*)b), &bView, byte_array_cptr((lean_object*)bias0), &b0View,
    has_bias1 ? byte_array_cptr((lean_object*)bias1) : NULL, has_bias1 ? &b1View : NULL, outShape,
    nat_to_size(k), do_relu, do_scale, scaleBits);
  view_info_free(&aView);
  view_info_free(&bView);
  view_info_free(&b0View);
  view_info_free(&b1View);
  return out;
}

static void add_bias_f32(uint8_t* out, const uint8_t* bias, size_t m, size_t n,
    b_lean_obj_arg biasShape) {
  size_t rank = array_size(biasShape);
  size_t* dims = malloc(rank * sizeof(size_t));
  size_t* strides = malloc(rank * sizeof(size_t));
  size_t* out_idx = malloc(2 * sizeof(size_t));
  for (size_t i = 0; i < rank; ++i) dims[i] = nat_array_get(biasShape, i);
  make_strides(dims, rank, strides);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      out_idx[0] = i;
      out_idx[1] = j;
      size_t b_flat = broadcast_flat_index(out_idx, 2, dims, rank, strides);
      float v = read_f32(out, i * n + j) + read_f32(bias, b_flat);
      write_f32(out, i * n + j, v);
    }
  }
  free(dims);
  free(strides);
  free(out_idx);
}

static lean_object* matmul_bias_core(b_lean_obj_arg a, b_lean_obj_arg b, b_lean_obj_arg bias,
    b_lean_obj_arg biasShape, b_lean_obj_arg m, b_lean_obj_arg k, b_lean_obj_arg n,
    int do_relu, int do_scale, uint32_t scaleBits) {
  size_t m_n = nat_to_size(m);
  size_t k_n = nat_to_size(k);
  size_t n_n = nat_to_size(n);
  float scale = do_scale ? f32_from_bits(scaleBits) : 1.0f;
  lean_object* out = mk_byte_array(m_n * n_n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  const uint8_t* y = byte_array_cptr((lean_object*)b);
  for (size_t i = 0; i < m_n; ++i) {
    for (size_t j = 0; j < n_n; ++j) {
      float sum = 0.0f;
      for (size_t t = 0; t < k_n; ++t) {
        sum += read_f32(x, i * k_n + t) * read_f32(y, t * n_n + j);
      }
      write_f32(o, i * n_n + j, sum * scale);
    }
  }
  add_bias_f32(o, byte_array_cptr((lean_object*)bias), m_n, n_n, biasShape);
  if (do_relu) {
    for (size_t i = 0; i < m_n * n_n; ++i) {
      float v = read_f32(o, i);
      write_f32(o, i, v > 0.0f ? v : 0.0f);
    }
  }
  return out;
}

static lean_object* matmul_bias2_core(b_lean_obj_arg a, b_lean_obj_arg b, b_lean_obj_arg bias0,
    b_lean_obj_arg bias0Shape, b_lean_obj_arg bias1, b_lean_obj_arg bias1Shape,
    b_lean_obj_arg m, b_lean_obj_arg k, b_lean_obj_arg n,
    int do_relu, int do_scale, uint32_t scaleBits) {
  lean_object* out = matmul_bias_core(a, b, bias0, bias0Shape, m, k, n, 0, do_scale, scaleBits);
  size_t m_n = nat_to_size(m);
  size_t n_n = nat_to_size(n);
  add_bias_f32(byte_array_cptr(out), byte_array_cptr((lean_object*)bias1), m_n, n_n, bias1Shape);
  if (do_relu) {
    for (size_t i = 0; i < m_n * n_n; ++i) {
      float v = read_f32(byte_array_cptr(out), i);
      write_f32(byte_array_cptr(out), i, v > 0.0f ? v : 0.0f);
    }
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_bias_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg bias, b_lean_obj_arg biasShape, b_lean_obj_arg m, b_lean_obj_arg k,
    b_lean_obj_arg n) {
  return matmul_bias_core(a, b, bias, biasShape, m, k, n, 0, 0, 0);
}

LEAN_EXPORT lean_obj_res tg4_matmul_bias_scale_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg bias, b_lean_obj_arg biasShape, b_lean_obj_arg m, b_lean_obj_arg k,
    b_lean_obj_arg n, b_lean_obj_arg scaleBits) {
  return matmul_bias_core(a, b, bias, biasShape, m, k, n, 0, 1, unbox_u32(scaleBits));
}

LEAN_EXPORT lean_obj_res tg4_matmul_bias_relu_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg bias, b_lean_obj_arg biasShape, b_lean_obj_arg m, b_lean_obj_arg k,
    b_lean_obj_arg n) {
  return matmul_bias_core(a, b, bias, biasShape, m, k, n, 1, 0, 0);
}

LEAN_EXPORT lean_obj_res tg4_matmul_bias_scale_relu_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg bias, b_lean_obj_arg biasShape, b_lean_obj_arg m, b_lean_obj_arg k,
    b_lean_obj_arg n, b_lean_obj_arg scaleBits) {
  return matmul_bias_core(a, b, bias, biasShape, m, k, n, 1, 1, unbox_u32(scaleBits));
}

LEAN_EXPORT lean_obj_res tg4_matmul_bias2_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg bias0, b_lean_obj_arg bias0Shape, b_lean_obj_arg bias1, b_lean_obj_arg bias1Shape,
    b_lean_obj_arg m, b_lean_obj_arg k, b_lean_obj_arg n) {
  return matmul_bias2_core(a, b, bias0, bias0Shape, bias1, bias1Shape, m, k, n, 0, 0, 0);
}

LEAN_EXPORT lean_obj_res tg4_matmul_bias2_scale_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg bias0, b_lean_obj_arg bias0Shape, b_lean_obj_arg bias1, b_lean_obj_arg bias1Shape,
    b_lean_obj_arg m, b_lean_obj_arg k, b_lean_obj_arg n, b_lean_obj_arg scaleBits) {
  return matmul_bias2_core(a, b, bias0, bias0Shape, bias1, bias1Shape, m, k, n, 0, 1,
    unbox_u32(scaleBits));
}

LEAN_EXPORT lean_obj_res tg4_matmul_bias2_relu_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg bias0, b_lean_obj_arg bias0Shape, b_lean_obj_arg bias1, b_lean_obj_arg bias1Shape,
    b_lean_obj_arg m, b_lean_obj_arg k, b_lean_obj_arg n) {
  return matmul_bias2_core(a, b, bias0, bias0Shape, bias1, bias1Shape, m, k, n, 1, 0, 0);
}

LEAN_EXPORT lean_obj_res tg4_matmul_bias2_scale_relu_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg bias0, b_lean_obj_arg bias0Shape, b_lean_obj_arg bias1, b_lean_obj_arg bias1Shape,
    b_lean_obj_arg m, b_lean_obj_arg k, b_lean_obj_arg n, b_lean_obj_arg scaleBits) {
  return matmul_bias2_core(a, b, bias0, bias0Shape, bias1, bias1Shape, m, k, n, 1, 1,
    unbox_u32(scaleBits));
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_bias_f32(lean_obj_arg a, lean_obj_arg b, lean_obj_arg bias,
    lean_obj_arg aStrides, lean_obj_arg aOffset, lean_obj_arg aMaskStarts, lean_obj_arg aMaskEnds,
    lean_obj_arg bStrides, lean_obj_arg bOffset, lean_obj_arg bMaskStarts, lean_obj_arg bMaskEnds,
    lean_obj_arg biasStrides, lean_obj_arg biasOffset, lean_obj_arg biasMaskStarts,
    lean_obj_arg biasMaskEnds, lean_obj_arg outShape, lean_obj_arg k) {
  return matmul_view_bias_core(a, b, bias, NULL, aStrides, aOffset, aMaskStarts, aMaskEnds,
    bStrides, bOffset, bMaskStarts, bMaskEnds, biasStrides, biasOffset, biasMaskStarts,
    biasMaskEnds, NULL, NULL, NULL, NULL, outShape, k, 0, 0, 0, 0);
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_bias_scale_f32(lean_obj_arg a, lean_obj_arg b,
    lean_obj_arg bias, lean_obj_arg aStrides, lean_obj_arg aOffset, lean_obj_arg aMaskStarts,
    lean_obj_arg aMaskEnds, lean_obj_arg bStrides, lean_obj_arg bOffset, lean_obj_arg bMaskStarts,
    lean_obj_arg bMaskEnds, lean_obj_arg biasStrides, lean_obj_arg biasOffset,
    lean_obj_arg biasMaskStarts, lean_obj_arg biasMaskEnds, lean_obj_arg outShape, lean_obj_arg k,
    lean_obj_arg scaleBits) {
  return matmul_view_bias_core(a, b, bias, NULL, aStrides, aOffset, aMaskStarts, aMaskEnds,
    bStrides, bOffset, bMaskStarts, bMaskEnds, biasStrides, biasOffset, biasMaskStarts,
    biasMaskEnds, NULL, NULL, NULL, NULL, outShape, k, 0, 1, unbox_u32(scaleBits), 0);
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_bias_relu_f32(lean_obj_arg a, lean_obj_arg b,
    lean_obj_arg bias, lean_obj_arg aStrides, lean_obj_arg aOffset, lean_obj_arg aMaskStarts,
    lean_obj_arg aMaskEnds, lean_obj_arg bStrides, lean_obj_arg bOffset, lean_obj_arg bMaskStarts,
    lean_obj_arg bMaskEnds, lean_obj_arg biasStrides, lean_obj_arg biasOffset,
    lean_obj_arg biasMaskStarts, lean_obj_arg biasMaskEnds, lean_obj_arg outShape, lean_obj_arg k) {
  return matmul_view_bias_core(a, b, bias, NULL, aStrides, aOffset, aMaskStarts, aMaskEnds,
    bStrides, bOffset, bMaskStarts, bMaskEnds, biasStrides, biasOffset, biasMaskStarts,
    biasMaskEnds, NULL, NULL, NULL, NULL, outShape, k, 1, 0, 0, 0);
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_bias_scale_relu_f32(lean_obj_arg a, lean_obj_arg b,
    lean_obj_arg bias, lean_obj_arg aStrides, lean_obj_arg aOffset, lean_obj_arg aMaskStarts,
    lean_obj_arg aMaskEnds, lean_obj_arg bStrides, lean_obj_arg bOffset, lean_obj_arg bMaskStarts,
    lean_obj_arg bMaskEnds, lean_obj_arg biasStrides, lean_obj_arg biasOffset,
    lean_obj_arg biasMaskStarts, lean_obj_arg biasMaskEnds, lean_obj_arg outShape, lean_obj_arg k,
    lean_obj_arg scaleBits) {
  return matmul_view_bias_core(a, b, bias, NULL, aStrides, aOffset, aMaskStarts, aMaskEnds,
    bStrides, bOffset, bMaskStarts, bMaskEnds, biasStrides, biasOffset, biasMaskStarts,
    biasMaskEnds, NULL, NULL, NULL, NULL, outShape, k, 1, 1, unbox_u32(scaleBits), 0);
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_bias2_f32(lean_obj_arg a, lean_obj_arg b, lean_obj_arg bias0,
    lean_obj_arg bias1, lean_obj_arg aStrides, lean_obj_arg aOffset, lean_obj_arg aMaskStarts,
    lean_obj_arg aMaskEnds, lean_obj_arg bStrides, lean_obj_arg bOffset, lean_obj_arg bMaskStarts,
    lean_obj_arg bMaskEnds, lean_obj_arg bias0Strides, lean_obj_arg bias0Offset,
    lean_obj_arg bias0MaskStarts, lean_obj_arg bias0MaskEnds, lean_obj_arg bias1Strides,
    lean_obj_arg bias1Offset, lean_obj_arg bias1MaskStarts, lean_obj_arg bias1MaskEnds,
    lean_obj_arg outShape, lean_obj_arg k) {
  return matmul_view_bias_core(a, b, bias0, bias1, aStrides, aOffset, aMaskStarts, aMaskEnds,
    bStrides, bOffset, bMaskStarts, bMaskEnds, bias0Strides, bias0Offset, bias0MaskStarts,
    bias0MaskEnds, bias1Strides, bias1Offset, bias1MaskStarts, bias1MaskEnds, outShape, k, 0, 0,
    0, 1);
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_bias2_scale_f32(lean_obj_arg a, lean_obj_arg b,
    lean_obj_arg bias0, lean_obj_arg bias1, lean_obj_arg aStrides, lean_obj_arg aOffset,
    lean_obj_arg aMaskStarts, lean_obj_arg aMaskEnds, lean_obj_arg bStrides, lean_obj_arg bOffset,
    lean_obj_arg bMaskStarts, lean_obj_arg bMaskEnds, lean_obj_arg bias0Strides,
    lean_obj_arg bias0Offset, lean_obj_arg bias0MaskStarts, lean_obj_arg bias0MaskEnds,
    lean_obj_arg bias1Strides, lean_obj_arg bias1Offset, lean_obj_arg bias1MaskStarts,
    lean_obj_arg bias1MaskEnds, lean_obj_arg outShape, lean_obj_arg k, lean_obj_arg scaleBits) {
  return matmul_view_bias_core(a, b, bias0, bias1, aStrides, aOffset, aMaskStarts, aMaskEnds,
    bStrides, bOffset, bMaskStarts, bMaskEnds, bias0Strides, bias0Offset, bias0MaskStarts,
    bias0MaskEnds, bias1Strides, bias1Offset, bias1MaskStarts, bias1MaskEnds, outShape, k, 0, 1,
    unbox_u32(scaleBits), 1);
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_bias2_relu_f32(lean_obj_arg a, lean_obj_arg b,
    lean_obj_arg bias0, lean_obj_arg bias1, lean_obj_arg aStrides, lean_obj_arg aOffset,
    lean_obj_arg aMaskStarts, lean_obj_arg aMaskEnds, lean_obj_arg bStrides, lean_obj_arg bOffset,
    lean_obj_arg bMaskStarts, lean_obj_arg bMaskEnds, lean_obj_arg bias0Strides,
    lean_obj_arg bias0Offset, lean_obj_arg bias0MaskStarts, lean_obj_arg bias0MaskEnds,
    lean_obj_arg bias1Strides, lean_obj_arg bias1Offset, lean_obj_arg bias1MaskStarts,
    lean_obj_arg bias1MaskEnds, lean_obj_arg outShape, lean_obj_arg k) {
  return matmul_view_bias_core(a, b, bias0, bias1, aStrides, aOffset, aMaskStarts, aMaskEnds,
    bStrides, bOffset, bMaskStarts, bMaskEnds, bias0Strides, bias0Offset, bias0MaskStarts,
    bias0MaskEnds, bias1Strides, bias1Offset, bias1MaskStarts, bias1MaskEnds, outShape, k, 1, 0,
    0, 1);
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_bias2_scale_relu_f32(lean_obj_arg a, lean_obj_arg b,
    lean_obj_arg bias0, lean_obj_arg bias1, lean_obj_arg aStrides, lean_obj_arg aOffset,
    lean_obj_arg aMaskStarts, lean_obj_arg aMaskEnds, lean_obj_arg bStrides, lean_obj_arg bOffset,
    lean_obj_arg bMaskStarts, lean_obj_arg bMaskEnds, lean_obj_arg bias0Strides,
    lean_obj_arg bias0Offset, lean_obj_arg bias0MaskStarts, lean_obj_arg bias0MaskEnds,
    lean_obj_arg bias1Strides, lean_obj_arg bias1Offset, lean_obj_arg bias1MaskStarts,
    lean_obj_arg bias1MaskEnds, lean_obj_arg outShape, lean_obj_arg k, lean_obj_arg scaleBits) {
  return matmul_view_bias_core(a, b, bias0, bias1, aStrides, aOffset, aMaskStarts, aMaskEnds,
    bStrides, bOffset, bMaskStarts, bMaskEnds, bias0Strides, bias0Offset, bias0MaskStarts,
    bias0MaskEnds, bias1Strides, bias1Offset, bias1MaskStarts, bias1MaskEnds, outShape, k, 1, 1,
    unbox_u32(scaleBits), 1);
}

LEAN_EXPORT lean_obj_res tg4_matmul_batched_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg aStarts, b_lean_obj_arg bStarts, b_lean_obj_arg m, b_lean_obj_arg k,
    b_lean_obj_arg n) {
  size_t batch = array_size(aStarts);
  size_t m_n = nat_to_size(m);
  size_t k_n = nat_to_size(k);
  size_t n_n = nat_to_size(n);
  lean_object* out = mk_byte_array(batch * m_n * n_n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  const uint8_t* y = byte_array_cptr((lean_object*)b);

  for (size_t bidx = 0; bidx < batch; ++bidx) {
    size_t a_start = nat_array_get(aStarts, bidx) / 4;
    size_t b_start = nat_array_get(bStarts, bidx) / 4;
    uint8_t* out_batch = o + bidx * m_n * n_n * 4;
    for (size_t i = 0; i < m_n; ++i) {
      for (size_t j = 0; j < n_n; ++j) {
        float sum = 0.0f;
        for (size_t t = 0; t < k_n; ++t) {
          float av = read_f32(x, a_start + i * k_n + t);
          float bv = read_f32(y, b_start + t * n_n + j);
          sum += av * bv;
        }
        write_f32(out_batch, i * n_n + j, sum);
      }
    }
  }

  return out;
}

static lean_object* matmul_batched_bias_core(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg bias, b_lean_obj_arg biasShape, b_lean_obj_arg aStarts, b_lean_obj_arg bStarts,
    b_lean_obj_arg biasStarts, b_lean_obj_arg m, b_lean_obj_arg k, b_lean_obj_arg n,
    int do_relu, int do_scale, uint32_t scaleBits) {
  size_t batch = array_size(aStarts);
  size_t m_n = nat_to_size(m);
  size_t k_n = nat_to_size(k);
  size_t n_n = nat_to_size(n);
  float scale = do_scale ? f32_from_bits(scaleBits) : 1.0f;
  lean_object* out = mk_byte_array(batch * m_n * n_n * 4);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* x = byte_array_cptr((lean_object*)a);
  const uint8_t* y = byte_array_cptr((lean_object*)b);
  const uint8_t* b0 = byte_array_cptr((lean_object*)bias);

  for (size_t bidx = 0; bidx < batch; ++bidx) {
    size_t a_start = nat_array_get(aStarts, bidx) / 4;
    size_t b_start = nat_array_get(bStarts, bidx) / 4;
    size_t bias_start = nat_array_get(biasStarts, bidx) / 4;
    uint8_t* out_batch = o + bidx * m_n * n_n * 4;
    for (size_t i = 0; i < m_n; ++i) {
      for (size_t j = 0; j < n_n; ++j) {
        float sum = 0.0f;
        for (size_t t = 0; t < k_n; ++t) {
          float av = read_f32(x, a_start + i * k_n + t);
          float bv = read_f32(y, b_start + t * n_n + j);
          sum += av * bv;
        }
        write_f32(out_batch, i * n_n + j, sum * scale);
      }
    }
    add_bias_f32(out_batch, b0 + bias_start * 4, m_n, n_n, biasShape);
    if (do_relu) {
      for (size_t i = 0; i < m_n * n_n; ++i) {
        float v = read_f32(out_batch, i);
        write_f32(out_batch, i, v > 0.0f ? v : 0.0f);
      }
    }
  }

  return out;
}

static lean_object* matmul_batched_bias2_core(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg bias0, b_lean_obj_arg bias0Shape, b_lean_obj_arg bias1,
    b_lean_obj_arg bias1Shape, b_lean_obj_arg aStarts, b_lean_obj_arg bStarts,
    b_lean_obj_arg bias0Starts, b_lean_obj_arg bias1Starts, b_lean_obj_arg m, b_lean_obj_arg k,
    b_lean_obj_arg n, int do_relu, int do_scale, uint32_t scaleBits) {
  lean_object* out = matmul_batched_bias_core(a, b, bias0, bias0Shape, aStarts, bStarts,
    bias0Starts, m, k, n, 0, do_scale, scaleBits);
  size_t batch = array_size(aStarts);
  size_t m_n = nat_to_size(m);
  size_t n_n = nat_to_size(n);
  uint8_t* o = byte_array_cptr(out);
  const uint8_t* b1 = byte_array_cptr((lean_object*)bias1);

  for (size_t bidx = 0; bidx < batch; ++bidx) {
    size_t bias_start = nat_array_get(bias1Starts, bidx) / 4;
    uint8_t* out_batch = o + bidx * m_n * n_n * 4;
    add_bias_f32(out_batch, b1 + bias_start * 4, m_n, n_n, bias1Shape);
    if (do_relu) {
      for (size_t i = 0; i < m_n * n_n; ++i) {
        float v = read_f32(out_batch, i);
        write_f32(out_batch, i, v > 0.0f ? v : 0.0f);
      }
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_batched_bias_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg bias, b_lean_obj_arg biasShape, b_lean_obj_arg aStarts, b_lean_obj_arg bStarts,
    b_lean_obj_arg biasStarts, b_lean_obj_arg m, b_lean_obj_arg k, b_lean_obj_arg n) {
  return matmul_batched_bias_core(a, b, bias, biasShape, aStarts, bStarts, biasStarts, m, k, n, 0, 0, 0);
}

LEAN_EXPORT lean_obj_res tg4_matmul_batched_bias_scale_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg bias, b_lean_obj_arg biasShape, b_lean_obj_arg aStarts, b_lean_obj_arg bStarts,
    b_lean_obj_arg biasStarts, b_lean_obj_arg m, b_lean_obj_arg k, b_lean_obj_arg n,
    b_lean_obj_arg scaleBits) {
  return matmul_batched_bias_core(a, b, bias, biasShape, aStarts, bStarts, biasStarts, m, k, n,
    0, 1, unbox_u32(scaleBits));
}

LEAN_EXPORT lean_obj_res tg4_matmul_batched_bias_relu_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg bias, b_lean_obj_arg biasShape, b_lean_obj_arg aStarts, b_lean_obj_arg bStarts,
    b_lean_obj_arg biasStarts, b_lean_obj_arg m, b_lean_obj_arg k, b_lean_obj_arg n) {
  return matmul_batched_bias_core(a, b, bias, biasShape, aStarts, bStarts, biasStarts, m, k, n, 1, 0, 0);
}

LEAN_EXPORT lean_obj_res tg4_matmul_batched_bias_scale_relu_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg bias, b_lean_obj_arg biasShape, b_lean_obj_arg aStarts, b_lean_obj_arg bStarts,
    b_lean_obj_arg biasStarts, b_lean_obj_arg m, b_lean_obj_arg k, b_lean_obj_arg n,
    b_lean_obj_arg scaleBits) {
  return matmul_batched_bias_core(a, b, bias, biasShape, aStarts, bStarts, biasStarts, m, k, n,
    1, 1, unbox_u32(scaleBits));
}

LEAN_EXPORT lean_obj_res tg4_matmul_batched_bias2_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg bias0, b_lean_obj_arg bias0Shape, b_lean_obj_arg bias1, b_lean_obj_arg bias1Shape,
    b_lean_obj_arg aStarts, b_lean_obj_arg bStarts, b_lean_obj_arg bias0Starts,
    b_lean_obj_arg bias1Starts, b_lean_obj_arg m, b_lean_obj_arg k, b_lean_obj_arg n) {
  return matmul_batched_bias2_core(a, b, bias0, bias0Shape, bias1, bias1Shape, aStarts, bStarts,
    bias0Starts, bias1Starts, m, k, n, 0, 0, 0);
}

LEAN_EXPORT lean_obj_res tg4_matmul_batched_bias2_scale_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg bias0, b_lean_obj_arg bias0Shape, b_lean_obj_arg bias1, b_lean_obj_arg bias1Shape,
    b_lean_obj_arg aStarts, b_lean_obj_arg bStarts, b_lean_obj_arg bias0Starts,
    b_lean_obj_arg bias1Starts, b_lean_obj_arg m, b_lean_obj_arg k, b_lean_obj_arg n,
    b_lean_obj_arg scaleBits) {
  return matmul_batched_bias2_core(a, b, bias0, bias0Shape, bias1, bias1Shape, aStarts, bStarts,
    bias0Starts, bias1Starts, m, k, n, 0, 1, unbox_u32(scaleBits));
}

LEAN_EXPORT lean_obj_res tg4_matmul_batched_bias2_relu_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg bias0, b_lean_obj_arg bias0Shape, b_lean_obj_arg bias1, b_lean_obj_arg bias1Shape,
    b_lean_obj_arg aStarts, b_lean_obj_arg bStarts, b_lean_obj_arg bias0Starts,
    b_lean_obj_arg bias1Starts, b_lean_obj_arg m, b_lean_obj_arg k, b_lean_obj_arg n) {
  return matmul_batched_bias2_core(a, b, bias0, bias0Shape, bias1, bias1Shape, aStarts, bStarts,
    bias0Starts, bias1Starts, m, k, n, 1, 0, 0);
}

LEAN_EXPORT lean_obj_res tg4_matmul_batched_bias2_scale_relu_f32(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg bias0, b_lean_obj_arg bias0Shape, b_lean_obj_arg bias1, b_lean_obj_arg bias1Shape,
    b_lean_obj_arg aStarts, b_lean_obj_arg bStarts, b_lean_obj_arg bias0Starts,
    b_lean_obj_arg bias1Starts, b_lean_obj_arg m, b_lean_obj_arg k, b_lean_obj_arg n,
    b_lean_obj_arg scaleBits) {
  return matmul_batched_bias2_core(a, b, bias0, bias0Shape, bias1, bias1Shape, aStarts, bStarts,
    bias0Starts, bias1Starts, m, k, n, 1, 1, unbox_u32(scaleBits));
}

LEAN_EXPORT lean_obj_res tg4_matmul_f64(b_lean_obj_arg a, b_lean_obj_arg b,
    b_lean_obj_arg m, b_lean_obj_arg k, b_lean_obj_arg n) {
  size_t m_n = nat_to_size(m);
  size_t k_n = nat_to_size(k);
  size_t n_n = nat_to_size(n);
  size_t total = m_n * n_n;
  lean_object* out = mk_float_array(total);
  double* o = float_array_cptr(out);
  const double* x = float_array_cptr((lean_object*)a);
  const double* y = float_array_cptr((lean_object*)b);
  for (size_t i = 0; i < m_n; ++i) {
    for (size_t j = 0; j < n_n; ++j) {
      double sum = 0.0;
      for (size_t t = 0; t < k_n; ++t) {
        sum += x[i * k_n + t] * y[t * n_n + j];
      }
      o[i * n_n + j] = sum;
    }
  }
  return out;
}
