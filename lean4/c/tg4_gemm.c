#include <lean/lean.h>

// Portable (no BLAS) blocked GEMM for Lean FloatArray (Float = double).

extern float sqrtf(float); // NOLINT
extern float exp2f(float); // NOLINT
extern float log2f(float); // NOLINT
extern float sinf(float); // NOLINT
extern void *alloca(size_t); // NOLINT

static inline size_t min_size(size_t a, size_t b) { return a < b ? a : b; }

static inline size_t f32_elems(b_lean_obj_arg bytesObj) {
  const size_t bytes = lean_sarray_size(bytesObj);
  return bytes / 4;
}

static inline const float *f32_ptr(b_lean_obj_arg bytesObj) { return (const float *)lean_sarray_cptr(bytesObj); } // NOLINT
static inline float *f32_ptr_mut(b_lean_obj_arg bytesObj) { return (float *)lean_sarray_cptr(bytesObj); } // NOLINT

static inline bool read_shape_dims(b_lean_obj_arg shapeObj, size_t *dims) {
  const size_t rank = lean_array_size(shapeObj);
  for (size_t i = 0; i < rank; i++) {
    b_lean_obj_arg dObj = lean_array_get_core(shapeObj, i);
    if (!lean_is_scalar(dObj)) {
      return false;
    }
    dims[i] = lean_unbox(dObj);
  }
  return true;
}

static inline bool read_padded_dims(b_lean_obj_arg shapeObj, size_t outRank, size_t *dims) {
  const size_t rank = lean_array_size(shapeObj);
  if (rank > outRank) {
    return false;
  }
  const size_t pad = outRank - rank;
  for (size_t i = 0; i < pad; i++) dims[i] = 1;
  for (size_t i = 0; i < rank; i++) {
    b_lean_obj_arg dObj = lean_array_get_core(shapeObj, i);
    if (!lean_is_scalar(dObj)) {
      return false;
    }
    dims[pad + i] = lean_unbox(dObj);
  }
  return true;
}

static inline size_t prod_dims(const size_t *dims, size_t rank) {
  size_t p = 1;
  for (size_t i = 0; i < rank; i++) p *= dims[i];
  return p;
}

static inline void contiguous_strides(const size_t *dims, size_t rank, size_t *strides) {
  size_t running = 1;
  for (size_t i = rank; i-- > 0;) {
    strides[i] = running;
    running *= dims[i];
  }
}

static inline void broadcast_strides(const size_t *dims, size_t rank, size_t *strides) {
  for (size_t i = 0; i < rank; i++) {
    if (dims[i] == 1) strides[i] = 0;
  }
}

static inline bool is_broadcastable(const size_t *dims, const size_t *outDims, size_t rank) {
  for (size_t i = 0; i < rank; i++) {
    const size_t d = dims[i];
    const size_t o = outDims[i];
    if (d != 1 && d != o) return false;
  }
  return true;
}

static inline bool read_perm(b_lean_obj_arg permObj, size_t rank, size_t *perm) {
  if (lean_array_size(permObj) != rank) return false;
  bool *seen = LEAN_ALLOCA(sizeof(bool) * rank); // NOLINT
  for (size_t i = 0; i < rank; i++) seen[i] = false;
  for (size_t i = 0; i < rank; i++) {
    b_lean_obj_arg pObj = lean_array_get_core(permObj, i);
    if (!lean_is_scalar(pObj)) return false;
    const size_t p = lean_unbox(pObj);
    if (p >= rank) return false;
    if (seen[p]) return false;
    seen[p] = true;
    perm[i] = p;
  }
  return true;
}

static inline bool read_nat_array_len(b_lean_obj_arg arrObj, size_t len, size_t *out) {
  if (lean_array_size(arrObj) != len) return false;
  for (size_t i = 0; i < len; i++) {
    b_lean_obj_arg xObj = lean_array_get_core(arrObj, i);
    if (!lean_is_scalar(xObj)) return false;
    out[i] = lean_unbox(xObj);
  }
  return true;
}

static inline bool read_i64_array_len(b_lean_obj_arg arrObj, size_t len, int64_t *out) {
  if (lean_array_size(arrObj) != len) return false;
  for (size_t i = 0; i < len; i++) {
    b_lean_obj_arg xObj = lean_array_get_core(arrObj, i);
    if (!lean_is_ctor(xObj)) return false;
    out[i] = (int64_t)lean_unbox_uint64(xObj);
  }
  return true;
}

static inline int64_t i64_from_u64(uint64_t bits) {
  union {
    uint64_t u;
    int64_t i;
  } conv;
  conv.u = bits;
  return conv.i;
}

static inline bool read_axes_mask(b_lean_obj_arg axesObj, size_t rank, bool *mask) {
  for (size_t i = 0; i < rank; i++) mask[i] = false;
  const size_t n = lean_array_size(axesObj);
  for (size_t i = 0; i < n; i++) {
    b_lean_obj_arg axObj = lean_array_get_core(axesObj, i);
    if (!lean_is_scalar(axObj)) return false;
    const size_t ax = lean_unbox(axObj);
    if (ax >= rank) return false;
    mask[ax] = true;
  }
  return true;
}

LEAN_EXPORT lean_obj_res tg4_full_f32(b_lean_obj_arg nObj, double v) {
  if (!lean_is_scalar(nObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }
  const size_t n = lean_unbox(nObj);
  const float fv = (float)v;
  const size_t outBytes = n * 4;

  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  for (size_t i = 0; i < n; i++) dst[i] = fv;
  return out;
}

LEAN_EXPORT lean_obj_res tg4_full_f32_bits(b_lean_obj_arg nObj, uint32_t bits) {
  if (!lean_is_scalar(nObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }
  const size_t n = lean_unbox(nObj);
  union {
    uint32_t u;
    float f;
  } conv;
  conv.u = bits;
  const float fv = conv.f;
  const size_t outBytes = n * 4;

  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  for (size_t i = 0; i < n; i++) dst[i] = fv;
  return out;
}

LEAN_EXPORT lean_obj_res tg4_expand_scalar_f32(b_lean_obj_arg scalarBytes, b_lean_obj_arg nObj) {
  if (!lean_is_scalar(nObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }
  const size_t n = lean_unbox(nObj);
  if (lean_sarray_size(scalarBytes) < 4) {
    return lean_mk_empty_byte_array(lean_box(0));
  }
  const float v = f32_ptr(scalarBytes)[0];
  const size_t outBytes = n * 4;

  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  for (size_t i = 0; i < n; i++) dst[i] = v;
  return out;
}

LEAN_EXPORT lean_obj_res tg4_expand_bcast_f32(b_lean_obj_arg aBytes, b_lean_obj_arg aShapeObj, b_lean_obj_arg outShapeObj) {
  const size_t rank = lean_array_size(outShapeObj);
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_padded_dims(aShapeObj, rank, aDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!is_broadcastable(aDims, outDims, rank)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  const size_t aNumel = prod_dims(aDims, rank);
  if (lean_sarray_size(aBytes) < aNumel * 4) return lean_mk_empty_byte_array(lean_box(0));

  contiguous_strides(aDims, rank, aStrides);
  broadcast_strides(aDims, rank, aStrides);

  for (size_t i = 0; i < rank; i++) idx[i] = 0;

  const float *a = f32_ptr(aBytes);
  size_t aOff = 0;
  for (size_t i = 0; i < outNumel; i++) {
    dst[i] = a[aOff];
    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aOff += aStrides[d];
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      aOff -= aStrides[d] * outDims[d];
    }
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_expand_bcast_u8(b_lean_obj_arg aBytes, b_lean_obj_arg aShapeObj, b_lean_obj_arg outShapeObj) {
  const size_t rank = lean_array_size(outShapeObj);
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_padded_dims(aShapeObj, rank, aDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!is_broadcastable(aDims, outDims, rank)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t outNumel = prod_dims(outDims, rank);
  lean_obj_res out = lean_alloc_sarray(1, outNumel, outNumel);
  uint8_t *dst = (uint8_t *)lean_sarray_cptr(out); // NOLINT
  if (outNumel == 0) return out;

  const size_t aNumel = prod_dims(aDims, rank);
  if (lean_sarray_size(aBytes) < aNumel) return lean_mk_empty_byte_array(lean_box(0));

  contiguous_strides(aDims, rank, aStrides);
  broadcast_strides(aDims, rank, aStrides);

  for (size_t i = 0; i < rank; i++) idx[i] = 0;

  const uint8_t *a = (const uint8_t *)lean_sarray_cptr(aBytes); // NOLINT
  size_t aOff = 0;
  for (size_t i = 0; i < outNumel; i++) {
    dst[i] = a[aOff];
    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aOff += aStrides[d];
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      aOff -= aStrides[d] * outDims[d];
    }
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_neg_f32(b_lean_obj_arg aBytes) {
  const size_t n = f32_elems(aBytes);
  const size_t outBytes = n * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  const float *a = f32_ptr(aBytes);
  float *dst = f32_ptr_mut(out);
  for (size_t i = 0; i < n; i++) dst[i] = -a[i];
  return out;
}

static inline void tg4_relu_f32_inplace_ptr(float *dst, size_t n) {
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float x0 = dst[i + 0];
    float x1 = dst[i + 1];
    float x2 = dst[i + 2];
    float x3 = dst[i + 3];
    dst[i + 0] = (x0 > 0.0f) ? x0 : 0.0f;
    dst[i + 1] = (x1 > 0.0f) ? x1 : 0.0f;
    dst[i + 2] = (x2 > 0.0f) ? x2 : 0.0f;
    dst[i + 3] = (x3 > 0.0f) ? x3 : 0.0f;
  }
  for (; i < n; i++) {
    const float x = dst[i];
    dst[i] = (x > 0.0f) ? x : 0.0f;
  }
}

LEAN_EXPORT lean_obj_res tg4_relu_f32(b_lean_obj_arg aBytes) {
  const size_t n = f32_elems(aBytes);
  const size_t outBytes = n * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  const float *a = f32_ptr(aBytes);
  float *dst = f32_ptr_mut(out);
  for (size_t i = 0; i < n; i++) {
    const float x = a[i];
    dst[i] = (x > 0.0f) ? x : 0.0f;
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_sqrt_f32(b_lean_obj_arg aBytes) {
  const size_t n = f32_elems(aBytes);
  const size_t outBytes = n * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  const float *a = f32_ptr(aBytes);
  float *dst = f32_ptr_mut(out);
  for (size_t i = 0; i < n; i++) dst[i] = sqrtf(a[i]);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_reciprocal_f32(b_lean_obj_arg aBytes) {
  const size_t n = f32_elems(aBytes);
  const size_t outBytes = n * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  const float *a = f32_ptr(aBytes);
  float *dst = f32_ptr_mut(out);
  for (size_t i = 0; i < n; i++) dst[i] = 1.0f / a[i];
  return out;
}

LEAN_EXPORT lean_obj_res tg4_exp2_f32(b_lean_obj_arg aBytes) {
  const size_t n = f32_elems(aBytes);
  const size_t outBytes = n * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  const float *a = f32_ptr(aBytes);
  float *dst = f32_ptr_mut(out);
  for (size_t i = 0; i < n; i++) dst[i] = exp2f(a[i]);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_log2_f32(b_lean_obj_arg aBytes) {
  const size_t n = f32_elems(aBytes);
  const size_t outBytes = n * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  const float *a = f32_ptr(aBytes);
  float *dst = f32_ptr_mut(out);
  for (size_t i = 0; i < n; i++) dst[i] = log2f(a[i]);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_sin_f32(b_lean_obj_arg aBytes) {
  const size_t n = f32_elems(aBytes);
  const size_t outBytes = n * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  const float *a = f32_ptr(aBytes);
  float *dst = f32_ptr_mut(out);
  for (size_t i = 0; i < n; i++) dst[i] = sinf(a[i]);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_add_f32(b_lean_obj_arg aBytes, b_lean_obj_arg bBytes) {
  const size_t n = min_size(f32_elems(aBytes), f32_elems(bBytes));
  const size_t outBytes = n * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  const float *a = f32_ptr(aBytes);
  const float *b = f32_ptr(bBytes);
  float *dst = f32_ptr_mut(out);
  for (size_t i = 0; i < n; i++) dst[i] = a[i] + b[i];
  return out;
}

LEAN_EXPORT lean_obj_res tg4_sub_f32(b_lean_obj_arg aBytes, b_lean_obj_arg bBytes) {
  const size_t n = min_size(f32_elems(aBytes), f32_elems(bBytes));
  const size_t outBytes = n * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  const float *a = f32_ptr(aBytes);
  const float *b = f32_ptr(bBytes);
  float *dst = f32_ptr_mut(out);
  for (size_t i = 0; i < n; i++) dst[i] = a[i] - b[i];
  return out;
}

LEAN_EXPORT lean_obj_res tg4_mul_f32(b_lean_obj_arg aBytes, b_lean_obj_arg bBytes) {
  const size_t n = min_size(f32_elems(aBytes), f32_elems(bBytes));
  const size_t outBytes = n * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  const float *a = f32_ptr(aBytes);
  const float *b = f32_ptr(bBytes);
  float *dst = f32_ptr_mut(out);
  for (size_t i = 0; i < n; i++) dst[i] = a[i] * b[i];
  return out;
}

// SGD update: w <- w - lr * grad
LEAN_EXPORT lean_obj_res tg4_sgd_update_f32(b_lean_obj_arg wBytes, b_lean_obj_arg gradBytes, double lr) {
  const size_t wSz = lean_sarray_size(wBytes);
  const size_t gSz = lean_sarray_size(gradBytes);
  if (wSz != gSz || (wSz & 3) != 0) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  lean_obj_res out = lean_alloc_sarray(1, wSz, wSz);
  const float *w = f32_ptr(wBytes);
  const float *g = f32_ptr(gradBytes);
  float *dst = f32_ptr_mut(out);

  const size_t n = wSz / 4;
  const float flr = (float)lr;
  for (size_t i = 0; i < n; i++) dst[i] = w[i] - flr * g[i];
  return out;
}

LEAN_EXPORT lean_obj_res tg4_div_f32(b_lean_obj_arg aBytes, b_lean_obj_arg bBytes) {
  const size_t n = min_size(f32_elems(aBytes), f32_elems(bBytes));
  const size_t outBytes = n * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  const float *a = f32_ptr(aBytes);
  const float *b = f32_ptr(bBytes);
  float *dst = f32_ptr_mut(out);
  for (size_t i = 0; i < n; i++) dst[i] = a[i] / b[i];
  return out;
}

LEAN_EXPORT lean_obj_res tg4_max_f32(b_lean_obj_arg aBytes, b_lean_obj_arg bBytes) {
  const size_t n = min_size(f32_elems(aBytes), f32_elems(bBytes));
  const size_t outBytes = n * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  const float *a = f32_ptr(aBytes);
  const float *b = f32_ptr(bBytes);
  float *dst = f32_ptr_mut(out);
  for (size_t i = 0; i < n; i++) dst[i] = (a[i] >= b[i]) ? a[i] : b[i];
  return out;
}

LEAN_EXPORT lean_obj_res tg4_add_bcast_f32(
    b_lean_obj_arg aBytes,
    b_lean_obj_arg bBytes,
    b_lean_obj_arg aShapeObj,
    b_lean_obj_arg bShapeObj,
    b_lean_obj_arg outShapeObj) {
  const size_t rank = lean_array_size(outShapeObj);
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_padded_dims(aShapeObj, rank, aDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_padded_dims(bShapeObj, rank, bDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!is_broadcastable(aDims, outDims, rank) || !is_broadcastable(bDims, outDims, rank)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  const size_t aNumel = prod_dims(aDims, rank);
  const size_t bNumel = prod_dims(bDims, rank);
  if (lean_sarray_size(aBytes) < aNumel * 4 || lean_sarray_size(bBytes) < bNumel * 4) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  contiguous_strides(aDims, rank, aStrides);
  contiguous_strides(bDims, rank, bStrides);
  broadcast_strides(aDims, rank, aStrides);
  broadcast_strides(bDims, rank, bStrides);

  for (size_t i = 0; i < rank; i++) idx[i] = 0;

  const float *a = f32_ptr(aBytes);
  const float *b = f32_ptr(bBytes);
  size_t aOff = 0;
  size_t bOff = 0;
  for (size_t i = 0; i < outNumel; i++) {
    dst[i] = a[aOff] + b[bOff];
    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aOff += aStrides[d];
      bOff += bStrides[d];
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      aOff -= aStrides[d] * outDims[d];
      bOff -= bStrides[d] * outDims[d];
    }
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_sub_bcast_f32(
    b_lean_obj_arg aBytes,
    b_lean_obj_arg bBytes,
    b_lean_obj_arg aShapeObj,
    b_lean_obj_arg bShapeObj,
    b_lean_obj_arg outShapeObj) {
  const size_t rank = lean_array_size(outShapeObj);
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_padded_dims(aShapeObj, rank, aDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_padded_dims(bShapeObj, rank, bDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!is_broadcastable(aDims, outDims, rank) || !is_broadcastable(bDims, outDims, rank)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  const size_t aNumel = prod_dims(aDims, rank);
  const size_t bNumel = prod_dims(bDims, rank);
  if (lean_sarray_size(aBytes) < aNumel * 4 || lean_sarray_size(bBytes) < bNumel * 4) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  contiguous_strides(aDims, rank, aStrides);
  contiguous_strides(bDims, rank, bStrides);
  broadcast_strides(aDims, rank, aStrides);
  broadcast_strides(bDims, rank, bStrides);

  for (size_t i = 0; i < rank; i++) idx[i] = 0;

  const float *a = f32_ptr(aBytes);
  const float *b = f32_ptr(bBytes);
  size_t aOff = 0;
  size_t bOff = 0;
  for (size_t i = 0; i < outNumel; i++) {
    dst[i] = a[aOff] - b[bOff];
    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aOff += aStrides[d];
      bOff += bStrides[d];
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      aOff -= aStrides[d] * outDims[d];
      bOff -= bStrides[d] * outDims[d];
    }
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_mul_bcast_f32(
    b_lean_obj_arg aBytes,
    b_lean_obj_arg bBytes,
    b_lean_obj_arg aShapeObj,
    b_lean_obj_arg bShapeObj,
    b_lean_obj_arg outShapeObj) {
  const size_t rank = lean_array_size(outShapeObj);
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_padded_dims(aShapeObj, rank, aDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_padded_dims(bShapeObj, rank, bDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!is_broadcastable(aDims, outDims, rank) || !is_broadcastable(bDims, outDims, rank)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  const size_t aNumel = prod_dims(aDims, rank);
  const size_t bNumel = prod_dims(bDims, rank);
  if (lean_sarray_size(aBytes) < aNumel * 4 || lean_sarray_size(bBytes) < bNumel * 4) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  contiguous_strides(aDims, rank, aStrides);
  contiguous_strides(bDims, rank, bStrides);
  broadcast_strides(aDims, rank, aStrides);
  broadcast_strides(bDims, rank, bStrides);

  for (size_t i = 0; i < rank; i++) idx[i] = 0;

  const float *a = f32_ptr(aBytes);
  const float *b = f32_ptr(bBytes);
  size_t aOff = 0;
  size_t bOff = 0;
  for (size_t i = 0; i < outNumel; i++) {
    dst[i] = a[aOff] * b[bOff];
    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aOff += aStrides[d];
      bOff += bStrides[d];
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      aOff -= aStrides[d] * outDims[d];
      bOff -= bStrides[d] * outDims[d];
    }
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_div_bcast_f32(
    b_lean_obj_arg aBytes,
    b_lean_obj_arg bBytes,
    b_lean_obj_arg aShapeObj,
    b_lean_obj_arg bShapeObj,
    b_lean_obj_arg outShapeObj) {
  const size_t rank = lean_array_size(outShapeObj);
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_padded_dims(aShapeObj, rank, aDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_padded_dims(bShapeObj, rank, bDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!is_broadcastable(aDims, outDims, rank) || !is_broadcastable(bDims, outDims, rank)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  const size_t aNumel = prod_dims(aDims, rank);
  const size_t bNumel = prod_dims(bDims, rank);
  if (lean_sarray_size(aBytes) < aNumel * 4 || lean_sarray_size(bBytes) < bNumel * 4) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  contiguous_strides(aDims, rank, aStrides);
  contiguous_strides(bDims, rank, bStrides);
  broadcast_strides(aDims, rank, aStrides);
  broadcast_strides(bDims, rank, bStrides);

  for (size_t i = 0; i < rank; i++) idx[i] = 0;

  const float *a = f32_ptr(aBytes);
  const float *b = f32_ptr(bBytes);
  size_t aOff = 0;
  size_t bOff = 0;
  for (size_t i = 0; i < outNumel; i++) {
    dst[i] = a[aOff] / b[bOff];
    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aOff += aStrides[d];
      bOff += bStrides[d];
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      aOff -= aStrides[d] * outDims[d];
      bOff -= bStrides[d] * outDims[d];
    }
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_max_bcast_f32(
    b_lean_obj_arg aBytes,
    b_lean_obj_arg bBytes,
    b_lean_obj_arg aShapeObj,
    b_lean_obj_arg bShapeObj,
    b_lean_obj_arg outShapeObj) {
  const size_t rank = lean_array_size(outShapeObj);
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_padded_dims(aShapeObj, rank, aDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_padded_dims(bShapeObj, rank, bDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!is_broadcastable(aDims, outDims, rank) || !is_broadcastable(bDims, outDims, rank)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  const size_t aNumel = prod_dims(aDims, rank);
  const size_t bNumel = prod_dims(bDims, rank);
  if (lean_sarray_size(aBytes) < aNumel * 4 || lean_sarray_size(bBytes) < bNumel * 4) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  contiguous_strides(aDims, rank, aStrides);
  contiguous_strides(bDims, rank, bStrides);
  broadcast_strides(aDims, rank, aStrides);
  broadcast_strides(bDims, rank, bStrides);

  for (size_t i = 0; i < rank; i++) idx[i] = 0;

  const float *a = f32_ptr(aBytes);
  const float *b = f32_ptr(bBytes);
  size_t aOff = 0;
  size_t bOff = 0;
  for (size_t i = 0; i < outNumel; i++) {
    const float av = a[aOff];
    const float bv = b[bOff];
    dst[i] = (av >= bv) ? av : bv;
    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aOff += aStrides[d];
      bOff += bStrides[d];
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      aOff -= aStrides[d] * outDims[d];
      bOff -= bStrides[d] * outDims[d];
    }
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_cmplt_f32(b_lean_obj_arg aBytes, b_lean_obj_arg bBytes) {
  const size_t n = min_size(f32_elems(aBytes), f32_elems(bBytes));
  lean_obj_res out = lean_alloc_sarray(1, n, n);
  const float *a = f32_ptr(aBytes);
  const float *b = f32_ptr(bBytes);
  uint8_t *dst = lean_sarray_cptr(out);
  for (size_t i = 0; i < n; i++) dst[i] = (a[i] < b[i]) ? 1 : 0;
  return out;
}

LEAN_EXPORT lean_obj_res tg4_cmplt_bcast_f32(
    b_lean_obj_arg aBytes,
    b_lean_obj_arg bBytes,
    b_lean_obj_arg aShapeObj,
    b_lean_obj_arg bShapeObj,
    b_lean_obj_arg outShapeObj) {
  const size_t rank = lean_array_size(outShapeObj);
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_padded_dims(aShapeObj, rank, aDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_padded_dims(bShapeObj, rank, bDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!is_broadcastable(aDims, outDims, rank) || !is_broadcastable(bDims, outDims, rank)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t outNumel = prod_dims(outDims, rank);
  lean_obj_res out = lean_alloc_sarray(1, outNumel, outNumel);
  uint8_t *dst = lean_sarray_cptr(out);
  if (outNumel == 0) return out;

  const size_t aNumel = prod_dims(aDims, rank);
  const size_t bNumel = prod_dims(bDims, rank);
  if (lean_sarray_size(aBytes) < aNumel * 4 || lean_sarray_size(bBytes) < bNumel * 4) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  contiguous_strides(aDims, rank, aStrides);
  contiguous_strides(bDims, rank, bStrides);
  broadcast_strides(aDims, rank, aStrides);
  broadcast_strides(bDims, rank, bStrides);

  for (size_t i = 0; i < rank; i++) idx[i] = 0;

  const float *a = f32_ptr(aBytes);
  const float *b = f32_ptr(bBytes);
  size_t aOff = 0;
  size_t bOff = 0;
  for (size_t i = 0; i < outNumel; i++) {
    dst[i] = (a[aOff] < b[bOff]) ? 1 : 0;
    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aOff += aStrides[d];
      bOff += bStrides[d];
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      aOff -= aStrides[d] * outDims[d];
      bOff -= bStrides[d] * outDims[d];
    }
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_where_f32(b_lean_obj_arg condBytes, b_lean_obj_arg xBytes, b_lean_obj_arg yBytes, b_lean_obj_arg nObj) {
  if (!lean_is_scalar(nObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t n = lean_unbox(nObj);
  const size_t condSz = lean_sarray_size(condBytes);
  const size_t xSz = lean_sarray_size(xBytes);
  const size_t ySz = lean_sarray_size(yBytes);
  const uint8_t *cond = lean_sarray_cptr(condBytes);
  const float *x = f32_ptr(xBytes);
  const float *y = f32_ptr(yBytes);
  const bool condScalar = (condSz == 1);
  const bool xScalar = (xSz == 4);
  const bool yScalar = (ySz == 4);

  const size_t outBytes = n * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  for (size_t i = 0; i < n; i++) {
    const uint8_t c = cond[condScalar ? 0 : i];
    const float xv = x[xScalar ? 0 : i];
    const float yv = y[yScalar ? 0 : i];
    dst[i] = c ? xv : yv;
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_where_bcast_f32(
    b_lean_obj_arg condBytes,
    b_lean_obj_arg xBytes,
    b_lean_obj_arg yBytes,
    b_lean_obj_arg condShapeObj,
    b_lean_obj_arg xShapeObj,
    b_lean_obj_arg yShapeObj,
    b_lean_obj_arg outShapeObj) {
  const size_t rank = lean_array_size(outShapeObj);
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *cDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *xDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *yDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *cStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *xStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *yStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_padded_dims(condShapeObj, rank, cDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_padded_dims(xShapeObj, rank, xDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_padded_dims(yShapeObj, rank, yDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!is_broadcastable(cDims, outDims, rank) || !is_broadcastable(xDims, outDims, rank) || !is_broadcastable(yDims, outDims, rank)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  const size_t cNumel = prod_dims(cDims, rank);
  const size_t xNumel = prod_dims(xDims, rank);
  const size_t yNumel = prod_dims(yDims, rank);
  if (lean_sarray_size(condBytes) < cNumel || lean_sarray_size(xBytes) < xNumel * 4 || lean_sarray_size(yBytes) < yNumel * 4) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  contiguous_strides(cDims, rank, cStrides);
  contiguous_strides(xDims, rank, xStrides);
  contiguous_strides(yDims, rank, yStrides);
  broadcast_strides(cDims, rank, cStrides);
  broadcast_strides(xDims, rank, xStrides);
  broadcast_strides(yDims, rank, yStrides);

  for (size_t i = 0; i < rank; i++) idx[i] = 0;

  const uint8_t *cond = lean_sarray_cptr(condBytes);
  const float *x = f32_ptr(xBytes);
  const float *y = f32_ptr(yBytes);
  size_t cOff = 0;
  size_t xOff = 0;
  size_t yOff = 0;
  for (size_t i = 0; i < outNumel; i++) {
    dst[i] = cond[cOff] ? x[xOff] : y[yOff];
    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      cOff += cStrides[d];
      xOff += xStrides[d];
      yOff += yStrides[d];
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      cOff -= cStrides[d] * outDims[d];
      xOff -= xStrides[d] * outDims[d];
      yOff -= yStrides[d] * outDims[d];
    }
  }
  return out;
}

static inline float f32_from_bits(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } conv;
  conv.u = bits;
  return conv.f;
}

LEAN_EXPORT lean_obj_res tg4_fused_ewise_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg inputShapesObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg outShapeObj,
    b_lean_obj_arg progObj) {
  const size_t rank = lean_array_size(outShapeObj);
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(inputShapesObj) != nInputs || lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) {
    return lean_mk_empty_byte_array(lean_box(0));
  }
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  size_t *inDims = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *inStrides = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *inOff = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  uint8_t *inScalar = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  float *inScalarVals = LEAN_ALLOCA(sizeof(float) * nInputs); // NOLINT

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    b_lean_obj_arg shapeObj = lean_array_get_core(inputShapesObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT

    size_t *dims = inDims + i * rank;
    size_t *strides = inStrides + i * rank;
    if (!read_padded_dims(shapeObj, rank, dims)) return lean_mk_empty_byte_array(lean_box(0));
    if (!is_broadcastable(dims, outDims, rank)) return lean_mk_empty_byte_array(lean_box(0));

    const size_t numel = prod_dims(dims, rank);
    const size_t needed = (dt == 1) ? numel : (numel * 4);
    if (lean_sarray_size(bytesObj) < needed) return lean_mk_empty_byte_array(lean_box(0));

    contiguous_strides(dims, rank, strides);
    broadcast_strides(dims, rank, strides);
    inOff[i] = 0;
    uint8_t scalar = 1;
    for (size_t d = 0; d < rank; d++) {
      if (strides[d] != 0) { scalar = 0; break; }
    }
    inScalar[i] = scalar;
    if (scalar) {
      if (dt == 1) { // bool/u8
        const uint8_t v = inPtrs[i][0];
        inScalarVals[i] = v ? 1.0f : 0.0f;
      } else { // f32
        const float *p = (const float *)inPtrs[i]; // NOLINT
        inScalarVals[i] = p[0];
      }
    }
  }

  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT

  for (size_t o = 0; o < outNumel; o++) {
    size_t sp = 0;
    for (size_t ip = 0; ip < instsLen; ip++) {
      const uint64_t inst = prog[ip];

      const uint8_t kind = (uint8_t)(inst & 0xFFu);
      const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
      const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
      const uint32_t imm = (uint32_t)(inst >> 32);

      switch (kind) {
        case 0: // const f32 bits
          stack[sp++] = f32_from_bits(imm);
          break;
        case 1: { // load input[arg]
          if (arg >= nInputs) {
            stack[sp++] = 0.0f;
            break;
          }
          if (inScalar[arg]) {
            stack[sp++] = inScalarVals[arg];
          } else {
            const size_t off = inOff[arg];
            if (inDtypes[arg] == 1) { // bool/u8
              const uint8_t v = inPtrs[arg][off];
              stack[sp++] = v ? 1.0f : 0.0f;
            } else { // f32
              const float *p = (const float *)inPtrs[arg]; // NOLINT
              stack[sp++] = p[off];
            }
          }
          break;
        }
        case 2: { // unary
          if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
          const float x = stack[sp - 1];
          switch (op) {
            case 0: stack[sp - 1] = -x; break;
            case 1: stack[sp - 1] = sqrtf(x); break;
            case 2: stack[sp - 1] = 1.0f / x; break;
            case 3: stack[sp - 1] = exp2f(x); break;
            case 4: stack[sp - 1] = log2f(x); break;
            case 5: stack[sp - 1] = sinf(x); break;
            default: return lean_mk_empty_byte_array(lean_box(0));
          }
          break;
        }
        case 3: { // binary
          if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
          const float y = stack[--sp];
          const float x = stack[--sp];
          switch (op) {
            case 0: stack[sp++] = x + y; break;
            case 1: stack[sp++] = x - y; break;
            case 2: stack[sp++] = x * y; break;
            case 3: stack[sp++] = x / y; break;
            case 4: stack[sp++] = (x >= y) ? x : y; break;
            case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
            default: return lean_mk_empty_byte_array(lean_box(0));
          }
          break;
        }
        case 4: { // where(cond, x, y)
          if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
          const float y = stack[--sp];
          const float x = stack[--sp];
          const float c = stack[--sp];
          stack[sp++] = (c != 0.0f) ? x : y;
          break;
        }
        default:
          return lean_mk_empty_byte_array(lean_box(0));
      }
    }

    if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
    dst[o] = stack[0];

    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] += inStrides[i * rank + d];
      }
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] -= inStrides[i * rank + d] * outDims[d];
      }
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_ewise_view_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg inputStridesObj,
    b_lean_obj_arg inputOffsetsObj,
    b_lean_obj_arg inputMaskStartsObj,
    b_lean_obj_arg inputMaskEndsObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg outShapeObj,
    b_lean_obj_arg progObj) {
  const size_t rank = lean_array_size(outShapeObj);
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(inputStridesObj) != nInputs || lean_array_size(inputOffsetsObj) != nInputs ||
      lean_array_size(inputMaskStartsObj) != nInputs || lean_array_size(inputMaskEndsObj) != nInputs ||
      lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  int64_t *inStrides = LEAN_ALLOCA(sizeof(int64_t) * nInputs * rank); // NOLINT
  int64_t *inOff = LEAN_ALLOCA(sizeof(int64_t) * nInputs); // NOLINT
  size_t *maskStart = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *maskEnd = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  size_t *inCap = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  uint8_t *maskFull = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  uint8_t *inScalar = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  float *inScalarVals = LEAN_ALLOCA(sizeof(float) * nInputs); // NOLINT

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT
    const size_t byteSize = lean_sarray_size(bytesObj);
    inCap[i] = (dt == 1) ? byteSize : (byteSize / 4);

    b_lean_obj_arg stridesObj = lean_array_get_core(inputStridesObj, i);
    if (!read_i64_array_len(stridesObj, rank, inStrides + i * rank)) return lean_mk_empty_byte_array(lean_box(0));

    b_lean_obj_arg offObj = lean_array_get_core(inputOffsetsObj, i);
    if (!lean_is_ctor(offObj)) return lean_mk_empty_byte_array(lean_box(0));
    inOff[i] = (int64_t)lean_unbox_uint64(offObj);

    b_lean_obj_arg msObj = lean_array_get_core(inputMaskStartsObj, i);
    b_lean_obj_arg meObj = lean_array_get_core(inputMaskEndsObj, i);
    if (!read_nat_array_len(msObj, rank, maskStart + i * rank)) return lean_mk_empty_byte_array(lean_box(0));
    if (!read_nat_array_len(meObj, rank, maskEnd + i * rank)) return lean_mk_empty_byte_array(lean_box(0));
    uint8_t full = 1;
    for (size_t d = 0; d < rank; d++) {
      const size_t ms = maskStart[i * rank + d];
      const size_t me = maskEnd[i * rank + d];
      if (ms != 0 || me != outDims[d]) full = 0;
      if (ms > me || me > outDims[d]) return lean_mk_empty_byte_array(lean_box(0));
    }
    maskFull[i] = full;

    uint8_t scalar = 1;
    for (size_t d = 0; d < rank; d++) {
      if (inStrides[i * rank + d] != 0) { scalar = 0; break; }
    }
    inScalar[i] = scalar;
    if (scalar) {
      const int64_t off = inOff[i];
      float val = 0.0f;
      if (off >= 0 && (uint64_t)off < inCap[i]) {
        if (inDtypes[i] == 1) {
          const uint8_t v = inPtrs[i][(size_t)off];
          val = v ? 1.0f : 0.0f;
        } else {
          const float *p = (const float *)inPtrs[i]; // NOLINT
          val = p[(size_t)off];
        }
      }
      inScalarVals[i] = val;
    }
  }

  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  uint8_t *valid = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT

  for (size_t o = 0; o < outNumel; o++) {
    for (size_t i = 0; i < nInputs; i++) {
      if (maskFull[i]) {
        valid[i] = 1;
        continue;
      }
      uint8_t ok = 1;
      for (size_t d = 0; d < rank; d++) {
        const size_t x = idx[d];
        const size_t ms = maskStart[i * rank + d];
        const size_t me = maskEnd[i * rank + d];
        if (x < ms || x >= me) {
          ok = 0;
          break;
        }
      }
      valid[i] = ok;
    }

    size_t sp = 0;
    for (size_t ip = 0; ip < instsLen; ip++) {
      const uint64_t inst = prog[ip];

      const uint8_t kind = (uint8_t)(inst & 0xFFu);
      const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
      const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
      const uint32_t imm = (uint32_t)(inst >> 32);

      switch (kind) {
        case 0:
          stack[sp++] = f32_from_bits(imm);
          break;
        case 1: {
          if (arg >= nInputs || !valid[arg]) {
            stack[sp++] = 0.0f;
            break;
          }
          if (inScalar[arg]) {
            stack[sp++] = inScalarVals[arg];
            break;
          }
          const int64_t off = inOff[arg];
          if (off < 0 || (uint64_t)off >= inCap[arg]) {
            stack[sp++] = 0.0f;
            break;
          }
          if (inDtypes[arg] == 1) {
            const uint8_t v = inPtrs[arg][(size_t)off];
            stack[sp++] = v ? 1.0f : 0.0f;
          } else {
            const float *p = (const float *)inPtrs[arg]; // NOLINT
            stack[sp++] = p[(size_t)off];
          }
          break;
        }
        case 2: {
          if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
          const float x = stack[sp - 1];
          switch (op) {
            case 0: stack[sp - 1] = -x; break;
            case 1: stack[sp - 1] = sqrtf(x); break;
            case 2: stack[sp - 1] = 1.0f / x; break;
            case 3: stack[sp - 1] = exp2f(x); break;
            case 4: stack[sp - 1] = log2f(x); break;
            case 5: stack[sp - 1] = sinf(x); break;
            default: return lean_mk_empty_byte_array(lean_box(0));
          }
          break;
        }
        case 3: {
          if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
          const float y = stack[--sp];
          const float x = stack[--sp];
          switch (op) {
            case 0: stack[sp++] = x + y; break;
            case 1: stack[sp++] = x - y; break;
            case 2: stack[sp++] = x * y; break;
            case 3: stack[sp++] = x / y; break;
            case 4: stack[sp++] = (x >= y) ? x : y; break;
            case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
            default: return lean_mk_empty_byte_array(lean_box(0));
          }
          break;
        }
        case 4: {
          if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
          const float y = stack[--sp];
          const float x = stack[--sp];
          const float c = stack[--sp];
          stack[sp++] = (c != 0.0f) ? x : y;
          break;
        }
        default:
          return lean_mk_empty_byte_array(lean_box(0));
      }
    }

    if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
    dst[o] = stack[0];

    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] += inStrides[i * rank + d];
      }
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] -= inStrides[i * rank + d] * (int64_t)outDims[d];
      }
    }
  }

  return out;
}

// ---- View stack (ShapeTracker-style) ----

typedef struct tg4_view_stack_view {
  size_t rank;
  size_t numel;
  const size_t *contigStrides;
  const int64_t *strides;
  int64_t offset;
  const size_t *maskStart;
  const size_t *maskEnd;
} tg4_view_stack_view;

typedef struct tg4_view_stack {
  size_t nViews;
  const tg4_view_stack_view *views; // base -> top
} tg4_view_stack;

static inline bool tg4_view_stack_offset(
    const tg4_view_stack *stk,
    const size_t *outIdx,
    size_t outRank,
    size_t *idxWork,
    int64_t *outOff) {
  if (stk->nViews == 0) return false;

  const tg4_view_stack_view *views = stk->views;
  const size_t topRank = views[stk->nViews - 1].rank;
  if (topRank != outRank) return false;

  for (size_t d = 0; d < outRank; d++) idxWork[d] = outIdx[d];

  for (size_t vi = stk->nViews; vi-- > 0;) {
    const tg4_view_stack_view *v = &views[vi];

    for (size_t d = 0; d < v->rank; d++) {
      const size_t x = idxWork[d];
      const size_t ms = v->maskStart[d];
      const size_t me = v->maskEnd[d];
      if (x < ms || x >= me) return false;
    }

    int64_t lin = v->offset;
    for (size_t d = 0; d < v->rank; d++) {
      lin += (int64_t)idxWork[d] * v->strides[d];
    }

    if (vi == 0) {
      *outOff = lin;
      return true;
    }

    const tg4_view_stack_view *next = &views[vi - 1];
    if (lin < 0 || (uint64_t)lin >= next->numel) return false;

    size_t linU = (size_t)lin;
    for (size_t d = 0; d < next->rank; d++) {
      const size_t stride = next->contigStrides[d];
      idxWork[d] = linU / stride;
      linU = linU % stride;
    }
  }

  return false;
}

LEAN_EXPORT lean_obj_res tg4_fused_ewise_view_stack_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg stackShapesObj,
    b_lean_obj_arg stackStridesObj,
    b_lean_obj_arg stackOffsetsObj,
    b_lean_obj_arg stackMaskStartsObj,
    b_lean_obj_arg stackMaskEndsObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg outShapeObj,
    b_lean_obj_arg progObj) {
  const size_t rank = lean_array_size(outShapeObj);
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(stackShapesObj) != nInputs || lean_array_size(stackStridesObj) != nInputs || lean_array_size(stackOffsetsObj) != nInputs ||
      lean_array_size(stackMaskStartsObj) != nInputs || lean_array_size(stackMaskEndsObj) != nInputs || lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  size_t *inCap = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  tg4_view_stack *stacks = LEAN_ALLOCA(sizeof(tg4_view_stack) * nInputs); // NOLINT
  size_t maxRank = rank;

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT
    const size_t byteSize = lean_sarray_size(bytesObj);
    inCap[i] = (dt == 1) ? byteSize : (byteSize / 4);

    b_lean_obj_arg shapesViewsObj = lean_array_get_core(stackShapesObj, i);
    b_lean_obj_arg stridesViewsObj = lean_array_get_core(stackStridesObj, i);
    b_lean_obj_arg offsetsViewsObj = lean_array_get_core(stackOffsetsObj, i);
    b_lean_obj_arg msViewsObj = lean_array_get_core(stackMaskStartsObj, i);
    b_lean_obj_arg meViewsObj = lean_array_get_core(stackMaskEndsObj, i);

    const size_t nViews = lean_array_size(shapesViewsObj);
    if (nViews == 0 || lean_array_size(stridesViewsObj) != nViews || lean_array_size(offsetsViewsObj) != nViews ||
        lean_array_size(msViewsObj) != nViews || lean_array_size(meViewsObj) != nViews) {
      return lean_mk_empty_byte_array(lean_box(0));
    }

    tg4_view_stack_view *views = LEAN_ALLOCA(sizeof(tg4_view_stack_view) * nViews); // NOLINT
    for (size_t vi = 0; vi < nViews; vi++) {
      b_lean_obj_arg shapeObj = lean_array_get_core(shapesViewsObj, vi);
      const size_t vRank = lean_array_size(shapeObj);

      size_t *dims = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      if (!read_shape_dims(shapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

      size_t *contig = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      contiguous_strides(dims, vRank, contig);

      const size_t vNumel = prod_dims(dims, vRank);

      b_lean_obj_arg stridesObj = lean_array_get_core(stridesViewsObj, vi);
      int64_t *strides = LEAN_ALLOCA(sizeof(int64_t) * vRank); // NOLINT
      if (!read_i64_array_len(stridesObj, vRank, strides)) return lean_mk_empty_byte_array(lean_box(0));

      b_lean_obj_arg offObj = lean_array_get_core(offsetsViewsObj, vi);
      if (!lean_is_ctor(offObj)) return lean_mk_empty_byte_array(lean_box(0));
      const int64_t offset = (int64_t)lean_unbox_uint64(offObj);

      b_lean_obj_arg msObj = lean_array_get_core(msViewsObj, vi);
      b_lean_obj_arg meObj = lean_array_get_core(meViewsObj, vi);
      size_t *maskStart = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      size_t *maskEnd = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      if (!read_nat_array_len(msObj, vRank, maskStart)) return lean_mk_empty_byte_array(lean_box(0));
      if (!read_nat_array_len(meObj, vRank, maskEnd)) return lean_mk_empty_byte_array(lean_box(0));
      for (size_t d = 0; d < vRank; d++) {
        const size_t ms = maskStart[d];
        const size_t me = maskEnd[d];
        if (ms > me || me > dims[d]) return lean_mk_empty_byte_array(lean_box(0));
      }

      views[vi].rank = vRank;
      views[vi].numel = vNumel;
      views[vi].contigStrides = contig;
      views[vi].strides = strides;
      views[vi].offset = offset;
      views[vi].maskStart = maskStart;
      views[vi].maskEnd = maskEnd;

      if (vRank > maxRank) maxRank = vRank;
    }

    stacks[i].nViews = nViews;
    stacks[i].views = views;

    // Basic sanity: top view should match the kernel output rank.
    if (views[nViews - 1].rank != rank) return lean_mk_empty_byte_array(lean_box(0));
  }

  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  size_t *idxWork = LEAN_ALLOCA(sizeof(size_t) * maxRank); // NOLINT
  size_t *zeroIdx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  uint8_t *valid = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  int64_t *inOff = LEAN_ALLOCA(sizeof(int64_t) * nInputs); // NOLINT
  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT
  uint8_t *inScalar = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  float *inScalarVals = LEAN_ALLOCA(sizeof(float) * nInputs); // NOLINT

  for (size_t d = 0; d < rank; d++) zeroIdx[d] = 0;
  for (size_t i = 0; i < nInputs; i++) {
    uint8_t scalar = 1;
    const tg4_view_stack_view *top = &stacks[i].views[stacks[i].nViews - 1];
    for (size_t d = 0; d < top->rank; d++) {
      if (top->strides[d] != 0) { scalar = 0; break; }
    }
    if (scalar) {
      for (size_t d = 0; d < top->rank; d++) {
        if (top->maskStart[d] != 0 || top->maskEnd[d] != outDims[d]) { scalar = 0; break; }
      }
    }
    inScalar[i] = scalar;
    if (scalar) {
      int64_t off = 0;
      const bool ok = tg4_view_stack_offset(&stacks[i], zeroIdx, rank, idxWork, &off);
      float val = 0.0f;
      if (ok && off >= 0 && (uint64_t)off < inCap[i]) {
        if (inDtypes[i] == 1) {
          const uint8_t v = inPtrs[i][(size_t)off];
          val = v ? 1.0f : 0.0f;
        } else {
          const float *p = (const float *)inPtrs[i]; // NOLINT
          val = p[(size_t)off];
        }
      }
      inScalarVals[i] = val;
    }
  }

  for (size_t o = 0; o < outNumel; o++) {
    for (size_t i = 0; i < nInputs; i++) {
      if (inScalar[i]) {
        valid[i] = 1;
        inOff[i] = 0;
        continue;
      }
      int64_t off = 0;
      const bool ok = tg4_view_stack_offset(&stacks[i], idx, rank, idxWork, &off);
      valid[i] = ok ? 1 : 0;
      inOff[i] = off;
    }

    size_t sp = 0;
    for (size_t ip = 0; ip < instsLen; ip++) {
      const uint64_t inst = prog[ip];

      const uint8_t kind = (uint8_t)(inst & 0xFFu);
      const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
      const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
      const uint32_t imm = (uint32_t)(inst >> 32);

      switch (kind) {
        case 0:
          stack[sp++] = f32_from_bits(imm);
          break;
        case 1: {
          if (arg >= nInputs || !valid[arg]) {
            stack[sp++] = 0.0f;
            break;
          }
          if (inScalar[arg]) {
            stack[sp++] = inScalarVals[arg];
            break;
          }
          const int64_t off = inOff[arg];
          if (off < 0 || (uint64_t)off >= inCap[arg]) {
            stack[sp++] = 0.0f;
            break;
          }
          if (inDtypes[arg] == 1) {
            const uint8_t v = inPtrs[arg][(size_t)off];
            stack[sp++] = v ? 1.0f : 0.0f;
          } else {
            const float *p = (const float *)inPtrs[arg]; // NOLINT
            stack[sp++] = p[(size_t)off];
          }
          break;
        }
        case 2: {
          if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
          const float x = stack[sp - 1];
          switch (op) {
            case 0: stack[sp - 1] = -x; break;
            case 1: stack[sp - 1] = sqrtf(x); break;
            case 2: stack[sp - 1] = 1.0f / x; break;
            case 3: stack[sp - 1] = exp2f(x); break;
            case 4: stack[sp - 1] = log2f(x); break;
            case 5: stack[sp - 1] = sinf(x); break;
            default: return lean_mk_empty_byte_array(lean_box(0));
          }
          break;
        }
        case 3: {
          if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
          const float y = stack[--sp];
          const float x = stack[--sp];
          switch (op) {
            case 0: stack[sp++] = x + y; break;
            case 1: stack[sp++] = x - y; break;
            case 2: stack[sp++] = x * y; break;
            case 3: stack[sp++] = x / y; break;
            case 4: stack[sp++] = (x >= y) ? x : y; break;
            case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
            default: return lean_mk_empty_byte_array(lean_box(0));
          }
          break;
        }
        case 4: {
          if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
          const float y = stack[--sp];
          const float x = stack[--sp];
          const float c = stack[--sp];
          stack[sp++] = (c != 0.0f) ? x : y;
          break;
        }
        default:
          return lean_mk_empty_byte_array(lean_box(0));
      }
    }

    if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
    dst[o] = stack[0];

    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_last_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg inputShapesObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg fullShapeObj,
    b_lean_obj_arg progObj) {
  const size_t rank = lean_array_size(fullShapeObj);
  if (rank == 0) return lean_mk_empty_byte_array(lean_box(0));

  size_t *dims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(fullShapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(inputShapesObj) != nInputs || lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  const size_t inner = dims[rank - 1];
  size_t outer = 1;
  for (size_t i = 0; i + 1 < rank; i++) outer *= dims[i];

  const size_t outBytes = outer * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outer == 0) return out;

  size_t *inDims = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *inStrides = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *inOffBase = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  size_t *inOff = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    b_lean_obj_arg shapeObj = lean_array_get_core(inputShapesObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT

    size_t *idims = inDims + i * rank;
    size_t *istrides = inStrides + i * rank;
    if (!read_padded_dims(shapeObj, rank, idims)) return lean_mk_empty_byte_array(lean_box(0));
    if (!is_broadcastable(idims, dims, rank)) return lean_mk_empty_byte_array(lean_box(0));

    const size_t numel = prod_dims(idims, rank);
    const size_t needed = (dt == 1) ? numel : (numel * 4);
    if (lean_sarray_size(bytesObj) < needed) return lean_mk_empty_byte_array(lean_box(0));

    contiguous_strides(idims, rank, istrides);
    broadcast_strides(idims, rank, istrides);
    inOffBase[i] = 0;
  }

  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT
  for (size_t o = 0; o < outer; o++) {
    for (size_t i = 0; i < nInputs; i++) inOff[i] = inOffBase[i];

    float acc = 0.0f;
    for (size_t r = 0; r < inner; r++) {
      size_t sp = 0;
      for (size_t ip = 0; ip < instsLen; ip++) {
        const uint64_t inst = prog[ip];

        const uint8_t kind = (uint8_t)(inst & 0xFFu);
        const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
        const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
        const uint32_t imm = (uint32_t)(inst >> 32);

        switch (kind) {
          case 0:
            stack[sp++] = f32_from_bits(imm);
            break;
          case 1: {
            if (arg >= nInputs) {
              stack[sp++] = 0.0f;
              break;
            }
            const size_t off = inOff[arg];
            if (inDtypes[arg] == 1) {
              const uint8_t v = inPtrs[arg][off];
              stack[sp++] = v ? 1.0f : 0.0f;
            } else {
              const float *p = (const float *)inPtrs[arg]; // NOLINT
              stack[sp++] = p[off];
            }
            break;
          }
          case 2: {
            if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
            const float x = stack[sp - 1];
            switch (op) {
              case 0: stack[sp - 1] = -x; break;
              case 1: stack[sp - 1] = sqrtf(x); break;
              case 2: stack[sp - 1] = 1.0f / x; break;
              case 3: stack[sp - 1] = exp2f(x); break;
              case 4: stack[sp - 1] = log2f(x); break;
              case 5: stack[sp - 1] = sinf(x); break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 3: {
            if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            switch (op) {
              case 0: stack[sp++] = x + y; break;
              case 1: stack[sp++] = x - y; break;
              case 2: stack[sp++] = x * y; break;
              case 3: stack[sp++] = x / y; break;
              case 4: stack[sp++] = (x >= y) ? x : y; break;
              case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 4: {
            if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            const float c = stack[--sp];
            stack[sp++] = (c != 0.0f) ? x : y;
            break;
          }
          default:
            return lean_mk_empty_byte_array(lean_box(0));
        }
      }

      if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
      acc += stack[0];
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] += inStrides[i * rank + (rank - 1)];
      }
    }
    dst[o] = acc;

    for (size_t d = rank - 1; d-- > 0;) {
      idx[d]++;
      for (size_t i = 0; i < nInputs; i++) {
        inOffBase[i] += inStrides[i * rank + d];
      }
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
      for (size_t i = 0; i < nInputs; i++) {
        inOffBase[i] -= inStrides[i * rank + d] * dims[d];
      }
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_last_view_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg inputStridesObj,
    b_lean_obj_arg inputOffsetsObj,
    b_lean_obj_arg inputMaskStartsObj,
    b_lean_obj_arg inputMaskEndsObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg fullShapeObj,
    b_lean_obj_arg progObj) {
  const size_t rank = lean_array_size(fullShapeObj);
  if (rank == 0) return lean_mk_empty_byte_array(lean_box(0));

  size_t *dims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(fullShapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(inputStridesObj) != nInputs || lean_array_size(inputOffsetsObj) != nInputs ||
      lean_array_size(inputMaskStartsObj) != nInputs || lean_array_size(inputMaskEndsObj) != nInputs ||
      lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  const size_t inner = dims[rank - 1];
  size_t outer = 1;
  for (size_t i = 0; i + 1 < rank; i++) outer *= dims[i];

  const size_t outBytes = outer * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outer == 0) return out;

  if (inner == 0) {
    for (size_t o = 0; o < outer; o++) dst[o] = 0.0f;
    return out;
  }

  int64_t *inStrides = LEAN_ALLOCA(sizeof(int64_t) * nInputs * rank); // NOLINT
  int64_t *inOffBase = LEAN_ALLOCA(sizeof(int64_t) * nInputs); // NOLINT
  int64_t *inOff = LEAN_ALLOCA(sizeof(int64_t) * nInputs); // NOLINT
  size_t *maskStart = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *maskEnd = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  size_t *inCap = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT
    const size_t byteSize = lean_sarray_size(bytesObj);
    inCap[i] = (dt == 1) ? byteSize : (byteSize / 4);

    b_lean_obj_arg stridesObj = lean_array_get_core(inputStridesObj, i);
    if (!read_i64_array_len(stridesObj, rank, inStrides + i * rank)) return lean_mk_empty_byte_array(lean_box(0));

    b_lean_obj_arg offObj = lean_array_get_core(inputOffsetsObj, i);
    if (!lean_is_ctor(offObj)) return lean_mk_empty_byte_array(lean_box(0));
    inOffBase[i] = (int64_t)lean_unbox_uint64(offObj);

    b_lean_obj_arg msObj = lean_array_get_core(inputMaskStartsObj, i);
    b_lean_obj_arg meObj = lean_array_get_core(inputMaskEndsObj, i);
    if (!read_nat_array_len(msObj, rank, maskStart + i * rank)) return lean_mk_empty_byte_array(lean_box(0));
    if (!read_nat_array_len(meObj, rank, maskEnd + i * rank)) return lean_mk_empty_byte_array(lean_box(0));
    for (size_t d = 0; d < rank; d++) {
      const size_t ms = maskStart[i * rank + d];
      const size_t me = maskEnd[i * rank + d];
      if (ms > me || me > dims[d]) return lean_mk_empty_byte_array(lean_box(0));
    }
  }

  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  uint8_t *validOuter = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  uint8_t *valid = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT

  const size_t redAxis = rank - 1;

  for (size_t o = 0; o < outer; o++) {
    for (size_t i = 0; i < nInputs; i++) inOff[i] = inOffBase[i];

    for (size_t i = 0; i < nInputs; i++) {
      uint8_t ok = 1;
      for (size_t d = 0; d + 1 < rank; d++) {
        const size_t x = idx[d];
        const size_t ms = maskStart[i * rank + d];
        const size_t me = maskEnd[i * rank + d];
        if (x < ms || x >= me) {
          ok = 0;
          break;
        }
      }
      validOuter[i] = ok;
    }

    float acc = 0.0f;
    for (size_t r = 0; r < inner; r++) {
      for (size_t i = 0; i < nInputs; i++) {
        if (!validOuter[i]) {
          valid[i] = 0;
          continue;
        }
        const size_t ms = maskStart[i * rank + redAxis];
        const size_t me = maskEnd[i * rank + redAxis];
        valid[i] = (r >= ms && r < me) ? 1 : 0;
      }

      size_t sp = 0;
      for (size_t ip = 0; ip < instsLen; ip++) {
        const uint64_t inst = prog[ip];

        const uint8_t kind = (uint8_t)(inst & 0xFFu);
        const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
        const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
        const uint32_t imm = (uint32_t)(inst >> 32);

        switch (kind) {
          case 0:
            stack[sp++] = f32_from_bits(imm);
            break;
          case 1: {
            if (arg >= nInputs || !valid[arg]) {
              stack[sp++] = 0.0f;
              break;
            }
            const int64_t off = inOff[arg];
            if (off < 0 || (uint64_t)off >= inCap[arg]) {
              stack[sp++] = 0.0f;
              break;
            }
            if (inDtypes[arg] == 1) {
              const uint8_t v = inPtrs[arg][(size_t)off];
              stack[sp++] = v ? 1.0f : 0.0f;
            } else {
              const float *p = (const float *)inPtrs[arg]; // NOLINT
              stack[sp++] = p[(size_t)off];
            }
            break;
          }
          case 2: {
            if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
            const float x = stack[sp - 1];
            switch (op) {
              case 0: stack[sp - 1] = -x; break;
              case 1: stack[sp - 1] = sqrtf(x); break;
              case 2: stack[sp - 1] = 1.0f / x; break;
              case 3: stack[sp - 1] = exp2f(x); break;
              case 4: stack[sp - 1] = log2f(x); break;
              case 5: stack[sp - 1] = sinf(x); break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 3: {
            if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            switch (op) {
              case 0: stack[sp++] = x + y; break;
              case 1: stack[sp++] = x - y; break;
              case 2: stack[sp++] = x * y; break;
              case 3: stack[sp++] = x / y; break;
              case 4: stack[sp++] = (x >= y) ? x : y; break;
              case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 4: {
            if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            const float c = stack[--sp];
            stack[sp++] = (c != 0.0f) ? x : y;
            break;
          }
          default:
            return lean_mk_empty_byte_array(lean_box(0));
        }
      }

      if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
      acc += stack[0];
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] += inStrides[i * rank + redAxis];
      }
    }
    dst[o] = acc;

    for (size_t d = rank - 1; d-- > 0;) {
      idx[d]++;
      for (size_t i = 0; i < nInputs; i++) {
        inOffBase[i] += inStrides[i * rank + d];
      }
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
      for (size_t i = 0; i < nInputs; i++) {
        inOffBase[i] -= inStrides[i * rank + d] * (int64_t)dims[d];
      }
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_last_view_stack_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg stackShapesObj,
    b_lean_obj_arg stackStridesObj,
    b_lean_obj_arg stackOffsetsObj,
    b_lean_obj_arg stackMaskStartsObj,
    b_lean_obj_arg stackMaskEndsObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg fullShapeObj,
    b_lean_obj_arg progObj) {
  const size_t rank = lean_array_size(fullShapeObj);
  if (rank == 0) return lean_mk_empty_byte_array(lean_box(0));

  size_t *dims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(fullShapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  const size_t inner = dims[rank - 1];
  size_t outer = 1;
  for (size_t i = 0; i + 1 < rank; i++) outer *= dims[i];

  const size_t outBytes = outer * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outer == 0) return out;

  if (inner == 0) {
    for (size_t o = 0; o < outer; o++) dst[o] = 0.0f;
    return out;
  }

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(stackShapesObj) != nInputs || lean_array_size(stackStridesObj) != nInputs || lean_array_size(stackOffsetsObj) != nInputs ||
      lean_array_size(stackMaskStartsObj) != nInputs || lean_array_size(stackMaskEndsObj) != nInputs || lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  size_t *inCap = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  tg4_view_stack *stacks = LEAN_ALLOCA(sizeof(tg4_view_stack) * nInputs); // NOLINT
  size_t maxRank = rank;

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT
    const size_t byteSize = lean_sarray_size(bytesObj);
    inCap[i] = (dt == 1) ? byteSize : (byteSize / 4);

    b_lean_obj_arg shapesViewsObj = lean_array_get_core(stackShapesObj, i);
    b_lean_obj_arg stridesViewsObj = lean_array_get_core(stackStridesObj, i);
    b_lean_obj_arg offsetsViewsObj = lean_array_get_core(stackOffsetsObj, i);
    b_lean_obj_arg msViewsObj = lean_array_get_core(stackMaskStartsObj, i);
    b_lean_obj_arg meViewsObj = lean_array_get_core(stackMaskEndsObj, i);

    const size_t nViews = lean_array_size(shapesViewsObj);
    if (nViews == 0 || lean_array_size(stridesViewsObj) != nViews || lean_array_size(offsetsViewsObj) != nViews ||
        lean_array_size(msViewsObj) != nViews || lean_array_size(meViewsObj) != nViews) {
      return lean_mk_empty_byte_array(lean_box(0));
    }

    tg4_view_stack_view *views = LEAN_ALLOCA(sizeof(tg4_view_stack_view) * nViews); // NOLINT
    for (size_t vi = 0; vi < nViews; vi++) {
      b_lean_obj_arg shapeObj = lean_array_get_core(shapesViewsObj, vi);
      const size_t vRank = lean_array_size(shapeObj);

      size_t *vDims = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      if (!read_shape_dims(shapeObj, vDims)) return lean_mk_empty_byte_array(lean_box(0));

      size_t *contig = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      contiguous_strides(vDims, vRank, contig);

      const size_t vNumel = prod_dims(vDims, vRank);

      b_lean_obj_arg stridesObj = lean_array_get_core(stridesViewsObj, vi);
      int64_t *strides = LEAN_ALLOCA(sizeof(int64_t) * vRank); // NOLINT
      if (!read_i64_array_len(stridesObj, vRank, strides)) return lean_mk_empty_byte_array(lean_box(0));

      b_lean_obj_arg offObj = lean_array_get_core(offsetsViewsObj, vi);
      if (!lean_is_ctor(offObj)) return lean_mk_empty_byte_array(lean_box(0));
      const int64_t offset = (int64_t)lean_unbox_uint64(offObj);

      b_lean_obj_arg msObj = lean_array_get_core(msViewsObj, vi);
      b_lean_obj_arg meObj = lean_array_get_core(meViewsObj, vi);
      size_t *maskStart = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      size_t *maskEnd = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      if (!read_nat_array_len(msObj, vRank, maskStart)) return lean_mk_empty_byte_array(lean_box(0));
      if (!read_nat_array_len(meObj, vRank, maskEnd)) return lean_mk_empty_byte_array(lean_box(0));
      for (size_t d = 0; d < vRank; d++) {
        const size_t ms = maskStart[d];
        const size_t me = maskEnd[d];
        if (ms > me || me > vDims[d]) return lean_mk_empty_byte_array(lean_box(0));
      }

      views[vi].rank = vRank;
      views[vi].numel = vNumel;
      views[vi].contigStrides = contig;
      views[vi].strides = strides;
      views[vi].offset = offset;
      views[vi].maskStart = maskStart;
      views[vi].maskEnd = maskEnd;

      if (vRank > maxRank) maxRank = vRank;
    }

    stacks[i].nViews = nViews;
    stacks[i].views = views;

    if (views[nViews - 1].rank != rank) return lean_mk_empty_byte_array(lean_box(0));
  }

  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  size_t *idxWork = LEAN_ALLOCA(sizeof(size_t) * maxRank); // NOLINT
  uint8_t *valid = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  int64_t *inOff = LEAN_ALLOCA(sizeof(int64_t) * nInputs); // NOLINT
  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT

  for (size_t o = 0; o < outer; o++) {
    float acc = 0.0f;
    for (size_t r = 0; r < inner; r++) {
      idx[rank - 1] = r;

      for (size_t i = 0; i < nInputs; i++) {
        int64_t off = 0;
        const bool ok = tg4_view_stack_offset(&stacks[i], idx, rank, idxWork, &off);
        valid[i] = ok ? 1 : 0;
        inOff[i] = off;
      }

      size_t sp = 0;
      for (size_t ip = 0; ip < instsLen; ip++) {
        const uint64_t inst = prog[ip];

        const uint8_t kind = (uint8_t)(inst & 0xFFu);
        const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
        const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
        const uint32_t imm = (uint32_t)(inst >> 32);

        switch (kind) {
          case 0:
            stack[sp++] = f32_from_bits(imm);
            break;
          case 1: {
            if (arg >= nInputs || !valid[arg]) {
              stack[sp++] = 0.0f;
              break;
            }
            const int64_t off = inOff[arg];
            if (off < 0 || (uint64_t)off >= inCap[arg]) {
              stack[sp++] = 0.0f;
              break;
            }
            if (inDtypes[arg] == 1) {
              const uint8_t v = inPtrs[arg][(size_t)off];
              stack[sp++] = v ? 1.0f : 0.0f;
            } else {
              const float *p = (const float *)inPtrs[arg]; // NOLINT
              stack[sp++] = p[(size_t)off];
            }
            break;
          }
          case 2: {
            if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
            const float x = stack[sp - 1];
            switch (op) {
              case 0: stack[sp - 1] = -x; break;
              case 1: stack[sp - 1] = sqrtf(x); break;
              case 2: stack[sp - 1] = 1.0f / x; break;
              case 3: stack[sp - 1] = exp2f(x); break;
              case 4: stack[sp - 1] = log2f(x); break;
              case 5: stack[sp - 1] = sinf(x); break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 3: {
            if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            switch (op) {
              case 0: stack[sp++] = x + y; break;
              case 1: stack[sp++] = x - y; break;
              case 2: stack[sp++] = x * y; break;
              case 3: stack[sp++] = x / y; break;
              case 4: stack[sp++] = (x >= y) ? x : y; break;
              case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 4: {
            if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            const float c = stack[--sp];
            stack[sp++] = (c != 0.0f) ? x : y;
            break;
          }
          default:
            return lean_mk_empty_byte_array(lean_box(0));
        }
      }

      if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
      acc += stack[0];
    }

    dst[o] = acc;
    idx[rank - 1] = 0;

    for (size_t d = rank - 1; d-- > 0;) {
      idx[d]++;
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_last_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg inputShapesObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg fullShapeObj,
    b_lean_obj_arg progObj) {
  const size_t rank = lean_array_size(fullShapeObj);
  if (rank == 0) return lean_mk_empty_byte_array(lean_box(0));

  size_t *dims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(fullShapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(inputShapesObj) != nInputs || lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  const size_t inner = dims[rank - 1];
  size_t outer = 1;
  for (size_t i = 0; i + 1 < rank; i++) outer *= dims[i];

  const size_t outBytes = outer * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outer == 0) return out;

  union {
    uint32_t u;
    float f;
  } conv;
  conv.u = 0xFF800000u; // -inf
  const float negInf = conv.f;

  if (inner == 0) {
    for (size_t o = 0; o < outer; o++) dst[o] = negInf;
    return out;
  }

  size_t *inDims = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *inStrides = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *inOffBase = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  size_t *inOff = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    b_lean_obj_arg shapeObj = lean_array_get_core(inputShapesObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT

    size_t *idims = inDims + i * rank;
    size_t *istrides = inStrides + i * rank;
    if (!read_padded_dims(shapeObj, rank, idims)) return lean_mk_empty_byte_array(lean_box(0));
    if (!is_broadcastable(idims, dims, rank)) return lean_mk_empty_byte_array(lean_box(0));

    const size_t numel = prod_dims(idims, rank);
    const size_t needed = (dt == 1) ? numel : (numel * 4);
    if (lean_sarray_size(bytesObj) < needed) return lean_mk_empty_byte_array(lean_box(0));

    contiguous_strides(idims, rank, istrides);
    broadcast_strides(idims, rank, istrides);
    inOffBase[i] = 0;
  }

  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT
  for (size_t o = 0; o < outer; o++) {
    for (size_t i = 0; i < nInputs; i++) inOff[i] = inOffBase[i];

    float acc = negInf;
    for (size_t r = 0; r < inner; r++) {
      size_t sp = 0;
      for (size_t ip = 0; ip < instsLen; ip++) {
        const uint64_t inst = prog[ip];

        const uint8_t kind = (uint8_t)(inst & 0xFFu);
        const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
        const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
        const uint32_t imm = (uint32_t)(inst >> 32);

        switch (kind) {
          case 0:
            stack[sp++] = f32_from_bits(imm);
            break;
          case 1: {
            if (arg >= nInputs) {
              stack[sp++] = 0.0f;
              break;
            }
            const size_t off = inOff[arg];
            if (inDtypes[arg] == 1) {
              const uint8_t v = inPtrs[arg][off];
              stack[sp++] = v ? 1.0f : 0.0f;
            } else {
              const float *p = (const float *)inPtrs[arg]; // NOLINT
              stack[sp++] = p[off];
            }
            break;
          }
          case 2: {
            if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
            const float x = stack[sp - 1];
            switch (op) {
              case 0: stack[sp - 1] = -x; break;
              case 1: stack[sp - 1] = sqrtf(x); break;
              case 2: stack[sp - 1] = 1.0f / x; break;
              case 3: stack[sp - 1] = exp2f(x); break;
              case 4: stack[sp - 1] = log2f(x); break;
              case 5: stack[sp - 1] = sinf(x); break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 3: {
            if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            switch (op) {
              case 0: stack[sp++] = x + y; break;
              case 1: stack[sp++] = x - y; break;
              case 2: stack[sp++] = x * y; break;
              case 3: stack[sp++] = x / y; break;
              case 4: stack[sp++] = (x >= y) ? x : y; break;
              case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 4: {
            if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            const float c = stack[--sp];
            stack[sp++] = (c != 0.0f) ? x : y;
            break;
          }
          default:
            return lean_mk_empty_byte_array(lean_box(0));
        }
      }

      if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
      const float v = stack[0];
      acc = (v >= acc) ? v : acc;
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] += inStrides[i * rank + (rank - 1)];
      }
    }
    dst[o] = acc;

    for (size_t d = rank - 1; d-- > 0;) {
      idx[d]++;
      for (size_t i = 0; i < nInputs; i++) {
        inOffBase[i] += inStrides[i * rank + d];
      }
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
      for (size_t i = 0; i < nInputs; i++) {
        inOffBase[i] -= inStrides[i * rank + d] * dims[d];
      }
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_last_view_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg inputStridesObj,
    b_lean_obj_arg inputOffsetsObj,
    b_lean_obj_arg inputMaskStartsObj,
    b_lean_obj_arg inputMaskEndsObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg fullShapeObj,
    b_lean_obj_arg progObj) {
  const size_t rank = lean_array_size(fullShapeObj);
  if (rank == 0) return lean_mk_empty_byte_array(lean_box(0));

  size_t *dims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(fullShapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(inputStridesObj) != nInputs || lean_array_size(inputOffsetsObj) != nInputs ||
      lean_array_size(inputMaskStartsObj) != nInputs || lean_array_size(inputMaskEndsObj) != nInputs ||
      lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  const size_t inner = dims[rank - 1];
  size_t outer = 1;
  for (size_t i = 0; i + 1 < rank; i++) outer *= dims[i];

  const size_t outBytes = outer * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outer == 0) return out;

  union {
    uint32_t u;
    float f;
  } conv;
  conv.u = 0xFF800000u; // -inf
  const float negInf = conv.f;

  if (inner == 0) {
    for (size_t o = 0; o < outer; o++) dst[o] = negInf;
    return out;
  }

  int64_t *inStrides = LEAN_ALLOCA(sizeof(int64_t) * nInputs * rank); // NOLINT
  int64_t *inOffBase = LEAN_ALLOCA(sizeof(int64_t) * nInputs); // NOLINT
  int64_t *inOff = LEAN_ALLOCA(sizeof(int64_t) * nInputs); // NOLINT
  size_t *maskStart = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *maskEnd = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  size_t *inCap = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT
    const size_t byteSize = lean_sarray_size(bytesObj);
    inCap[i] = (dt == 1) ? byteSize : (byteSize / 4);

    b_lean_obj_arg stridesObj = lean_array_get_core(inputStridesObj, i);
    if (!read_i64_array_len(stridesObj, rank, inStrides + i * rank)) return lean_mk_empty_byte_array(lean_box(0));

    b_lean_obj_arg offObj = lean_array_get_core(inputOffsetsObj, i);
    if (!lean_is_ctor(offObj)) return lean_mk_empty_byte_array(lean_box(0));
    inOffBase[i] = (int64_t)lean_unbox_uint64(offObj);

    b_lean_obj_arg msObj = lean_array_get_core(inputMaskStartsObj, i);
    b_lean_obj_arg meObj = lean_array_get_core(inputMaskEndsObj, i);
    if (!read_nat_array_len(msObj, rank, maskStart + i * rank)) return lean_mk_empty_byte_array(lean_box(0));
    if (!read_nat_array_len(meObj, rank, maskEnd + i * rank)) return lean_mk_empty_byte_array(lean_box(0));
    for (size_t d = 0; d < rank; d++) {
      const size_t ms = maskStart[i * rank + d];
      const size_t me = maskEnd[i * rank + d];
      if (ms > me || me > dims[d]) return lean_mk_empty_byte_array(lean_box(0));
    }
  }

  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  uint8_t *validOuter = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  uint8_t *valid = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT

  const size_t redAxis = rank - 1;

  for (size_t o = 0; o < outer; o++) {
    for (size_t i = 0; i < nInputs; i++) inOff[i] = inOffBase[i];

    for (size_t i = 0; i < nInputs; i++) {
      uint8_t ok = 1;
      for (size_t d = 0; d + 1 < rank; d++) {
        const size_t x = idx[d];
        const size_t ms = maskStart[i * rank + d];
        const size_t me = maskEnd[i * rank + d];
        if (x < ms || x >= me) {
          ok = 0;
          break;
        }
      }
      validOuter[i] = ok;
    }

    float acc = negInf;
    for (size_t r = 0; r < inner; r++) {
      for (size_t i = 0; i < nInputs; i++) {
        if (!validOuter[i]) {
          valid[i] = 0;
          continue;
        }
        const size_t ms = maskStart[i * rank + redAxis];
        const size_t me = maskEnd[i * rank + redAxis];
        valid[i] = (r >= ms && r < me) ? 1 : 0;
      }

      size_t sp = 0;
      for (size_t ip = 0; ip < instsLen; ip++) {
        const uint64_t inst = prog[ip];

        const uint8_t kind = (uint8_t)(inst & 0xFFu);
        const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
        const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
        const uint32_t imm = (uint32_t)(inst >> 32);

        switch (kind) {
          case 0:
            stack[sp++] = f32_from_bits(imm);
            break;
          case 1: {
            if (arg >= nInputs || !valid[arg]) {
              stack[sp++] = 0.0f;
              break;
            }
            const int64_t off = inOff[arg];
            if (off < 0 || (uint64_t)off >= inCap[arg]) {
              stack[sp++] = 0.0f;
              break;
            }
            if (inDtypes[arg] == 1) {
              const uint8_t v = inPtrs[arg][(size_t)off];
              stack[sp++] = v ? 1.0f : 0.0f;
            } else {
              const float *p = (const float *)inPtrs[arg]; // NOLINT
              stack[sp++] = p[(size_t)off];
            }
            break;
          }
          case 2: {
            if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
            const float x = stack[sp - 1];
            switch (op) {
              case 0: stack[sp - 1] = -x; break;
              case 1: stack[sp - 1] = sqrtf(x); break;
              case 2: stack[sp - 1] = 1.0f / x; break;
              case 3: stack[sp - 1] = exp2f(x); break;
              case 4: stack[sp - 1] = log2f(x); break;
              case 5: stack[sp - 1] = sinf(x); break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 3: {
            if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            switch (op) {
              case 0: stack[sp++] = x + y; break;
              case 1: stack[sp++] = x - y; break;
              case 2: stack[sp++] = x * y; break;
              case 3: stack[sp++] = x / y; break;
              case 4: stack[sp++] = (x >= y) ? x : y; break;
              case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 4: {
            if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            const float c = stack[--sp];
            stack[sp++] = (c != 0.0f) ? x : y;
            break;
          }
          default:
            return lean_mk_empty_byte_array(lean_box(0));
        }
      }

      if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
      const float v = stack[0];
      acc = (v >= acc) ? v : acc;
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] += inStrides[i * rank + redAxis];
      }
    }
    dst[o] = acc;

    for (size_t d = rank - 1; d-- > 0;) {
      idx[d]++;
      for (size_t i = 0; i < nInputs; i++) {
        inOffBase[i] += inStrides[i * rank + d];
      }
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
      for (size_t i = 0; i < nInputs; i++) {
        inOffBase[i] -= inStrides[i * rank + d] * (int64_t)dims[d];
      }
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_last_view_stack_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg stackShapesObj,
    b_lean_obj_arg stackStridesObj,
    b_lean_obj_arg stackOffsetsObj,
    b_lean_obj_arg stackMaskStartsObj,
    b_lean_obj_arg stackMaskEndsObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg fullShapeObj,
    b_lean_obj_arg progObj) {
  const size_t rank = lean_array_size(fullShapeObj);
  if (rank == 0) return lean_mk_empty_byte_array(lean_box(0));

  size_t *dims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(fullShapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  const size_t inner = dims[rank - 1];
  size_t outer = 1;
  for (size_t i = 0; i + 1 < rank; i++) outer *= dims[i];

  const size_t outBytes = outer * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outer == 0) return out;

  union {
    uint32_t u;
    float f;
  } conv;
  conv.u = 0xFF800000u; // -inf
  const float negInf = conv.f;

  if (inner == 0) {
    for (size_t o = 0; o < outer; o++) dst[o] = negInf;
    return out;
  }

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(stackShapesObj) != nInputs || lean_array_size(stackStridesObj) != nInputs || lean_array_size(stackOffsetsObj) != nInputs ||
      lean_array_size(stackMaskStartsObj) != nInputs || lean_array_size(stackMaskEndsObj) != nInputs || lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  size_t *inCap = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  tg4_view_stack *stacks = LEAN_ALLOCA(sizeof(tg4_view_stack) * nInputs); // NOLINT
  size_t maxRank = rank;

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT
    const size_t byteSize = lean_sarray_size(bytesObj);
    inCap[i] = (dt == 1) ? byteSize : (byteSize / 4);

    b_lean_obj_arg shapesViewsObj = lean_array_get_core(stackShapesObj, i);
    b_lean_obj_arg stridesViewsObj = lean_array_get_core(stackStridesObj, i);
    b_lean_obj_arg offsetsViewsObj = lean_array_get_core(stackOffsetsObj, i);
    b_lean_obj_arg msViewsObj = lean_array_get_core(stackMaskStartsObj, i);
    b_lean_obj_arg meViewsObj = lean_array_get_core(stackMaskEndsObj, i);

    const size_t nViews = lean_array_size(shapesViewsObj);
    if (nViews == 0 || lean_array_size(stridesViewsObj) != nViews || lean_array_size(offsetsViewsObj) != nViews ||
        lean_array_size(msViewsObj) != nViews || lean_array_size(meViewsObj) != nViews) {
      return lean_mk_empty_byte_array(lean_box(0));
    }

    tg4_view_stack_view *views = LEAN_ALLOCA(sizeof(tg4_view_stack_view) * nViews); // NOLINT
    for (size_t vi = 0; vi < nViews; vi++) {
      b_lean_obj_arg shapeObj = lean_array_get_core(shapesViewsObj, vi);
      const size_t vRank = lean_array_size(shapeObj);

      size_t *vDims = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      if (!read_shape_dims(shapeObj, vDims)) return lean_mk_empty_byte_array(lean_box(0));

      size_t *contig = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      contiguous_strides(vDims, vRank, contig);

      const size_t vNumel = prod_dims(vDims, vRank);

      b_lean_obj_arg stridesObj = lean_array_get_core(stridesViewsObj, vi);
      int64_t *strides = LEAN_ALLOCA(sizeof(int64_t) * vRank); // NOLINT
      if (!read_i64_array_len(stridesObj, vRank, strides)) return lean_mk_empty_byte_array(lean_box(0));

      b_lean_obj_arg offObj = lean_array_get_core(offsetsViewsObj, vi);
      if (!lean_is_ctor(offObj)) return lean_mk_empty_byte_array(lean_box(0));
      const int64_t offset = (int64_t)lean_unbox_uint64(offObj);

      b_lean_obj_arg msObj = lean_array_get_core(msViewsObj, vi);
      b_lean_obj_arg meObj = lean_array_get_core(meViewsObj, vi);
      size_t *maskStart = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      size_t *maskEnd = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      if (!read_nat_array_len(msObj, vRank, maskStart)) return lean_mk_empty_byte_array(lean_box(0));
      if (!read_nat_array_len(meObj, vRank, maskEnd)) return lean_mk_empty_byte_array(lean_box(0));
      for (size_t d = 0; d < vRank; d++) {
        const size_t ms = maskStart[d];
        const size_t me = maskEnd[d];
        if (ms > me || me > vDims[d]) return lean_mk_empty_byte_array(lean_box(0));
      }

      views[vi].rank = vRank;
      views[vi].numel = vNumel;
      views[vi].contigStrides = contig;
      views[vi].strides = strides;
      views[vi].offset = offset;
      views[vi].maskStart = maskStart;
      views[vi].maskEnd = maskEnd;

      if (vRank > maxRank) maxRank = vRank;
    }

    stacks[i].nViews = nViews;
    stacks[i].views = views;

    if (views[nViews - 1].rank != rank) return lean_mk_empty_byte_array(lean_box(0));
  }

  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  size_t *idxWork = LEAN_ALLOCA(sizeof(size_t) * maxRank); // NOLINT
  uint8_t *valid = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  int64_t *inOff = LEAN_ALLOCA(sizeof(int64_t) * nInputs); // NOLINT
  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT

  for (size_t o = 0; o < outer; o++) {
    float acc = negInf;
    for (size_t r = 0; r < inner; r++) {
      idx[rank - 1] = r;

      for (size_t i = 0; i < nInputs; i++) {
        int64_t off = 0;
        const bool ok = tg4_view_stack_offset(&stacks[i], idx, rank, idxWork, &off);
        valid[i] = ok ? 1 : 0;
        inOff[i] = off;
      }

      size_t sp = 0;
      for (size_t ip = 0; ip < instsLen; ip++) {
        const uint64_t inst = prog[ip];

        const uint8_t kind = (uint8_t)(inst & 0xFFu);
        const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
        const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
        const uint32_t imm = (uint32_t)(inst >> 32);

        switch (kind) {
          case 0:
            stack[sp++] = f32_from_bits(imm);
            break;
          case 1: {
            if (arg >= nInputs || !valid[arg]) {
              stack[sp++] = 0.0f;
              break;
            }
            const int64_t off = inOff[arg];
            if (off < 0 || (uint64_t)off >= inCap[arg]) {
              stack[sp++] = 0.0f;
              break;
            }
            if (inDtypes[arg] == 1) {
              const uint8_t v = inPtrs[arg][(size_t)off];
              stack[sp++] = v ? 1.0f : 0.0f;
            } else {
              const float *p = (const float *)inPtrs[arg]; // NOLINT
              stack[sp++] = p[(size_t)off];
            }
            break;
          }
          case 2: {
            if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
            const float x = stack[sp - 1];
            switch (op) {
              case 0: stack[sp - 1] = -x; break;
              case 1: stack[sp - 1] = sqrtf(x); break;
              case 2: stack[sp - 1] = 1.0f / x; break;
              case 3: stack[sp - 1] = exp2f(x); break;
              case 4: stack[sp - 1] = log2f(x); break;
              case 5: stack[sp - 1] = sinf(x); break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 3: {
            if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            switch (op) {
              case 0: stack[sp++] = x + y; break;
              case 1: stack[sp++] = x - y; break;
              case 2: stack[sp++] = x * y; break;
              case 3: stack[sp++] = x / y; break;
              case 4: stack[sp++] = (x >= y) ? x : y; break;
              case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 4: {
            if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            const float c = stack[--sp];
            stack[sp++] = (c != 0.0f) ? x : y;
            break;
          }
          default:
            return lean_mk_empty_byte_array(lean_box(0));
        }
      }

      if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
      const float v = stack[0];
      acc = (v >= acc) ? v : acc;
    }

    dst[o] = acc;
    idx[rank - 1] = 0;

    for (size_t d = rank - 1; d-- > 0;) {
      idx[d]++;
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_axis_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg inputShapesObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg fullShapeObj,
    b_lean_obj_arg axisObj,
    b_lean_obj_arg progObj) {
  if (!lean_is_scalar(axisObj)) return lean_mk_empty_byte_array(lean_box(0));
  const size_t axis = lean_unbox(axisObj);

  const size_t rank = lean_array_size(fullShapeObj);
  if (rank == 0 || axis >= rank) return lean_mk_empty_byte_array(lean_box(0));

  size_t *dims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(fullShapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t reduce = dims[axis];

  size_t outNumel = 1;
  for (size_t d = 0; d < rank; d++) {
    if (d != axis) outNumel *= dims[d];
  }

  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;
  if (reduce == 0) {
    for (size_t i = 0; i < outNumel; i++) dst[i] = 0.0f;
    return out;
  }

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(inputShapesObj) != nInputs || lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  size_t *inDims = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *inStrides = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *inOffBase = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  size_t *inOff = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    b_lean_obj_arg shapeObj = lean_array_get_core(inputShapesObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT

    size_t *idims = inDims + i * rank;
    size_t *istrides = inStrides + i * rank;
    if (!read_padded_dims(shapeObj, rank, idims)) return lean_mk_empty_byte_array(lean_box(0));
    if (!is_broadcastable(idims, dims, rank)) return lean_mk_empty_byte_array(lean_box(0));

    const size_t numel = prod_dims(idims, rank);
    const size_t needed = (dt == 1) ? numel : (numel * 4);
    if (lean_sarray_size(bytesObj) < needed) return lean_mk_empty_byte_array(lean_box(0));

    contiguous_strides(idims, rank, istrides);
    broadcast_strides(idims, rank, istrides);
    inOffBase[i] = 0;
  }

  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT

  for (size_t o = 0; o < outNumel; o++) {
    for (size_t i = 0; i < nInputs; i++) inOff[i] = inOffBase[i];

    float acc = 0.0f;
    for (size_t r = 0; r < reduce; r++) {
      size_t sp = 0;
      for (size_t ip = 0; ip < instsLen; ip++) {
        const uint64_t inst = prog[ip];

        const uint8_t kind = (uint8_t)(inst & 0xFFu);
        const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
        const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
        const uint32_t imm = (uint32_t)(inst >> 32);

        switch (kind) {
          case 0:
            stack[sp++] = f32_from_bits(imm);
            break;
          case 1: {
            if (arg >= nInputs) {
              stack[sp++] = 0.0f;
              break;
            }
            const size_t off = inOff[arg];
            if (inDtypes[arg] == 1) {
              const uint8_t v = inPtrs[arg][off];
              stack[sp++] = v ? 1.0f : 0.0f;
            } else {
              const float *p = (const float *)inPtrs[arg]; // NOLINT
              stack[sp++] = p[off];
            }
            break;
          }
          case 2: {
            if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
            const float x = stack[sp - 1];
            switch (op) {
              case 0: stack[sp - 1] = -x; break;
              case 1: stack[sp - 1] = sqrtf(x); break;
              case 2: stack[sp - 1] = 1.0f / x; break;
              case 3: stack[sp - 1] = exp2f(x); break;
              case 4: stack[sp - 1] = log2f(x); break;
              case 5: stack[sp - 1] = sinf(x); break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 3: {
            if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            switch (op) {
              case 0: stack[sp++] = x + y; break;
              case 1: stack[sp++] = x - y; break;
              case 2: stack[sp++] = x * y; break;
              case 3: stack[sp++] = x / y; break;
              case 4: stack[sp++] = (x >= y) ? x : y; break;
              case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 4: {
            if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            const float c = stack[--sp];
            stack[sp++] = (c != 0.0f) ? x : y;
            break;
          }
          default:
            return lean_mk_empty_byte_array(lean_box(0));
        }
      }

      if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
      acc += stack[0];
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] += inStrides[i * rank + axis];
      }
    }
    dst[o] = acc;

    for (size_t d = rank; d-- > 0;) {
      if (d == axis) continue;
      idx[d]++;
      for (size_t i = 0; i < nInputs; i++) {
        inOffBase[i] += inStrides[i * rank + d];
      }
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
      for (size_t i = 0; i < nInputs; i++) {
        inOffBase[i] -= inStrides[i * rank + d] * dims[d];
      }
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_axis_view_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg inputStridesObj,
    b_lean_obj_arg inputOffsetsObj,
    b_lean_obj_arg inputMaskStartsObj,
    b_lean_obj_arg inputMaskEndsObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg fullShapeObj,
    b_lean_obj_arg axisObj,
    b_lean_obj_arg progObj) {
  if (!lean_is_scalar(axisObj)) return lean_mk_empty_byte_array(lean_box(0));
  const size_t axis = lean_unbox(axisObj);

  const size_t rank = lean_array_size(fullShapeObj);
  if (rank == 0 || axis >= rank) return lean_mk_empty_byte_array(lean_box(0));

  size_t *dims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(fullShapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t reduce = dims[axis];

  size_t outNumel = 1;
  for (size_t d = 0; d < rank; d++) {
    if (d != axis) outNumel *= dims[d];
  }

  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;
  if (reduce == 0) {
    for (size_t i = 0; i < outNumel; i++) dst[i] = 0.0f;
    return out;
  }

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(inputStridesObj) != nInputs || lean_array_size(inputOffsetsObj) != nInputs ||
      lean_array_size(inputMaskStartsObj) != nInputs || lean_array_size(inputMaskEndsObj) != nInputs ||
      lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  int64_t *inStrides = LEAN_ALLOCA(sizeof(int64_t) * nInputs * rank); // NOLINT
  int64_t *inOffBase = LEAN_ALLOCA(sizeof(int64_t) * nInputs); // NOLINT
  int64_t *inOff = LEAN_ALLOCA(sizeof(int64_t) * nInputs); // NOLINT
  size_t *maskStart = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *maskEnd = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  size_t *inCap = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT
    const size_t byteSize = lean_sarray_size(bytesObj);
    inCap[i] = (dt == 1) ? byteSize : (byteSize / 4);

    b_lean_obj_arg stridesObj = lean_array_get_core(inputStridesObj, i);
    if (!read_i64_array_len(stridesObj, rank, inStrides + i * rank)) return lean_mk_empty_byte_array(lean_box(0));

    b_lean_obj_arg offObj = lean_array_get_core(inputOffsetsObj, i);
    if (!lean_is_ctor(offObj)) return lean_mk_empty_byte_array(lean_box(0));
    inOffBase[i] = (int64_t)lean_unbox_uint64(offObj);

    b_lean_obj_arg msObj = lean_array_get_core(inputMaskStartsObj, i);
    b_lean_obj_arg meObj = lean_array_get_core(inputMaskEndsObj, i);
    if (!read_nat_array_len(msObj, rank, maskStart + i * rank)) return lean_mk_empty_byte_array(lean_box(0));
    if (!read_nat_array_len(meObj, rank, maskEnd + i * rank)) return lean_mk_empty_byte_array(lean_box(0));
    for (size_t d = 0; d < rank; d++) {
      const size_t ms = maskStart[i * rank + d];
      const size_t me = maskEnd[i * rank + d];
      if (ms > me || me > dims[d]) return lean_mk_empty_byte_array(lean_box(0));
    }
  }

  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  uint8_t *validOuter = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  uint8_t *valid = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT

  for (size_t o = 0; o < outNumel; o++) {
    for (size_t i = 0; i < nInputs; i++) inOff[i] = inOffBase[i];

    for (size_t i = 0; i < nInputs; i++) {
      uint8_t ok = 1;
      for (size_t d = 0; d < rank; d++) {
        if (d == axis) continue;
        const size_t x = idx[d];
        const size_t ms = maskStart[i * rank + d];
        const size_t me = maskEnd[i * rank + d];
        if (x < ms || x >= me) {
          ok = 0;
          break;
        }
      }
      validOuter[i] = ok;
    }

    float acc = 0.0f;
    for (size_t r = 0; r < reduce; r++) {
      for (size_t i = 0; i < nInputs; i++) {
        if (!validOuter[i]) {
          valid[i] = 0;
          continue;
        }
        const size_t ms = maskStart[i * rank + axis];
        const size_t me = maskEnd[i * rank + axis];
        valid[i] = (r >= ms && r < me) ? 1 : 0;
      }

      size_t sp = 0;
      for (size_t ip = 0; ip < instsLen; ip++) {
        const uint64_t inst = prog[ip];

        const uint8_t kind = (uint8_t)(inst & 0xFFu);
        const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
        const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
        const uint32_t imm = (uint32_t)(inst >> 32);

        switch (kind) {
          case 0:
            stack[sp++] = f32_from_bits(imm);
            break;
          case 1: {
            if (arg >= nInputs || !valid[arg]) {
              stack[sp++] = 0.0f;
              break;
            }
            const int64_t off = inOff[arg];
            if (off < 0 || (uint64_t)off >= inCap[arg]) {
              stack[sp++] = 0.0f;
              break;
            }
            if (inDtypes[arg] == 1) {
              const uint8_t v = inPtrs[arg][(size_t)off];
              stack[sp++] = v ? 1.0f : 0.0f;
            } else {
              const float *p = (const float *)inPtrs[arg]; // NOLINT
              stack[sp++] = p[(size_t)off];
            }
            break;
          }
          case 2: {
            if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
            const float x = stack[sp - 1];
            switch (op) {
              case 0: stack[sp - 1] = -x; break;
              case 1: stack[sp - 1] = sqrtf(x); break;
              case 2: stack[sp - 1] = 1.0f / x; break;
              case 3: stack[sp - 1] = exp2f(x); break;
              case 4: stack[sp - 1] = log2f(x); break;
              case 5: stack[sp - 1] = sinf(x); break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 3: {
            if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            switch (op) {
              case 0: stack[sp++] = x + y; break;
              case 1: stack[sp++] = x - y; break;
              case 2: stack[sp++] = x * y; break;
              case 3: stack[sp++] = x / y; break;
              case 4: stack[sp++] = (x >= y) ? x : y; break;
              case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 4: {
            if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            const float c = stack[--sp];
            stack[sp++] = (c != 0.0f) ? x : y;
            break;
          }
          default:
            return lean_mk_empty_byte_array(lean_box(0));
        }
      }

      if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
      acc += stack[0];
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] += inStrides[i * rank + axis];
      }
    }
    dst[o] = acc;

    for (size_t d = rank; d-- > 0;) {
      if (d == axis) continue;
      idx[d]++;
      for (size_t i = 0; i < nInputs; i++) {
        inOffBase[i] += inStrides[i * rank + d];
      }
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
      for (size_t i = 0; i < nInputs; i++) {
        inOffBase[i] -= inStrides[i * rank + d] * (int64_t)dims[d];
      }
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_axis_view_stack_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg stackShapesObj,
    b_lean_obj_arg stackStridesObj,
    b_lean_obj_arg stackOffsetsObj,
    b_lean_obj_arg stackMaskStartsObj,
    b_lean_obj_arg stackMaskEndsObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg fullShapeObj,
    b_lean_obj_arg axisObj,
    b_lean_obj_arg progObj) {
  if (!lean_is_scalar(axisObj)) return lean_mk_empty_byte_array(lean_box(0));
  const size_t axis = lean_unbox(axisObj);

  const size_t rank = lean_array_size(fullShapeObj);
  if (rank == 0 || axis >= rank) return lean_mk_empty_byte_array(lean_box(0));

  size_t *dims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(fullShapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t reduce = dims[axis];
  size_t outNumel = 1;
  for (size_t d = 0; d < rank; d++) {
    if (d != axis) outNumel *= dims[d];
  }

  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;
  if (reduce == 0) {
    for (size_t i = 0; i < outNumel; i++) dst[i] = 0.0f;
    return out;
  }

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(stackShapesObj) != nInputs || lean_array_size(stackStridesObj) != nInputs || lean_array_size(stackOffsetsObj) != nInputs ||
      lean_array_size(stackMaskStartsObj) != nInputs || lean_array_size(stackMaskEndsObj) != nInputs || lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  size_t *inCap = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  tg4_view_stack *stacks = LEAN_ALLOCA(sizeof(tg4_view_stack) * nInputs); // NOLINT
  size_t maxRank = rank;

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT
    const size_t byteSize = lean_sarray_size(bytesObj);
    inCap[i] = (dt == 1) ? byteSize : (byteSize / 4);

    b_lean_obj_arg shapesViewsObj = lean_array_get_core(stackShapesObj, i);
    b_lean_obj_arg stridesViewsObj = lean_array_get_core(stackStridesObj, i);
    b_lean_obj_arg offsetsViewsObj = lean_array_get_core(stackOffsetsObj, i);
    b_lean_obj_arg msViewsObj = lean_array_get_core(stackMaskStartsObj, i);
    b_lean_obj_arg meViewsObj = lean_array_get_core(stackMaskEndsObj, i);

    const size_t nViews = lean_array_size(shapesViewsObj);
    if (nViews == 0 || lean_array_size(stridesViewsObj) != nViews || lean_array_size(offsetsViewsObj) != nViews ||
        lean_array_size(msViewsObj) != nViews || lean_array_size(meViewsObj) != nViews) {
      return lean_mk_empty_byte_array(lean_box(0));
    }

    tg4_view_stack_view *views = LEAN_ALLOCA(sizeof(tg4_view_stack_view) * nViews); // NOLINT
    for (size_t vi = 0; vi < nViews; vi++) {
      b_lean_obj_arg shapeObj = lean_array_get_core(shapesViewsObj, vi);
      const size_t vRank = lean_array_size(shapeObj);

      size_t *vDims = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      if (!read_shape_dims(shapeObj, vDims)) return lean_mk_empty_byte_array(lean_box(0));

      size_t *contig = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      contiguous_strides(vDims, vRank, contig);

      const size_t vNumel = prod_dims(vDims, vRank);

      b_lean_obj_arg stridesObj = lean_array_get_core(stridesViewsObj, vi);
      int64_t *strides = LEAN_ALLOCA(sizeof(int64_t) * vRank); // NOLINT
      if (!read_i64_array_len(stridesObj, vRank, strides)) return lean_mk_empty_byte_array(lean_box(0));

      b_lean_obj_arg offObj = lean_array_get_core(offsetsViewsObj, vi);
      if (!lean_is_ctor(offObj)) return lean_mk_empty_byte_array(lean_box(0));
      const int64_t offset = (int64_t)lean_unbox_uint64(offObj);

      b_lean_obj_arg msObj = lean_array_get_core(msViewsObj, vi);
      b_lean_obj_arg meObj = lean_array_get_core(meViewsObj, vi);
      size_t *maskStart = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      size_t *maskEnd = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      if (!read_nat_array_len(msObj, vRank, maskStart)) return lean_mk_empty_byte_array(lean_box(0));
      if (!read_nat_array_len(meObj, vRank, maskEnd)) return lean_mk_empty_byte_array(lean_box(0));
      for (size_t d = 0; d < vRank; d++) {
        const size_t ms = maskStart[d];
        const size_t me = maskEnd[d];
        if (ms > me || me > vDims[d]) return lean_mk_empty_byte_array(lean_box(0));
      }

      views[vi].rank = vRank;
      views[vi].numel = vNumel;
      views[vi].contigStrides = contig;
      views[vi].strides = strides;
      views[vi].offset = offset;
      views[vi].maskStart = maskStart;
      views[vi].maskEnd = maskEnd;

      if (vRank > maxRank) maxRank = vRank;
    }

    stacks[i].nViews = nViews;
    stacks[i].views = views;

    if (views[nViews - 1].rank != rank) return lean_mk_empty_byte_array(lean_box(0));
  }

  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  size_t *idxWork = LEAN_ALLOCA(sizeof(size_t) * maxRank); // NOLINT
  uint8_t *valid = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  int64_t *inOff = LEAN_ALLOCA(sizeof(int64_t) * nInputs); // NOLINT
  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT

  for (size_t o = 0; o < outNumel; o++) {
    float acc = 0.0f;
    for (size_t r = 0; r < reduce; r++) {
      idx[axis] = r;

      for (size_t i = 0; i < nInputs; i++) {
        int64_t off = 0;
        const bool ok = tg4_view_stack_offset(&stacks[i], idx, rank, idxWork, &off);
        valid[i] = ok ? 1 : 0;
        inOff[i] = off;
      }

      size_t sp = 0;
      for (size_t ip = 0; ip < instsLen; ip++) {
        const uint64_t inst = prog[ip];

        const uint8_t kind = (uint8_t)(inst & 0xFFu);
        const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
        const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
        const uint32_t imm = (uint32_t)(inst >> 32);

        switch (kind) {
          case 0:
            stack[sp++] = f32_from_bits(imm);
            break;
          case 1: {
            if (arg >= nInputs || !valid[arg]) {
              stack[sp++] = 0.0f;
              break;
            }
            const int64_t off = inOff[arg];
            if (off < 0 || (uint64_t)off >= inCap[arg]) {
              stack[sp++] = 0.0f;
              break;
            }
            if (inDtypes[arg] == 1) {
              const uint8_t v = inPtrs[arg][(size_t)off];
              stack[sp++] = v ? 1.0f : 0.0f;
            } else {
              const float *p = (const float *)inPtrs[arg]; // NOLINT
              stack[sp++] = p[(size_t)off];
            }
            break;
          }
          case 2: {
            if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
            const float x = stack[sp - 1];
            switch (op) {
              case 0: stack[sp - 1] = -x; break;
              case 1: stack[sp - 1] = sqrtf(x); break;
              case 2: stack[sp - 1] = 1.0f / x; break;
              case 3: stack[sp - 1] = exp2f(x); break;
              case 4: stack[sp - 1] = log2f(x); break;
              case 5: stack[sp - 1] = sinf(x); break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 3: {
            if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            switch (op) {
              case 0: stack[sp++] = x + y; break;
              case 1: stack[sp++] = x - y; break;
              case 2: stack[sp++] = x * y; break;
              case 3: stack[sp++] = x / y; break;
              case 4: stack[sp++] = (x >= y) ? x : y; break;
              case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 4: {
            if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            const float c = stack[--sp];
            stack[sp++] = (c != 0.0f) ? x : y;
            break;
          }
          default:
            return lean_mk_empty_byte_array(lean_box(0));
        }
      }

      if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
      acc += stack[0];
    }

    dst[o] = acc;
    idx[axis] = 0;

    for (size_t d = rank; d-- > 0;) {
      if (d == axis) continue;
      idx[d]++;
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_axis_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg inputShapesObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg fullShapeObj,
    b_lean_obj_arg axisObj,
    b_lean_obj_arg progObj) {
  if (!lean_is_scalar(axisObj)) return lean_mk_empty_byte_array(lean_box(0));
  const size_t axis = lean_unbox(axisObj);

  const size_t rank = lean_array_size(fullShapeObj);
  if (rank == 0 || axis >= rank) return lean_mk_empty_byte_array(lean_box(0));

  size_t *dims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(fullShapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t reduce = dims[axis];

  size_t outNumel = 1;
  for (size_t d = 0; d < rank; d++) {
    if (d != axis) outNumel *= dims[d];
  }

  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  union {
    uint32_t u;
    float f;
  } conv;
  conv.u = 0xFF800000u; // -inf
  const float negInf = conv.f;

  if (reduce == 0) {
    for (size_t i = 0; i < outNumel; i++) dst[i] = negInf;
    return out;
  }

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(inputShapesObj) != nInputs || lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  size_t *inDims = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *inStrides = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *inOffBase = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  size_t *inOff = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    b_lean_obj_arg shapeObj = lean_array_get_core(inputShapesObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT

    size_t *idims = inDims + i * rank;
    size_t *istrides = inStrides + i * rank;
    if (!read_padded_dims(shapeObj, rank, idims)) return lean_mk_empty_byte_array(lean_box(0));
    if (!is_broadcastable(idims, dims, rank)) return lean_mk_empty_byte_array(lean_box(0));

    const size_t numel = prod_dims(idims, rank);
    const size_t needed = (dt == 1) ? numel : (numel * 4);
    if (lean_sarray_size(bytesObj) < needed) return lean_mk_empty_byte_array(lean_box(0));

    contiguous_strides(idims, rank, istrides);
    broadcast_strides(idims, rank, istrides);
    inOffBase[i] = 0;
  }

  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT

  for (size_t o = 0; o < outNumel; o++) {
    for (size_t i = 0; i < nInputs; i++) inOff[i] = inOffBase[i];

    float acc = negInf;
    for (size_t r = 0; r < reduce; r++) {
      size_t sp = 0;
      for (size_t ip = 0; ip < instsLen; ip++) {
        const uint64_t inst = prog[ip];

        const uint8_t kind = (uint8_t)(inst & 0xFFu);
        const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
        const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
        const uint32_t imm = (uint32_t)(inst >> 32);

        switch (kind) {
          case 0:
            stack[sp++] = f32_from_bits(imm);
            break;
          case 1: {
            if (arg >= nInputs) {
              stack[sp++] = 0.0f;
              break;
            }
            const size_t off = inOff[arg];
            if (inDtypes[arg] == 1) {
              const uint8_t v = inPtrs[arg][off];
              stack[sp++] = v ? 1.0f : 0.0f;
            } else {
              const float *p = (const float *)inPtrs[arg]; // NOLINT
              stack[sp++] = p[off];
            }
            break;
          }
          case 2: {
            if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
            const float x = stack[sp - 1];
            switch (op) {
              case 0: stack[sp - 1] = -x; break;
              case 1: stack[sp - 1] = sqrtf(x); break;
              case 2: stack[sp - 1] = 1.0f / x; break;
              case 3: stack[sp - 1] = exp2f(x); break;
              case 4: stack[sp - 1] = log2f(x); break;
              case 5: stack[sp - 1] = sinf(x); break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 3: {
            if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            switch (op) {
              case 0: stack[sp++] = x + y; break;
              case 1: stack[sp++] = x - y; break;
              case 2: stack[sp++] = x * y; break;
              case 3: stack[sp++] = x / y; break;
              case 4: stack[sp++] = (x >= y) ? x : y; break;
              case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 4: {
            if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            const float c = stack[--sp];
            stack[sp++] = (c != 0.0f) ? x : y;
            break;
          }
          default:
            return lean_mk_empty_byte_array(lean_box(0));
        }
      }

      if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
      const float v = stack[0];
      acc = (v >= acc) ? v : acc;
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] += inStrides[i * rank + axis];
      }
    }
    dst[o] = acc;

    for (size_t d = rank; d-- > 0;) {
      if (d == axis) continue;
      idx[d]++;
      for (size_t i = 0; i < nInputs; i++) {
        inOffBase[i] += inStrides[i * rank + d];
      }
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
      for (size_t i = 0; i < nInputs; i++) {
        inOffBase[i] -= inStrides[i * rank + d] * dims[d];
      }
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_axis_view_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg inputStridesObj,
    b_lean_obj_arg inputOffsetsObj,
    b_lean_obj_arg inputMaskStartsObj,
    b_lean_obj_arg inputMaskEndsObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg fullShapeObj,
    b_lean_obj_arg axisObj,
    b_lean_obj_arg progObj) {
  if (!lean_is_scalar(axisObj)) return lean_mk_empty_byte_array(lean_box(0));
  const size_t axis = lean_unbox(axisObj);

  const size_t rank = lean_array_size(fullShapeObj);
  if (rank == 0 || axis >= rank) return lean_mk_empty_byte_array(lean_box(0));

  size_t *dims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(fullShapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t reduce = dims[axis];

  size_t outNumel = 1;
  for (size_t d = 0; d < rank; d++) {
    if (d != axis) outNumel *= dims[d];
  }

  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  union {
    uint32_t u;
    float f;
  } conv;
  conv.u = 0xFF800000u; // -inf
  const float negInf = conv.f;

  if (reduce == 0) {
    for (size_t i = 0; i < outNumel; i++) dst[i] = negInf;
    return out;
  }

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(inputStridesObj) != nInputs || lean_array_size(inputOffsetsObj) != nInputs ||
      lean_array_size(inputMaskStartsObj) != nInputs || lean_array_size(inputMaskEndsObj) != nInputs ||
      lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  int64_t *inStrides = LEAN_ALLOCA(sizeof(int64_t) * nInputs * rank); // NOLINT
  int64_t *inOffBase = LEAN_ALLOCA(sizeof(int64_t) * nInputs); // NOLINT
  int64_t *inOff = LEAN_ALLOCA(sizeof(int64_t) * nInputs); // NOLINT
  size_t *maskStart = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *maskEnd = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  size_t *inCap = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT
    const size_t byteSize = lean_sarray_size(bytesObj);
    inCap[i] = (dt == 1) ? byteSize : (byteSize / 4);

    b_lean_obj_arg stridesObj = lean_array_get_core(inputStridesObj, i);
    if (!read_i64_array_len(stridesObj, rank, inStrides + i * rank)) return lean_mk_empty_byte_array(lean_box(0));

    b_lean_obj_arg offObj = lean_array_get_core(inputOffsetsObj, i);
    if (!lean_is_ctor(offObj)) return lean_mk_empty_byte_array(lean_box(0));
    inOffBase[i] = (int64_t)lean_unbox_uint64(offObj);

    b_lean_obj_arg msObj = lean_array_get_core(inputMaskStartsObj, i);
    b_lean_obj_arg meObj = lean_array_get_core(inputMaskEndsObj, i);
    if (!read_nat_array_len(msObj, rank, maskStart + i * rank)) return lean_mk_empty_byte_array(lean_box(0));
    if (!read_nat_array_len(meObj, rank, maskEnd + i * rank)) return lean_mk_empty_byte_array(lean_box(0));
    for (size_t d = 0; d < rank; d++) {
      const size_t ms = maskStart[i * rank + d];
      const size_t me = maskEnd[i * rank + d];
      if (ms > me || me > dims[d]) return lean_mk_empty_byte_array(lean_box(0));
    }
  }

  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  uint8_t *validOuter = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  uint8_t *valid = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT

  for (size_t o = 0; o < outNumel; o++) {
    for (size_t i = 0; i < nInputs; i++) inOff[i] = inOffBase[i];

    for (size_t i = 0; i < nInputs; i++) {
      uint8_t ok = 1;
      for (size_t d = 0; d < rank; d++) {
        if (d == axis) continue;
        const size_t x = idx[d];
        const size_t ms = maskStart[i * rank + d];
        const size_t me = maskEnd[i * rank + d];
        if (x < ms || x >= me) {
          ok = 0;
          break;
        }
      }
      validOuter[i] = ok;
    }

    float acc = negInf;
    for (size_t r = 0; r < reduce; r++) {
      for (size_t i = 0; i < nInputs; i++) {
        if (!validOuter[i]) {
          valid[i] = 0;
          continue;
        }
        const size_t ms = maskStart[i * rank + axis];
        const size_t me = maskEnd[i * rank + axis];
        valid[i] = (r >= ms && r < me) ? 1 : 0;
      }

      size_t sp = 0;
      for (size_t ip = 0; ip < instsLen; ip++) {
        const uint64_t inst = prog[ip];

        const uint8_t kind = (uint8_t)(inst & 0xFFu);
        const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
        const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
        const uint32_t imm = (uint32_t)(inst >> 32);

        switch (kind) {
          case 0:
            stack[sp++] = f32_from_bits(imm);
            break;
          case 1: {
            if (arg >= nInputs || !valid[arg]) {
              stack[sp++] = 0.0f;
              break;
            }
            const int64_t off = inOff[arg];
            if (off < 0 || (uint64_t)off >= inCap[arg]) {
              stack[sp++] = 0.0f;
              break;
            }
            if (inDtypes[arg] == 1) {
              const uint8_t v = inPtrs[arg][(size_t)off];
              stack[sp++] = v ? 1.0f : 0.0f;
            } else {
              const float *p = (const float *)inPtrs[arg]; // NOLINT
              stack[sp++] = p[(size_t)off];
            }
            break;
          }
          case 2: {
            if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
            const float x = stack[sp - 1];
            switch (op) {
              case 0: stack[sp - 1] = -x; break;
              case 1: stack[sp - 1] = sqrtf(x); break;
              case 2: stack[sp - 1] = 1.0f / x; break;
              case 3: stack[sp - 1] = exp2f(x); break;
              case 4: stack[sp - 1] = log2f(x); break;
              case 5: stack[sp - 1] = sinf(x); break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 3: {
            if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            switch (op) {
              case 0: stack[sp++] = x + y; break;
              case 1: stack[sp++] = x - y; break;
              case 2: stack[sp++] = x * y; break;
              case 3: stack[sp++] = x / y; break;
              case 4: stack[sp++] = (x >= y) ? x : y; break;
              case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 4: {
            if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            const float c = stack[--sp];
            stack[sp++] = (c != 0.0f) ? x : y;
            break;
          }
          default:
            return lean_mk_empty_byte_array(lean_box(0));
        }
      }

      if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
      const float v = stack[0];
      acc = (v >= acc) ? v : acc;
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] += inStrides[i * rank + axis];
      }
    }
    dst[o] = acc;

    for (size_t d = rank; d-- > 0;) {
      if (d == axis) continue;
      idx[d]++;
      for (size_t i = 0; i < nInputs; i++) {
        inOffBase[i] += inStrides[i * rank + d];
      }
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
      for (size_t i = 0; i < nInputs; i++) {
        inOffBase[i] -= inStrides[i * rank + d] * (int64_t)dims[d];
      }
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_axis_view_stack_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg stackShapesObj,
    b_lean_obj_arg stackStridesObj,
    b_lean_obj_arg stackOffsetsObj,
    b_lean_obj_arg stackMaskStartsObj,
    b_lean_obj_arg stackMaskEndsObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg fullShapeObj,
    b_lean_obj_arg axisObj,
    b_lean_obj_arg progObj) {
  if (!lean_is_scalar(axisObj)) return lean_mk_empty_byte_array(lean_box(0));
  const size_t axis = lean_unbox(axisObj);

  const size_t rank = lean_array_size(fullShapeObj);
  if (rank == 0 || axis >= rank) return lean_mk_empty_byte_array(lean_box(0));

  size_t *dims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(fullShapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t reduce = dims[axis];
  size_t outNumel = 1;
  for (size_t d = 0; d < rank; d++) {
    if (d != axis) outNumel *= dims[d];
  }

  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  union {
    uint32_t u;
    float f;
  } conv;
  conv.u = 0xFF800000u; // -inf
  const float negInf = conv.f;

  if (reduce == 0) {
    for (size_t i = 0; i < outNumel; i++) dst[i] = negInf;
    return out;
  }

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(stackShapesObj) != nInputs || lean_array_size(stackStridesObj) != nInputs || lean_array_size(stackOffsetsObj) != nInputs ||
      lean_array_size(stackMaskStartsObj) != nInputs || lean_array_size(stackMaskEndsObj) != nInputs || lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  size_t *inCap = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  tg4_view_stack *stacks = LEAN_ALLOCA(sizeof(tg4_view_stack) * nInputs); // NOLINT
  size_t maxRank = rank;

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT
    const size_t byteSize = lean_sarray_size(bytesObj);
    inCap[i] = (dt == 1) ? byteSize : (byteSize / 4);

    b_lean_obj_arg shapesViewsObj = lean_array_get_core(stackShapesObj, i);
    b_lean_obj_arg stridesViewsObj = lean_array_get_core(stackStridesObj, i);
    b_lean_obj_arg offsetsViewsObj = lean_array_get_core(stackOffsetsObj, i);
    b_lean_obj_arg msViewsObj = lean_array_get_core(stackMaskStartsObj, i);
    b_lean_obj_arg meViewsObj = lean_array_get_core(stackMaskEndsObj, i);

    const size_t nViews = lean_array_size(shapesViewsObj);
    if (nViews == 0 || lean_array_size(stridesViewsObj) != nViews || lean_array_size(offsetsViewsObj) != nViews ||
        lean_array_size(msViewsObj) != nViews || lean_array_size(meViewsObj) != nViews) {
      return lean_mk_empty_byte_array(lean_box(0));
    }

    tg4_view_stack_view *views = LEAN_ALLOCA(sizeof(tg4_view_stack_view) * nViews); // NOLINT
    for (size_t vi = 0; vi < nViews; vi++) {
      b_lean_obj_arg shapeObj = lean_array_get_core(shapesViewsObj, vi);
      const size_t vRank = lean_array_size(shapeObj);

      size_t *vDims = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      if (!read_shape_dims(shapeObj, vDims)) return lean_mk_empty_byte_array(lean_box(0));

      size_t *contig = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      contiguous_strides(vDims, vRank, contig);

      const size_t vNumel = prod_dims(vDims, vRank);

      b_lean_obj_arg stridesObj = lean_array_get_core(stridesViewsObj, vi);
      int64_t *strides = LEAN_ALLOCA(sizeof(int64_t) * vRank); // NOLINT
      if (!read_i64_array_len(stridesObj, vRank, strides)) return lean_mk_empty_byte_array(lean_box(0));

      b_lean_obj_arg offObj = lean_array_get_core(offsetsViewsObj, vi);
      if (!lean_is_ctor(offObj)) return lean_mk_empty_byte_array(lean_box(0));
      const int64_t offset = (int64_t)lean_unbox_uint64(offObj);

      b_lean_obj_arg msObj = lean_array_get_core(msViewsObj, vi);
      b_lean_obj_arg meObj = lean_array_get_core(meViewsObj, vi);
      size_t *maskStart = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      size_t *maskEnd = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      if (!read_nat_array_len(msObj, vRank, maskStart)) return lean_mk_empty_byte_array(lean_box(0));
      if (!read_nat_array_len(meObj, vRank, maskEnd)) return lean_mk_empty_byte_array(lean_box(0));
      for (size_t d = 0; d < vRank; d++) {
        const size_t ms = maskStart[d];
        const size_t me = maskEnd[d];
        if (ms > me || me > vDims[d]) return lean_mk_empty_byte_array(lean_box(0));
      }

      views[vi].rank = vRank;
      views[vi].numel = vNumel;
      views[vi].contigStrides = contig;
      views[vi].strides = strides;
      views[vi].offset = offset;
      views[vi].maskStart = maskStart;
      views[vi].maskEnd = maskEnd;

      if (vRank > maxRank) maxRank = vRank;
    }

    stacks[i].nViews = nViews;
    stacks[i].views = views;

    if (views[nViews - 1].rank != rank) return lean_mk_empty_byte_array(lean_box(0));
  }

  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  size_t *idxWork = LEAN_ALLOCA(sizeof(size_t) * maxRank); // NOLINT
  uint8_t *valid = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  int64_t *inOff = LEAN_ALLOCA(sizeof(int64_t) * nInputs); // NOLINT
  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT

  for (size_t o = 0; o < outNumel; o++) {
    float acc = negInf;
    for (size_t r = 0; r < reduce; r++) {
      idx[axis] = r;

      for (size_t i = 0; i < nInputs; i++) {
        int64_t off = 0;
        const bool ok = tg4_view_stack_offset(&stacks[i], idx, rank, idxWork, &off);
        valid[i] = ok ? 1 : 0;
        inOff[i] = off;
      }

      size_t sp = 0;
      for (size_t ip = 0; ip < instsLen; ip++) {
        const uint64_t inst = prog[ip];

        const uint8_t kind = (uint8_t)(inst & 0xFFu);
        const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
        const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
        const uint32_t imm = (uint32_t)(inst >> 32);

        switch (kind) {
          case 0:
            stack[sp++] = f32_from_bits(imm);
            break;
          case 1: {
            if (arg >= nInputs || !valid[arg]) {
              stack[sp++] = 0.0f;
              break;
            }
            const int64_t off = inOff[arg];
            if (off < 0 || (uint64_t)off >= inCap[arg]) {
              stack[sp++] = 0.0f;
              break;
            }
            if (inDtypes[arg] == 1) {
              const uint8_t v = inPtrs[arg][(size_t)off];
              stack[sp++] = v ? 1.0f : 0.0f;
            } else {
              const float *p = (const float *)inPtrs[arg]; // NOLINT
              stack[sp++] = p[(size_t)off];
            }
            break;
          }
          case 2: {
            if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
            const float x = stack[sp - 1];
            switch (op) {
              case 0: stack[sp - 1] = -x; break;
              case 1: stack[sp - 1] = sqrtf(x); break;
              case 2: stack[sp - 1] = 1.0f / x; break;
              case 3: stack[sp - 1] = exp2f(x); break;
              case 4: stack[sp - 1] = log2f(x); break;
              case 5: stack[sp - 1] = sinf(x); break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 3: {
            if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            switch (op) {
              case 0: stack[sp++] = x + y; break;
              case 1: stack[sp++] = x - y; break;
              case 2: stack[sp++] = x * y; break;
              case 3: stack[sp++] = x / y; break;
              case 4: stack[sp++] = (x >= y) ? x : y; break;
              case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
              default: return lean_mk_empty_byte_array(lean_box(0));
            }
            break;
          }
          case 4: {
            if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
            const float y = stack[--sp];
            const float x = stack[--sp];
            const float c = stack[--sp];
            stack[sp++] = (c != 0.0f) ? x : y;
            break;
          }
          default:
            return lean_mk_empty_byte_array(lean_box(0));
        }
      }

      if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
      const float v = stack[0];
      acc = (v >= acc) ? v : acc;
    }

    dst[o] = acc;
    idx[axis] = 0;

    for (size_t d = rank; d-- > 0;) {
      if (d == axis) continue;
      idx[d]++;
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_all_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg inputShapesObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg fullShapeObj,
    b_lean_obj_arg progObj) {
  const size_t rank = lean_array_size(fullShapeObj);
  size_t *dims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(fullShapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(inputShapesObj) != nInputs || lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  lean_obj_res out = lean_alloc_sarray(1, 4, 4);
  float *dst = f32_ptr_mut(out);

  const size_t fullNumel = prod_dims(dims, rank);
  if (fullNumel == 0) {
    dst[0] = 0.0f;
    return out;
  }

  size_t *inDims = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *inStrides = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *inOff = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    b_lean_obj_arg shapeObj = lean_array_get_core(inputShapesObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT

    size_t *idims = inDims + i * rank;
    size_t *istrides = inStrides + i * rank;
    if (!read_padded_dims(shapeObj, rank, idims)) return lean_mk_empty_byte_array(lean_box(0));
    if (!is_broadcastable(idims, dims, rank)) return lean_mk_empty_byte_array(lean_box(0));

    const size_t numel = prod_dims(idims, rank);
    const size_t needed = (dt == 1) ? numel : (numel * 4);
    if (lean_sarray_size(bytesObj) < needed) return lean_mk_empty_byte_array(lean_box(0));

    contiguous_strides(idims, rank, istrides);
    broadcast_strides(idims, rank, istrides);
    inOff[i] = 0;
  }

  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT

  float acc = 0.0f;
  for (size_t o = 0; o < fullNumel; o++) {
    size_t sp = 0;
    for (size_t ip = 0; ip < instsLen; ip++) {
      const uint64_t inst = prog[ip];

      const uint8_t kind = (uint8_t)(inst & 0xFFu);
      const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
      const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
      const uint32_t imm = (uint32_t)(inst >> 32);

      switch (kind) {
        case 0:
          stack[sp++] = f32_from_bits(imm);
          break;
        case 1: {
          if (arg >= nInputs) {
            stack[sp++] = 0.0f;
            break;
          }
          const size_t off = inOff[arg];
          if (inDtypes[arg] == 1) {
            const uint8_t v = inPtrs[arg][off];
            stack[sp++] = v ? 1.0f : 0.0f;
          } else {
            const float *p = (const float *)inPtrs[arg]; // NOLINT
            stack[sp++] = p[off];
          }
          break;
        }
        case 2: {
          if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
          const float x = stack[sp - 1];
          switch (op) {
            case 0: stack[sp - 1] = -x; break;
            case 1: stack[sp - 1] = sqrtf(x); break;
            case 2: stack[sp - 1] = 1.0f / x; break;
            case 3: stack[sp - 1] = exp2f(x); break;
            case 4: stack[sp - 1] = log2f(x); break;
            case 5: stack[sp - 1] = sinf(x); break;
            default: return lean_mk_empty_byte_array(lean_box(0));
          }
          break;
        }
        case 3: {
          if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
          const float y = stack[--sp];
          const float x = stack[--sp];
          switch (op) {
            case 0: stack[sp++] = x + y; break;
            case 1: stack[sp++] = x - y; break;
            case 2: stack[sp++] = x * y; break;
            case 3: stack[sp++] = x / y; break;
            case 4: stack[sp++] = (x >= y) ? x : y; break;
            case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
            default: return lean_mk_empty_byte_array(lean_box(0));
          }
          break;
        }
        case 4: {
          if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
          const float y = stack[--sp];
          const float x = stack[--sp];
          const float c = stack[--sp];
          stack[sp++] = (c != 0.0f) ? x : y;
          break;
        }
        default:
          return lean_mk_empty_byte_array(lean_box(0));
      }
    }

    if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
    acc += stack[0];

    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] += inStrides[i * rank + d];
      }
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] -= inStrides[i * rank + d] * dims[d];
      }
    }
  }

  dst[0] = acc;
  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_all_view_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg inputStridesObj,
    b_lean_obj_arg inputOffsetsObj,
    b_lean_obj_arg inputMaskStartsObj,
    b_lean_obj_arg inputMaskEndsObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg fullShapeObj,
    b_lean_obj_arg progObj) {
  const size_t rank = lean_array_size(fullShapeObj);
  size_t *dims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(fullShapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(inputStridesObj) != nInputs || lean_array_size(inputOffsetsObj) != nInputs ||
      lean_array_size(inputMaskStartsObj) != nInputs || lean_array_size(inputMaskEndsObj) != nInputs ||
      lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  lean_obj_res out = lean_alloc_sarray(1, 4, 4);
  float *dst = f32_ptr_mut(out);

  const size_t fullNumel = prod_dims(dims, rank);
  if (fullNumel == 0) {
    dst[0] = 0.0f;
    return out;
  }

  int64_t *inStrides = LEAN_ALLOCA(sizeof(int64_t) * nInputs * rank); // NOLINT
  int64_t *inOff = LEAN_ALLOCA(sizeof(int64_t) * nInputs); // NOLINT
  size_t *maskStart = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *maskEnd = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  size_t *inCap = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT
    const size_t byteSize = lean_sarray_size(bytesObj);
    inCap[i] = (dt == 1) ? byteSize : (byteSize / 4);

    b_lean_obj_arg stridesObj = lean_array_get_core(inputStridesObj, i);
    if (!read_i64_array_len(stridesObj, rank, inStrides + i * rank)) return lean_mk_empty_byte_array(lean_box(0));

    b_lean_obj_arg offObj = lean_array_get_core(inputOffsetsObj, i);
    if (!lean_is_ctor(offObj)) return lean_mk_empty_byte_array(lean_box(0));
    inOff[i] = (int64_t)lean_unbox_uint64(offObj);

    b_lean_obj_arg msObj = lean_array_get_core(inputMaskStartsObj, i);
    b_lean_obj_arg meObj = lean_array_get_core(inputMaskEndsObj, i);
    if (!read_nat_array_len(msObj, rank, maskStart + i * rank)) return lean_mk_empty_byte_array(lean_box(0));
    if (!read_nat_array_len(meObj, rank, maskEnd + i * rank)) return lean_mk_empty_byte_array(lean_box(0));
    for (size_t d = 0; d < rank; d++) {
      const size_t ms = maskStart[i * rank + d];
      const size_t me = maskEnd[i * rank + d];
      if (ms > me || me > dims[d]) return lean_mk_empty_byte_array(lean_box(0));
    }
  }

  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  uint8_t *valid = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT

  float acc = 0.0f;
  for (size_t o = 0; o < fullNumel; o++) {
    for (size_t i = 0; i < nInputs; i++) {
      uint8_t ok = 1;
      for (size_t d = 0; d < rank; d++) {
        const size_t x = idx[d];
        const size_t ms = maskStart[i * rank + d];
        const size_t me = maskEnd[i * rank + d];
        if (x < ms || x >= me) {
          ok = 0;
          break;
        }
      }
      valid[i] = ok;
    }

    size_t sp = 0;
    for (size_t ip = 0; ip < instsLen; ip++) {
      const uint64_t inst = prog[ip];

      const uint8_t kind = (uint8_t)(inst & 0xFFu);
      const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
      const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
      const uint32_t imm = (uint32_t)(inst >> 32);

      switch (kind) {
        case 0:
          stack[sp++] = f32_from_bits(imm);
          break;
        case 1: {
          if (arg >= nInputs || !valid[arg]) {
            stack[sp++] = 0.0f;
            break;
          }
          const int64_t off = inOff[arg];
          if (off < 0 || (uint64_t)off >= inCap[arg]) {
            stack[sp++] = 0.0f;
            break;
          }
          if (inDtypes[arg] == 1) {
            const uint8_t v = inPtrs[arg][(size_t)off];
            stack[sp++] = v ? 1.0f : 0.0f;
          } else {
            const float *p = (const float *)inPtrs[arg]; // NOLINT
            stack[sp++] = p[(size_t)off];
          }
          break;
        }
        case 2: {
          if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
          const float x = stack[sp - 1];
          switch (op) {
            case 0: stack[sp - 1] = -x; break;
            case 1: stack[sp - 1] = sqrtf(x); break;
            case 2: stack[sp - 1] = 1.0f / x; break;
            case 3: stack[sp - 1] = exp2f(x); break;
            case 4: stack[sp - 1] = log2f(x); break;
            case 5: stack[sp - 1] = sinf(x); break;
            default: return lean_mk_empty_byte_array(lean_box(0));
          }
          break;
        }
        case 3: {
          if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
          const float y = stack[--sp];
          const float x = stack[--sp];
          switch (op) {
            case 0: stack[sp++] = x + y; break;
            case 1: stack[sp++] = x - y; break;
            case 2: stack[sp++] = x * y; break;
            case 3: stack[sp++] = x / y; break;
            case 4: stack[sp++] = (x >= y) ? x : y; break;
            case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
            default: return lean_mk_empty_byte_array(lean_box(0));
          }
          break;
        }
        case 4: {
          if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
          const float y = stack[--sp];
          const float x = stack[--sp];
          const float c = stack[--sp];
          stack[sp++] = (c != 0.0f) ? x : y;
          break;
        }
        default:
          return lean_mk_empty_byte_array(lean_box(0));
      }
    }

    if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
    acc += stack[0];

    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] += inStrides[i * rank + d];
      }
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] -= inStrides[i * rank + d] * (int64_t)dims[d];
      }
    }
  }

  dst[0] = acc;
  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_sum_all_view_stack_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg stackShapesObj,
    b_lean_obj_arg stackStridesObj,
    b_lean_obj_arg stackOffsetsObj,
    b_lean_obj_arg stackMaskStartsObj,
    b_lean_obj_arg stackMaskEndsObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg fullShapeObj,
    b_lean_obj_arg progObj) {
  const size_t rank = lean_array_size(fullShapeObj);
  size_t *dims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(fullShapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  lean_obj_res out = lean_alloc_sarray(1, 4, 4);
  float *dst = f32_ptr_mut(out);

  const size_t fullNumel = prod_dims(dims, rank);
  if (fullNumel == 0) {
    dst[0] = 0.0f;
    return out;
  }

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(stackShapesObj) != nInputs || lean_array_size(stackStridesObj) != nInputs || lean_array_size(stackOffsetsObj) != nInputs ||
      lean_array_size(stackMaskStartsObj) != nInputs || lean_array_size(stackMaskEndsObj) != nInputs || lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  size_t *inCap = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  tg4_view_stack *stacks = LEAN_ALLOCA(sizeof(tg4_view_stack) * nInputs); // NOLINT
  size_t maxRank = rank;

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT
    const size_t byteSize = lean_sarray_size(bytesObj);
    inCap[i] = (dt == 1) ? byteSize : (byteSize / 4);

    b_lean_obj_arg shapesViewsObj = lean_array_get_core(stackShapesObj, i);
    b_lean_obj_arg stridesViewsObj = lean_array_get_core(stackStridesObj, i);
    b_lean_obj_arg offsetsViewsObj = lean_array_get_core(stackOffsetsObj, i);
    b_lean_obj_arg msViewsObj = lean_array_get_core(stackMaskStartsObj, i);
    b_lean_obj_arg meViewsObj = lean_array_get_core(stackMaskEndsObj, i);

    const size_t nViews = lean_array_size(shapesViewsObj);
    if (nViews == 0 || lean_array_size(stridesViewsObj) != nViews || lean_array_size(offsetsViewsObj) != nViews ||
        lean_array_size(msViewsObj) != nViews || lean_array_size(meViewsObj) != nViews) {
      return lean_mk_empty_byte_array(lean_box(0));
    }

    tg4_view_stack_view *views = LEAN_ALLOCA(sizeof(tg4_view_stack_view) * nViews); // NOLINT
    for (size_t vi = 0; vi < nViews; vi++) {
      b_lean_obj_arg shapeObj = lean_array_get_core(shapesViewsObj, vi);
      const size_t vRank = lean_array_size(shapeObj);

      size_t *vDims = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      if (!read_shape_dims(shapeObj, vDims)) return lean_mk_empty_byte_array(lean_box(0));

      size_t *contig = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      contiguous_strides(vDims, vRank, contig);

      const size_t vNumel = prod_dims(vDims, vRank);

      b_lean_obj_arg stridesObj = lean_array_get_core(stridesViewsObj, vi);
      int64_t *strides = LEAN_ALLOCA(sizeof(int64_t) * vRank); // NOLINT
      if (!read_i64_array_len(stridesObj, vRank, strides)) return lean_mk_empty_byte_array(lean_box(0));

      b_lean_obj_arg offObj = lean_array_get_core(offsetsViewsObj, vi);
      if (!lean_is_ctor(offObj)) return lean_mk_empty_byte_array(lean_box(0));
      const int64_t offset = (int64_t)lean_unbox_uint64(offObj);

      b_lean_obj_arg msObj = lean_array_get_core(msViewsObj, vi);
      b_lean_obj_arg meObj = lean_array_get_core(meViewsObj, vi);
      size_t *maskStart = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      size_t *maskEnd = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      if (!read_nat_array_len(msObj, vRank, maskStart)) return lean_mk_empty_byte_array(lean_box(0));
      if (!read_nat_array_len(meObj, vRank, maskEnd)) return lean_mk_empty_byte_array(lean_box(0));
      for (size_t d = 0; d < vRank; d++) {
        const size_t ms = maskStart[d];
        const size_t me = maskEnd[d];
        if (ms > me || me > vDims[d]) return lean_mk_empty_byte_array(lean_box(0));
      }

      views[vi].rank = vRank;
      views[vi].numel = vNumel;
      views[vi].contigStrides = contig;
      views[vi].strides = strides;
      views[vi].offset = offset;
      views[vi].maskStart = maskStart;
      views[vi].maskEnd = maskEnd;

      if (vRank > maxRank) maxRank = vRank;
    }

    stacks[i].nViews = nViews;
    stacks[i].views = views;

    if (views[nViews - 1].rank != rank) return lean_mk_empty_byte_array(lean_box(0));
  }

  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  size_t *idxWork = LEAN_ALLOCA(sizeof(size_t) * maxRank); // NOLINT
  uint8_t *valid = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  int64_t *inOff = LEAN_ALLOCA(sizeof(int64_t) * nInputs); // NOLINT
  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT

  float acc = 0.0f;
  for (size_t o = 0; o < fullNumel; o++) {
    for (size_t i = 0; i < nInputs; i++) {
      int64_t off = 0;
      const bool ok = tg4_view_stack_offset(&stacks[i], idx, rank, idxWork, &off);
      valid[i] = ok ? 1 : 0;
      inOff[i] = off;
    }

    size_t sp = 0;
    for (size_t ip = 0; ip < instsLen; ip++) {
      const uint64_t inst = prog[ip];

      const uint8_t kind = (uint8_t)(inst & 0xFFu);
      const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
      const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
      const uint32_t imm = (uint32_t)(inst >> 32);

      switch (kind) {
        case 0:
          stack[sp++] = f32_from_bits(imm);
          break;
        case 1: {
          if (arg >= nInputs || !valid[arg]) {
            stack[sp++] = 0.0f;
            break;
          }
          const int64_t off = inOff[arg];
          if (off < 0 || (uint64_t)off >= inCap[arg]) {
            stack[sp++] = 0.0f;
            break;
          }
          if (inDtypes[arg] == 1) {
            const uint8_t v = inPtrs[arg][(size_t)off];
            stack[sp++] = v ? 1.0f : 0.0f;
          } else {
            const float *p = (const float *)inPtrs[arg]; // NOLINT
            stack[sp++] = p[(size_t)off];
          }
          break;
        }
        case 2: {
          if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
          const float x = stack[sp - 1];
          switch (op) {
            case 0: stack[sp - 1] = -x; break;
            case 1: stack[sp - 1] = sqrtf(x); break;
            case 2: stack[sp - 1] = 1.0f / x; break;
            case 3: stack[sp - 1] = exp2f(x); break;
            case 4: stack[sp - 1] = log2f(x); break;
            case 5: stack[sp - 1] = sinf(x); break;
            default: return lean_mk_empty_byte_array(lean_box(0));
          }
          break;
        }
        case 3: {
          if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
          const float y = stack[--sp];
          const float x = stack[--sp];
          switch (op) {
            case 0: stack[sp++] = x + y; break;
            case 1: stack[sp++] = x - y; break;
            case 2: stack[sp++] = x * y; break;
            case 3: stack[sp++] = x / y; break;
            case 4: stack[sp++] = (x >= y) ? x : y; break;
            case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
            default: return lean_mk_empty_byte_array(lean_box(0));
          }
          break;
        }
        case 4: {
          if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
          const float y = stack[--sp];
          const float x = stack[--sp];
          const float c = stack[--sp];
          stack[sp++] = (c != 0.0f) ? x : y;
          break;
        }
        default:
          return lean_mk_empty_byte_array(lean_box(0));
      }
    }

    if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
    acc += stack[0];

    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
    }
  }

  dst[0] = acc;
  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_all_view_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg inputStridesObj,
    b_lean_obj_arg inputOffsetsObj,
    b_lean_obj_arg inputMaskStartsObj,
    b_lean_obj_arg inputMaskEndsObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg fullShapeObj,
    b_lean_obj_arg progObj) {
  const size_t rank = lean_array_size(fullShapeObj);
  size_t *dims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(fullShapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(inputStridesObj) != nInputs || lean_array_size(inputOffsetsObj) != nInputs ||
      lean_array_size(inputMaskStartsObj) != nInputs || lean_array_size(inputMaskEndsObj) != nInputs ||
      lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  lean_obj_res out = lean_alloc_sarray(1, 4, 4);
  float *dst = f32_ptr_mut(out);

  union {
    uint32_t u;
    float f;
  } conv;
  conv.u = 0xFF800000u; // -inf
  const float negInf = conv.f;

  const size_t fullNumel = prod_dims(dims, rank);
  if (fullNumel == 0) {
    dst[0] = negInf;
    return out;
  }

  int64_t *inStrides = LEAN_ALLOCA(sizeof(int64_t) * nInputs * rank); // NOLINT
  int64_t *inOff = LEAN_ALLOCA(sizeof(int64_t) * nInputs); // NOLINT
  size_t *maskStart = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *maskEnd = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  size_t *inCap = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT
    const size_t byteSize = lean_sarray_size(bytesObj);
    inCap[i] = (dt == 1) ? byteSize : (byteSize / 4);

    b_lean_obj_arg stridesObj = lean_array_get_core(inputStridesObj, i);
    if (!read_i64_array_len(stridesObj, rank, inStrides + i * rank)) return lean_mk_empty_byte_array(lean_box(0));

    b_lean_obj_arg offObj = lean_array_get_core(inputOffsetsObj, i);
    if (!lean_is_ctor(offObj)) return lean_mk_empty_byte_array(lean_box(0));
    inOff[i] = (int64_t)lean_unbox_uint64(offObj);

    b_lean_obj_arg msObj = lean_array_get_core(inputMaskStartsObj, i);
    b_lean_obj_arg meObj = lean_array_get_core(inputMaskEndsObj, i);
    if (!read_nat_array_len(msObj, rank, maskStart + i * rank)) return lean_mk_empty_byte_array(lean_box(0));
    if (!read_nat_array_len(meObj, rank, maskEnd + i * rank)) return lean_mk_empty_byte_array(lean_box(0));
    for (size_t d = 0; d < rank; d++) {
      const size_t ms = maskStart[i * rank + d];
      const size_t me = maskEnd[i * rank + d];
      if (ms > me || me > dims[d]) return lean_mk_empty_byte_array(lean_box(0));
    }
  }

  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  uint8_t *valid = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT

  float acc = negInf;
  for (size_t o = 0; o < fullNumel; o++) {
    for (size_t i = 0; i < nInputs; i++) {
      uint8_t ok = 1;
      for (size_t d = 0; d < rank; d++) {
        const size_t x = idx[d];
        const size_t ms = maskStart[i * rank + d];
        const size_t me = maskEnd[i * rank + d];
        if (x < ms || x >= me) {
          ok = 0;
          break;
        }
      }
      valid[i] = ok;
    }

    size_t sp = 0;
    for (size_t ip = 0; ip < instsLen; ip++) {
      const uint64_t inst = prog[ip];

      const uint8_t kind = (uint8_t)(inst & 0xFFu);
      const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
      const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
      const uint32_t imm = (uint32_t)(inst >> 32);

      switch (kind) {
        case 0:
          stack[sp++] = f32_from_bits(imm);
          break;
        case 1: {
          if (arg >= nInputs || !valid[arg]) {
            stack[sp++] = 0.0f;
            break;
          }
          const int64_t off = inOff[arg];
          if (off < 0 || (uint64_t)off >= inCap[arg]) {
            stack[sp++] = 0.0f;
            break;
          }
          if (inDtypes[arg] == 1) {
            const uint8_t v = inPtrs[arg][(size_t)off];
            stack[sp++] = v ? 1.0f : 0.0f;
          } else {
            const float *p = (const float *)inPtrs[arg]; // NOLINT
            stack[sp++] = p[(size_t)off];
          }
          break;
        }
        case 2: {
          if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
          const float x = stack[sp - 1];
          switch (op) {
            case 0: stack[sp - 1] = -x; break;
            case 1: stack[sp - 1] = sqrtf(x); break;
            case 2: stack[sp - 1] = 1.0f / x; break;
            case 3: stack[sp - 1] = exp2f(x); break;
            case 4: stack[sp - 1] = log2f(x); break;
            case 5: stack[sp - 1] = sinf(x); break;
            default: return lean_mk_empty_byte_array(lean_box(0));
          }
          break;
        }
        case 3: {
          if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
          const float y = stack[--sp];
          const float x = stack[--sp];
          switch (op) {
            case 0: stack[sp++] = x + y; break;
            case 1: stack[sp++] = x - y; break;
            case 2: stack[sp++] = x * y; break;
            case 3: stack[sp++] = x / y; break;
            case 4: stack[sp++] = (x >= y) ? x : y; break;
            case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
            default: return lean_mk_empty_byte_array(lean_box(0));
          }
          break;
        }
        case 4: {
          if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
          const float y = stack[--sp];
          const float x = stack[--sp];
          const float c = stack[--sp];
          stack[sp++] = (c != 0.0f) ? x : y;
          break;
        }
        default:
          return lean_mk_empty_byte_array(lean_box(0));
      }
    }

    if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
    const float v = stack[0];
    acc = (v >= acc) ? v : acc;

    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] += inStrides[i * rank + d];
      }
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] -= inStrides[i * rank + d] * (int64_t)dims[d];
      }
    }
  }

  dst[0] = acc;
  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_all_view_stack_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg stackShapesObj,
    b_lean_obj_arg stackStridesObj,
    b_lean_obj_arg stackOffsetsObj,
    b_lean_obj_arg stackMaskStartsObj,
    b_lean_obj_arg stackMaskEndsObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg fullShapeObj,
    b_lean_obj_arg progObj) {
  const size_t rank = lean_array_size(fullShapeObj);
  size_t *dims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(fullShapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  lean_obj_res out = lean_alloc_sarray(1, 4, 4);
  float *dst = f32_ptr_mut(out);

  union {
    uint32_t u;
    float f;
  } conv;
  conv.u = 0xFF800000u; // -inf
  const float negInf = conv.f;

  const size_t fullNumel = prod_dims(dims, rank);
  if (fullNumel == 0) {
    dst[0] = negInf;
    return out;
  }

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(stackShapesObj) != nInputs || lean_array_size(stackStridesObj) != nInputs || lean_array_size(stackOffsetsObj) != nInputs ||
      lean_array_size(stackMaskStartsObj) != nInputs || lean_array_size(stackMaskEndsObj) != nInputs || lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  size_t *inCap = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  tg4_view_stack *stacks = LEAN_ALLOCA(sizeof(tg4_view_stack) * nInputs); // NOLINT
  size_t maxRank = rank;

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT
    const size_t byteSize = lean_sarray_size(bytesObj);
    inCap[i] = (dt == 1) ? byteSize : (byteSize / 4);

    b_lean_obj_arg shapesViewsObj = lean_array_get_core(stackShapesObj, i);
    b_lean_obj_arg stridesViewsObj = lean_array_get_core(stackStridesObj, i);
    b_lean_obj_arg offsetsViewsObj = lean_array_get_core(stackOffsetsObj, i);
    b_lean_obj_arg msViewsObj = lean_array_get_core(stackMaskStartsObj, i);
    b_lean_obj_arg meViewsObj = lean_array_get_core(stackMaskEndsObj, i);

    const size_t nViews = lean_array_size(shapesViewsObj);
    if (nViews == 0 || lean_array_size(stridesViewsObj) != nViews || lean_array_size(offsetsViewsObj) != nViews ||
        lean_array_size(msViewsObj) != nViews || lean_array_size(meViewsObj) != nViews) {
      return lean_mk_empty_byte_array(lean_box(0));
    }

    tg4_view_stack_view *views = LEAN_ALLOCA(sizeof(tg4_view_stack_view) * nViews); // NOLINT
    for (size_t vi = 0; vi < nViews; vi++) {
      b_lean_obj_arg shapeObj = lean_array_get_core(shapesViewsObj, vi);
      const size_t vRank = lean_array_size(shapeObj);

      size_t *vDims = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      if (!read_shape_dims(shapeObj, vDims)) return lean_mk_empty_byte_array(lean_box(0));

      size_t *contig = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      contiguous_strides(vDims, vRank, contig);

      const size_t vNumel = prod_dims(vDims, vRank);

      b_lean_obj_arg stridesObj = lean_array_get_core(stridesViewsObj, vi);
      int64_t *strides = LEAN_ALLOCA(sizeof(int64_t) * vRank); // NOLINT
      if (!read_i64_array_len(stridesObj, vRank, strides)) return lean_mk_empty_byte_array(lean_box(0));

      b_lean_obj_arg offObj = lean_array_get_core(offsetsViewsObj, vi);
      if (!lean_is_ctor(offObj)) return lean_mk_empty_byte_array(lean_box(0));
      const int64_t offset = (int64_t)lean_unbox_uint64(offObj);

      b_lean_obj_arg msObj = lean_array_get_core(msViewsObj, vi);
      b_lean_obj_arg meObj = lean_array_get_core(meViewsObj, vi);
      size_t *maskStart = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      size_t *maskEnd = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
      if (!read_nat_array_len(msObj, vRank, maskStart)) return lean_mk_empty_byte_array(lean_box(0));
      if (!read_nat_array_len(meObj, vRank, maskEnd)) return lean_mk_empty_byte_array(lean_box(0));
      for (size_t d = 0; d < vRank; d++) {
        const size_t ms = maskStart[d];
        const size_t me = maskEnd[d];
        if (ms > me || me > vDims[d]) return lean_mk_empty_byte_array(lean_box(0));
      }

      views[vi].rank = vRank;
      views[vi].numel = vNumel;
      views[vi].contigStrides = contig;
      views[vi].strides = strides;
      views[vi].offset = offset;
      views[vi].maskStart = maskStart;
      views[vi].maskEnd = maskEnd;

      if (vRank > maxRank) maxRank = vRank;
    }

    stacks[i].nViews = nViews;
    stacks[i].views = views;

    if (views[nViews - 1].rank != rank) return lean_mk_empty_byte_array(lean_box(0));
  }

  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  size_t *idxWork = LEAN_ALLOCA(sizeof(size_t) * maxRank); // NOLINT
  uint8_t *valid = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT
  int64_t *inOff = LEAN_ALLOCA(sizeof(int64_t) * nInputs); // NOLINT
  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT

  float acc = negInf;
  for (size_t o = 0; o < fullNumel; o++) {
    for (size_t i = 0; i < nInputs; i++) {
      int64_t off = 0;
      const bool ok = tg4_view_stack_offset(&stacks[i], idx, rank, idxWork, &off);
      valid[i] = ok ? 1 : 0;
      inOff[i] = off;
    }

    size_t sp = 0;
    for (size_t ip = 0; ip < instsLen; ip++) {
      const uint64_t inst = prog[ip];

      const uint8_t kind = (uint8_t)(inst & 0xFFu);
      const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
      const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
      const uint32_t imm = (uint32_t)(inst >> 32);

      switch (kind) {
        case 0:
          stack[sp++] = f32_from_bits(imm);
          break;
        case 1: {
          if (arg >= nInputs || !valid[arg]) {
            stack[sp++] = 0.0f;
            break;
          }
          const int64_t off = inOff[arg];
          if (off < 0 || (uint64_t)off >= inCap[arg]) {
            stack[sp++] = 0.0f;
            break;
          }
          if (inDtypes[arg] == 1) {
            const uint8_t v = inPtrs[arg][(size_t)off];
            stack[sp++] = v ? 1.0f : 0.0f;
          } else {
            const float *p = (const float *)inPtrs[arg]; // NOLINT
            stack[sp++] = p[(size_t)off];
          }
          break;
        }
        case 2: {
          if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
          const float x = stack[sp - 1];
          switch (op) {
            case 0: stack[sp - 1] = -x; break;
            case 1: stack[sp - 1] = sqrtf(x); break;
            case 2: stack[sp - 1] = 1.0f / x; break;
            case 3: stack[sp - 1] = exp2f(x); break;
            case 4: stack[sp - 1] = log2f(x); break;
            case 5: stack[sp - 1] = sinf(x); break;
            default: return lean_mk_empty_byte_array(lean_box(0));
          }
          break;
        }
        case 3: {
          if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
          const float y = stack[--sp];
          const float x = stack[--sp];
          switch (op) {
            case 0: stack[sp++] = x + y; break;
            case 1: stack[sp++] = x - y; break;
            case 2: stack[sp++] = x * y; break;
            case 3: stack[sp++] = x / y; break;
            case 4: stack[sp++] = (x >= y) ? x : y; break;
            case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
            default: return lean_mk_empty_byte_array(lean_box(0));
          }
          break;
        }
        case 4: {
          if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
          const float y = stack[--sp];
          const float x = stack[--sp];
          const float c = stack[--sp];
          stack[sp++] = (c != 0.0f) ? x : y;
          break;
        }
        default:
          return lean_mk_empty_byte_array(lean_box(0));
      }
    }

    if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
    const float v = stack[0];
    acc = (v >= acc) ? v : acc;

    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
    }
  }

  dst[0] = acc;
  return out;
}

LEAN_EXPORT lean_obj_res tg4_fused_reduce_max_all_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg inputShapesObj,
    b_lean_obj_arg inputDtypesObj,
    b_lean_obj_arg fullShapeObj,
    b_lean_obj_arg progObj) {
  const size_t rank = lean_array_size(fullShapeObj);
  size_t *dims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(fullShapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t nInputs = lean_array_size(inputsObj);
  if (lean_array_size(inputShapesObj) != nInputs || lean_array_size(inputDtypesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t instsLen = lean_array_size(progObj);
  if (instsLen == 0) return lean_mk_empty_byte_array(lean_box(0));
  uint64_t *prog = LEAN_ALLOCA(sizeof(uint64_t) * instsLen); // NOLINT
  for (size_t ip = 0; ip < instsLen; ip++) {
    b_lean_obj_arg instObj = lean_array_get_core(progObj, ip);
    prog[ip] = lean_unbox_uint64(instObj);
  }

  lean_obj_res out = lean_alloc_sarray(1, 4, 4);
  float *dst = f32_ptr_mut(out);

  union {
    uint32_t u;
    float f;
  } conv;
  conv.u = 0xFF800000u; // -inf
  const float negInf = conv.f;

  const size_t fullNumel = prod_dims(dims, rank);
  if (fullNumel == 0) {
    dst[0] = negInf;
    return out;
  }

  size_t *inDims = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *inStrides = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  size_t *inOff = LEAN_ALLOCA(sizeof(size_t) * nInputs); // NOLINT
  uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT
  uint8_t *inDtypes = LEAN_ALLOCA(sizeof(uint8_t) * nInputs); // NOLINT

  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg dtObj = lean_array_get_core(inputDtypesObj, i);
    if (!lean_is_scalar(dtObj)) return lean_mk_empty_byte_array(lean_box(0));
    const size_t dt = lean_unbox(dtObj);
    inDtypes[i] = (uint8_t)dt;

    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    b_lean_obj_arg shapeObj = lean_array_get_core(inputShapesObj, i);
    inPtrs[i] = (uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT

    size_t *idims = inDims + i * rank;
    size_t *istrides = inStrides + i * rank;
    if (!read_padded_dims(shapeObj, rank, idims)) return lean_mk_empty_byte_array(lean_box(0));
    if (!is_broadcastable(idims, dims, rank)) return lean_mk_empty_byte_array(lean_box(0));

    const size_t numel = prod_dims(idims, rank);
    const size_t needed = (dt == 1) ? numel : (numel * 4);
    if (lean_sarray_size(bytesObj) < needed) return lean_mk_empty_byte_array(lean_box(0));

    contiguous_strides(idims, rank, istrides);
    broadcast_strides(idims, rank, istrides);
    inOff[i] = 0;
  }

  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  float *stack = LEAN_ALLOCA(sizeof(float) * instsLen); // NOLINT

  float acc = negInf;
  for (size_t o = 0; o < fullNumel; o++) {
    size_t sp = 0;
    for (size_t ip = 0; ip < instsLen; ip++) {
      const uint64_t inst = prog[ip];

      const uint8_t kind = (uint8_t)(inst & 0xFFu);
      const uint8_t op = (uint8_t)((inst >> 8) & 0xFFu);
      const uint16_t arg = (uint16_t)((inst >> 16) & 0xFFFFu);
      const uint32_t imm = (uint32_t)(inst >> 32);

      switch (kind) {
        case 0:
          stack[sp++] = f32_from_bits(imm);
          break;
        case 1: {
          if (arg >= nInputs) {
            stack[sp++] = 0.0f;
            break;
          }
          const size_t off = inOff[arg];
          if (inDtypes[arg] == 1) {
            const uint8_t v = inPtrs[arg][off];
            stack[sp++] = v ? 1.0f : 0.0f;
          } else {
            const float *p = (const float *)inPtrs[arg]; // NOLINT
            stack[sp++] = p[off];
          }
          break;
        }
        case 2: {
          if (sp < 1) return lean_mk_empty_byte_array(lean_box(0));
          const float x = stack[sp - 1];
          switch (op) {
            case 0: stack[sp - 1] = -x; break;
            case 1: stack[sp - 1] = sqrtf(x); break;
            case 2: stack[sp - 1] = 1.0f / x; break;
            case 3: stack[sp - 1] = exp2f(x); break;
            case 4: stack[sp - 1] = log2f(x); break;
            case 5: stack[sp - 1] = sinf(x); break;
            default: return lean_mk_empty_byte_array(lean_box(0));
          }
          break;
        }
        case 3: {
          if (sp < 2) return lean_mk_empty_byte_array(lean_box(0));
          const float y = stack[--sp];
          const float x = stack[--sp];
          switch (op) {
            case 0: stack[sp++] = x + y; break;
            case 1: stack[sp++] = x - y; break;
            case 2: stack[sp++] = x * y; break;
            case 3: stack[sp++] = x / y; break;
            case 4: stack[sp++] = (x >= y) ? x : y; break;
            case 5: stack[sp++] = (x < y) ? 1.0f : 0.0f; break;
            default: return lean_mk_empty_byte_array(lean_box(0));
          }
          break;
        }
        case 4: {
          if (sp < 3) return lean_mk_empty_byte_array(lean_box(0));
          const float y = stack[--sp];
          const float x = stack[--sp];
          const float c = stack[--sp];
          stack[sp++] = (c != 0.0f) ? x : y;
          break;
        }
        default:
          return lean_mk_empty_byte_array(lean_box(0));
      }
    }

    if (sp != 1) return lean_mk_empty_byte_array(lean_box(0));
    const float v = stack[0];
    acc = (v >= acc) ? v : acc;

    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] += inStrides[i * rank + d];
      }
      if (idx[d] < dims[d]) break;
      idx[d] = 0;
      for (size_t i = 0; i < nInputs; i++) {
        inOff[i] -= inStrides[i * rank + d] * dims[d];
      }
    }
  }

  dst[0] = acc;
  return out;
}

LEAN_EXPORT lean_obj_res tg4_sum_all_f32(b_lean_obj_arg aBytes) {
  const size_t n = f32_elems(aBytes);
  const float *a = f32_ptr(aBytes);
  float acc = 0.0f;
  for (size_t i = 0; i < n; i++) acc += a[i];

  lean_obj_res out = lean_alloc_sarray(1, 4, 4);
  f32_ptr_mut(out)[0] = acc;
  return out;
}

LEAN_EXPORT lean_obj_res tg4_reduce_sum_last_f32(b_lean_obj_arg aBytes, b_lean_obj_arg outerObj, b_lean_obj_arg innerObj) {
  if (!lean_is_scalar(outerObj) || !lean_is_scalar(innerObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t outer = lean_unbox(outerObj);
  const size_t inner = lean_unbox(innerObj);
  const size_t outBytes = outer * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);

  if (inner == 0) {
    for (size_t i = 0; i < outer; i++) dst[i] = 0.0f;
    return out;
  }

  const size_t aBytesSz = lean_sarray_size(aBytes);
  if (aBytesSz < outer * inner * 4) {
    for (size_t i = 0; i < outer; i++) dst[i] = 0.0f;
    return out;
  }

  const float *a = f32_ptr(aBytes);
  for (size_t i = 0; i < outer; i++) {
    float acc = 0.0f;
    const float *row = a + i * inner;
    for (size_t j = 0; j < inner; j++) acc += row[j];
    dst[i] = acc;
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_reduce_max_last_f32(b_lean_obj_arg aBytes, b_lean_obj_arg outerObj, b_lean_obj_arg innerObj) {
  if (!lean_is_scalar(outerObj) || !lean_is_scalar(innerObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t outer = lean_unbox(outerObj);
  const size_t inner = lean_unbox(innerObj);
  const size_t outBytes = outer * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);

  if (inner == 0) {
    union {
      uint32_t u;
      float f;
    } conv;
    conv.u = 0xFF800000u; // -inf
    const float negInf = conv.f;
    for (size_t i = 0; i < outer; i++) dst[i] = negInf;
    return out;
  }

  const size_t aBytesSz = lean_sarray_size(aBytes);
  if (aBytesSz < outer * inner * 4) {
    for (size_t i = 0; i < outer; i++) dst[i] = 0.0f;
    return out;
  }

  const float *a = f32_ptr(aBytes);
  for (size_t i = 0; i < outer; i++) {
    const float *row = a + i * inner;
    float acc = row[0];
    for (size_t j = 1; j < inner; j++) acc = (row[j] >= acc) ? row[j] : acc;
    dst[i] = acc;
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_reduce_sum_axis_f32(
    b_lean_obj_arg aBytes,
    b_lean_obj_arg outerObj,
    b_lean_obj_arg reduceObj,
    b_lean_obj_arg innerObj) {
  if (!lean_is_scalar(outerObj) || !lean_is_scalar(reduceObj) || !lean_is_scalar(innerObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t outer = lean_unbox(outerObj);
  const size_t reduce = lean_unbox(reduceObj);
  const size_t inner = lean_unbox(innerObj);
  const size_t outFloats = outer * inner;
  const size_t outBytes = outFloats * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);

  if (reduce == 0) {
    for (size_t i = 0; i < outFloats; i++) dst[i] = 0.0f;
    return out;
  }

  const size_t aBytesSz = lean_sarray_size(aBytes);
  if (aBytesSz < outer * reduce * inner * 4) {
    for (size_t i = 0; i < outFloats; i++) dst[i] = 0.0f;
    return out;
  }

  const float *a = f32_ptr(aBytes);
  for (size_t o = 0; o < outer; o++) {
    for (size_t i = 0; i < inner; i++) {
      float acc = 0.0f;
      for (size_t r = 0; r < reduce; r++) {
        acc += a[(o * reduce + r) * inner + i];
      }
      dst[o * inner + i] = acc;
    }
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_reduce_max_axis_f32(
    b_lean_obj_arg aBytes,
    b_lean_obj_arg outerObj,
    b_lean_obj_arg reduceObj,
    b_lean_obj_arg innerObj) {
  if (!lean_is_scalar(outerObj) || !lean_is_scalar(reduceObj) || !lean_is_scalar(innerObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t outer = lean_unbox(outerObj);
  const size_t reduce = lean_unbox(reduceObj);
  const size_t inner = lean_unbox(innerObj);
  const size_t outFloats = outer * inner;
  const size_t outBytes = outFloats * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);

  if (reduce == 0) {
    union {
      uint32_t u;
      float f;
    } conv;
    conv.u = 0xFF800000u; // -inf
    const float negInf = conv.f;
    for (size_t i = 0; i < outFloats; i++) dst[i] = negInf;
    return out;
  }

  const size_t aBytesSz = lean_sarray_size(aBytes);
  if (aBytesSz < outer * reduce * inner * 4) {
    for (size_t i = 0; i < outFloats; i++) dst[i] = 0.0f;
    return out;
  }

  const float *a = f32_ptr(aBytes);
  for (size_t o = 0; o < outer; o++) {
    for (size_t i = 0; i < inner; i++) {
      float acc = a[(o * reduce + 0) * inner + i];
      for (size_t r = 1; r < reduce; r++) {
        const float v = a[(o * reduce + r) * inner + i];
        acc = (v >= acc) ? v : acc;
      }
      dst[o * inner + i] = acc;
    }
  }
  return out;
}

// Softmax/log-softmax over the last axis for a [outer, inner] contiguous float32 buffer.
// Uses exp2/log2 to match the TinyGrad4 Float32 graph encoding.
LEAN_EXPORT lean_obj_res tg4_softmax_last_f32(
    b_lean_obj_arg aBytes,
    b_lean_obj_arg outerObj,
    b_lean_obj_arg innerObj,
    uint32_t scaleBits) {
  if (!lean_is_scalar(outerObj) || !lean_is_scalar(innerObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t outer = lean_unbox(outerObj);
  const size_t inner = lean_unbox(innerObj);
  const float scale = f32_from_bits(scaleBits);

  const size_t outFloats = outer * inner;
  const size_t outBytes = outFloats * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outFloats == 0) return out;

  const size_t aBytesSz = lean_sarray_size(aBytes);
  if (aBytesSz < outBytes) return lean_mk_empty_byte_array(lean_box(0));
  const float *a = f32_ptr(aBytes);

  for (size_t o = 0; o < outer; o++) {
    const float *row = a + o * inner;
    float *outRow = dst + o * inner;

    float maxv = row[0];
    for (size_t j = 1; j < inner; j++) {
      const float v = row[j];
      maxv = (v >= maxv) ? v : maxv;
    }

    float sum = 0.0f;
    for (size_t j = 0; j < inner; j++) {
      const float shifted = row[j] - maxv;
      const float e = exp2f(shifted * scale);
      outRow[j] = e;
      sum += e;
    }
    const float inv = 1.0f / sum;
    for (size_t j = 0; j < inner; j++) {
      outRow[j] *= inv;
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_logsoftmax_last_f32(
    b_lean_obj_arg aBytes,
    b_lean_obj_arg outerObj,
    b_lean_obj_arg innerObj,
    uint32_t scaleBits,
    uint32_t ln2Bits) {
  if (!lean_is_scalar(outerObj) || !lean_is_scalar(innerObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t outer = lean_unbox(outerObj);
  const size_t inner = lean_unbox(innerObj);
  const float scale = f32_from_bits(scaleBits);
  const float ln2 = f32_from_bits(ln2Bits);

  const size_t outFloats = outer * inner;
  const size_t outBytes = outFloats * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outFloats == 0) return out;

  const size_t aBytesSz = lean_sarray_size(aBytes);
  if (aBytesSz < outBytes) return lean_mk_empty_byte_array(lean_box(0));
  const float *a = f32_ptr(aBytes);

  for (size_t o = 0; o < outer; o++) {
    const float *row = a + o * inner;
    float *outRow = dst + o * inner;

    float maxv = row[0];
    for (size_t j = 1; j < inner; j++) {
      const float v = row[j];
      maxv = (v >= maxv) ? v : maxv;
    }

    float sum = 0.0f;
    for (size_t j = 0; j < inner; j++) {
      const float shifted = row[j] - maxv;
      sum += exp2f(shifted * scale);
    }
    const float logSumNat = log2f(sum) * ln2;

    for (size_t j = 0; j < inner; j++) {
      const float shifted = row[j] - maxv;
      outRow[j] = shifted - logSumNat;
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_transpose2d_f32(b_lean_obj_arg aBytes, b_lean_obj_arg mObj, b_lean_obj_arg nObj) {
  if (!lean_is_scalar(mObj) || !lean_is_scalar(nObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t m = lean_unbox(mObj);
  const size_t n = lean_unbox(nObj);
  const size_t elems = m * n;
  const size_t outBytes = elems * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  const float *a = f32_ptr(aBytes);
  float *dst = f32_ptr_mut(out);

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      dst[j * m + i] = a[i * n + j];
    }
  }
  return out;
}

LEAN_EXPORT lean_obj_res tg4_permute_f32(b_lean_obj_arg aBytes, b_lean_obj_arg aShapeObj, b_lean_obj_arg permObj) {
  const size_t rank = lean_array_size(aShapeObj);
  fprintf(stderr, "[DEBUG] tg4_permute_f32: rank=%zu\n", rank);
  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *perm = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aStridesPerm = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_shape_dims(aShapeObj, aDims)) {
    fprintf(stderr, "[DEBUG] tg4_permute_f32: read_shape_dims failed\n");
    return lean_mk_empty_byte_array(lean_box(0));
  }
  fprintf(stderr, "[DEBUG] tg4_permute_f32: aDims OK, trying perm\n");
  if (!read_perm(permObj, rank, perm)) {
    fprintf(stderr, "[DEBUG] tg4_permute_f32: read_perm failed\n");
    return lean_mk_empty_byte_array(lean_box(0));
  }
  fprintf(stderr, "[DEBUG] tg4_permute_f32: both OK\n");

  for (size_t i = 0; i < rank; i++) outDims[i] = aDims[perm[i]];

  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  const size_t aNumel = prod_dims(aDims, rank);
  if (lean_sarray_size(aBytes) < aNumel * 4) return lean_mk_empty_byte_array(lean_box(0));

  contiguous_strides(aDims, rank, aStrides);
  for (size_t i = 0; i < rank; i++) aStridesPerm[i] = aStrides[perm[i]];
  for (size_t i = 0; i < rank; i++) idx[i] = 0;

  const float *a = f32_ptr(aBytes);
  size_t aOff = 0;
  for (size_t i = 0; i < outNumel; i++) {
    dst[i] = a[aOff];
    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aOff += aStridesPerm[d];
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      aOff -= aStridesPerm[d] * outDims[d];
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_permute_u8(b_lean_obj_arg aBytes, b_lean_obj_arg aShapeObj, b_lean_obj_arg permObj) {
  const size_t rank = lean_array_size(aShapeObj);
  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *perm = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aStridesPerm = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_shape_dims(aShapeObj, aDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_perm(permObj, rank, perm)) return lean_mk_empty_byte_array(lean_box(0));

  for (size_t i = 0; i < rank; i++) outDims[i] = aDims[perm[i]];

  const size_t outNumel = prod_dims(outDims, rank);
  lean_obj_res out = lean_alloc_sarray(1, outNumel, outNumel);
  uint8_t *dst = (uint8_t *)lean_sarray_cptr(out); // NOLINT
  if (outNumel == 0) return out;

  const size_t aNumel = prod_dims(aDims, rank);
  if (lean_sarray_size(aBytes) < aNumel) return lean_mk_empty_byte_array(lean_box(0));

  contiguous_strides(aDims, rank, aStrides);
  for (size_t i = 0; i < rank; i++) aStridesPerm[i] = aStrides[perm[i]];
  for (size_t i = 0; i < rank; i++) idx[i] = 0;

  const uint8_t *a = (const uint8_t *)lean_sarray_cptr(aBytes); // NOLINT
  size_t aOff = 0;
  for (size_t i = 0; i < outNumel; i++) {
    dst[i] = a[aOff];
    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aOff += aStridesPerm[d];
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      aOff -= aStridesPerm[d] * outDims[d];
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_pad_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg aShapeObj,
    b_lean_obj_arg padLeftObj, b_lean_obj_arg padRightObj) {
  const size_t rank = lean_array_size(aShapeObj);
  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *padL = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *padR = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *outStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_shape_dims(aShapeObj, aDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_nat_array_len(padLeftObj, rank, padL)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_nat_array_len(padRightObj, rank, padR)) return lean_mk_empty_byte_array(lean_box(0));

  for (size_t i = 0; i < rank; i++) outDims[i] = aDims[i] + padL[i] + padR[i];

  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = (float *)lean_sarray_cptr(out); // NOLINT
  for (size_t i = 0; i < outNumel; i++) dst[i] = 0.0f;

  const size_t aNumel = prod_dims(aDims, rank);
  if (aNumel == 0) return out;
  if (lean_sarray_size(aBytes) < aNumel * 4) return lean_mk_empty_byte_array(lean_box(0));

  contiguous_strides(aDims, rank, aStrides);
  contiguous_strides(outDims, rank, outStrides);

  for (size_t i = 0; i < rank; i++) idx[i] = 0;

  const float *a = f32_ptr(aBytes);
  size_t aOff = 0;
  size_t outOff = 0;
  for (size_t i = 0; i < rank; i++) outOff += padL[i] * outStrides[i];

  for (size_t i = 0; i < aNumel; i++) {
    dst[outOff] = a[aOff];
    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aOff += aStrides[d];
      outOff += outStrides[d];
      if (idx[d] < aDims[d]) break;
      idx[d] = 0;
      aOff -= aStrides[d] * aDims[d];
      outOff -= outStrides[d] * aDims[d];
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_pad_u8(
    b_lean_obj_arg aBytes, b_lean_obj_arg aShapeObj,
    b_lean_obj_arg padLeftObj, b_lean_obj_arg padRightObj) {
  const size_t rank = lean_array_size(aShapeObj);
  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *padL = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *padR = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *outStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_shape_dims(aShapeObj, aDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_nat_array_len(padLeftObj, rank, padL)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_nat_array_len(padRightObj, rank, padR)) return lean_mk_empty_byte_array(lean_box(0));

  for (size_t i = 0; i < rank; i++) outDims[i] = aDims[i] + padL[i] + padR[i];

  const size_t outNumel = prod_dims(outDims, rank);
  lean_obj_res out = lean_alloc_sarray(1, outNumel, outNumel);
  uint8_t *dst = (uint8_t *)lean_sarray_cptr(out); // NOLINT
  for (size_t i = 0; i < outNumel; i++) dst[i] = 0;

  const size_t aNumel = prod_dims(aDims, rank);
  if (aNumel == 0) return out;
  if (lean_sarray_size(aBytes) < aNumel) return lean_mk_empty_byte_array(lean_box(0));

  contiguous_strides(aDims, rank, aStrides);
  contiguous_strides(outDims, rank, outStrides);

  for (size_t i = 0; i < rank; i++) idx[i] = 0;

  const uint8_t *a = (const uint8_t *)lean_sarray_cptr(aBytes); // NOLINT
  size_t aOff = 0;
  size_t outOff = 0;
  for (size_t i = 0; i < rank; i++) outOff += padL[i] * outStrides[i];

  for (size_t i = 0; i < aNumel; i++) {
    dst[outOff] = a[aOff];
    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aOff += aStrides[d];
      outOff += outStrides[d];
      if (idx[d] < aDims[d]) break;
      idx[d] = 0;
      aOff -= aStrides[d] * aDims[d];
      outOff -= outStrides[d] * aDims[d];
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_shrink_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg aShapeObj,
    b_lean_obj_arg startsObj, b_lean_obj_arg stopsObj) {
  const size_t rank = lean_array_size(aShapeObj);
  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *starts = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *stops = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_shape_dims(aShapeObj, aDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_nat_array_len(startsObj, rank, starts)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_nat_array_len(stopsObj, rank, stops)) return lean_mk_empty_byte_array(lean_box(0));

  for (size_t i = 0; i < rank; i++) {
    if (starts[i] > stops[i] || stops[i] > aDims[i]) return lean_mk_empty_byte_array(lean_box(0));
    outDims[i] = stops[i] - starts[i];
  }

  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  const size_t aNumel = prod_dims(aDims, rank);
  if (lean_sarray_size(aBytes) < aNumel * 4) return lean_mk_empty_byte_array(lean_box(0));

  contiguous_strides(aDims, rank, aStrides);
  for (size_t i = 0; i < rank; i++) idx[i] = 0;

  size_t baseOff = 0;
  for (size_t i = 0; i < rank; i++) baseOff += starts[i] * aStrides[i];

  const float *a = f32_ptr(aBytes);
  size_t aOff = baseOff;
  for (size_t i = 0; i < outNumel; i++) {
    dst[i] = a[aOff];
    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aOff += aStrides[d];
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      aOff -= aStrides[d] * outDims[d];
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_shrink_u8(
    b_lean_obj_arg aBytes, b_lean_obj_arg aShapeObj,
    b_lean_obj_arg startsObj, b_lean_obj_arg stopsObj) {
  const size_t rank = lean_array_size(aShapeObj);
  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *starts = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *stops = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_shape_dims(aShapeObj, aDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_nat_array_len(startsObj, rank, starts)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_nat_array_len(stopsObj, rank, stops)) return lean_mk_empty_byte_array(lean_box(0));

  for (size_t i = 0; i < rank; i++) {
    if (starts[i] > stops[i] || stops[i] > aDims[i]) return lean_mk_empty_byte_array(lean_box(0));
    outDims[i] = stops[i] - starts[i];
  }

  const size_t outNumel = prod_dims(outDims, rank);
  lean_obj_res out = lean_alloc_sarray(1, outNumel, outNumel);
  uint8_t *dst = (uint8_t *)lean_sarray_cptr(out); // NOLINT
  if (outNumel == 0) return out;

  const size_t aNumel = prod_dims(aDims, rank);
  if (lean_sarray_size(aBytes) < aNumel) return lean_mk_empty_byte_array(lean_box(0));

  contiguous_strides(aDims, rank, aStrides);
  for (size_t i = 0; i < rank; i++) idx[i] = 0;

  size_t baseOff = 0;
  for (size_t i = 0; i < rank; i++) baseOff += starts[i] * aStrides[i];

  const uint8_t *a = (const uint8_t *)lean_sarray_cptr(aBytes); // NOLINT
  size_t aOff = baseOff;
  for (size_t i = 0; i < outNumel; i++) {
    dst[i] = a[aOff];
    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aOff += aStrides[d];
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      aOff -= aStrides[d] * outDims[d];
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_cat_f32(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg inputShapesObj,
    b_lean_obj_arg axisObj,
    b_lean_obj_arg outShapeObj) {
  if (!lean_is_scalar(axisObj)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t axis = lean_unbox(axisObj);
  const size_t nInputs = lean_array_size(inputsObj);
  if (nInputs == 0 || lean_array_size(inputShapesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t rank = lean_array_size(outShapeObj);
  if (rank == 0 || axis >= rank) return lean_mk_empty_byte_array(lean_box(0));

  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *inDims = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  const float **inPtrs = LEAN_ALLOCA(sizeof(float *) * nInputs); // NOLINT

  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));

  size_t axisSum = 0;
  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg shapeObj = lean_array_get_core(inputShapesObj, i);
    if (lean_array_size(shapeObj) != rank) return lean_mk_empty_byte_array(lean_box(0));
    size_t *dims = inDims + i * rank;
    if (!read_shape_dims(shapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));
    for (size_t d = 0; d < rank; d++) {
      if (d != axis && dims[d] != outDims[d]) return lean_mk_empty_byte_array(lean_box(0));
    }
    axisSum += dims[axis];
    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    const size_t numel = prod_dims(dims, rank);
    if (lean_sarray_size(bytesObj) < numel * 4) return lean_mk_empty_byte_array(lean_box(0));
    inPtrs[i] = f32_ptr(bytesObj);
  }

  const size_t outAxis = outDims[axis];
  if (axisSum != outAxis) return lean_mk_empty_byte_array(lean_box(0));

  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  size_t outer = 1;
  for (size_t d = 0; d < axis; d++) outer *= outDims[d];
  size_t inner = 1;
  for (size_t d = axis + 1; d < rank; d++) inner *= outDims[d];

  for (size_t o = 0; o < outer; o++) {
    size_t outBase = o * outAxis * inner;
    size_t axisOff = 0;
    for (size_t i = 0; i < nInputs; i++) {
      const size_t *dims = inDims + i * rank;
      const size_t axisDim = dims[axis];
      const size_t block = axisDim * inner;
      const size_t inBase = o * axisDim * inner;
      for (size_t j = 0; j < block; j++) {
        dst[outBase + axisOff * inner + j] = inPtrs[i][inBase + j];
      }
      axisOff += axisDim;
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_cat_u8(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg inputShapesObj,
    b_lean_obj_arg axisObj,
    b_lean_obj_arg outShapeObj) {
  if (!lean_is_scalar(axisObj)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t axis = lean_unbox(axisObj);
  const size_t nInputs = lean_array_size(inputsObj);
  if (nInputs == 0 || lean_array_size(inputShapesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t rank = lean_array_size(outShapeObj);
  if (rank == 0 || axis >= rank) return lean_mk_empty_byte_array(lean_box(0));

  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *inDims = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  const uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT

  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));

  size_t axisSum = 0;
  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg shapeObj = lean_array_get_core(inputShapesObj, i);
    if (lean_array_size(shapeObj) != rank) return lean_mk_empty_byte_array(lean_box(0));
    size_t *dims = inDims + i * rank;
    if (!read_shape_dims(shapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));
    for (size_t d = 0; d < rank; d++) {
      if (d != axis && dims[d] != outDims[d]) return lean_mk_empty_byte_array(lean_box(0));
    }
    axisSum += dims[axis];
    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    const size_t numel = prod_dims(dims, rank);
    if (lean_sarray_size(bytesObj) < numel) return lean_mk_empty_byte_array(lean_box(0));
    inPtrs[i] = (const uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT
  }

  const size_t outAxis = outDims[axis];
  if (axisSum != outAxis) return lean_mk_empty_byte_array(lean_box(0));

  const size_t outNumel = prod_dims(outDims, rank);
  lean_obj_res out = lean_alloc_sarray(1, outNumel, outNumel);
  uint8_t *dst = (uint8_t *)lean_sarray_cptr(out); // NOLINT
  if (outNumel == 0) return out;

  size_t outer = 1;
  for (size_t d = 0; d < axis; d++) outer *= outDims[d];
  size_t inner = 1;
  for (size_t d = axis + 1; d < rank; d++) inner *= outDims[d];

  for (size_t o = 0; o < outer; o++) {
    size_t outBase = o * outAxis * inner;
    size_t axisOff = 0;
    for (size_t i = 0; i < nInputs; i++) {
      const size_t *dims = inDims + i * rank;
      const size_t axisDim = dims[axis];
      const size_t block = axisDim * inner;
      const size_t inBase = o * axisDim * inner;
      for (size_t j = 0; j < block; j++) {
        dst[outBase + axisOff * inner + j] = inPtrs[i][inBase + j];
      }
      axisOff += axisDim;
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_cat_bytes(
    b_lean_obj_arg inputsObj,
    b_lean_obj_arg inputShapesObj,
    b_lean_obj_arg axisObj,
    b_lean_obj_arg outShapeObj,
    b_lean_obj_arg elemSizeObj) {
  if (!lean_is_scalar(axisObj) || !lean_is_scalar(elemSizeObj)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t axis = lean_unbox(axisObj);
  const size_t elemSize = lean_unbox(elemSizeObj);
  if (elemSize == 0) return lean_mk_empty_byte_array(lean_box(0));

  const size_t nInputs = lean_array_size(inputsObj);
  if (nInputs == 0 || lean_array_size(inputShapesObj) != nInputs) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t rank = lean_array_size(outShapeObj);
  if (rank == 0 || axis >= rank) return lean_mk_empty_byte_array(lean_box(0));

  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *inDims = LEAN_ALLOCA(sizeof(size_t) * nInputs * rank); // NOLINT
  const uint8_t **inPtrs = LEAN_ALLOCA(sizeof(uint8_t *) * nInputs); // NOLINT

  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));

  size_t axisSum = 0;
  for (size_t i = 0; i < nInputs; i++) {
    b_lean_obj_arg shapeObj = lean_array_get_core(inputShapesObj, i);
    if (lean_array_size(shapeObj) != rank) return lean_mk_empty_byte_array(lean_box(0));
    size_t *dims = inDims + i * rank;
    if (!read_shape_dims(shapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));
    for (size_t d = 0; d < rank; d++) {
      if (d != axis && dims[d] != outDims[d]) return lean_mk_empty_byte_array(lean_box(0));
    }
    axisSum += dims[axis];
    b_lean_obj_arg bytesObj = lean_array_get_core(inputsObj, i);
    const size_t numel = prod_dims(dims, rank);
    if (lean_sarray_size(bytesObj) < numel * elemSize) return lean_mk_empty_byte_array(lean_box(0));
    inPtrs[i] = (const uint8_t *)lean_sarray_cptr(bytesObj); // NOLINT
  }

  const size_t outAxis = outDims[axis];
  if (axisSum != outAxis) return lean_mk_empty_byte_array(lean_box(0));

  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * elemSize;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  uint8_t *dst = (uint8_t *)lean_sarray_cptr(out); // NOLINT
  if (outNumel == 0) return out;

  size_t outer = 1;
  for (size_t d = 0; d < axis; d++) outer *= outDims[d];
  size_t inner = 1;
  for (size_t d = axis + 1; d < rank; d++) inner *= outDims[d];

  for (size_t o = 0; o < outer; o++) {
    size_t outBase = o * outAxis * inner * elemSize;
    size_t axisOff = 0;
    for (size_t i = 0; i < nInputs; i++) {
      const size_t *dims = inDims + i * rank;
      const size_t axisDim = dims[axis];
      const size_t blockElems = axisDim * inner;
      const size_t blockBytes = blockElems * elemSize;
      const size_t inBase = o * axisDim * inner * elemSize;
      const size_t outOff = outBase + axisOff * inner * elemSize;
      for (size_t j = 0; j < blockBytes; j++) {
        dst[outOff + j] = inPtrs[i][inBase + j];
      }
      axisOff += axisDim;
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_flip_f32(b_lean_obj_arg aBytes, b_lean_obj_arg aShapeObj, b_lean_obj_arg axesObj) {
  const size_t rank = lean_array_size(aShapeObj);
  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  int64_t *aStridesSigned = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  bool *flipMask = LEAN_ALLOCA(sizeof(bool) * rank); // NOLINT

  if (!read_shape_dims(aShapeObj, aDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_axes_mask(axesObj, rank, flipMask)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t aNumel = prod_dims(aDims, rank);
  const size_t outBytes = aNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (aNumel == 0) return out;
  if (lean_sarray_size(aBytes) < aNumel * 4) return lean_mk_empty_byte_array(lean_box(0));

  contiguous_strides(aDims, rank, aStrides);
  int64_t baseOff = 0;
  for (size_t i = 0; i < rank; i++) {
    if (flipMask[i] && aDims[i] > 0) baseOff += (int64_t)(aDims[i] - 1) * (int64_t)aStrides[i];
    aStridesSigned[i] = flipMask[i] ? -(int64_t)aStrides[i] : (int64_t)aStrides[i];
    idx[i] = 0;
  }

  const float *a = f32_ptr(aBytes);
  int64_t aOff = baseOff;
  for (size_t i = 0; i < aNumel; i++) {
    dst[i] = a[(size_t)aOff];
    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aOff += aStridesSigned[d];
      if (idx[d] < aDims[d]) break;
      idx[d] = 0;
      aOff -= aStridesSigned[d] * (int64_t)aDims[d];
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_flip_u8(b_lean_obj_arg aBytes, b_lean_obj_arg aShapeObj, b_lean_obj_arg axesObj) {
  const size_t rank = lean_array_size(aShapeObj);
  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aStrides = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  int64_t *aStridesSigned = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  bool *flipMask = LEAN_ALLOCA(sizeof(bool) * rank); // NOLINT

  if (!read_shape_dims(aShapeObj, aDims)) return lean_mk_empty_byte_array(lean_box(0));
  if (!read_axes_mask(axesObj, rank, flipMask)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t aNumel = prod_dims(aDims, rank);
  lean_obj_res out = lean_alloc_sarray(1, aNumel, aNumel);
  uint8_t *dst = (uint8_t *)lean_sarray_cptr(out); // NOLINT
  if (aNumel == 0) return out;
  if (lean_sarray_size(aBytes) < aNumel) return lean_mk_empty_byte_array(lean_box(0));

  contiguous_strides(aDims, rank, aStrides);
  int64_t baseOff = 0;
  for (size_t i = 0; i < rank; i++) {
    if (flipMask[i] && aDims[i] > 0) baseOff += (int64_t)(aDims[i] - 1) * (int64_t)aStrides[i];
    aStridesSigned[i] = flipMask[i] ? -(int64_t)aStrides[i] : (int64_t)aStrides[i];
    idx[i] = 0;
  }

  const uint8_t *a = (const uint8_t *)lean_sarray_cptr(aBytes); // NOLINT
  int64_t aOff = baseOff;
  for (size_t i = 0; i < aNumel; i++) {
    dst[i] = a[(size_t)aOff];
    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aOff += aStridesSigned[d];
      if (idx[d] < aDims[d]) break;
      idx[d] = 0;
      aOff -= aStridesSigned[d] * (int64_t)aDims[d];
    }
  }

  return out;
}

// FloatArray (f64) -> ByteArray (raw f32 bytes).
LEAN_EXPORT lean_obj_res tg4_pack_f32_from_f64(b_lean_obj_arg aObj) {
  const size_t n = lean_sarray_size(aObj);
  const size_t outBytes = n * 4;

  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = (float *)lean_sarray_cptr(out); // NOLINT
  const double *src = lean_float_array_cptr(aObj);

  for (size_t i = 0; i < n; i++) dst[i] = (float)src[i];
  return out;
}

// ByteArray (raw f32 bytes) -> FloatArray (f64).
LEAN_EXPORT lean_obj_res tg4_unpack_f64_from_f32(b_lean_obj_arg bytesObj) {
  const size_t inBytes = lean_sarray_size(bytesObj);
  if ((inBytes & 3) != 0) {
    return lean_alloc_sarray(sizeof(double), 0, 0); // NOLINT
  }
  const size_t n = inBytes / 4;

  lean_obj_res out = lean_alloc_sarray(sizeof(double), n, n); // NOLINT
  const float *src = (const float *)lean_sarray_cptr(bytesObj); // NOLINT
  double *dst = lean_float_array_cptr(out);

  for (size_t i = 0; i < n; i++) dst[i] = (double)src[i];
  return out;
}

#if defined(__clang__) || defined(__GNUC__)
#define TG4_HAS_V4F 1
typedef float tg4_v4f __attribute__((vector_size(16)));
static inline tg4_v4f tg4_v4f_load(const float *p) {
  tg4_v4f v;
  __builtin_memcpy(&v, p, sizeof(v)); // NOLINT
  return v;
}
static inline void tg4_v4f_store(float *p, tg4_v4f v) { __builtin_memcpy(p, &v, sizeof(v)); } // NOLINT
static inline tg4_v4f tg4_v4f_splat(float x) { return (tg4_v4f){x, x, x, x}; }
#else
#define TG4_HAS_V4F 0
#endif

static inline void tg4_gemm_blocked_f32(float *C, const float *A, const float *B, size_t m, size_t k, size_t n) {
  const size_t BM = 32;
  const size_t BN = 64;
  const size_t BK = 32;

  for (size_t i0 = 0; i0 < m; i0 += BM) {
    const size_t iMax = min_size(i0 + BM, m);
    for (size_t k0 = 0; k0 < k; k0 += BK) {
      const size_t kMax = min_size(k0 + BK, k);
      for (size_t j0 = 0; j0 < n; j0 += BN) {
        const size_t jMax = min_size(j0 + BN, n);
        for (size_t i = i0; i < iMax; i++) {
          const float *aRow = A + i * k;
          float *cRow = C + i * n;
          size_t j = j0;

#if TG4_HAS_V4F
          for (; j + 16 <= jMax; j += 16) {
            tg4_v4f acc0 = tg4_v4f_load(cRow + j + 0);
            tg4_v4f acc1 = tg4_v4f_load(cRow + j + 4);
            tg4_v4f acc2 = tg4_v4f_load(cRow + j + 8);
            tg4_v4f acc3 = tg4_v4f_load(cRow + j + 12);
            for (size_t t = k0; t < kMax; t++) {
              const tg4_v4f av = tg4_v4f_splat(aRow[t]);
              const float *bRow = B + t * n + j;
              acc0 += tg4_v4f_load(bRow + 0) * av;
              acc1 += tg4_v4f_load(bRow + 4) * av;
              acc2 += tg4_v4f_load(bRow + 8) * av;
              acc3 += tg4_v4f_load(bRow + 12) * av;
            }
            tg4_v4f_store(cRow + j + 0, acc0);
            tg4_v4f_store(cRow + j + 4, acc1);
            tg4_v4f_store(cRow + j + 8, acc2);
            tg4_v4f_store(cRow + j + 12, acc3);
          }
          for (; j + 4 <= jMax; j += 4) {
            tg4_v4f acc = tg4_v4f_load(cRow + j);
            for (size_t t = k0; t < kMax; t++) {
              const tg4_v4f av = tg4_v4f_splat(aRow[t]);
              const float *bRow = B + t * n + j;
              acc += tg4_v4f_load(bRow) * av;
            }
            tg4_v4f_store(cRow + j, acc);
          }
#else
          for (size_t t = k0; t < kMax; t++) {
            const float a = aRow[t];
            const float *bRow = B + t * n;
            size_t j2 = j0;
            for (; j2 + 4 <= jMax; j2 += 4) {
              cRow[j2 + 0] += a * bRow[j2 + 0];
              cRow[j2 + 1] += a * bRow[j2 + 1];
              cRow[j2 + 2] += a * bRow[j2 + 2];
              cRow[j2 + 3] += a * bRow[j2 + 3];
            }
            for (; j2 < jMax; j2++) {
              cRow[j2] += a * bRow[j2];
            }
          }
          j = jMax;
#endif
          for (; j < jMax; j++) {
            float acc = cRow[j];
            for (size_t t = k0; t < kMax; t++) {
              acc += aRow[t] * B[t * n + j];
            }
            cRow[j] = acc;
          }
        }
      }
    }
  }
}

// GEMM when B is stored transposed as (n, k) row-major, i.e. we want:
//   C(m, n) += A(m, k) @ (B_T(n, k))^T
static inline void tg4_gemm_blocked_bt_f32(float *C, const float *A, const float *B_T, size_t m, size_t k, size_t n) {
  const size_t BM = 32;
  const size_t BN = 64;
  const size_t BK = 32;

  float *Bpack = LEAN_ALLOCA(sizeof(float) * BK * BN); // NOLINT

  for (size_t j0 = 0; j0 < n; j0 += BN) {
    const size_t jMax = min_size(j0 + BN, n);
    const size_t jLen = jMax - j0;
    for (size_t k0 = 0; k0 < k; k0 += BK) {
      const size_t kMax = min_size(k0 + BK, k);
      const size_t kLen = kMax - k0;

      // Pack and transpose B_T block into row-major (kLen, jLen) with row stride BN.
      for (size_t j = 0; j < jLen; j++) {
        const float *srcRow = B_T + (j0 + j) * k + k0;
        for (size_t t = 0; t < kLen; t++) {
          Bpack[t * BN + j] = srcRow[t];
        }
      }

      for (size_t i0 = 0; i0 < m; i0 += BM) {
        const size_t iMax = min_size(i0 + BM, m);
        for (size_t i = i0; i < iMax; i++) {
          const float *aRow = A + i * k + k0;
          float *cRow = C + i * n + j0;
          size_t j = 0;

#if TG4_HAS_V4F
          for (; j + 16 <= jLen; j += 16) {
            tg4_v4f acc0 = tg4_v4f_load(cRow + j + 0);
            tg4_v4f acc1 = tg4_v4f_load(cRow + j + 4);
            tg4_v4f acc2 = tg4_v4f_load(cRow + j + 8);
            tg4_v4f acc3 = tg4_v4f_load(cRow + j + 12);
            for (size_t t = 0; t < kLen; t++) {
              const tg4_v4f av = tg4_v4f_splat(aRow[t]);
              const float *bRow = Bpack + t * BN + j;
              acc0 += tg4_v4f_load(bRow + 0) * av;
              acc1 += tg4_v4f_load(bRow + 4) * av;
              acc2 += tg4_v4f_load(bRow + 8) * av;
              acc3 += tg4_v4f_load(bRow + 12) * av;
            }
            tg4_v4f_store(cRow + j + 0, acc0);
            tg4_v4f_store(cRow + j + 4, acc1);
            tg4_v4f_store(cRow + j + 8, acc2);
            tg4_v4f_store(cRow + j + 12, acc3);
          }
          for (; j + 4 <= jLen; j += 4) {
            tg4_v4f acc = tg4_v4f_load(cRow + j);
            for (size_t t = 0; t < kLen; t++) {
              const tg4_v4f av = tg4_v4f_splat(aRow[t]);
              const float *bRow = Bpack + t * BN + j;
              acc += tg4_v4f_load(bRow) * av;
            }
            tg4_v4f_store(cRow + j, acc);
          }
#else
          for (size_t t = 0; t < kLen; t++) {
            const float a = aRow[t];
            const float *bRow = Bpack + t * BN;
            size_t j2 = 0;
            for (; j2 + 4 <= jLen; j2 += 4) {
              cRow[j2 + 0] += a * bRow[j2 + 0];
              cRow[j2 + 1] += a * bRow[j2 + 1];
              cRow[j2 + 2] += a * bRow[j2 + 2];
              cRow[j2 + 3] += a * bRow[j2 + 3];
            }
            for (; j2 < jLen; j2++) {
              cRow[j2] += a * bRow[j2];
            }
          }
          j = jLen;
#endif
          for (; j < jLen; j++) {
            float acc = cRow[j];
            for (size_t t = 0; t < kLen; t++) {
              acc += aRow[t] * Bpack[t * BN + j];
            }
            cRow[j] = acc;
          }
        }
      }
    }
  }
}

static inline void tg4_gemm_blocked_alpha_f32(float *C, const float *A, const float *B, size_t m, size_t k, size_t n, float alpha) {
  const size_t BM = 32;
  const size_t BN = 64;
  const size_t BK = 32;

  for (size_t i0 = 0; i0 < m; i0 += BM) {
    const size_t iMax = min_size(i0 + BM, m);
    for (size_t k0 = 0; k0 < k; k0 += BK) {
      const size_t kMax = min_size(k0 + BK, k);
      for (size_t j0 = 0; j0 < n; j0 += BN) {
        const size_t jMax = min_size(j0 + BN, n);
        for (size_t i = i0; i < iMax; i++) {
          const float *aRow = A + i * k;
          float *cRow = C + i * n;
          size_t j = j0;

#if TG4_HAS_V4F
          for (; j + 16 <= jMax; j += 16) {
            tg4_v4f acc0 = tg4_v4f_load(cRow + j + 0);
            tg4_v4f acc1 = tg4_v4f_load(cRow + j + 4);
            tg4_v4f acc2 = tg4_v4f_load(cRow + j + 8);
            tg4_v4f acc3 = tg4_v4f_load(cRow + j + 12);
            for (size_t t = k0; t < kMax; t++) {
              const tg4_v4f av = tg4_v4f_splat(aRow[t] * alpha);
              const float *bRow = B + t * n + j;
              acc0 += tg4_v4f_load(bRow + 0) * av;
              acc1 += tg4_v4f_load(bRow + 4) * av;
              acc2 += tg4_v4f_load(bRow + 8) * av;
              acc3 += tg4_v4f_load(bRow + 12) * av;
            }
            tg4_v4f_store(cRow + j + 0, acc0);
            tg4_v4f_store(cRow + j + 4, acc1);
            tg4_v4f_store(cRow + j + 8, acc2);
            tg4_v4f_store(cRow + j + 12, acc3);
          }
          for (; j + 4 <= jMax; j += 4) {
            tg4_v4f acc = tg4_v4f_load(cRow + j);
            for (size_t t = k0; t < kMax; t++) {
              const tg4_v4f av = tg4_v4f_splat(aRow[t] * alpha);
              const float *bRow = B + t * n + j;
              acc += tg4_v4f_load(bRow) * av;
            }
            tg4_v4f_store(cRow + j, acc);
          }
#else
          for (size_t t = k0; t < kMax; t++) {
            const float a = aRow[t] * alpha;
            const float *bRow = B + t * n;
            size_t j2 = j0;
            for (; j2 + 4 <= jMax; j2 += 4) {
              cRow[j2 + 0] += a * bRow[j2 + 0];
              cRow[j2 + 1] += a * bRow[j2 + 1];
              cRow[j2 + 2] += a * bRow[j2 + 2];
              cRow[j2 + 3] += a * bRow[j2 + 3];
            }
            for (; j2 < jMax; j2++) {
              cRow[j2] += a * bRow[j2];
            }
          }
          j = jMax;
#endif
          for (; j < jMax; j++) {
            float acc = cRow[j];
            for (size_t t = k0; t < kMax; t++) {
              acc += (aRow[t] * alpha) * B[t * n + j];
            }
            cRow[j] = acc;
          }
        }
      }
    }
  }
}

static inline void tg4_gemm_blocked_bt_alpha_f32(float *C, const float *A, const float *B_T, size_t m, size_t k, size_t n, float alpha) {
  const size_t BM = 32;
  const size_t BN = 64;
  const size_t BK = 32;

  float *Bpack = LEAN_ALLOCA(sizeof(float) * BK * BN); // NOLINT

  for (size_t j0 = 0; j0 < n; j0 += BN) {
    const size_t jMax = min_size(j0 + BN, n);
    const size_t jLen = jMax - j0;
    for (size_t k0 = 0; k0 < k; k0 += BK) {
      const size_t kMax = min_size(k0 + BK, k);
      const size_t kLen = kMax - k0;

      for (size_t j = 0; j < jLen; j++) {
        const float *srcRow = B_T + (j0 + j) * k + k0;
        for (size_t t = 0; t < kLen; t++) {
          Bpack[t * BN + j] = srcRow[t];
        }
      }

      for (size_t i0 = 0; i0 < m; i0 += BM) {
        const size_t iMax = min_size(i0 + BM, m);
        for (size_t i = i0; i < iMax; i++) {
          const float *aRow = A + i * k + k0;
          float *cRow = C + i * n + j0;
          size_t j = 0;

#if TG4_HAS_V4F
          for (; j + 16 <= jLen; j += 16) {
            tg4_v4f acc0 = tg4_v4f_load(cRow + j + 0);
            tg4_v4f acc1 = tg4_v4f_load(cRow + j + 4);
            tg4_v4f acc2 = tg4_v4f_load(cRow + j + 8);
            tg4_v4f acc3 = tg4_v4f_load(cRow + j + 12);
            for (size_t t = 0; t < kLen; t++) {
              const tg4_v4f av = tg4_v4f_splat(aRow[t] * alpha);
              const float *bRow = Bpack + t * BN + j;
              acc0 += tg4_v4f_load(bRow + 0) * av;
              acc1 += tg4_v4f_load(bRow + 4) * av;
              acc2 += tg4_v4f_load(bRow + 8) * av;
              acc3 += tg4_v4f_load(bRow + 12) * av;
            }
            tg4_v4f_store(cRow + j + 0, acc0);
            tg4_v4f_store(cRow + j + 4, acc1);
            tg4_v4f_store(cRow + j + 8, acc2);
            tg4_v4f_store(cRow + j + 12, acc3);
          }
          for (; j + 4 <= jLen; j += 4) {
            tg4_v4f acc = tg4_v4f_load(cRow + j);
            for (size_t t = 0; t < kLen; t++) {
              const tg4_v4f av = tg4_v4f_splat(aRow[t] * alpha);
              const float *bRow = Bpack + t * BN + j;
              acc += tg4_v4f_load(bRow) * av;
            }
            tg4_v4f_store(cRow + j, acc);
          }
#else
          for (size_t t = 0; t < kLen; t++) {
            const float a = aRow[t] * alpha;
            const float *bRow = Bpack + t * BN;
            size_t j2 = 0;
            for (; j2 + 4 <= jLen; j2 += 4) {
              cRow[j2 + 0] += a * bRow[j2 + 0];
              cRow[j2 + 1] += a * bRow[j2 + 1];
              cRow[j2 + 2] += a * bRow[j2 + 2];
              cRow[j2 + 3] += a * bRow[j2 + 3];
            }
            for (; j2 < jLen; j2++) {
              cRow[j2] += a * bRow[j2];
            }
          }
          j = jLen;
#endif
          for (; j < jLen; j++) {
            float acc = cRow[j];
            for (size_t t = 0; t < kLen; t++) {
              acc += (aRow[t] * alpha) * Bpack[t * BN + j];
            }
            cRow[j] = acc;
          }
        }
      }
    }
  }
}

static inline bool validate_mask(const size_t *ms, const size_t *me, const size_t *dims, size_t rank) {
  for (size_t d = 0; d < rank; d++) {
    if (ms[d] > me[d] || me[d] > dims[d]) return false;
  }
  return true;
}

static inline bool is_full_mask(const size_t *ms, const size_t *me, const size_t *dims, size_t rank) {
  for (size_t d = 0; d < rank; d++) {
    if (ms[d] != 0 || me[d] != dims[d]) return false;
  }
  return true;
}

static inline bool view_range_in_bounds(int64_t offset, const int64_t *strides, const size_t *dims, size_t rank, size_t cap) {
  if (cap == 0) return false;
  int64_t minOff = offset;
  int64_t maxOff = offset;
  for (size_t d = 0; d < rank; d++) {
    const size_t dim = dims[d];
    if (dim == 0) continue;
    const int64_t span = (int64_t)(dim - 1) * strides[d];
    if (span >= 0) maxOff += span; else minOff += span;
  }
  return minOff >= 0 && (uint64_t)maxOff < cap;
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes,
    b_lean_obj_arg aStridesObj, uint64_t aOffsetBits, b_lean_obj_arg aMaskStartsObj, b_lean_obj_arg aMaskEndsObj,
    b_lean_obj_arg bStridesObj, uint64_t bOffsetBits, b_lean_obj_arg bMaskStartsObj, b_lean_obj_arg bMaskEndsObj,
    b_lean_obj_arg outShapeObj, b_lean_obj_arg kObj) {
  if (!lean_is_scalar(kObj)) return lean_mk_empty_byte_array(lean_box(0));
  const size_t k = lean_unbox(kObj);

  const size_t rank = lean_array_size(outShapeObj);
  if (rank < 2) return lean_mk_empty_byte_array(lean_box(0));
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t m = outDims[rank - 2];
  const size_t n = outDims[rank - 1];
  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  int64_t *aStrides = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  int64_t *bStrides = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  if (!read_i64_array_len(aStridesObj, rank, aStrides) || !read_i64_array_len(bStridesObj, rank, bStrides)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const int64_t aOffset = i64_from_u64(aOffsetBits);
  const int64_t bOffset = i64_from_u64(bOffsetBits);

  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d + 2 < rank; d++) {
    aDims[d] = outDims[d];
    bDims[d] = outDims[d];
  }
  aDims[rank - 2] = m;
  aDims[rank - 1] = k;
  bDims[rank - 2] = k;
  bDims[rank - 1] = n;

  size_t *aMaskStart = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aMaskEnd = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bMaskStart = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bMaskEnd = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_nat_array_len(aMaskStartsObj, rank, aMaskStart) || !read_nat_array_len(aMaskEndsObj, rank, aMaskEnd) ||
      !read_nat_array_len(bMaskStartsObj, rank, bMaskStart) || !read_nat_array_len(bMaskEndsObj, rank, bMaskEnd)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }
  if (!validate_mask(aMaskStart, aMaskEnd, aDims, rank) || !validate_mask(bMaskStart, bMaskEnd, bDims, rank)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const float *A = f32_ptr(aBytes);
  const float *B = f32_ptr(bBytes);
  const size_t aCap = f32_elems(aBytes);
  const size_t bCap = f32_elems(bBytes);

  const bool aFast = is_full_mask(aMaskStart, aMaskEnd, aDims, rank) && view_range_in_bounds(aOffset, aStrides, aDims, rank, aCap);
  const bool bFast = is_full_mask(bMaskStart, bMaskEnd, bDims, rank) && view_range_in_bounds(bOffset, bStrides, bDims, rank, bCap);

  const int64_t aStrideM = aStrides[rank - 2];
  const int64_t aStrideK = aStrides[rank - 1];
  const int64_t bStrideK = bStrides[rank - 2];
  const int64_t bStrideN = bStrides[rank - 1];

  const bool aRowMajor = aStrideK == 1 && aStrideM == (int64_t)k;
  const bool bRowMajor = bStrideN == 1 && bStrideK == (int64_t)n;
  const bool bTransposed = bStrideK == 1 && bStrideN == (int64_t)k;

  if (aFast && bFast && aRowMajor && (bRowMajor || bTransposed)) {
    size_t batchNumel = 1;
    for (size_t d = 0; d + 2 < rank; d++) batchNumel *= outDims[d];

    size_t *batchIdx = NULL;
    if (rank > 2) {
      batchIdx = LEAN_ALLOCA(sizeof(size_t) * (rank - 2)); // NOLINT
      for (size_t d = 0; d + 2 < rank; d++) batchIdx[d] = 0;
    }

    const size_t outFloats = m * n;
    int64_t aBatchBase = aOffset;
    int64_t bBatchBase = bOffset;

    for (size_t bi = 0; bi < batchNumel; bi++) {
      float *C = dst + bi * outFloats;
      const float *Ab = A + (size_t)aBatchBase;
      const float *Bb = B + (size_t)bBatchBase;

      for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;

      if (bRowMajor) {
        tg4_gemm_blocked_f32(C, Ab, Bb, m, k, n);
      } else {
        tg4_gemm_blocked_bt_f32(C, Ab, Bb, m, k, n);
      }

      for (size_t d = rank - 2; d-- > 0;) {
        batchIdx[d]++;
        aBatchBase += aStrides[d];
        bBatchBase += bStrides[d];
        if (batchIdx[d] < outDims[d]) break;
        batchIdx[d] = 0;
        aBatchBase -= aStrides[d] * (int64_t)outDims[d];
        bBatchBase -= bStrides[d] * (int64_t)outDims[d];
      }
    }
    return out;
  }

  int64_t *stepA = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  int64_t *stepB = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) {
    stepA[d] = (d == rank - 1) ? 0 : aStrides[d];
    stepB[d] = (d == rank - 2) ? 0 : bStrides[d];
  }

  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  int64_t aBase = aOffset;
  int64_t bBase = bOffset;

  if (aFast && bFast) {
    for (size_t o = 0; o < outNumel; o++) {
      float acc = 0.0f;
      for (size_t t = 0; t < k; t++) {
        const float av = A[(size_t)(aBase + (int64_t)t * aStrideK)];
        const float bv = B[(size_t)(bBase + (int64_t)t * bStrideK)];
        acc += av * bv;
      }
      dst[o] = acc;

      for (size_t d = rank; d-- > 0;) {
        idx[d]++;
        aBase += stepA[d];
        bBase += stepB[d];
        if (idx[d] < outDims[d]) break;
        idx[d] = 0;
        aBase -= stepA[d] * (int64_t)outDims[d];
        bBase -= stepB[d] * (int64_t)outDims[d];
      }
    }
    return out;
  }

  for (size_t o = 0; o < outNumel; o++) {
    uint8_t validAOuter = 1;
    for (size_t d = 0; d + 2 < rank; d++) {
      const size_t x = idx[d];
      if (x < aMaskStart[d] || x >= aMaskEnd[d]) {
        validAOuter = 0;
        break;
      }
    }
    if (validAOuter) {
      const size_t xm = idx[rank - 2];
      if (xm < aMaskStart[rank - 2] || xm >= aMaskEnd[rank - 2]) validAOuter = 0;
    }

    uint8_t validBOuter = 1;
    for (size_t d = 0; d + 2 < rank; d++) {
      const size_t x = idx[d];
      if (x < bMaskStart[d] || x >= bMaskEnd[d]) {
        validBOuter = 0;
        break;
      }
    }
    if (validBOuter) {
      const size_t xn = idx[rank - 1];
      if (xn < bMaskStart[rank - 1] || xn >= bMaskEnd[rank - 1]) validBOuter = 0;
    }

    float acc = 0.0f;
    for (size_t t = 0; t < k; t++) {
      float av = 0.0f;
      float bv = 0.0f;

      if (validAOuter && t >= aMaskStart[rank - 1] && t < aMaskEnd[rank - 1]) {
        const int64_t off = aBase + (int64_t)t * aStrideK;
        if (off >= 0 && (uint64_t)off < aCap) av = A[(size_t)off];
      }
      if (validBOuter && t >= bMaskStart[rank - 2] && t < bMaskEnd[rank - 2]) {
        const int64_t off = bBase + (int64_t)t * bStrideK;
        if (off >= 0 && (uint64_t)off < bCap) bv = B[(size_t)off];
      }

      acc += av * bv;
    }
    dst[o] = acc;

    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aBase += stepA[d];
      bBase += stepB[d];
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      aBase -= stepA[d] * (int64_t)outDims[d];
      bBase -= stepB[d] * (int64_t)outDims[d];
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_stack_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes,
    b_lean_obj_arg aStackShapesObj,
    b_lean_obj_arg aStackStridesObj,
    b_lean_obj_arg aStackOffsetsObj,
    b_lean_obj_arg aStackMaskStartsObj,
    b_lean_obj_arg aStackMaskEndsObj,
    b_lean_obj_arg bStackShapesObj,
    b_lean_obj_arg bStackStridesObj,
    b_lean_obj_arg bStackOffsetsObj,
    b_lean_obj_arg bStackMaskStartsObj,
    b_lean_obj_arg bStackMaskEndsObj,
    b_lean_obj_arg outShapeObj, b_lean_obj_arg kObj) {
  if (!lean_is_scalar(kObj)) return lean_mk_empty_byte_array(lean_box(0));
  const size_t k = lean_unbox(kObj);

  const size_t rank = lean_array_size(outShapeObj);
  if (rank < 2) return lean_mk_empty_byte_array(lean_box(0));
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t m = outDims[rank - 2];
  const size_t n = outDims[rank - 1];

  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  size_t maxRank = rank;

  const size_t aViewsLen = lean_array_size(aStackShapesObj);
  if (aViewsLen == 0 || lean_array_size(aStackStridesObj) != aViewsLen || lean_array_size(aStackOffsetsObj) != aViewsLen ||
      lean_array_size(aStackMaskStartsObj) != aViewsLen || lean_array_size(aStackMaskEndsObj) != aViewsLen) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  tg4_view_stack_view *aViews = LEAN_ALLOCA(sizeof(tg4_view_stack_view) * aViewsLen); // NOLINT
  for (size_t vi = 0; vi < aViewsLen; vi++) {
    b_lean_obj_arg shapeObj = lean_array_get_core(aStackShapesObj, vi);
    const size_t vRank = lean_array_size(shapeObj);

    size_t *dims = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
    if (!read_shape_dims(shapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));
    size_t *contig = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
    contiguous_strides(dims, vRank, contig);
    const size_t vNumel = prod_dims(dims, vRank);

    b_lean_obj_arg stridesObj = lean_array_get_core(aStackStridesObj, vi);
    int64_t *strides = LEAN_ALLOCA(sizeof(int64_t) * vRank); // NOLINT
    if (!read_i64_array_len(stridesObj, vRank, strides)) return lean_mk_empty_byte_array(lean_box(0));

    b_lean_obj_arg offObj = lean_array_get_core(aStackOffsetsObj, vi);
    if (!lean_is_ctor(offObj)) return lean_mk_empty_byte_array(lean_box(0));
    const int64_t offset = (int64_t)lean_unbox_uint64(offObj);

    b_lean_obj_arg msObj = lean_array_get_core(aStackMaskStartsObj, vi);
    b_lean_obj_arg meObj = lean_array_get_core(aStackMaskEndsObj, vi);
    size_t *maskStart = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
    size_t *maskEnd = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
    if (!read_nat_array_len(msObj, vRank, maskStart)) return lean_mk_empty_byte_array(lean_box(0));
    if (!read_nat_array_len(meObj, vRank, maskEnd)) return lean_mk_empty_byte_array(lean_box(0));
    for (size_t d = 0; d < vRank; d++) {
      const size_t ms = maskStart[d];
      const size_t me = maskEnd[d];
      if (ms > me || me > dims[d]) return lean_mk_empty_byte_array(lean_box(0));
    }

    aViews[vi].rank = vRank;
    aViews[vi].numel = vNumel;
    aViews[vi].contigStrides = contig;
    aViews[vi].strides = strides;
    aViews[vi].offset = offset;
    aViews[vi].maskStart = maskStart;
    aViews[vi].maskEnd = maskEnd;

    if (vRank > maxRank) maxRank = vRank;
  }

  const size_t bViewsLen = lean_array_size(bStackShapesObj);
  if (bViewsLen == 0 || lean_array_size(bStackStridesObj) != bViewsLen || lean_array_size(bStackOffsetsObj) != bViewsLen ||
      lean_array_size(bStackMaskStartsObj) != bViewsLen || lean_array_size(bStackMaskEndsObj) != bViewsLen) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  tg4_view_stack_view *bViews = LEAN_ALLOCA(sizeof(tg4_view_stack_view) * bViewsLen); // NOLINT
  for (size_t vi = 0; vi < bViewsLen; vi++) {
    b_lean_obj_arg shapeObj = lean_array_get_core(bStackShapesObj, vi);
    const size_t vRank = lean_array_size(shapeObj);

    size_t *dims = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
    if (!read_shape_dims(shapeObj, dims)) return lean_mk_empty_byte_array(lean_box(0));
    size_t *contig = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
    contiguous_strides(dims, vRank, contig);
    const size_t vNumel = prod_dims(dims, vRank);

    b_lean_obj_arg stridesObj = lean_array_get_core(bStackStridesObj, vi);
    int64_t *strides = LEAN_ALLOCA(sizeof(int64_t) * vRank); // NOLINT
    if (!read_i64_array_len(stridesObj, vRank, strides)) return lean_mk_empty_byte_array(lean_box(0));

    b_lean_obj_arg offObj = lean_array_get_core(bStackOffsetsObj, vi);
    if (!lean_is_ctor(offObj)) return lean_mk_empty_byte_array(lean_box(0));
    const int64_t offset = (int64_t)lean_unbox_uint64(offObj);

    b_lean_obj_arg msObj = lean_array_get_core(bStackMaskStartsObj, vi);
    b_lean_obj_arg meObj = lean_array_get_core(bStackMaskEndsObj, vi);
    size_t *maskStart = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
    size_t *maskEnd = LEAN_ALLOCA(sizeof(size_t) * vRank); // NOLINT
    if (!read_nat_array_len(msObj, vRank, maskStart)) return lean_mk_empty_byte_array(lean_box(0));
    if (!read_nat_array_len(meObj, vRank, maskEnd)) return lean_mk_empty_byte_array(lean_box(0));
    for (size_t d = 0; d < vRank; d++) {
      const size_t ms = maskStart[d];
      const size_t me = maskEnd[d];
      if (ms > me || me > dims[d]) return lean_mk_empty_byte_array(lean_box(0));
    }

    bViews[vi].rank = vRank;
    bViews[vi].numel = vNumel;
    bViews[vi].contigStrides = contig;
    bViews[vi].strides = strides;
    bViews[vi].offset = offset;
    bViews[vi].maskStart = maskStart;
    bViews[vi].maskEnd = maskEnd;

    if (vRank > maxRank) maxRank = vRank;
  }

  if (aViews[aViewsLen - 1].rank != rank || bViews[bViewsLen - 1].rank != rank) return lean_mk_empty_byte_array(lean_box(0));

  size_t *aTopDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bTopDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  b_lean_obj_arg aTopShapeObj = lean_array_get_core(aStackShapesObj, aViewsLen - 1);
  b_lean_obj_arg bTopShapeObj = lean_array_get_core(bStackShapesObj, bViewsLen - 1);
  if (!read_shape_dims(aTopShapeObj, aTopDims) || !read_shape_dims(bTopShapeObj, bTopDims)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  for (size_t d = 0; d + 2 < rank; d++) {
    if (aTopDims[d] != outDims[d] || bTopDims[d] != outDims[d]) return lean_mk_empty_byte_array(lean_box(0));
  }
  if (aTopDims[rank - 2] != m || aTopDims[rank - 1] != k) return lean_mk_empty_byte_array(lean_box(0));
  if (bTopDims[rank - 2] != k || bTopDims[rank - 1] != n) return lean_mk_empty_byte_array(lean_box(0));

  tg4_view_stack aStack;
  tg4_view_stack bStack;
  aStack.nViews = aViewsLen;
  aStack.views = aViews;
  bStack.nViews = bViewsLen;
  bStack.views = bViews;

  const float *A = f32_ptr(aBytes);
  const float *B = f32_ptr(bBytes);
  const size_t aCap = f32_elems(aBytes);
  const size_t bCap = f32_elems(bBytes);

  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  size_t *idxWork = LEAN_ALLOCA(sizeof(size_t) * maxRank); // NOLINT
  size_t *aIdx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bIdx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  for (size_t o = 0; o < outNumel; o++) {
    for (size_t d = 0; d < rank; d++) {
      aIdx[d] = idx[d];
      bIdx[d] = idx[d];
    }

    float acc = 0.0f;
    for (size_t t = 0; t < k; t++) {
      aIdx[rank - 1] = t;
      bIdx[rank - 2] = t;

      float av = 0.0f;
      float bv = 0.0f;

      int64_t aOff = 0;
      if (tg4_view_stack_offset(&aStack, aIdx, rank, idxWork, &aOff)) {
        if (aOff >= 0 && (uint64_t)aOff < aCap) av = A[(size_t)aOff];
      }

      int64_t bOff = 0;
      if (tg4_view_stack_offset(&bStack, bIdx, rank, idxWork, &bOff)) {
        if (bOff >= 0 && (uint64_t)bOff < bCap) bv = B[(size_t)bOff];
      }

      acc += av * bv;
    }
    dst[o] = acc;

    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_bias_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes, b_lean_obj_arg biasBytes,
    b_lean_obj_arg aStridesObj, uint64_t aOffsetBits, b_lean_obj_arg aMaskStartsObj, b_lean_obj_arg aMaskEndsObj,
    b_lean_obj_arg bStridesObj, uint64_t bOffsetBits, b_lean_obj_arg bMaskStartsObj, b_lean_obj_arg bMaskEndsObj,
    b_lean_obj_arg biasStridesObj, uint64_t biasOffsetBits, b_lean_obj_arg biasMaskStartsObj, b_lean_obj_arg biasMaskEndsObj,
    b_lean_obj_arg outShapeObj, b_lean_obj_arg kObj) {
  if (!lean_is_scalar(kObj)) return lean_mk_empty_byte_array(lean_box(0));
  const size_t k = lean_unbox(kObj);

  const size_t rank = lean_array_size(outShapeObj);
  if (rank < 2) return lean_mk_empty_byte_array(lean_box(0));
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t m = outDims[rank - 2];
  const size_t n = outDims[rank - 1];
  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  int64_t *aStrides = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  int64_t *bStrides = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  int64_t *biasStrides = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  if (!read_i64_array_len(aStridesObj, rank, aStrides) || !read_i64_array_len(bStridesObj, rank, bStrides) ||
      !read_i64_array_len(biasStridesObj, rank, biasStrides)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const int64_t aOffset = i64_from_u64(aOffsetBits);
  const int64_t bOffset = i64_from_u64(bOffsetBits);
  const int64_t biasOffset = i64_from_u64(biasOffsetBits);

  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d + 2 < rank; d++) {
    aDims[d] = outDims[d];
    bDims[d] = outDims[d];
  }
  aDims[rank - 2] = m;
  aDims[rank - 1] = k;
  bDims[rank - 2] = k;
  bDims[rank - 1] = n;

  size_t *aMaskStart = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aMaskEnd = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bMaskStart = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bMaskEnd = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *biasMaskStart = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *biasMaskEnd = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_nat_array_len(aMaskStartsObj, rank, aMaskStart) || !read_nat_array_len(aMaskEndsObj, rank, aMaskEnd) ||
      !read_nat_array_len(bMaskStartsObj, rank, bMaskStart) || !read_nat_array_len(bMaskEndsObj, rank, bMaskEnd) ||
      !read_nat_array_len(biasMaskStartsObj, rank, biasMaskStart) || !read_nat_array_len(biasMaskEndsObj, rank, biasMaskEnd)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }
  if (!validate_mask(aMaskStart, aMaskEnd, aDims, rank) || !validate_mask(bMaskStart, bMaskEnd, bDims, rank) ||
      !validate_mask(biasMaskStart, biasMaskEnd, outDims, rank)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const float *A = f32_ptr(aBytes);
  const float *B = f32_ptr(bBytes);
  const float *bias = f32_ptr(biasBytes);
  const size_t aCap = f32_elems(aBytes);
  const size_t bCap = f32_elems(bBytes);
  const size_t biasCap = f32_elems(biasBytes);

  const bool aFast = is_full_mask(aMaskStart, aMaskEnd, aDims, rank) && view_range_in_bounds(aOffset, aStrides, aDims, rank, aCap);
  const bool bFast = is_full_mask(bMaskStart, bMaskEnd, bDims, rank) && view_range_in_bounds(bOffset, bStrides, bDims, rank, bCap);
  const bool biasFast =
      is_full_mask(biasMaskStart, biasMaskEnd, outDims, rank) && view_range_in_bounds(biasOffset, biasStrides, outDims, rank, biasCap);

  const int64_t aStrideM = aStrides[rank - 2];
  const int64_t aStrideK = aStrides[rank - 1];
  const int64_t bStrideK = bStrides[rank - 2];
  const int64_t bStrideN = bStrides[rank - 1];
  const int64_t biasStrideM = biasStrides[rank - 2];
  const int64_t biasStrideN = biasStrides[rank - 1];

  const bool aRowMajor = aStrideK == 1 && aStrideM == (int64_t)k;
  const bool bRowMajor = bStrideN == 1 && bStrideK == (int64_t)n;
  const bool bTransposed = bStrideK == 1 && bStrideN == (int64_t)k;

  if (aFast && bFast && biasFast && aRowMajor && (bRowMajor || bTransposed)) {
    size_t batchNumel = 1;
    for (size_t d = 0; d + 2 < rank; d++) batchNumel *= outDims[d];

    size_t *batchIdx = NULL;
    if (rank > 2) {
      batchIdx = LEAN_ALLOCA(sizeof(size_t) * (rank - 2)); // NOLINT
      for (size_t d = 0; d + 2 < rank; d++) batchIdx[d] = 0;
    }

    int64_t aBatchBase = aOffset;
    int64_t bBatchBase = bOffset;
    int64_t biasBatchBase = biasOffset;

    for (size_t bi = 0; bi < batchNumel; bi++) {
      float *C = dst + bi * (m * n);
      const float *Ab = A + (size_t)aBatchBase;
      const float *Bb = B + (size_t)bBatchBase;

      for (size_t i = 0; i < m; i++) {
        const int64_t bRow = biasBatchBase + (int64_t)i * biasStrideM;
        float *cRow = C + i * n;
        int64_t off = bRow;
        for (size_t j = 0; j < n; j++) {
          cRow[j] = bias[(size_t)off];
          off += biasStrideN;
        }
      }

      if (bRowMajor) {
        tg4_gemm_blocked_f32(C, Ab, Bb, m, k, n);
      } else {
        tg4_gemm_blocked_bt_f32(C, Ab, Bb, m, k, n);
      }

      for (size_t d = rank - 2; d-- > 0;) {
        batchIdx[d]++;
        aBatchBase += aStrides[d];
        bBatchBase += bStrides[d];
        biasBatchBase += biasStrides[d];
        if (batchIdx[d] < outDims[d]) break;
        batchIdx[d] = 0;
        aBatchBase -= aStrides[d] * (int64_t)outDims[d];
        bBatchBase -= bStrides[d] * (int64_t)outDims[d];
        biasBatchBase -= biasStrides[d] * (int64_t)outDims[d];
      }
    }
    return out;
  }

  int64_t *stepA = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  int64_t *stepB = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) {
    stepA[d] = (d == rank - 1) ? 0 : aStrides[d];
    stepB[d] = (d == rank - 2) ? 0 : bStrides[d];
  }

  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  int64_t aBase = aOffset;
  int64_t bBase = bOffset;
  int64_t biasBase = biasOffset;

  if (aFast && bFast && biasFast) {
    for (size_t o = 0; o < outNumel; o++) {
      float acc = bias[(size_t)biasBase];
      for (size_t t = 0; t < k; t++) {
        const float av = A[(size_t)(aBase + (int64_t)t * aStrideK)];
        const float bv = B[(size_t)(bBase + (int64_t)t * bStrideK)];
        acc += av * bv;
      }
      dst[o] = acc;

      for (size_t d = rank; d-- > 0;) {
        idx[d]++;
        aBase += stepA[d];
        bBase += stepB[d];
        biasBase += biasStrides[d];
        if (idx[d] < outDims[d]) break;
        idx[d] = 0;
        aBase -= stepA[d] * (int64_t)outDims[d];
        bBase -= stepB[d] * (int64_t)outDims[d];
        biasBase -= biasStrides[d] * (int64_t)outDims[d];
      }
    }
    return out;
  }

  for (size_t o = 0; o < outNumel; o++) {
    uint8_t validBias = 1;
    for (size_t d = 0; d < rank; d++) {
      const size_t x = idx[d];
      if (x < biasMaskStart[d] || x >= biasMaskEnd[d]) {
        validBias = 0;
        break;
      }
    }
    float acc = 0.0f;
    if (validBias) {
      const int64_t off = biasBase;
      if (off >= 0 && (uint64_t)off < biasCap) {
        acc = bias[(size_t)off];
      }
    }

    uint8_t validAOuter = 1;
    for (size_t d = 0; d + 2 < rank; d++) {
      const size_t x = idx[d];
      if (x < aMaskStart[d] || x >= aMaskEnd[d]) {
        validAOuter = 0;
        break;
      }
    }
    if (validAOuter) {
      const size_t xm = idx[rank - 2];
      if (xm < aMaskStart[rank - 2] || xm >= aMaskEnd[rank - 2]) validAOuter = 0;
    }

    uint8_t validBOuter = 1;
    for (size_t d = 0; d + 2 < rank; d++) {
      const size_t x = idx[d];
      if (x < bMaskStart[d] || x >= bMaskEnd[d]) {
        validBOuter = 0;
        break;
      }
    }
    if (validBOuter) {
      const size_t xn = idx[rank - 1];
      if (xn < bMaskStart[rank - 1] || xn >= bMaskEnd[rank - 1]) validBOuter = 0;
    }

    for (size_t t = 0; t < k; t++) {
      float av = 0.0f;
      float bv = 0.0f;

      if (validAOuter && t >= aMaskStart[rank - 1] && t < aMaskEnd[rank - 1]) {
        const int64_t off = aBase + (int64_t)t * aStrideK;
        if (off >= 0 && (uint64_t)off < aCap) av = A[(size_t)off];
      }
      if (validBOuter && t >= bMaskStart[rank - 2] && t < bMaskEnd[rank - 2]) {
        const int64_t off = bBase + (int64_t)t * bStrideK;
        if (off >= 0 && (uint64_t)off < bCap) bv = B[(size_t)off];
      }

      acc += av * bv;
    }
    dst[o] = acc;

    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aBase += stepA[d];
      bBase += stepB[d];
      biasBase += biasStrides[d];
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      aBase -= stepA[d] * (int64_t)outDims[d];
      bBase -= stepB[d] * (int64_t)outDims[d];
      biasBase -= biasStrides[d] * (int64_t)outDims[d];
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_bias_scale_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes, b_lean_obj_arg biasBytes,
    b_lean_obj_arg aStridesObj, uint64_t aOffsetBits, b_lean_obj_arg aMaskStartsObj, b_lean_obj_arg aMaskEndsObj,
    b_lean_obj_arg bStridesObj, uint64_t bOffsetBits, b_lean_obj_arg bMaskStartsObj, b_lean_obj_arg bMaskEndsObj,
    b_lean_obj_arg biasStridesObj, uint64_t biasOffsetBits, b_lean_obj_arg biasMaskStartsObj, b_lean_obj_arg biasMaskEndsObj,
    b_lean_obj_arg outShapeObj, b_lean_obj_arg kObj,
    uint32_t scaleBits) {
  if (!lean_is_scalar(kObj)) return lean_mk_empty_byte_array(lean_box(0));
  const size_t k = lean_unbox(kObj);
  const float alpha = f32_from_bits(scaleBits);

  const size_t rank = lean_array_size(outShapeObj);
  if (rank < 2) return lean_mk_empty_byte_array(lean_box(0));
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t m = outDims[rank - 2];
  const size_t n = outDims[rank - 1];
  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  int64_t *aStrides = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  int64_t *bStrides = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  int64_t *biasStrides = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  if (!read_i64_array_len(aStridesObj, rank, aStrides) || !read_i64_array_len(bStridesObj, rank, bStrides) ||
      !read_i64_array_len(biasStridesObj, rank, biasStrides)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const int64_t aOffset = i64_from_u64(aOffsetBits);
  const int64_t bOffset = i64_from_u64(bOffsetBits);
  const int64_t biasOffset = i64_from_u64(biasOffsetBits);

  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d + 2 < rank; d++) {
    aDims[d] = outDims[d];
    bDims[d] = outDims[d];
  }
  aDims[rank - 2] = m;
  aDims[rank - 1] = k;
  bDims[rank - 2] = k;
  bDims[rank - 1] = n;

  size_t *aMaskStart = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aMaskEnd = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bMaskStart = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bMaskEnd = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *biasMaskStart = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *biasMaskEnd = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_nat_array_len(aMaskStartsObj, rank, aMaskStart) || !read_nat_array_len(aMaskEndsObj, rank, aMaskEnd) ||
      !read_nat_array_len(bMaskStartsObj, rank, bMaskStart) || !read_nat_array_len(bMaskEndsObj, rank, bMaskEnd) ||
      !read_nat_array_len(biasMaskStartsObj, rank, biasMaskStart) || !read_nat_array_len(biasMaskEndsObj, rank, biasMaskEnd)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }
  if (!validate_mask(aMaskStart, aMaskEnd, aDims, rank) || !validate_mask(bMaskStart, bMaskEnd, bDims, rank) ||
      !validate_mask(biasMaskStart, biasMaskEnd, outDims, rank)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const float *A = f32_ptr(aBytes);
  const float *B = f32_ptr(bBytes);
  const float *bias = f32_ptr(biasBytes);
  const size_t aCap = f32_elems(aBytes);
  const size_t bCap = f32_elems(bBytes);
  const size_t biasCap = f32_elems(biasBytes);

  const bool aFast = is_full_mask(aMaskStart, aMaskEnd, aDims, rank) && view_range_in_bounds(aOffset, aStrides, aDims, rank, aCap);
  const bool bFast = is_full_mask(bMaskStart, bMaskEnd, bDims, rank) && view_range_in_bounds(bOffset, bStrides, bDims, rank, bCap);
  const bool biasFast =
      is_full_mask(biasMaskStart, biasMaskEnd, outDims, rank) && view_range_in_bounds(biasOffset, biasStrides, outDims, rank, biasCap);

  const int64_t aStrideM = aStrides[rank - 2];
  const int64_t aStrideK = aStrides[rank - 1];
  const int64_t bStrideK = bStrides[rank - 2];
  const int64_t bStrideN = bStrides[rank - 1];
  const int64_t biasStrideM = biasStrides[rank - 2];
  const int64_t biasStrideN = biasStrides[rank - 1];

  const bool aRowMajor = aStrideK == 1 && aStrideM == (int64_t)k;
  const bool bRowMajor = bStrideN == 1 && bStrideK == (int64_t)n;
  const bool bTransposed = bStrideK == 1 && bStrideN == (int64_t)k;

  if (aFast && bFast && biasFast && aRowMajor && (bRowMajor || bTransposed)) {
    size_t batchNumel = 1;
    for (size_t d = 0; d + 2 < rank; d++) batchNumel *= outDims[d];

    size_t *batchIdx = NULL;
    if (rank > 2) {
      batchIdx = LEAN_ALLOCA(sizeof(size_t) * (rank - 2)); // NOLINT
      for (size_t d = 0; d + 2 < rank; d++) batchIdx[d] = 0;
    }

    int64_t aBatchBase = aOffset;
    int64_t bBatchBase = bOffset;
    int64_t biasBatchBase = biasOffset;

    for (size_t bi = 0; bi < batchNumel; bi++) {
      float *C = dst + bi * (m * n);
      const float *Ab = A + (size_t)aBatchBase;
      const float *Bb = B + (size_t)bBatchBase;

      for (size_t i = 0; i < m; i++) {
        const int64_t bRow = biasBatchBase + (int64_t)i * biasStrideM;
        float *cRow = C + i * n;
        int64_t off = bRow;
        for (size_t j = 0; j < n; j++) {
          cRow[j] = bias[(size_t)off];
          off += biasStrideN;
        }
      }

      if (bRowMajor) {
        tg4_gemm_blocked_alpha_f32(C, Ab, Bb, m, k, n, alpha);
      } else {
        tg4_gemm_blocked_bt_alpha_f32(C, Ab, Bb, m, k, n, alpha);
      }

      for (size_t d = rank - 2; d-- > 0;) {
        batchIdx[d]++;
        aBatchBase += aStrides[d];
        bBatchBase += bStrides[d];
        biasBatchBase += biasStrides[d];
        if (batchIdx[d] < outDims[d]) break;
        batchIdx[d] = 0;
        aBatchBase -= aStrides[d] * (int64_t)outDims[d];
        bBatchBase -= bStrides[d] * (int64_t)outDims[d];
        biasBatchBase -= biasStrides[d] * (int64_t)outDims[d];
      }
    }
    return out;
  }

  int64_t *stepA = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  int64_t *stepB = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) {
    stepA[d] = (d == rank - 1) ? 0 : aStrides[d];
    stepB[d] = (d == rank - 2) ? 0 : bStrides[d];
  }

  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  int64_t aBase = aOffset;
  int64_t bBase = bOffset;
  int64_t biasBase = biasOffset;

  if (aFast && bFast && biasFast) {
    for (size_t o = 0; o < outNumel; o++) {
      float acc = bias[(size_t)biasBase];
      for (size_t t = 0; t < k; t++) {
        const float av = A[(size_t)(aBase + (int64_t)t * aStrideK)] * alpha;
        const float bv = B[(size_t)(bBase + (int64_t)t * bStrideK)];
        acc += av * bv;
      }
      dst[o] = acc;

      for (size_t d = rank; d-- > 0;) {
        idx[d]++;
        aBase += stepA[d];
        bBase += stepB[d];
        biasBase += biasStrides[d];
        if (idx[d] < outDims[d]) break;
        idx[d] = 0;
        aBase -= stepA[d] * (int64_t)outDims[d];
        bBase -= stepB[d] * (int64_t)outDims[d];
        biasBase -= biasStrides[d] * (int64_t)outDims[d];
      }
    }
    return out;
  }

  for (size_t o = 0; o < outNumel; o++) {
    uint8_t validBias = 1;
    for (size_t d = 0; d < rank; d++) {
      const size_t x = idx[d];
      if (x < biasMaskStart[d] || x >= biasMaskEnd[d]) {
        validBias = 0;
        break;
      }
    }
    float acc = 0.0f;
    if (validBias) {
      const int64_t off = biasBase;
      if (off >= 0 && (uint64_t)off < biasCap) {
        acc = bias[(size_t)off];
      }
    }

    uint8_t validAOuter = 1;
    for (size_t d = 0; d + 2 < rank; d++) {
      const size_t x = idx[d];
      if (x < aMaskStart[d] || x >= aMaskEnd[d]) {
        validAOuter = 0;
        break;
      }
    }
    if (validAOuter) {
      const size_t xm = idx[rank - 2];
      if (xm < aMaskStart[rank - 2] || xm >= aMaskEnd[rank - 2]) validAOuter = 0;
    }

    uint8_t validBOuter = 1;
    for (size_t d = 0; d + 2 < rank; d++) {
      const size_t x = idx[d];
      if (x < bMaskStart[d] || x >= bMaskEnd[d]) {
        validBOuter = 0;
        break;
      }
    }
    if (validBOuter) {
      const size_t xn = idx[rank - 1];
      if (xn < bMaskStart[rank - 1] || xn >= bMaskEnd[rank - 1]) validBOuter = 0;
    }

    for (size_t t = 0; t < k; t++) {
      float av = 0.0f;
      float bv = 0.0f;

      if (validAOuter && t >= aMaskStart[rank - 1] && t < aMaskEnd[rank - 1]) {
        const int64_t off = aBase + (int64_t)t * aStrideK;
        if (off >= 0 && (uint64_t)off < aCap) av = A[(size_t)off] * alpha;
      }
      if (validBOuter && t >= bMaskStart[rank - 2] && t < bMaskEnd[rank - 2]) {
        const int64_t off = bBase + (int64_t)t * bStrideK;
        if (off >= 0 && (uint64_t)off < bCap) bv = B[(size_t)off];
      }

      acc += av * bv;
    }
    dst[o] = acc;

    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aBase += stepA[d];
      bBase += stepB[d];
      biasBase += biasStrides[d];
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      aBase -= stepA[d] * (int64_t)outDims[d];
      bBase -= stepB[d] * (int64_t)outDims[d];
      biasBase -= biasStrides[d] * (int64_t)outDims[d];
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_bias_scale_relu_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes, b_lean_obj_arg biasBytes,
    b_lean_obj_arg aStridesObj, uint64_t aOffsetBits, b_lean_obj_arg aMaskStartsObj, b_lean_obj_arg aMaskEndsObj,
    b_lean_obj_arg bStridesObj, uint64_t bOffsetBits, b_lean_obj_arg bMaskStartsObj, b_lean_obj_arg bMaskEndsObj,
    b_lean_obj_arg biasStridesObj, uint64_t biasOffsetBits, b_lean_obj_arg biasMaskStartsObj, b_lean_obj_arg biasMaskEndsObj,
    b_lean_obj_arg outShapeObj, b_lean_obj_arg kObj,
    uint32_t scaleBits) {
  lean_obj_res out =
      tg4_matmul_view_bias_scale_f32(
          aBytes, bBytes, biasBytes,
          aStridesObj, aOffsetBits, aMaskStartsObj, aMaskEndsObj,
          bStridesObj, bOffsetBits, bMaskStartsObj, bMaskEndsObj,
          biasStridesObj, biasOffsetBits, biasMaskStartsObj, biasMaskEndsObj,
          outShapeObj, kObj,
          scaleBits);
  const size_t outBytes = lean_sarray_size(out);
  if ((outBytes & 3) != 0) return out;
  const size_t outFloats = outBytes / 4;
  tg4_relu_f32_inplace_ptr(f32_ptr_mut(out), outFloats);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_bias2_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes, b_lean_obj_arg bias0Bytes, b_lean_obj_arg bias1Bytes,
    b_lean_obj_arg aStridesObj, uint64_t aOffsetBits, b_lean_obj_arg aMaskStartsObj, b_lean_obj_arg aMaskEndsObj,
    b_lean_obj_arg bStridesObj, uint64_t bOffsetBits, b_lean_obj_arg bMaskStartsObj, b_lean_obj_arg bMaskEndsObj,
    b_lean_obj_arg bias0StridesObj, uint64_t bias0OffsetBits, b_lean_obj_arg bias0MaskStartsObj, b_lean_obj_arg bias0MaskEndsObj,
    b_lean_obj_arg bias1StridesObj, uint64_t bias1OffsetBits, b_lean_obj_arg bias1MaskStartsObj, b_lean_obj_arg bias1MaskEndsObj,
    b_lean_obj_arg outShapeObj, b_lean_obj_arg kObj) {
  if (!lean_is_scalar(kObj)) return lean_mk_empty_byte_array(lean_box(0));
  const size_t k = lean_unbox(kObj);

  const size_t rank = lean_array_size(outShapeObj);
  if (rank < 2) return lean_mk_empty_byte_array(lean_box(0));
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t m = outDims[rank - 2];
  const size_t n = outDims[rank - 1];
  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  int64_t *aStrides = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  int64_t *bStrides = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  int64_t *bias0Strides = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  int64_t *bias1Strides = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  if (!read_i64_array_len(aStridesObj, rank, aStrides) || !read_i64_array_len(bStridesObj, rank, bStrides) ||
      !read_i64_array_len(bias0StridesObj, rank, bias0Strides) || !read_i64_array_len(bias1StridesObj, rank, bias1Strides)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const int64_t aOffset = i64_from_u64(aOffsetBits);
  const int64_t bOffset = i64_from_u64(bOffsetBits);
  const int64_t bias0Offset = i64_from_u64(bias0OffsetBits);
  const int64_t bias1Offset = i64_from_u64(bias1OffsetBits);

  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d + 2 < rank; d++) {
    aDims[d] = outDims[d];
    bDims[d] = outDims[d];
  }
  aDims[rank - 2] = m;
  aDims[rank - 1] = k;
  bDims[rank - 2] = k;
  bDims[rank - 1] = n;

  size_t *aMaskStart = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aMaskEnd = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bMaskStart = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bMaskEnd = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bias0MaskStart = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bias0MaskEnd = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bias1MaskStart = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bias1MaskEnd = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_nat_array_len(aMaskStartsObj, rank, aMaskStart) || !read_nat_array_len(aMaskEndsObj, rank, aMaskEnd) ||
      !read_nat_array_len(bMaskStartsObj, rank, bMaskStart) || !read_nat_array_len(bMaskEndsObj, rank, bMaskEnd) ||
      !read_nat_array_len(bias0MaskStartsObj, rank, bias0MaskStart) || !read_nat_array_len(bias0MaskEndsObj, rank, bias0MaskEnd) ||
      !read_nat_array_len(bias1MaskStartsObj, rank, bias1MaskStart) || !read_nat_array_len(bias1MaskEndsObj, rank, bias1MaskEnd)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }
  if (!validate_mask(aMaskStart, aMaskEnd, aDims, rank) || !validate_mask(bMaskStart, bMaskEnd, bDims, rank) ||
      !validate_mask(bias0MaskStart, bias0MaskEnd, outDims, rank) || !validate_mask(bias1MaskStart, bias1MaskEnd, outDims, rank)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const float *A = f32_ptr(aBytes);
  const float *B = f32_ptr(bBytes);
  const float *bias0 = f32_ptr(bias0Bytes);
  const float *bias1 = f32_ptr(bias1Bytes);
  const size_t aCap = f32_elems(aBytes);
  const size_t bCap = f32_elems(bBytes);
  const size_t bias0Cap = f32_elems(bias0Bytes);
  const size_t bias1Cap = f32_elems(bias1Bytes);

  const bool aFast = is_full_mask(aMaskStart, aMaskEnd, aDims, rank) && view_range_in_bounds(aOffset, aStrides, aDims, rank, aCap);
  const bool bFast = is_full_mask(bMaskStart, bMaskEnd, bDims, rank) && view_range_in_bounds(bOffset, bStrides, bDims, rank, bCap);
  const bool bias0Fast =
      is_full_mask(bias0MaskStart, bias0MaskEnd, outDims, rank) && view_range_in_bounds(bias0Offset, bias0Strides, outDims, rank, bias0Cap);
  const bool bias1Fast =
      is_full_mask(bias1MaskStart, bias1MaskEnd, outDims, rank) && view_range_in_bounds(bias1Offset, bias1Strides, outDims, rank, bias1Cap);

  const int64_t aStrideM = aStrides[rank - 2];
  const int64_t aStrideK = aStrides[rank - 1];
  const int64_t bStrideK = bStrides[rank - 2];
  const int64_t bStrideN = bStrides[rank - 1];
  const int64_t bias0StrideM = bias0Strides[rank - 2];
  const int64_t bias0StrideN = bias0Strides[rank - 1];
  const int64_t bias1StrideM = bias1Strides[rank - 2];
  const int64_t bias1StrideN = bias1Strides[rank - 1];

  const bool aRowMajor = aStrideK == 1 && aStrideM == (int64_t)k;
  const bool bRowMajor = bStrideN == 1 && bStrideK == (int64_t)n;
  const bool bTransposed = bStrideK == 1 && bStrideN == (int64_t)k;

  if (aFast && bFast && bias0Fast && bias1Fast && aRowMajor && (bRowMajor || bTransposed)) {
    size_t batchNumel = 1;
    for (size_t d = 0; d + 2 < rank; d++) batchNumel *= outDims[d];

    size_t *batchIdx = NULL;
    if (rank > 2) {
      batchIdx = LEAN_ALLOCA(sizeof(size_t) * (rank - 2)); // NOLINT
      for (size_t d = 0; d + 2 < rank; d++) batchIdx[d] = 0;
    }

    int64_t aBatchBase = aOffset;
    int64_t bBatchBase = bOffset;
    int64_t bias0BatchBase = bias0Offset;
    int64_t bias1BatchBase = bias1Offset;

    for (size_t bi = 0; bi < batchNumel; bi++) {
      float *C = dst + bi * (m * n);
      const float *Ab = A + (size_t)aBatchBase;
      const float *Bb = B + (size_t)bBatchBase;

      for (size_t i = 0; i < m; i++) {
        const int64_t b0Row = bias0BatchBase + (int64_t)i * bias0StrideM;
        const int64_t b1Row = bias1BatchBase + (int64_t)i * bias1StrideM;
        float *cRow = C + i * n;
        int64_t off0 = b0Row;
        int64_t off1 = b1Row;
        for (size_t j = 0; j < n; j++) {
          cRow[j] = bias0[(size_t)off0] + bias1[(size_t)off1];
          off0 += bias0StrideN;
          off1 += bias1StrideN;
        }
      }

      if (bRowMajor) {
        tg4_gemm_blocked_f32(C, Ab, Bb, m, k, n);
      } else {
        tg4_gemm_blocked_bt_f32(C, Ab, Bb, m, k, n);
      }

      for (size_t d = rank - 2; d-- > 0;) {
        batchIdx[d]++;
        aBatchBase += aStrides[d];
        bBatchBase += bStrides[d];
        bias0BatchBase += bias0Strides[d];
        bias1BatchBase += bias1Strides[d];
        if (batchIdx[d] < outDims[d]) break;
        batchIdx[d] = 0;
        aBatchBase -= aStrides[d] * (int64_t)outDims[d];
        bBatchBase -= bStrides[d] * (int64_t)outDims[d];
        bias0BatchBase -= bias0Strides[d] * (int64_t)outDims[d];
        bias1BatchBase -= bias1Strides[d] * (int64_t)outDims[d];
      }
    }
    return out;
  }

  int64_t *stepA = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  int64_t *stepB = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) {
    stepA[d] = (d == rank - 1) ? 0 : aStrides[d];
    stepB[d] = (d == rank - 2) ? 0 : bStrides[d];
  }

  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  int64_t aBase = aOffset;
  int64_t bBase = bOffset;
  int64_t bias0Base = bias0Offset;
  int64_t bias1Base = bias1Offset;

  if (aFast && bFast && bias0Fast && bias1Fast) {
    for (size_t o = 0; o < outNumel; o++) {
      float acc = bias0[(size_t)bias0Base] + bias1[(size_t)bias1Base];
      for (size_t t = 0; t < k; t++) {
        const float av = A[(size_t)(aBase + (int64_t)t * aStrideK)];
        const float bv = B[(size_t)(bBase + (int64_t)t * bStrideK)];
        acc += av * bv;
      }
      dst[o] = acc;

      for (size_t d = rank; d-- > 0;) {
        idx[d]++;
        aBase += stepA[d];
        bBase += stepB[d];
        bias0Base += bias0Strides[d];
        bias1Base += bias1Strides[d];
        if (idx[d] < outDims[d]) break;
        idx[d] = 0;
        aBase -= stepA[d] * (int64_t)outDims[d];
        bBase -= stepB[d] * (int64_t)outDims[d];
        bias0Base -= bias0Strides[d] * (int64_t)outDims[d];
        bias1Base -= bias1Strides[d] * (int64_t)outDims[d];
      }
    }
    return out;
  }

  for (size_t o = 0; o < outNumel; o++) {
    uint8_t validBias0 = 1;
    uint8_t validBias1 = 1;
    for (size_t d = 0; d < rank; d++) {
      const size_t x = idx[d];
      if (x < bias0MaskStart[d] || x >= bias0MaskEnd[d]) validBias0 = 0;
      if (x < bias1MaskStart[d] || x >= bias1MaskEnd[d]) validBias1 = 0;
      if (!validBias0 && !validBias1) break;
    }
    float acc = 0.0f;
    if (validBias0) {
      const int64_t off = bias0Base;
      if (off >= 0 && (uint64_t)off < bias0Cap) acc += bias0[(size_t)off];
    }
    if (validBias1) {
      const int64_t off = bias1Base;
      if (off >= 0 && (uint64_t)off < bias1Cap) acc += bias1[(size_t)off];
    }

    uint8_t validAOuter = 1;
    for (size_t d = 0; d + 2 < rank; d++) {
      const size_t x = idx[d];
      if (x < aMaskStart[d] || x >= aMaskEnd[d]) {
        validAOuter = 0;
        break;
      }
    }
    if (validAOuter) {
      const size_t xm = idx[rank - 2];
      if (xm < aMaskStart[rank - 2] || xm >= aMaskEnd[rank - 2]) validAOuter = 0;
    }

    uint8_t validBOuter = 1;
    for (size_t d = 0; d + 2 < rank; d++) {
      const size_t x = idx[d];
      if (x < bMaskStart[d] || x >= bMaskEnd[d]) {
        validBOuter = 0;
        break;
      }
    }
    if (validBOuter) {
      const size_t xn = idx[rank - 1];
      if (xn < bMaskStart[rank - 1] || xn >= bMaskEnd[rank - 1]) validBOuter = 0;
    }

    for (size_t t = 0; t < k; t++) {
      float av = 0.0f;
      float bv = 0.0f;

      if (validAOuter && t >= aMaskStart[rank - 1] && t < aMaskEnd[rank - 1]) {
        const int64_t off = aBase + (int64_t)t * aStrideK;
        if (off >= 0 && (uint64_t)off < aCap) av = A[(size_t)off];
      }
      if (validBOuter && t >= bMaskStart[rank - 2] && t < bMaskEnd[rank - 2]) {
        const int64_t off = bBase + (int64_t)t * bStrideK;
        if (off >= 0 && (uint64_t)off < bCap) bv = B[(size_t)off];
      }

      acc += av * bv;
    }
    dst[o] = acc;

    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aBase += stepA[d];
      bBase += stepB[d];
      bias0Base += bias0Strides[d];
      bias1Base += bias1Strides[d];
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      aBase -= stepA[d] * (int64_t)outDims[d];
      bBase -= stepB[d] * (int64_t)outDims[d];
      bias0Base -= bias0Strides[d] * (int64_t)outDims[d];
      bias1Base -= bias1Strides[d] * (int64_t)outDims[d];
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_bias2_scale_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes, b_lean_obj_arg bias0Bytes, b_lean_obj_arg bias1Bytes,
    b_lean_obj_arg aStridesObj, uint64_t aOffsetBits, b_lean_obj_arg aMaskStartsObj, b_lean_obj_arg aMaskEndsObj,
    b_lean_obj_arg bStridesObj, uint64_t bOffsetBits, b_lean_obj_arg bMaskStartsObj, b_lean_obj_arg bMaskEndsObj,
    b_lean_obj_arg bias0StridesObj, uint64_t bias0OffsetBits, b_lean_obj_arg bias0MaskStartsObj, b_lean_obj_arg bias0MaskEndsObj,
    b_lean_obj_arg bias1StridesObj, uint64_t bias1OffsetBits, b_lean_obj_arg bias1MaskStartsObj, b_lean_obj_arg bias1MaskEndsObj,
    b_lean_obj_arg outShapeObj, b_lean_obj_arg kObj,
    uint32_t scaleBits) {
  if (!lean_is_scalar(kObj)) return lean_mk_empty_byte_array(lean_box(0));
  const size_t k = lean_unbox(kObj);
  const float alpha = f32_from_bits(scaleBits);

  const size_t rank = lean_array_size(outShapeObj);
  if (rank < 2) return lean_mk_empty_byte_array(lean_box(0));
  size_t *outDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  if (!read_shape_dims(outShapeObj, outDims)) return lean_mk_empty_byte_array(lean_box(0));

  const size_t m = outDims[rank - 2];
  const size_t n = outDims[rank - 1];
  const size_t outNumel = prod_dims(outDims, rank);
  const size_t outBytes = outNumel * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *dst = f32_ptr_mut(out);
  if (outNumel == 0) return out;

  int64_t *aStrides = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  int64_t *bStrides = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  int64_t *bias0Strides = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  int64_t *bias1Strides = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  if (!read_i64_array_len(aStridesObj, rank, aStrides) || !read_i64_array_len(bStridesObj, rank, bStrides) ||
      !read_i64_array_len(bias0StridesObj, rank, bias0Strides) || !read_i64_array_len(bias1StridesObj, rank, bias1Strides)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const int64_t aOffset = i64_from_u64(aOffsetBits);
  const int64_t bOffset = i64_from_u64(bOffsetBits);
  const int64_t bias0Offset = i64_from_u64(bias0OffsetBits);
  const int64_t bias1Offset = i64_from_u64(bias1OffsetBits);

  size_t *aDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bDims = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d + 2 < rank; d++) {
    aDims[d] = outDims[d];
    bDims[d] = outDims[d];
  }
  aDims[rank - 2] = m;
  aDims[rank - 1] = k;
  bDims[rank - 2] = k;
  bDims[rank - 1] = n;

  size_t *aMaskStart = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *aMaskEnd = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bMaskStart = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bMaskEnd = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bias0MaskStart = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bias0MaskEnd = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bias1MaskStart = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  size_t *bias1MaskEnd = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT

  if (!read_nat_array_len(aMaskStartsObj, rank, aMaskStart) || !read_nat_array_len(aMaskEndsObj, rank, aMaskEnd) ||
      !read_nat_array_len(bMaskStartsObj, rank, bMaskStart) || !read_nat_array_len(bMaskEndsObj, rank, bMaskEnd) ||
      !read_nat_array_len(bias0MaskStartsObj, rank, bias0MaskStart) || !read_nat_array_len(bias0MaskEndsObj, rank, bias0MaskEnd) ||
      !read_nat_array_len(bias1MaskStartsObj, rank, bias1MaskStart) || !read_nat_array_len(bias1MaskEndsObj, rank, bias1MaskEnd)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }
  if (!validate_mask(aMaskStart, aMaskEnd, aDims, rank) || !validate_mask(bMaskStart, bMaskEnd, bDims, rank) ||
      !validate_mask(bias0MaskStart, bias0MaskEnd, outDims, rank) || !validate_mask(bias1MaskStart, bias1MaskEnd, outDims, rank)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const float *A = f32_ptr(aBytes);
  const float *B = f32_ptr(bBytes);
  const float *bias0 = f32_ptr(bias0Bytes);
  const float *bias1 = f32_ptr(bias1Bytes);
  const size_t aCap = f32_elems(aBytes);
  const size_t bCap = f32_elems(bBytes);
  const size_t bias0Cap = f32_elems(bias0Bytes);
  const size_t bias1Cap = f32_elems(bias1Bytes);

  const bool aFast = is_full_mask(aMaskStart, aMaskEnd, aDims, rank) && view_range_in_bounds(aOffset, aStrides, aDims, rank, aCap);
  const bool bFast = is_full_mask(bMaskStart, bMaskEnd, bDims, rank) && view_range_in_bounds(bOffset, bStrides, bDims, rank, bCap);
  const bool bias0Fast =
      is_full_mask(bias0MaskStart, bias0MaskEnd, outDims, rank) && view_range_in_bounds(bias0Offset, bias0Strides, outDims, rank, bias0Cap);
  const bool bias1Fast =
      is_full_mask(bias1MaskStart, bias1MaskEnd, outDims, rank) && view_range_in_bounds(bias1Offset, bias1Strides, outDims, rank, bias1Cap);

  const int64_t aStrideM = aStrides[rank - 2];
  const int64_t aStrideK = aStrides[rank - 1];
  const int64_t bStrideK = bStrides[rank - 2];
  const int64_t bStrideN = bStrides[rank - 1];
  const int64_t bias0StrideM = bias0Strides[rank - 2];
  const int64_t bias0StrideN = bias0Strides[rank - 1];
  const int64_t bias1StrideM = bias1Strides[rank - 2];
  const int64_t bias1StrideN = bias1Strides[rank - 1];

  const bool aRowMajor = aStrideK == 1 && aStrideM == (int64_t)k;
  const bool bRowMajor = bStrideN == 1 && bStrideK == (int64_t)n;
  const bool bTransposed = bStrideK == 1 && bStrideN == (int64_t)k;

  if (aFast && bFast && bias0Fast && bias1Fast && aRowMajor && (bRowMajor || bTransposed)) {
    size_t batchNumel = 1;
    for (size_t d = 0; d + 2 < rank; d++) batchNumel *= outDims[d];

    size_t *batchIdx = NULL;
    if (rank > 2) {
      batchIdx = LEAN_ALLOCA(sizeof(size_t) * (rank - 2)); // NOLINT
      for (size_t d = 0; d + 2 < rank; d++) batchIdx[d] = 0;
    }

    int64_t aBatchBase = aOffset;
    int64_t bBatchBase = bOffset;
    int64_t bias0BatchBase = bias0Offset;
    int64_t bias1BatchBase = bias1Offset;

    for (size_t bi = 0; bi < batchNumel; bi++) {
      float *C = dst + bi * (m * n);
      const float *Ab = A + (size_t)aBatchBase;
      const float *Bb = B + (size_t)bBatchBase;

      for (size_t i = 0; i < m; i++) {
        const int64_t b0Row = bias0BatchBase + (int64_t)i * bias0StrideM;
        const int64_t b1Row = bias1BatchBase + (int64_t)i * bias1StrideM;
        float *cRow = C + i * n;
        int64_t off0 = b0Row;
        int64_t off1 = b1Row;
        for (size_t j = 0; j < n; j++) {
          cRow[j] = bias0[(size_t)off0] + bias1[(size_t)off1];
          off0 += bias0StrideN;
          off1 += bias1StrideN;
        }
      }

      if (bRowMajor) {
        tg4_gemm_blocked_alpha_f32(C, Ab, Bb, m, k, n, alpha);
      } else {
        tg4_gemm_blocked_bt_alpha_f32(C, Ab, Bb, m, k, n, alpha);
      }

      for (size_t d = rank - 2; d-- > 0;) {
        batchIdx[d]++;
        aBatchBase += aStrides[d];
        bBatchBase += bStrides[d];
        bias0BatchBase += bias0Strides[d];
        bias1BatchBase += bias1Strides[d];
        if (batchIdx[d] < outDims[d]) break;
        batchIdx[d] = 0;
        aBatchBase -= aStrides[d] * (int64_t)outDims[d];
        bBatchBase -= bStrides[d] * (int64_t)outDims[d];
        bias0BatchBase -= bias0Strides[d] * (int64_t)outDims[d];
        bias1BatchBase -= bias1Strides[d] * (int64_t)outDims[d];
      }
    }
    return out;
  }

  int64_t *stepA = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  int64_t *stepB = LEAN_ALLOCA(sizeof(int64_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) {
    stepA[d] = (d == rank - 1) ? 0 : aStrides[d];
    stepB[d] = (d == rank - 2) ? 0 : bStrides[d];
  }

  size_t *idx = LEAN_ALLOCA(sizeof(size_t) * rank); // NOLINT
  for (size_t d = 0; d < rank; d++) idx[d] = 0;

  int64_t aBase = aOffset;
  int64_t bBase = bOffset;
  int64_t bias0Base = bias0Offset;
  int64_t bias1Base = bias1Offset;

  if (aFast && bFast && bias0Fast && bias1Fast) {
    for (size_t o = 0; o < outNumel; o++) {
      float acc = bias0[(size_t)bias0Base] + bias1[(size_t)bias1Base];
      for (size_t t = 0; t < k; t++) {
        const float av = A[(size_t)(aBase + (int64_t)t * aStrideK)] * alpha;
        const float bv = B[(size_t)(bBase + (int64_t)t * bStrideK)];
        acc += av * bv;
      }
      dst[o] = acc;

      for (size_t d = rank; d-- > 0;) {
        idx[d]++;
        aBase += stepA[d];
        bBase += stepB[d];
        bias0Base += bias0Strides[d];
        bias1Base += bias1Strides[d];
        if (idx[d] < outDims[d]) break;
        idx[d] = 0;
        aBase -= stepA[d] * (int64_t)outDims[d];
        bBase -= stepB[d] * (int64_t)outDims[d];
        bias0Base -= bias0Strides[d] * (int64_t)outDims[d];
        bias1Base -= bias1Strides[d] * (int64_t)outDims[d];
      }
    }
    return out;
  }

  for (size_t o = 0; o < outNumel; o++) {
    uint8_t validBias0 = 1;
    uint8_t validBias1 = 1;
    for (size_t d = 0; d < rank; d++) {
      const size_t x = idx[d];
      if (x < bias0MaskStart[d] || x >= bias0MaskEnd[d]) validBias0 = 0;
      if (x < bias1MaskStart[d] || x >= bias1MaskEnd[d]) validBias1 = 0;
      if (!validBias0 && !validBias1) break;
    }
    float acc = 0.0f;
    if (validBias0) {
      const int64_t off = bias0Base;
      if (off >= 0 && (uint64_t)off < bias0Cap) acc += bias0[(size_t)off];
    }
    if (validBias1) {
      const int64_t off = bias1Base;
      if (off >= 0 && (uint64_t)off < bias1Cap) acc += bias1[(size_t)off];
    }

    uint8_t validAOuter = 1;
    for (size_t d = 0; d + 2 < rank; d++) {
      const size_t x = idx[d];
      if (x < aMaskStart[d] || x >= aMaskEnd[d]) {
        validAOuter = 0;
        break;
      }
    }
    if (validAOuter) {
      const size_t xm = idx[rank - 2];
      if (xm < aMaskStart[rank - 2] || xm >= aMaskEnd[rank - 2]) validAOuter = 0;
    }

    uint8_t validBOuter = 1;
    for (size_t d = 0; d + 2 < rank; d++) {
      const size_t x = idx[d];
      if (x < bMaskStart[d] || x >= bMaskEnd[d]) {
        validBOuter = 0;
        break;
      }
    }
    if (validBOuter) {
      const size_t xn = idx[rank - 1];
      if (xn < bMaskStart[rank - 1] || xn >= bMaskEnd[rank - 1]) validBOuter = 0;
    }

    for (size_t t = 0; t < k; t++) {
      float av = 0.0f;
      float bv = 0.0f;

      if (validAOuter && t >= aMaskStart[rank - 1] && t < aMaskEnd[rank - 1]) {
        const int64_t off = aBase + (int64_t)t * aStrideK;
        if (off >= 0 && (uint64_t)off < aCap) av = A[(size_t)off] * alpha;
      }
      if (validBOuter && t >= bMaskStart[rank - 2] && t < bMaskEnd[rank - 2]) {
        const int64_t off = bBase + (int64_t)t * bStrideK;
        if (off >= 0 && (uint64_t)off < bCap) bv = B[(size_t)off];
      }

      acc += av * bv;
    }
    dst[o] = acc;

    for (size_t d = rank; d-- > 0;) {
      idx[d]++;
      aBase += stepA[d];
      bBase += stepB[d];
      bias0Base += bias0Strides[d];
      bias1Base += bias1Strides[d];
      if (idx[d] < outDims[d]) break;
      idx[d] = 0;
      aBase -= stepA[d] * (int64_t)outDims[d];
      bBase -= stepB[d] * (int64_t)outDims[d];
      bias0Base -= bias0Strides[d] * (int64_t)outDims[d];
      bias1Base -= bias1Strides[d] * (int64_t)outDims[d];
    }
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_bias2_scale_relu_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes, b_lean_obj_arg bias0Bytes, b_lean_obj_arg bias1Bytes,
    b_lean_obj_arg aStridesObj, uint64_t aOffsetBits, b_lean_obj_arg aMaskStartsObj, b_lean_obj_arg aMaskEndsObj,
    b_lean_obj_arg bStridesObj, uint64_t bOffsetBits, b_lean_obj_arg bMaskStartsObj, b_lean_obj_arg bMaskEndsObj,
    b_lean_obj_arg bias0StridesObj, uint64_t bias0OffsetBits, b_lean_obj_arg bias0MaskStartsObj, b_lean_obj_arg bias0MaskEndsObj,
    b_lean_obj_arg bias1StridesObj, uint64_t bias1OffsetBits, b_lean_obj_arg bias1MaskStartsObj, b_lean_obj_arg bias1MaskEndsObj,
    b_lean_obj_arg outShapeObj, b_lean_obj_arg kObj,
    uint32_t scaleBits) {
  lean_obj_res out =
      tg4_matmul_view_bias2_scale_f32(
          aBytes, bBytes, bias0Bytes, bias1Bytes,
          aStridesObj, aOffsetBits, aMaskStartsObj, aMaskEndsObj,
          bStridesObj, bOffsetBits, bMaskStartsObj, bMaskEndsObj,
          bias0StridesObj, bias0OffsetBits, bias0MaskStartsObj, bias0MaskEndsObj,
          bias1StridesObj, bias1OffsetBits, bias1MaskStartsObj, bias1MaskEndsObj,
          outShapeObj, kObj,
          scaleBits);
  const size_t outBytes = lean_sarray_size(out);
  if ((outBytes & 3) != 0) return out;
  const size_t outFloats = outBytes / 4;
  tg4_relu_f32_inplace_ptr(f32_ptr_mut(out), outFloats);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_bias_relu_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes, b_lean_obj_arg biasBytes,
    b_lean_obj_arg aStridesObj, uint64_t aOffsetBits, b_lean_obj_arg aMaskStartsObj, b_lean_obj_arg aMaskEndsObj,
    b_lean_obj_arg bStridesObj, uint64_t bOffsetBits, b_lean_obj_arg bMaskStartsObj, b_lean_obj_arg bMaskEndsObj,
    b_lean_obj_arg biasStridesObj, uint64_t biasOffsetBits, b_lean_obj_arg biasMaskStartsObj, b_lean_obj_arg biasMaskEndsObj,
    b_lean_obj_arg outShapeObj, b_lean_obj_arg kObj) {
  lean_obj_res out = tg4_matmul_view_bias_f32(
      aBytes, bBytes, biasBytes,
      aStridesObj, aOffsetBits, aMaskStartsObj, aMaskEndsObj,
      bStridesObj, bOffsetBits, bMaskStartsObj, bMaskEndsObj,
      biasStridesObj, biasOffsetBits, biasMaskStartsObj, biasMaskEndsObj,
      outShapeObj, kObj);
  const size_t outBytes = lean_sarray_size(out);
  if ((outBytes & 3) != 0) return out;
  const size_t outFloats = outBytes / 4;
  tg4_relu_f32_inplace_ptr(f32_ptr_mut(out), outFloats);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_view_bias2_relu_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes, b_lean_obj_arg bias0Bytes, b_lean_obj_arg bias1Bytes,
    b_lean_obj_arg aStridesObj, uint64_t aOffsetBits, b_lean_obj_arg aMaskStartsObj, b_lean_obj_arg aMaskEndsObj,
    b_lean_obj_arg bStridesObj, uint64_t bOffsetBits, b_lean_obj_arg bMaskStartsObj, b_lean_obj_arg bMaskEndsObj,
    b_lean_obj_arg bias0StridesObj, uint64_t bias0OffsetBits, b_lean_obj_arg bias0MaskStartsObj, b_lean_obj_arg bias0MaskEndsObj,
    b_lean_obj_arg bias1StridesObj, uint64_t bias1OffsetBits, b_lean_obj_arg bias1MaskStartsObj, b_lean_obj_arg bias1MaskEndsObj,
    b_lean_obj_arg outShapeObj, b_lean_obj_arg kObj) {
  lean_obj_res out = tg4_matmul_view_bias2_f32(
      aBytes, bBytes, bias0Bytes, bias1Bytes,
      aStridesObj, aOffsetBits, aMaskStartsObj, aMaskEndsObj,
      bStridesObj, bOffsetBits, bMaskStartsObj, bMaskEndsObj,
      bias0StridesObj, bias0OffsetBits, bias0MaskStartsObj, bias0MaskEndsObj,
      bias1StridesObj, bias1OffsetBits, bias1MaskStartsObj, bias1MaskEndsObj,
      outShapeObj, kObj);
  const size_t outBytes = lean_sarray_size(out);
  if ((outBytes & 3) != 0) return out;
  const size_t outFloats = outBytes / 4;
  tg4_relu_f32_inplace_ptr(f32_ptr_mut(out), outFloats);
  return out;
}

// ByteArray (raw f32 bytes) matmul.
LEAN_EXPORT lean_obj_res tg4_matmul_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj) {
  if (!lean_is_scalar(mObj) || !lean_is_scalar(kObj) || !lean_is_scalar(nObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t m = lean_unbox(mObj);
  const size_t k = lean_unbox(kObj);
  const size_t n = lean_unbox(nObj);
  const size_t outFloats = m * n;
  const size_t outBytes = outFloats * 4;

  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *C = (float *)lean_sarray_cptr(out); // NOLINT
  const float *A = (const float *)lean_sarray_cptr(aBytes); // NOLINT
  const float *B = (const float *)lean_sarray_cptr(bBytes); // NOLINT

  const size_t aBytesSz = lean_sarray_size(aBytes);
  const size_t bBytesSz = lean_sarray_size(bBytes);
  if (aBytesSz < m * k * 4 || bBytesSz < k * n * 4) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }

  for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
  tg4_gemm_blocked_f32(C, A, B, m, k, n);

  return out;
}

// ByteArray (raw f32 bytes) matmul with broadcasted bias add.
// Computes: C = A @ B + bias, where bias broadcasts to [m, n].
LEAN_EXPORT lean_obj_res tg4_matmul_bias_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes, b_lean_obj_arg biasBytes,
    b_lean_obj_arg biasShapeObj,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj) {
  if (!lean_is_scalar(mObj) || !lean_is_scalar(kObj) || !lean_is_scalar(nObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t m = lean_unbox(mObj);
  const size_t k = lean_unbox(kObj);
  const size_t n = lean_unbox(nObj);
  const size_t outFloats = m * n;
  const size_t outBytes = outFloats * 4;

  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *C = (float *)lean_sarray_cptr(out); // NOLINT
  if (outFloats == 0) return out;

  const float *A = (const float *)lean_sarray_cptr(aBytes); // NOLINT
  const float *B = (const float *)lean_sarray_cptr(bBytes); // NOLINT
  const size_t aBytesSz = lean_sarray_size(aBytes);
  const size_t bBytesSz = lean_sarray_size(bBytes);
  if (aBytesSz < m * k * 4 || bBytesSz < k * n * 4) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }

  size_t outDims[2] = {m, n};
  size_t biasDims[2];
  size_t biasStrides[2];
  if (!read_padded_dims(biasShapeObj, 2, biasDims)) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }
  if (!is_broadcastable(biasDims, outDims, 2)) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }
  contiguous_strides(biasDims, 2, biasStrides);
  broadcast_strides(biasDims, 2, biasStrides);

  const size_t biasNumel = prod_dims(biasDims, 2);
  if (lean_sarray_size(biasBytes) < biasNumel * 4) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }
  const float *bias = (const float *)lean_sarray_cptr(biasBytes); // NOLINT

  for (size_t i = 0; i < m; i++) {
    const size_t bOffRow = i * biasStrides[0];
    float *cRow = C + i * n;
    for (size_t j = 0; j < n; j++) {
      cRow[j] = bias[bOffRow + j * biasStrides[1]];
    }
  }
  tg4_gemm_blocked_f32(C, A, B, m, k, n);

  return out;
}

// ByteArray (raw f32 bytes) matmul with broadcasted bias add, scaling the matmul term.
// Computes: C = (A @ B) * scale + bias, where bias broadcasts to [m, n].
LEAN_EXPORT lean_obj_res tg4_matmul_bias_scale_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes, b_lean_obj_arg biasBytes,
    b_lean_obj_arg biasShapeObj,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj,
    uint32_t scaleBits) {
  if (!lean_is_scalar(mObj) || !lean_is_scalar(kObj) || !lean_is_scalar(nObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t m = lean_unbox(mObj);
  const size_t k = lean_unbox(kObj);
  const size_t n = lean_unbox(nObj);
  const float alpha = f32_from_bits(scaleBits);
  const size_t outFloats = m * n;
  const size_t outBytes = outFloats * 4;

  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *C = (float *)lean_sarray_cptr(out); // NOLINT
  if (outFloats == 0) return out;

  const float *A = (const float *)lean_sarray_cptr(aBytes); // NOLINT
  const float *B = (const float *)lean_sarray_cptr(bBytes); // NOLINT
  const size_t aBytesSz = lean_sarray_size(aBytes);
  const size_t bBytesSz = lean_sarray_size(bBytes);
  if (aBytesSz < m * k * 4 || bBytesSz < k * n * 4) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }

  size_t outDims[2] = {m, n};
  size_t biasDims[2];
  size_t biasStrides[2];
  if (!read_padded_dims(biasShapeObj, 2, biasDims)) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }
  if (!is_broadcastable(biasDims, outDims, 2)) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }
  contiguous_strides(biasDims, 2, biasStrides);
  broadcast_strides(biasDims, 2, biasStrides);

  const size_t biasNumel = prod_dims(biasDims, 2);
  if (lean_sarray_size(biasBytes) < biasNumel * 4) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }
  const float *bias = (const float *)lean_sarray_cptr(biasBytes); // NOLINT

  for (size_t i = 0; i < m; i++) {
    const size_t bOffRow = i * biasStrides[0];
    float *cRow = C + i * n;
    for (size_t j = 0; j < n; j++) {
      cRow[j] = bias[bOffRow + j * biasStrides[1]];
    }
  }
  tg4_gemm_blocked_alpha_f32(C, A, B, m, k, n, alpha);

  return out;
}

// ByteArray (raw f32 bytes) matmul with two broadcasted adds.
// Computes: C = A @ B + bias0 + bias1, where both bias tensors broadcast to [m, n].
LEAN_EXPORT lean_obj_res tg4_matmul_bias2_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes,
    b_lean_obj_arg bias0Bytes, b_lean_obj_arg bias0ShapeObj,
    b_lean_obj_arg bias1Bytes, b_lean_obj_arg bias1ShapeObj,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj) {
  if (!lean_is_scalar(mObj) || !lean_is_scalar(kObj) || !lean_is_scalar(nObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t m = lean_unbox(mObj);
  const size_t k = lean_unbox(kObj);
  const size_t n = lean_unbox(nObj);
  const size_t outFloats = m * n;
  const size_t outBytes = outFloats * 4;

  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *C = (float *)lean_sarray_cptr(out); // NOLINT
  if (outFloats == 0) return out;

  const float *A = (const float *)lean_sarray_cptr(aBytes); // NOLINT
  const float *B = (const float *)lean_sarray_cptr(bBytes); // NOLINT
  const size_t aBytesSz = lean_sarray_size(aBytes);
  const size_t bBytesSz = lean_sarray_size(bBytes);
  if (aBytesSz < m * k * 4 || bBytesSz < k * n * 4) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }

  size_t outDims[2] = {m, n};

  size_t bias0Dims[2];
  size_t bias0Strides[2];
  if (!read_padded_dims(bias0ShapeObj, 2, bias0Dims)) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }
  if (!is_broadcastable(bias0Dims, outDims, 2)) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }
  contiguous_strides(bias0Dims, 2, bias0Strides);
  broadcast_strides(bias0Dims, 2, bias0Strides);
  const size_t bias0Numel = prod_dims(bias0Dims, 2);
  if (lean_sarray_size(bias0Bytes) < bias0Numel * 4) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }
  const float *bias0 = (const float *)lean_sarray_cptr(bias0Bytes); // NOLINT

  size_t bias1Dims[2];
  size_t bias1Strides[2];
  if (!read_padded_dims(bias1ShapeObj, 2, bias1Dims)) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }
  if (!is_broadcastable(bias1Dims, outDims, 2)) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }
  contiguous_strides(bias1Dims, 2, bias1Strides);
  broadcast_strides(bias1Dims, 2, bias1Strides);
  const size_t bias1Numel = prod_dims(bias1Dims, 2);
  if (lean_sarray_size(bias1Bytes) < bias1Numel * 4) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }
  const float *bias1 = (const float *)lean_sarray_cptr(bias1Bytes); // NOLINT

  for (size_t i = 0; i < m; i++) {
    const size_t b0OffRow = i * bias0Strides[0];
    const size_t b1OffRow = i * bias1Strides[0];
    float *cRow = C + i * n;
    for (size_t j = 0; j < n; j++) {
      const float v0 = bias0[b0OffRow + j * bias0Strides[1]];
      const float v1 = bias1[b1OffRow + j * bias1Strides[1]];
      cRow[j] = v0 + v1;
    }
  }
  tg4_gemm_blocked_f32(C, A, B, m, k, n);

  return out;
}

// ByteArray (raw f32 bytes) matmul with two broadcasted adds, scaling the matmul term.
// Computes: C = (A @ B) * scale + bias0 + bias1, where both bias tensors broadcast to [m, n].
LEAN_EXPORT lean_obj_res tg4_matmul_bias2_scale_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes,
    b_lean_obj_arg bias0Bytes, b_lean_obj_arg bias0ShapeObj,
    b_lean_obj_arg bias1Bytes, b_lean_obj_arg bias1ShapeObj,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj,
    uint32_t scaleBits) {
  if (!lean_is_scalar(mObj) || !lean_is_scalar(kObj) || !lean_is_scalar(nObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t m = lean_unbox(mObj);
  const size_t k = lean_unbox(kObj);
  const size_t n = lean_unbox(nObj);
  const float alpha = f32_from_bits(scaleBits);
  const size_t outFloats = m * n;
  const size_t outBytes = outFloats * 4;

  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *C = (float *)lean_sarray_cptr(out); // NOLINT
  if (outFloats == 0) return out;

  const float *A = (const float *)lean_sarray_cptr(aBytes); // NOLINT
  const float *B = (const float *)lean_sarray_cptr(bBytes); // NOLINT
  const size_t aBytesSz = lean_sarray_size(aBytes);
  const size_t bBytesSz = lean_sarray_size(bBytes);
  if (aBytesSz < m * k * 4 || bBytesSz < k * n * 4) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }

  size_t outDims[2] = {m, n};

  size_t bias0Dims[2];
  size_t bias0Strides[2];
  if (!read_padded_dims(bias0ShapeObj, 2, bias0Dims)) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }
  if (!is_broadcastable(bias0Dims, outDims, 2)) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }
  contiguous_strides(bias0Dims, 2, bias0Strides);
  broadcast_strides(bias0Dims, 2, bias0Strides);
  const size_t bias0Numel = prod_dims(bias0Dims, 2);
  if (lean_sarray_size(bias0Bytes) < bias0Numel * 4) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }
  const float *bias0 = (const float *)lean_sarray_cptr(bias0Bytes); // NOLINT

  size_t bias1Dims[2];
  size_t bias1Strides[2];
  if (!read_padded_dims(bias1ShapeObj, 2, bias1Dims)) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }
  if (!is_broadcastable(bias1Dims, outDims, 2)) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }
  contiguous_strides(bias1Dims, 2, bias1Strides);
  broadcast_strides(bias1Dims, 2, bias1Strides);
  const size_t bias1Numel = prod_dims(bias1Dims, 2);
  if (lean_sarray_size(bias1Bytes) < bias1Numel * 4) {
    for (size_t i = 0; i < outFloats; i++) C[i] = 0.0f;
    return out;
  }
  const float *bias1 = (const float *)lean_sarray_cptr(bias1Bytes); // NOLINT

  for (size_t i = 0; i < m; i++) {
    const size_t b0OffRow = i * bias0Strides[0];
    const size_t b1OffRow = i * bias1Strides[0];
    float *cRow = C + i * n;
    for (size_t j = 0; j < n; j++) {
      const float v0 = bias0[b0OffRow + j * bias0Strides[1]];
      const float v1 = bias1[b1OffRow + j * bias1Strides[1]];
      cRow[j] = v0 + v1;
    }
  }
  tg4_gemm_blocked_alpha_f32(C, A, B, m, k, n, alpha);

  return out;
}

// ByteArray (raw f32 bytes) matmul with two broadcasted adds + ReLU.
// Computes: C = max(0, A @ B + bias0 + bias1), where both bias tensors broadcast to [m, n].
LEAN_EXPORT lean_obj_res tg4_matmul_bias2_relu_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes,
    b_lean_obj_arg bias0Bytes, b_lean_obj_arg bias0ShapeObj,
    b_lean_obj_arg bias1Bytes, b_lean_obj_arg bias1ShapeObj,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj) {
  lean_obj_res out = tg4_matmul_bias2_f32(aBytes, bBytes, bias0Bytes, bias0ShapeObj, bias1Bytes, bias1ShapeObj, mObj, kObj, nObj);
  const size_t outBytes = lean_sarray_size(out);
  if ((outBytes & 3) != 0) return out;
  const size_t outFloats = outBytes / 4;
  tg4_relu_f32_inplace_ptr(f32_ptr_mut(out), outFloats);
  return out;
}

// ByteArray (raw f32 bytes) matmul with two broadcasted adds, scaling the matmul term + ReLU.
// Computes: C = max(0, (A @ B) * scale + bias0 + bias1), where both bias tensors broadcast to [m, n].
LEAN_EXPORT lean_obj_res tg4_matmul_bias2_scale_relu_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes,
    b_lean_obj_arg bias0Bytes, b_lean_obj_arg bias0ShapeObj,
    b_lean_obj_arg bias1Bytes, b_lean_obj_arg bias1ShapeObj,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj,
    uint32_t scaleBits) {
  lean_obj_res out =
      tg4_matmul_bias2_scale_f32(aBytes, bBytes, bias0Bytes, bias0ShapeObj, bias1Bytes, bias1ShapeObj, mObj, kObj, nObj, scaleBits);
  const size_t outBytes = lean_sarray_size(out);
  if ((outBytes & 3) != 0) return out;
  const size_t outFloats = outBytes / 4;
  tg4_relu_f32_inplace_ptr(f32_ptr_mut(out), outFloats);
  return out;
}

// ByteArray (raw f32 bytes) matmul with broadcasted bias add + ReLU.
// Computes: C = max(0, A @ B + bias), where bias broadcasts to [m, n].
LEAN_EXPORT lean_obj_res tg4_matmul_bias_relu_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes, b_lean_obj_arg biasBytes,
    b_lean_obj_arg biasShapeObj,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj) {
  lean_obj_res out = tg4_matmul_bias_f32(aBytes, bBytes, biasBytes, biasShapeObj, mObj, kObj, nObj);
  const size_t outBytes = lean_sarray_size(out);
  if ((outBytes & 3) != 0) return out;
  const size_t outFloats = outBytes / 4;
  tg4_relu_f32_inplace_ptr(f32_ptr_mut(out), outFloats);
  return out;
}

// ByteArray (raw f32 bytes) matmul with broadcasted bias add, scaling the matmul term + ReLU.
// Computes: C = max(0, (A @ B) * scale + bias), where bias broadcasts to [m, n].
LEAN_EXPORT lean_obj_res tg4_matmul_bias_scale_relu_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes, b_lean_obj_arg biasBytes,
    b_lean_obj_arg biasShapeObj,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj,
    uint32_t scaleBits) {
  lean_obj_res out = tg4_matmul_bias_scale_f32(aBytes, bBytes, biasBytes, biasShapeObj, mObj, kObj, nObj, scaleBits);
  const size_t outBytes = lean_sarray_size(out);
  if ((outBytes & 3) != 0) return out;
  const size_t outFloats = outBytes / 4;
  tg4_relu_f32_inplace_ptr(f32_ptr_mut(out), outFloats);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_batched_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes,
    b_lean_obj_arg aStartsObj, b_lean_obj_arg bStartsObj,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj) {
  if (!lean_is_scalar(mObj) || !lean_is_scalar(kObj) || !lean_is_scalar(nObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t m = lean_unbox(mObj);
  const size_t k = lean_unbox(kObj);
  const size_t n = lean_unbox(nObj);
  const size_t outFloatsPer = m * n;

  const size_t batch = lean_array_size(aStartsObj);
  if (batch != lean_array_size(bStartsObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t outFloats = batch * outFloatsPer;
  const size_t outBytes = outFloats * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *Cbase = (float *)lean_sarray_cptr(out); // NOLINT

  const uint8_t *aBase = lean_sarray_cptr(aBytes);
  const uint8_t *bBase = lean_sarray_cptr(bBytes);
  const size_t aBytesSz = lean_sarray_size(aBytes);
  const size_t bBytesSz = lean_sarray_size(bBytes);
  const size_t aMatBytes = m * k * 4;
  const size_t bMatBytes = k * n * 4;

  for (size_t bi = 0; bi < batch; bi++) {
    float *C = Cbase + bi * outFloatsPer;
    for (size_t i = 0; i < outFloatsPer; i++) C[i] = 0.0f;

    b_lean_obj_arg aStartObj = lean_array_get_core(aStartsObj, bi);
    b_lean_obj_arg bStartObj = lean_array_get_core(bStartsObj, bi);
    if (!lean_is_scalar(aStartObj) || !lean_is_scalar(bStartObj)) {
      continue;
    }

    const size_t aStart = lean_unbox(aStartObj);
    const size_t bStart = lean_unbox(bStartObj);
    if (aStart + aMatBytes > aBytesSz || bStart + bMatBytes > bBytesSz) {
      continue;
    }

    const float *A = (const float *)(aBase + aStart); // NOLINT
    const float *B = (const float *)(bBase + bStart); // NOLINT
    tg4_gemm_blocked_f32(C, A, B, m, k, n);
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_batched_bias_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes,
    b_lean_obj_arg biasBytes, b_lean_obj_arg biasShapeObj,
    b_lean_obj_arg aStartsObj, b_lean_obj_arg bStartsObj, b_lean_obj_arg biasStartsObj,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj) {
  if (!lean_is_scalar(mObj) || !lean_is_scalar(kObj) || !lean_is_scalar(nObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t m = lean_unbox(mObj);
  const size_t k = lean_unbox(kObj);
  const size_t n = lean_unbox(nObj);
  const size_t outFloatsPer = m * n;

  const size_t batch = lean_array_size(aStartsObj);
  if (batch != lean_array_size(bStartsObj) || batch != lean_array_size(biasStartsObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t outFloats = batch * outFloatsPer;
  const size_t outBytes = outFloats * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *Cbase = (float *)lean_sarray_cptr(out); // NOLINT
  if (outFloats == 0) return out;

  const uint8_t *aBase = lean_sarray_cptr(aBytes);
  const uint8_t *bBase = lean_sarray_cptr(bBytes);
  const uint8_t *biasBaseBytes = lean_sarray_cptr(biasBytes);
  const size_t aBytesSz = lean_sarray_size(aBytes);
  const size_t bBytesSz = lean_sarray_size(bBytes);
  const size_t biasBytesSz = lean_sarray_size(biasBytes);
  const size_t aMatBytes = m * k * 4;
  const size_t bMatBytes = k * n * 4;

  size_t outDims[2] = {m, n};
  size_t biasDims[2];
  size_t biasStrides[2];
  if (!read_padded_dims(biasShapeObj, 2, biasDims)) {
    for (size_t i = 0; i < outFloats; i++) Cbase[i] = 0.0f;
    return out;
  }
  if (!is_broadcastable(biasDims, outDims, 2)) {
    for (size_t i = 0; i < outFloats; i++) Cbase[i] = 0.0f;
    return out;
  }
  contiguous_strides(biasDims, 2, biasStrides);
  broadcast_strides(biasDims, 2, biasStrides);
  const size_t biasNumel = prod_dims(biasDims, 2);
  const size_t biasMatBytes = biasNumel * 4;

  for (size_t bi = 0; bi < batch; bi++) {
    float *C = Cbase + bi * outFloatsPer;
    for (size_t i = 0; i < outFloatsPer; i++) C[i] = 0.0f;

    b_lean_obj_arg aStartObj = lean_array_get_core(aStartsObj, bi);
    b_lean_obj_arg bStartObj = lean_array_get_core(bStartsObj, bi);
    b_lean_obj_arg biasStartObj = lean_array_get_core(biasStartsObj, bi);
    if (!lean_is_scalar(aStartObj) || !lean_is_scalar(bStartObj) || !lean_is_scalar(biasStartObj)) {
      continue;
    }

    const size_t aStart = lean_unbox(aStartObj);
    const size_t bStart = lean_unbox(bStartObj);
    const size_t biasStart = lean_unbox(biasStartObj);
    if (aStart + aMatBytes > aBytesSz || bStart + bMatBytes > bBytesSz || biasStart + biasMatBytes > biasBytesSz) {
      continue;
    }

    const float *A = (const float *)(aBase + aStart);      // NOLINT
    const float *B = (const float *)(bBase + bStart);      // NOLINT
    const float *bias = (const float *)(biasBaseBytes + biasStart); // NOLINT

    for (size_t i = 0; i < m; i++) {
      const size_t bOffRow = i * biasStrides[0];
      float *cRow = C + i * n;
      for (size_t j = 0; j < n; j++) {
        cRow[j] = bias[bOffRow + j * biasStrides[1]];
      }
    }

    tg4_gemm_blocked_f32(C, A, B, m, k, n);
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_batched_bias_scale_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes,
    b_lean_obj_arg biasBytes, b_lean_obj_arg biasShapeObj,
    b_lean_obj_arg aStartsObj, b_lean_obj_arg bStartsObj, b_lean_obj_arg biasStartsObj,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj,
    uint32_t scaleBits) {
  if (!lean_is_scalar(mObj) || !lean_is_scalar(kObj) || !lean_is_scalar(nObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t m = lean_unbox(mObj);
  const size_t k = lean_unbox(kObj);
  const size_t n = lean_unbox(nObj);
  const float alpha = f32_from_bits(scaleBits);
  const size_t outFloatsPer = m * n;

  const size_t batch = lean_array_size(aStartsObj);
  if (batch != lean_array_size(bStartsObj) || batch != lean_array_size(biasStartsObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t outFloats = batch * outFloatsPer;
  const size_t outBytes = outFloats * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *Cbase = (float *)lean_sarray_cptr(out); // NOLINT
  if (outFloats == 0) return out;

  const uint8_t *aBase = lean_sarray_cptr(aBytes);
  const uint8_t *bBase = lean_sarray_cptr(bBytes);
  const uint8_t *biasBaseBytes = lean_sarray_cptr(biasBytes);
  const size_t aBytesSz = lean_sarray_size(aBytes);
  const size_t bBytesSz = lean_sarray_size(bBytes);
  const size_t biasBytesSz = lean_sarray_size(biasBytes);
  const size_t aMatBytes = m * k * 4;
  const size_t bMatBytes = k * n * 4;

  size_t outDims[2] = {m, n};
  size_t biasDims[2];
  size_t biasStrides[2];
  if (!read_padded_dims(biasShapeObj, 2, biasDims)) {
    for (size_t i = 0; i < outFloats; i++) Cbase[i] = 0.0f;
    return out;
  }
  if (!is_broadcastable(biasDims, outDims, 2)) {
    for (size_t i = 0; i < outFloats; i++) Cbase[i] = 0.0f;
    return out;
  }
  contiguous_strides(biasDims, 2, biasStrides);
  broadcast_strides(biasDims, 2, biasStrides);
  const size_t biasNumel = prod_dims(biasDims, 2);
  const size_t biasMatBytes = biasNumel * 4;

  for (size_t bi = 0; bi < batch; bi++) {
    float *C = Cbase + bi * outFloatsPer;
    for (size_t i = 0; i < outFloatsPer; i++) C[i] = 0.0f;

    b_lean_obj_arg aStartObj = lean_array_get_core(aStartsObj, bi);
    b_lean_obj_arg bStartObj = lean_array_get_core(bStartsObj, bi);
    b_lean_obj_arg biasStartObj = lean_array_get_core(biasStartsObj, bi);
    if (!lean_is_scalar(aStartObj) || !lean_is_scalar(bStartObj) || !lean_is_scalar(biasStartObj)) {
      continue;
    }

    const size_t aStart = lean_unbox(aStartObj);
    const size_t bStart = lean_unbox(bStartObj);
    const size_t biasStart = lean_unbox(biasStartObj);
    if (aStart + aMatBytes > aBytesSz || bStart + bMatBytes > bBytesSz || biasStart + biasMatBytes > biasBytesSz) {
      continue;
    }

    const float *A = (const float *)(aBase + aStart);      // NOLINT
    const float *B = (const float *)(bBase + bStart);      // NOLINT
    const float *bias = (const float *)(biasBaseBytes + biasStart); // NOLINT

    for (size_t i = 0; i < m; i++) {
      const size_t bOffRow = i * biasStrides[0];
      float *cRow = C + i * n;
      for (size_t j = 0; j < n; j++) {
        cRow[j] = bias[bOffRow + j * biasStrides[1]];
      }
    }

    tg4_gemm_blocked_alpha_f32(C, A, B, m, k, n, alpha);
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_batched_bias_relu_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes,
    b_lean_obj_arg biasBytes, b_lean_obj_arg biasShapeObj,
    b_lean_obj_arg aStartsObj, b_lean_obj_arg bStartsObj, b_lean_obj_arg biasStartsObj,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj) {
  lean_obj_res out = tg4_matmul_batched_bias_f32(aBytes, bBytes, biasBytes, biasShapeObj, aStartsObj, bStartsObj, biasStartsObj, mObj, kObj, nObj);
  const size_t outBytes = lean_sarray_size(out);
  if ((outBytes & 3) != 0) return out;
  const size_t outFloats = outBytes / 4;
  tg4_relu_f32_inplace_ptr(f32_ptr_mut(out), outFloats);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_batched_bias_scale_relu_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes,
    b_lean_obj_arg biasBytes, b_lean_obj_arg biasShapeObj,
    b_lean_obj_arg aStartsObj, b_lean_obj_arg bStartsObj, b_lean_obj_arg biasStartsObj,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj,
    uint32_t scaleBits) {
  lean_obj_res out =
      tg4_matmul_batched_bias_scale_f32(aBytes, bBytes, biasBytes, biasShapeObj, aStartsObj, bStartsObj, biasStartsObj, mObj, kObj, nObj, scaleBits);
  const size_t outBytes = lean_sarray_size(out);
  if ((outBytes & 3) != 0) return out;
  const size_t outFloats = outBytes / 4;
  tg4_relu_f32_inplace_ptr(f32_ptr_mut(out), outFloats);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_batched_bias2_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes,
    b_lean_obj_arg bias0Bytes, b_lean_obj_arg bias0ShapeObj,
    b_lean_obj_arg bias1Bytes, b_lean_obj_arg bias1ShapeObj,
    b_lean_obj_arg aStartsObj, b_lean_obj_arg bStartsObj,
    b_lean_obj_arg bias0StartsObj, b_lean_obj_arg bias1StartsObj,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj) {
  if (!lean_is_scalar(mObj) || !lean_is_scalar(kObj) || !lean_is_scalar(nObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t m = lean_unbox(mObj);
  const size_t k = lean_unbox(kObj);
  const size_t n = lean_unbox(nObj);
  const size_t outFloatsPer = m * n;

  const size_t batch = lean_array_size(aStartsObj);
  if (batch != lean_array_size(bStartsObj) || batch != lean_array_size(bias0StartsObj) || batch != lean_array_size(bias1StartsObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t outFloats = batch * outFloatsPer;
  const size_t outBytes = outFloats * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *Cbase = (float *)lean_sarray_cptr(out); // NOLINT
  if (outFloats == 0) return out;

  const uint8_t *aBase = lean_sarray_cptr(aBytes);
  const uint8_t *bBase = lean_sarray_cptr(bBytes);
  const uint8_t *bias0BaseBytes = lean_sarray_cptr(bias0Bytes);
  const uint8_t *bias1BaseBytes = lean_sarray_cptr(bias1Bytes);
  const size_t aBytesSz = lean_sarray_size(aBytes);
  const size_t bBytesSz = lean_sarray_size(bBytes);
  const size_t bias0BytesSz = lean_sarray_size(bias0Bytes);
  const size_t bias1BytesSz = lean_sarray_size(bias1Bytes);
  const size_t aMatBytes = m * k * 4;
  const size_t bMatBytes = k * n * 4;

  size_t outDims[2] = {m, n};

  size_t bias0Dims[2];
  size_t bias0Strides[2];
  if (!read_padded_dims(bias0ShapeObj, 2, bias0Dims)) {
    for (size_t i = 0; i < outFloats; i++) Cbase[i] = 0.0f;
    return out;
  }
  if (!is_broadcastable(bias0Dims, outDims, 2)) {
    for (size_t i = 0; i < outFloats; i++) Cbase[i] = 0.0f;
    return out;
  }
  contiguous_strides(bias0Dims, 2, bias0Strides);
  broadcast_strides(bias0Dims, 2, bias0Strides);
  const size_t bias0Numel = prod_dims(bias0Dims, 2);
  const size_t bias0MatBytes = bias0Numel * 4;

  size_t bias1Dims[2];
  size_t bias1Strides[2];
  if (!read_padded_dims(bias1ShapeObj, 2, bias1Dims)) {
    for (size_t i = 0; i < outFloats; i++) Cbase[i] = 0.0f;
    return out;
  }
  if (!is_broadcastable(bias1Dims, outDims, 2)) {
    for (size_t i = 0; i < outFloats; i++) Cbase[i] = 0.0f;
    return out;
  }
  contiguous_strides(bias1Dims, 2, bias1Strides);
  broadcast_strides(bias1Dims, 2, bias1Strides);
  const size_t bias1Numel = prod_dims(bias1Dims, 2);
  const size_t bias1MatBytes = bias1Numel * 4;

  for (size_t bi = 0; bi < batch; bi++) {
    float *C = Cbase + bi * outFloatsPer;
    for (size_t i = 0; i < outFloatsPer; i++) C[i] = 0.0f;

    b_lean_obj_arg aStartObj = lean_array_get_core(aStartsObj, bi);
    b_lean_obj_arg bStartObj = lean_array_get_core(bStartsObj, bi);
    b_lean_obj_arg bias0StartObj = lean_array_get_core(bias0StartsObj, bi);
    b_lean_obj_arg bias1StartObj = lean_array_get_core(bias1StartsObj, bi);
    if (!lean_is_scalar(aStartObj) || !lean_is_scalar(bStartObj) || !lean_is_scalar(bias0StartObj) || !lean_is_scalar(bias1StartObj)) {
      continue;
    }

    const size_t aStart = lean_unbox(aStartObj);
    const size_t bStart = lean_unbox(bStartObj);
    const size_t bias0Start = lean_unbox(bias0StartObj);
    const size_t bias1Start = lean_unbox(bias1StartObj);
    if (aStart + aMatBytes > aBytesSz || bStart + bMatBytes > bBytesSz ||
        bias0Start + bias0MatBytes > bias0BytesSz || bias1Start + bias1MatBytes > bias1BytesSz) {
      continue;
    }

    const float *A = (const float *)(aBase + aStart);          // NOLINT
    const float *B = (const float *)(bBase + bStart);          // NOLINT
    const float *bias0 = (const float *)(bias0BaseBytes + bias0Start); // NOLINT
    const float *bias1 = (const float *)(bias1BaseBytes + bias1Start); // NOLINT

    for (size_t i = 0; i < m; i++) {
      const size_t b0OffRow = i * bias0Strides[0];
      const size_t b1OffRow = i * bias1Strides[0];
      float *cRow = C + i * n;
      for (size_t j = 0; j < n; j++) {
        const size_t off0 = b0OffRow + j * bias0Strides[1];
        const size_t off1 = b1OffRow + j * bias1Strides[1];
        cRow[j] = bias0[off0] + bias1[off1];
      }
    }

    tg4_gemm_blocked_f32(C, A, B, m, k, n);
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_batched_bias2_scale_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes,
    b_lean_obj_arg bias0Bytes, b_lean_obj_arg bias0ShapeObj,
    b_lean_obj_arg bias1Bytes, b_lean_obj_arg bias1ShapeObj,
    b_lean_obj_arg aStartsObj, b_lean_obj_arg bStartsObj,
    b_lean_obj_arg bias0StartsObj, b_lean_obj_arg bias1StartsObj,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj,
    uint32_t scaleBits) {
  if (!lean_is_scalar(mObj) || !lean_is_scalar(kObj) || !lean_is_scalar(nObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t m = lean_unbox(mObj);
  const size_t k = lean_unbox(kObj);
  const size_t n = lean_unbox(nObj);
  const float alpha = f32_from_bits(scaleBits);
  const size_t outFloatsPer = m * n;

  const size_t batch = lean_array_size(aStartsObj);
  if (batch != lean_array_size(bStartsObj) || batch != lean_array_size(bias0StartsObj) || batch != lean_array_size(bias1StartsObj)) {
    return lean_mk_empty_byte_array(lean_box(0));
  }

  const size_t outFloats = batch * outFloatsPer;
  const size_t outBytes = outFloats * 4;
  lean_obj_res out = lean_alloc_sarray(1, outBytes, outBytes);
  float *Cbase = (float *)lean_sarray_cptr(out); // NOLINT
  if (outFloats == 0) return out;

  const uint8_t *aBase = lean_sarray_cptr(aBytes);
  const uint8_t *bBase = lean_sarray_cptr(bBytes);
  const uint8_t *bias0BaseBytes = lean_sarray_cptr(bias0Bytes);
  const uint8_t *bias1BaseBytes = lean_sarray_cptr(bias1Bytes);
  const size_t aBytesSz = lean_sarray_size(aBytes);
  const size_t bBytesSz = lean_sarray_size(bBytes);
  const size_t bias0BytesSz = lean_sarray_size(bias0Bytes);
  const size_t bias1BytesSz = lean_sarray_size(bias1Bytes);
  const size_t aMatBytes = m * k * 4;
  const size_t bMatBytes = k * n * 4;

  size_t outDims[2] = {m, n};

  size_t bias0Dims[2];
  size_t bias0Strides[2];
  if (!read_padded_dims(bias0ShapeObj, 2, bias0Dims)) {
    for (size_t i = 0; i < outFloats; i++) Cbase[i] = 0.0f;
    return out;
  }
  if (!is_broadcastable(bias0Dims, outDims, 2)) {
    for (size_t i = 0; i < outFloats; i++) Cbase[i] = 0.0f;
    return out;
  }
  contiguous_strides(bias0Dims, 2, bias0Strides);
  broadcast_strides(bias0Dims, 2, bias0Strides);
  const size_t bias0Numel = prod_dims(bias0Dims, 2);
  const size_t bias0MatBytes = bias0Numel * 4;

  size_t bias1Dims[2];
  size_t bias1Strides[2];
  if (!read_padded_dims(bias1ShapeObj, 2, bias1Dims)) {
    for (size_t i = 0; i < outFloats; i++) Cbase[i] = 0.0f;
    return out;
  }
  if (!is_broadcastable(bias1Dims, outDims, 2)) {
    for (size_t i = 0; i < outFloats; i++) Cbase[i] = 0.0f;
    return out;
  }
  contiguous_strides(bias1Dims, 2, bias1Strides);
  broadcast_strides(bias1Dims, 2, bias1Strides);
  const size_t bias1Numel = prod_dims(bias1Dims, 2);
  const size_t bias1MatBytes = bias1Numel * 4;

  for (size_t bi = 0; bi < batch; bi++) {
    float *C = Cbase + bi * outFloatsPer;
    for (size_t i = 0; i < outFloatsPer; i++) C[i] = 0.0f;

    b_lean_obj_arg aStartObj = lean_array_get_core(aStartsObj, bi);
    b_lean_obj_arg bStartObj = lean_array_get_core(bStartsObj, bi);
    b_lean_obj_arg bias0StartObj = lean_array_get_core(bias0StartsObj, bi);
    b_lean_obj_arg bias1StartObj = lean_array_get_core(bias1StartsObj, bi);
    if (!lean_is_scalar(aStartObj) || !lean_is_scalar(bStartObj) || !lean_is_scalar(bias0StartObj) || !lean_is_scalar(bias1StartObj)) {
      continue;
    }

    const size_t aStart = lean_unbox(aStartObj);
    const size_t bStart = lean_unbox(bStartObj);
    const size_t bias0Start = lean_unbox(bias0StartObj);
    const size_t bias1Start = lean_unbox(bias1StartObj);
    if (aStart + aMatBytes > aBytesSz || bStart + bMatBytes > bBytesSz ||
        bias0Start + bias0MatBytes > bias0BytesSz || bias1Start + bias1MatBytes > bias1BytesSz) {
      continue;
    }

    const float *A = (const float *)(aBase + aStart);          // NOLINT
    const float *B = (const float *)(bBase + bStart);          // NOLINT
    const float *bias0 = (const float *)(bias0BaseBytes + bias0Start); // NOLINT
    const float *bias1 = (const float *)(bias1BaseBytes + bias1Start); // NOLINT

    for (size_t i = 0; i < m; i++) {
      const size_t b0OffRow = i * bias0Strides[0];
      const size_t b1OffRow = i * bias1Strides[0];
      float *cRow = C + i * n;
      for (size_t j = 0; j < n; j++) {
        const size_t off0 = b0OffRow + j * bias0Strides[1];
        const size_t off1 = b1OffRow + j * bias1Strides[1];
        cRow[j] = bias0[off0] + bias1[off1];
      }
    }

    tg4_gemm_blocked_alpha_f32(C, A, B, m, k, n, alpha);
  }

  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_batched_bias2_relu_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes,
    b_lean_obj_arg bias0Bytes, b_lean_obj_arg bias0ShapeObj,
    b_lean_obj_arg bias1Bytes, b_lean_obj_arg bias1ShapeObj,
    b_lean_obj_arg aStartsObj, b_lean_obj_arg bStartsObj,
    b_lean_obj_arg bias0StartsObj, b_lean_obj_arg bias1StartsObj,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj) {
  lean_obj_res out =
      tg4_matmul_batched_bias2_f32(
          aBytes, bBytes,
          bias0Bytes, bias0ShapeObj,
          bias1Bytes, bias1ShapeObj,
          aStartsObj, bStartsObj,
          bias0StartsObj, bias1StartsObj,
          mObj, kObj, nObj);
  const size_t outBytes = lean_sarray_size(out);
  if ((outBytes & 3) != 0) return out;
  const size_t outFloats = outBytes / 4;
  tg4_relu_f32_inplace_ptr(f32_ptr_mut(out), outFloats);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_batched_bias2_scale_relu_f32(
    b_lean_obj_arg aBytes, b_lean_obj_arg bBytes,
    b_lean_obj_arg bias0Bytes, b_lean_obj_arg bias0ShapeObj,
    b_lean_obj_arg bias1Bytes, b_lean_obj_arg bias1ShapeObj,
    b_lean_obj_arg aStartsObj, b_lean_obj_arg bStartsObj,
    b_lean_obj_arg bias0StartsObj, b_lean_obj_arg bias1StartsObj,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj,
    uint32_t scaleBits) {
  lean_obj_res out =
      tg4_matmul_batched_bias2_scale_f32(
          aBytes, bBytes,
          bias0Bytes, bias0ShapeObj,
          bias1Bytes, bias1ShapeObj,
          aStartsObj, bStartsObj,
          bias0StartsObj, bias1StartsObj,
          mObj, kObj, nObj,
          scaleBits);
  const size_t outBytes = lean_sarray_size(out);
  if ((outBytes & 3) != 0) return out;
  const size_t outFloats = outBytes / 4;
  tg4_relu_f32_inplace_ptr(f32_ptr_mut(out), outFloats);
  return out;
}

LEAN_EXPORT lean_obj_res tg4_matmul_f64(
    b_lean_obj_arg aObj, b_lean_obj_arg bObj,
    b_lean_obj_arg mObj, b_lean_obj_arg kObj, b_lean_obj_arg nObj) {
  if (!lean_is_scalar(mObj) || !lean_is_scalar(kObj) || !lean_is_scalar(nObj)) {
    return lean_mk_empty_float_array(lean_box(0));
  }

  const size_t m = lean_unbox(mObj);
  const size_t k = lean_unbox(kObj);
  const size_t n = lean_unbox(nObj);
  const size_t outSize = m * n;

  lean_obj_res out = lean_alloc_sarray(sizeof(double), outSize, outSize); // NOLINT
  double *C = lean_float_array_cptr(out);
  const double *A = lean_float_array_cptr(aObj);
  const double *B = lean_float_array_cptr(bObj);

  // Basic bounds check for safety; if wrong, return zeros.
  const size_t aSize = lean_sarray_size(aObj);
  const size_t bSize = lean_sarray_size(bObj);
  if (aSize < m * k || bSize < k * n) {
    for (size_t i = 0; i < outSize; i++) C[i] = 0.0;
    return out;
  }

  for (size_t i = 0; i < outSize; i++) C[i] = 0.0;

  // Simple cache blocking. These constants are conservative and portable.
  const size_t BM = 32;
  const size_t BN = 64;
  const size_t BK = 32;

  for (size_t i0 = 0; i0 < m; i0 += BM) {
    const size_t iMax = min_size(i0 + BM, m);
    for (size_t k0 = 0; k0 < k; k0 += BK) {
      const size_t kMax = min_size(k0 + BK, k);
      for (size_t j0 = 0; j0 < n; j0 += BN) {
        const size_t jMax = min_size(j0 + BN, n);
        for (size_t i = i0; i < iMax; i++) {
          const double *aRow = A + i * k;
          double *cRow = C + i * n;
          for (size_t t = k0; t < kMax; t++) {
            const double a = aRow[t];
            const double *bRow = B + t * n;
            for (size_t j = j0; j < jMax; j++) {
              cRow[j] += a * bRow[j];
            }
          }
        }
      }
    }
  }

  return out;
}
