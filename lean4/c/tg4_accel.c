#include <lean/lean.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

static inline const uint8_t* byte_array_cptr(b_lean_obj_arg a) {
  return (const uint8_t*)lean_sarray_cptr(a);
}

static inline uint8_t* byte_array_cptr_mut(lean_object* a) {
  return (uint8_t*)lean_sarray_cptr(a);
}

static inline size_t byte_array_size(b_lean_obj_arg a) {
  return lean_sarray_size(a);
}

static inline lean_object* mk_byte_array(size_t size) {
  return lean_alloc_sarray(1, size, size);
}

static inline void accel_panic(const char* msg) {
  lean_internal_panic(msg);
}

static inline int mul_overflow_size(size_t a, size_t b, size_t* out) {
  if (a == 0 || b == 0) {
    *out = 0;
    return 0;
  }
  if (a > SIZE_MAX / b) {
    return 1;
  }
  *out = a * b;
  return 0;
}

static inline int add_overflow_size(size_t a, size_t b, size_t* out) {
  if (a > SIZE_MAX - b) {
    return 1;
  }
  *out = a + b;
  return 0;
}

static inline void accel_bounds_check(const char* fn, size_t size, size_t offset, size_t len) {
  if (offset > size || len > size - offset) {
    accel_panic(fn);
  }
}

// Sum all bytes in a ByteArray slice.
LEAN_EXPORT uint64_t lean_accel_sum_u8(b_lean_obj_arg ba, size_t offset, size_t len) {
  size_t size = byte_array_size(ba);
  accel_bounds_check("lean_accel_sum_u8: offset+len out of bounds", size, offset, len);
  const uint8_t* data = byte_array_cptr(ba) + offset;
  uint64_t sum = 0;
  for (size_t i = 0; i < len; ++i) {
    sum += data[i];
  }
  return sum;
}

// Sum float32 values in a ByteArray slice (offset/len are in elements).
LEAN_EXPORT double lean_accel_sum_f32(b_lean_obj_arg ba, size_t offset, size_t num_floats) {
  size_t size = byte_array_size(ba);
  size_t byte_offset = 0;
  size_t byte_len = 0;
  if (mul_overflow_size(offset, sizeof(float), &byte_offset)) {
    accel_panic("lean_accel_sum_f32: byte offset overflow");
  }
  if (mul_overflow_size(num_floats, sizeof(float), &byte_len)) {
    accel_panic("lean_accel_sum_f32: byte length overflow");
  }
  accel_bounds_check("lean_accel_sum_f32: offset+len out of bounds", size, byte_offset, byte_len);
  const float* data = (const float*)(byte_array_cptr(ba) + byte_offset);
  double sum = 0.0;
  for (size_t i = 0; i < num_floats; ++i) {
    sum += (double)data[i];
  }
  return sum;
}

// Convert uint8 to float32 and normalize by 255. Returns ByteArray.
LEAN_EXPORT lean_obj_res lean_accel_normalize_u8_to_f32(b_lean_obj_arg ba, size_t offset, size_t len) {
  size_t size = byte_array_size(ba);
  accel_bounds_check("lean_accel_normalize_u8_to_f32: offset+len out of bounds", size, offset, len);
  const uint8_t* data = byte_array_cptr(ba) + offset;
  size_t out_bytes = 0;
  if (mul_overflow_size(len, sizeof(float), &out_bytes)) {
    accel_panic("lean_accel_normalize_u8_to_f32: output size overflow");
  }
  lean_object* out = mk_byte_array(out_bytes);
  float* out_f = (float*)byte_array_cptr_mut(out);
  const float scale = 1.0f / 255.0f;
  for (size_t i = 0; i < len; ++i) {
    out_f[i] = (float)data[i] * scale;
  }
  return out;
}

// Fused normalize + sum: u8 -> f32/255 -> sum.
LEAN_EXPORT double lean_accel_normalize_sum_u8(b_lean_obj_arg ba, size_t offset, size_t len) {
  size_t size = byte_array_size(ba);
  accel_bounds_check("lean_accel_normalize_sum_u8: offset+len out of bounds", size, offset, len);
  const uint8_t* data = byte_array_cptr(ba) + offset;
  const double scale = 1.0 / 255.0;
  double sum = 0.0;
  for (size_t i = 0; i < len; ++i) {
    sum += (double)data[i] * scale;
  }
  return sum;
}

// Matrix multiply C = A @ B, float32 row-major. Offsets are in elements.
LEAN_EXPORT lean_obj_res lean_accel_matmul_f32(b_lean_obj_arg a, size_t a_offset,
    b_lean_obj_arg b, size_t b_offset, size_t M, size_t K, size_t N) {
  size_t a_size = byte_array_size(a);
  size_t b_size = byte_array_size(b);
  size_t a_byte_offset = 0;
  size_t b_byte_offset = 0;
  size_t a_elems = 0;
  size_t b_elems = 0;
  size_t out_elems = 0;
  if (mul_overflow_size(a_offset, sizeof(float), &a_byte_offset)) {
    accel_panic("lean_accel_matmul_f32: A byte offset overflow");
  }
  if (mul_overflow_size(b_offset, sizeof(float), &b_byte_offset)) {
    accel_panic("lean_accel_matmul_f32: B byte offset overflow");
  }
  if (mul_overflow_size(M, K, &a_elems)) {
    accel_panic("lean_accel_matmul_f32: A element count overflow");
  }
  if (mul_overflow_size(K, N, &b_elems)) {
    accel_panic("lean_accel_matmul_f32: B element count overflow");
  }
  if (mul_overflow_size(M, N, &out_elems)) {
    accel_panic("lean_accel_matmul_f32: output element count overflow");
  }
  size_t a_need_bytes = 0;
  size_t b_need_bytes = 0;
  size_t out_need_bytes = 0;
  size_t a_total_elems = 0;
  size_t b_total_elems = 0;
  if (add_overflow_size(a_offset, a_elems, &a_total_elems)) {
    accel_panic("lean_accel_matmul_f32: A size overflow");
  }
  if (add_overflow_size(b_offset, b_elems, &b_total_elems)) {
    accel_panic("lean_accel_matmul_f32: B size overflow");
  }
  if (mul_overflow_size(a_total_elems, sizeof(float), &a_need_bytes)) {
    accel_panic("lean_accel_matmul_f32: A size overflow");
  }
  if (mul_overflow_size(b_total_elems, sizeof(float), &b_need_bytes)) {
    accel_panic("lean_accel_matmul_f32: B size overflow");
  }
  if (mul_overflow_size(out_elems, sizeof(float), &out_need_bytes)) {
    accel_panic("lean_accel_matmul_f32: output size overflow");
  }
  accel_bounds_check("lean_accel_matmul_f32: A out of bounds", a_size, a_byte_offset, a_need_bytes - a_byte_offset);
  accel_bounds_check("lean_accel_matmul_f32: B out of bounds", b_size, b_byte_offset, b_need_bytes - b_byte_offset);
  const float* a_data = (const float*)(byte_array_cptr(a) + a_byte_offset);
  const float* b_data = (const float*)(byte_array_cptr(b) + b_byte_offset);
  lean_object* out = mk_byte_array(out_need_bytes);
  float* out_f = (float*)byte_array_cptr_mut(out);

#if defined(__APPLE__)
  // Use Accelerate if available.
  vDSP_mmul(a_data, 1, b_data, 1, out_f, 1, M, N, K);
#else
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        acc += a_data[i * K + k] * b_data[k * N + j];
      }
      out_f[i * N + j] = acc;
    }
  }
#endif
  return out;
}

// Check if Accelerate is available.
LEAN_EXPORT uint8_t lean_accel_available(b_lean_obj_arg unit) {
  (void)unit;
#if defined(__APPLE__)
  return 1;
#else
  return 0;
#endif
}

// Return a conservative default batch size.
LEAN_EXPORT size_t lean_accel_optimal_batch_size(b_lean_obj_arg unit) {
  (void)unit;
  return 256;
}
