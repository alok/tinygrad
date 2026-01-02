/**
 * tg4_accel.c - Accelerate/vDSP FFI for vectorized operations
 *
 * Uses Apple's Accelerate framework for SIMD operations on CPU.
 * These are the same primitives numpy uses on macOS.
 */

#include <lean/lean.h>
#include <stdint.h>
#include <string.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define HAS_ACCELERATE 1
#else
#define HAS_ACCELERATE 0
#endif

// ============================================================================
// Sum operations
// ============================================================================

/**
 * Sum all bytes in a ByteArray slice.
 * Uses vDSP_sve (vectorized sum) after converting to float.
 */
LEAN_EXPORT uint64_t lean_accel_sum_u8(b_lean_obj_arg ba, size_t offset, size_t len) {
    uint8_t* data = lean_sarray_cptr(ba);

    if (offset + len > lean_sarray_size(ba)) {
        len = lean_sarray_size(ba) - offset;
    }

#if HAS_ACCELERATE
    // For small arrays, scalar is faster than vDSP setup overhead
    if (len < 256) {
        uint64_t sum = 0;
        for (size_t i = 0; i < len; i++) {
            sum += data[offset + i];
        }
        return sum;
    }

    // Convert u8 to float and sum with vDSP
    // vDSP_vfltu8 converts uint8 to float
    // vDSP_sve sums the float array
    float* temp = (float*)malloc(len * sizeof(float));
    if (!temp) {
        // Fallback to scalar
        uint64_t sum = 0;
        for (size_t i = 0; i < len; i++) {
            sum += data[offset + i];
        }
        return sum;
    }

    vDSP_vfltu8(data + offset, 1, temp, 1, len);

    float result;
    vDSP_sve(temp, 1, &result, len);

    free(temp);
    return (uint64_t)result;
#else
    // Scalar fallback for non-Apple platforms
    uint64_t sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum += data[offset + i];
    }
    return sum;
#endif
}

/**
 * Sum all floats in a ByteArray (interpreted as float32).
 */
LEAN_EXPORT double lean_accel_sum_f32(b_lean_obj_arg ba, size_t offset, size_t num_floats) {
    uint8_t* raw = lean_sarray_cptr(ba);
    float* data = (float*)(raw + offset);

    size_t byte_end = offset + num_floats * sizeof(float);
    if (byte_end > lean_sarray_size(ba)) {
        num_floats = (lean_sarray_size(ba) - offset) / sizeof(float);
    }

#if HAS_ACCELERATE
    float result;
    vDSP_sve(data, 1, &result, num_floats);
    return (double)result;
#else
    double sum = 0.0;
    for (size_t i = 0; i < num_floats; i++) {
        sum += data[i];
    }
    return sum;
#endif
}

// ============================================================================
// Normalization operations
// ============================================================================

/**
 * Convert uint8 array to float32 and normalize by 255.
 * Returns a new ByteArray containing float32 data.
 *
 * This is the hot path for image preprocessing.
 */
LEAN_EXPORT lean_obj_res lean_accel_normalize_u8_to_f32(b_lean_obj_arg ba, size_t offset, size_t len) {
    uint8_t* src = lean_sarray_cptr(ba);

    if (offset + len > lean_sarray_size(ba)) {
        len = lean_sarray_size(ba) - offset;
    }

    // Allocate output ByteArray for float32 data
    size_t out_size = len * sizeof(float);
    lean_object* out = lean_alloc_sarray(1, out_size, out_size);
    float* dst = (float*)lean_sarray_cptr(out);

#if HAS_ACCELERATE
    // Convert uint8 to float
    vDSP_vfltu8(src + offset, 1, dst, 1, len);

    // Divide by 255.0
    float scale = 255.0f;
    vDSP_vsdiv(dst, 1, &scale, dst, 1, len);
#else
    // Scalar fallback
    for (size_t i = 0; i < len; i++) {
        dst[i] = (float)src[offset + i] / 255.0f;
    }
#endif

    return out;
}

/**
 * Sum normalized values (u8 -> f32/255 -> sum) in one pass.
 * More efficient than normalize + sum separately.
 */
LEAN_EXPORT double lean_accel_normalize_sum_u8(b_lean_obj_arg ba, size_t offset, size_t len) {
    uint8_t* data = lean_sarray_cptr(ba);

    if (offset + len > lean_sarray_size(ba)) {
        len = lean_sarray_size(ba) - offset;
    }

#if HAS_ACCELERATE
    // Allocate temp buffer for floats
    float* temp = (float*)malloc(len * sizeof(float));
    if (!temp) {
        // Fallback
        double sum = 0.0;
        for (size_t i = 0; i < len; i++) {
            sum += (double)data[offset + i] / 255.0;
        }
        return sum;
    }

    // Convert u8 to float
    vDSP_vfltu8(data + offset, 1, temp, 1, len);

    // Sum
    float result;
    vDSP_sve(temp, 1, &result, len);

    free(temp);

    // Divide by 255 (faster to do once at the end)
    return (double)result / 255.0;
#else
    double sum = 0.0;
    for (size_t i = 0; i < len; i++) {
        sum += (double)data[offset + i] / 255.0;
    }
    return sum;
#endif
}

// ============================================================================
// Matrix operations (for forward pass benchmarking)
// ============================================================================

/**
 * Matrix multiply: C = A @ B
 * A: [M, K], B: [K, N], C: [M, N]
 * All matrices are float32 in row-major order.
 */
LEAN_EXPORT lean_obj_res lean_accel_matmul_f32(
    b_lean_obj_arg a_ba, size_t a_offset,
    b_lean_obj_arg b_ba, size_t b_offset,
    size_t M, size_t K, size_t N
) {
    float* A = (float*)(lean_sarray_cptr(a_ba) + a_offset);
    float* B = (float*)(lean_sarray_cptr(b_ba) + b_offset);

    // Allocate output
    size_t out_size = M * N * sizeof(float);
    lean_object* out = lean_alloc_sarray(1, out_size, out_size);
    float* C = (float*)lean_sarray_cptr(out);

#if HAS_ACCELERATE
    // cblas_sgemm for matrix multiply
    // C = alpha * A * B + beta * C
    cblas_sgemm(
        CblasRowMajor,  // Row-major order
        CblasNoTrans,   // Don't transpose A
        CblasNoTrans,   // Don't transpose B
        (int)M, (int)N, (int)K,  // Dimensions
        1.0f,           // alpha
        A, (int)K,      // A matrix, leading dimension
        B, (int)N,      // B matrix, leading dimension
        0.0f,           // beta
        C, (int)N       // C matrix, leading dimension
    );
#else
    // Naive matmul fallback
    memset(C, 0, out_size);
    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            for (size_t j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
#endif

    return out;
}

// ============================================================================
// Utility functions
// ============================================================================

/**
 * Check if Accelerate is available.
 */
LEAN_EXPORT uint8_t lean_accel_available(lean_object* _unused) {
#if HAS_ACCELERATE
    return 1;
#else
    return 0;
#endif
}

/**
 * Get optimal batch size for current hardware.
 * Returns a reasonable default based on cache sizes.
 */
LEAN_EXPORT size_t lean_accel_optimal_batch_size(lean_object* _unused) {
    // L2 cache is typically 256KB-1MB per core
    // For float32 operations, aim to fit working set in L2
    // 64 images * 784 pixels * 4 bytes = 200KB
    return 64;
}
