#include <metal_stdlib>
using namespace metal;

// Simple matmul for benchmarking (non-tiled)
kernel void matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Tiled matmul with shared memory (16x16 tiles)
kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    threadgroup float As[16][16];
    threadgroup float Bs[16][16];

    uint row = tgid.y * 16 + tid.y;
    uint col = tgid.x * 16 + tid.x;

    float sum = 0.0f;
    for (uint t = 0; t < K; t += 16) {
        // Load tiles into shared memory
        As[tid.y][tid.x] = (row < M && t + tid.x < K) ? A[row * K + t + tid.x] : 0.0f;
        Bs[tid.y][tid.x] = (t + tid.y < K && col < N) ? B[(t + tid.y) * N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (uint k = 0; k < 16; k++) {
            sum += As[tid.y][k] * Bs[k][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
