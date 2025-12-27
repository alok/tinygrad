#include <metal_stdlib>
using namespace metal;

kernel void test_add(
    device const float* buf0 [[buffer(0)]],
    device float* buf1 [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
  float v2 = buf0[gid] + buf1[gid];
  buf1[gid] = v2;
}
