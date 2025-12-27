// matmul_runner.m - Metal matmul benchmark
// Compile: clang -framework Metal -framework Foundation -O3 -o matmul_runner matmul_runner.m

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

static double get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

int main(int argc, char* argv[]) {
    @autoreleasepool {
        uint32_t M = argc > 1 ? atoi(argv[1]) : 1024;
        uint32_t K = argc > 2 ? atoi(argv[2]) : 1024;
        uint32_t N = argc > 3 ? atoi(argv[3]) : 1024;
        int use_tiled = argc > 4 ? atoi(argv[4]) : 1;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        printf("Device: %s\n", [[device name] UTF8String]);
        printf("Matmul: [%u, %u] @ [%u, %u] = [%u, %u]\n", M, K, K, N, M, N);

        // Load shader
        NSError* error = nil;
        NSString* source = [NSString stringWithContentsOfFile:@"test_matmul.metal"
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (error) {
            fprintf(stderr, "Failed to read shader\n");
            return 1;
        }

        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (error) {
            fprintf(stderr, "Compile error: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }

        NSString* kernelName = use_tiled ? @"matmul_tiled" : @"matmul";
        id<MTLFunction> function = [library newFunctionWithName:kernelName];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        printf("Using kernel: %s\n", [kernelName UTF8String]);

        // Allocate buffers
        id<MTLBuffer> bufA = [device newBufferWithLength:M * K * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithLength:K * N * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [device newBufferWithLength:M * N * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufM = [device newBufferWithBytes:&M length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK = [device newBufferWithBytes:&K length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufN = [device newBufferWithBytes:&N length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        // Initialize
        float* A = (float*)[bufA contents];
        float* B = (float*)[bufB contents];
        for (uint32_t i = 0; i < M * K; i++) A[i] = (float)(i % 100) / 100.0f;
        for (uint32_t i = 0; i < K * N; i++) B[i] = (float)(i % 100) / 100.0f;

        id<MTLCommandQueue> queue = [device newCommandQueue];

        // Warmup
        for (int w = 0; w < 3; w++) {
            id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:bufA offset:0 atIndex:0];
            [encoder setBuffer:bufB offset:0 atIndex:1];
            [encoder setBuffer:bufC offset:0 atIndex:2];
            [encoder setBuffer:bufM offset:0 atIndex:3];
            [encoder setBuffer:bufK offset:0 atIndex:4];
            [encoder setBuffer:bufN offset:0 atIndex:5];

            if (use_tiled) {
                MTLSize threadsPerGroup = MTLSizeMake(16, 16, 1);
                MTLSize threadgroups = MTLSizeMake((N + 15) / 16, (M + 15) / 16, 1);
                [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
            } else {
                MTLSize threads = MTLSizeMake(N, M, 1);
                MTLSize threadsPerGroup = MTLSizeMake(16, 16, 1);
                [encoder dispatchThreads:threads threadsPerThreadgroup:threadsPerGroup];
            }
            [encoder endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
        }

        // Benchmark
        int iterations = 20;
        double total_time = 0;

        for (int iter = 0; iter < iterations; iter++) {
            id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:bufA offset:0 atIndex:0];
            [encoder setBuffer:bufB offset:0 atIndex:1];
            [encoder setBuffer:bufC offset:0 atIndex:2];
            [encoder setBuffer:bufM offset:0 atIndex:3];
            [encoder setBuffer:bufK offset:0 atIndex:4];
            [encoder setBuffer:bufN offset:0 atIndex:5];

            if (use_tiled) {
                MTLSize threadsPerGroup = MTLSizeMake(16, 16, 1);
                MTLSize threadgroups = MTLSizeMake((N + 15) / 16, (M + 15) / 16, 1);
                [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerGroup];
            } else {
                MTLSize threads = MTLSizeMake(N, M, 1);
                MTLSize threadsPerGroup = MTLSizeMake(16, 16, 1);
                [encoder dispatchThreads:threads threadsPerThreadgroup:threadsPerGroup];
            }

            double start = get_time_us();
            [encoder endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
            total_time += get_time_us() - start;
        }

        double avg_time_ms = total_time / iterations / 1000.0;
        double flops = 2.0 * M * K * N;  // 2 ops per multiply-add
        double gflops = (flops / (avg_time_ms / 1000.0)) / 1e9;

        printf("\n=== Results ===\n");
        printf("Time: %.3f ms\n", avg_time_ms);
        printf("Throughput: %.2f GFLOP/s\n", gflops);

        // Verify
        float* C = (float*)[bufC contents];
        printf("C[0,0] = %.4f\n", C[0]);

        return 0;
    }
}
