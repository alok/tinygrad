// metal_runner.m - Minimal Metal kernel runner for TinyGrad4
// Compile: clang -framework Metal -framework Foundation -o metal_runner metal_runner.m

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

static double get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

int main(int argc, char* argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "Usage: %s <shader.metal> <kernel_name> [size]\n", argv[0]);
            return 1;
        }

        const char* shader_path = argv[1];
        const char* kernel_name = argv[2];
        size_t size = argc > 3 ? atol(argv[3]) : 1024 * 1024;  // 1M elements default

        // Get Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Metal not supported\n");
            return 1;
        }
        printf("Device: %s\n", [[device name] UTF8String]);

        // Read shader source
        NSString* path = [NSString stringWithUTF8String:shader_path];
        NSError* error = nil;
        NSString* source = [NSString stringWithContentsOfFile:path
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (error) {
            fprintf(stderr, "Failed to read shader: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }

        // Compile shader
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;

        double compile_start = get_time_us();
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                      options:options
                                                        error:&error];
        double compile_time = get_time_us() - compile_start;

        if (error) {
            fprintf(stderr, "Shader compilation failed: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }
        printf("Compile time: %.2f ms\n", compile_time / 1000.0);

        // Get kernel function
        NSString* funcName = [NSString stringWithUTF8String:kernel_name];
        id<MTLFunction> function = [library newFunctionWithName:funcName];
        if (!function) {
            fprintf(stderr, "Kernel '%s' not found\n", kernel_name);
            // List available functions
            NSArray* names = [library functionNames];
            fprintf(stderr, "Available: ");
            for (NSString* name in names) {
                fprintf(stderr, "%s ", [name UTF8String]);
            }
            fprintf(stderr, "\n");
            return 1;
        }

        // Create pipeline
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                     error:&error];
        if (error) {
            fprintf(stderr, "Pipeline creation failed: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }

        // Allocate buffers
        size_t buf_size = size * sizeof(float);
        id<MTLBuffer> buf0 = [device newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf1 = [device newBufferWithLength:buf_size options:MTLResourceStorageModeShared];

        // Initialize input data
        float* data0 = (float*)[buf0 contents];
        float* data1 = (float*)[buf1 contents];
        for (size_t i = 0; i < size; i++) {
            data0[i] = (float)(i % 1000) / 1000.0f;
            data1[i] = (float)((i + 500) % 1000) / 1000.0f;
        }

        // Create command queue
        id<MTLCommandQueue> queue = [device newCommandQueue];

        // Warmup
        for (int w = 0; w < 3; w++) {
            id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:buf0 offset:0 atIndex:0];
            [encoder setBuffer:buf1 offset:0 atIndex:1];

            NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
            MTLSize threads = MTLSizeMake(size, 1, 1);
            MTLSize threadgroups = MTLSizeMake((size + threadGroupSize - 1) / threadGroupSize, 1, 1);
            MTLSize groupSize = MTLSizeMake(threadGroupSize, 1, 1);

            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:groupSize];
            [encoder endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
        }

        // Benchmark
        int iterations = 100;
        double total_time = 0;

        for (int iter = 0; iter < iterations; iter++) {
            id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:buf0 offset:0 atIndex:0];
            [encoder setBuffer:buf1 offset:0 atIndex:1];

            NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
            MTLSize threadgroups = MTLSizeMake((size + threadGroupSize - 1) / threadGroupSize, 1, 1);
            MTLSize groupSize = MTLSizeMake(threadGroupSize, 1, 1);

            double start = get_time_us();
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:groupSize];
            [encoder endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
            double elapsed = get_time_us() - start;
            total_time += elapsed;
        }

        double avg_time_us = total_time / iterations;
        double flops = (double)size;  // 1 flop per element for simple ops
        double gflops = (flops / avg_time_us) * 1e6 / 1e9;
        double bandwidth_gb = (2.0 * buf_size / avg_time_us) * 1e6 / 1e9;  // read + write

        printf("\n=== Results ===\n");
        printf("Size: %zu elements (%.2f MB)\n", size, buf_size / 1e6);
        printf("Time: %.2f μs\n", avg_time_us);
        printf("Throughput: %.2f GFLOP/s\n", gflops);
        printf("Bandwidth: %.2f GB/s\n", bandwidth_gb);

        // Verify first few results
        printf("\nFirst 5 outputs: ");
        for (int i = 0; i < 5 && i < (int)size; i++) {
            printf("%.4f ", data1[i]);
        }
        printf("\n");

        return 0;
    }
}
