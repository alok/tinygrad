// tg4_metal.m - Metal backend FFI for Lean TinyGrad4
// Compile with: clang -framework Metal -framework Foundation -c tg4_metal.m

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <lean/lean.h>
#include <string.h>

// ============================================================================
// Global Metal State
// ============================================================================

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLCommandBuffer> g_last_cmd_buf = nil;  // Track last command buffer for sync

// ============================================================================
// Pipeline Cache
// ============================================================================

static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* g_pipeline_cache = nil;
static uint64_t g_cache_hits = 0;
static uint64_t g_cache_misses = 0;

static void ensure_metal_init(void) {
    if (g_device == nil) {
        g_device = MTLCreateSystemDefaultDevice();
        g_queue = [g_device newCommandQueue];
        g_pipeline_cache = [NSMutableDictionary dictionary];
    }
}

// ============================================================================
// Buffer Type (opaque to Lean)
// ============================================================================

typedef struct {
    id<MTLBuffer> buffer;
    size_t numel;
} TG4MetalBuffer;

// External class for Lean FFI
static void tg4_metal_buffer_finalize(void* ptr) {
    TG4MetalBuffer* buf = (TG4MetalBuffer*)ptr;
    if (buf) {
        buf->buffer = nil;  // Release MTLBuffer
        free(buf);
    }
}

static void tg4_metal_buffer_foreach(void* ptr, b_lean_obj_arg fn) {
    (void)ptr;
    (void)fn;
}

static lean_external_class* g_metal_buffer_class = NULL;

static lean_external_class* get_metal_buffer_class(void) {
    if (g_metal_buffer_class == NULL) {
        g_metal_buffer_class = lean_register_external_class(
            tg4_metal_buffer_finalize,
            tg4_metal_buffer_foreach
        );
    }
    return g_metal_buffer_class;
}

static inline TG4MetalBuffer* metal_buffer_unbox(lean_object* obj) {
    return (TG4MetalBuffer*)lean_get_external_data(obj);
}

static inline lean_object* metal_buffer_box(TG4MetalBuffer* buf) {
    return lean_alloc_external(get_metal_buffer_class(), buf);
}

// ============================================================================
// Program Type (opaque to Lean)
// ============================================================================

typedef struct {
    id<MTLComputePipelineState> pipeline;
    NSString* name;
} TG4MetalProgram;

static void tg4_metal_program_finalize(void* ptr) {
    TG4MetalProgram* prog = (TG4MetalProgram*)ptr;
    if (prog) {
        prog->pipeline = nil;
        prog->name = nil;
        free(prog);
    }
}

static void tg4_metal_program_foreach(void* ptr, b_lean_obj_arg fn) {
    (void)ptr;
    (void)fn;
}

static lean_external_class* g_metal_program_class = NULL;

static lean_external_class* get_metal_program_class(void) {
    if (g_metal_program_class == NULL) {
        g_metal_program_class = lean_register_external_class(
            tg4_metal_program_finalize,
            tg4_metal_program_foreach
        );
    }
    return g_metal_program_class;
}

static inline TG4MetalProgram* metal_program_unbox(lean_object* obj) {
    return (TG4MetalProgram*)lean_get_external_data(obj);
}

static inline lean_object* metal_program_box(TG4MetalProgram* prog) {
    return lean_alloc_external(get_metal_program_class(), prog);
}

// ============================================================================
// Allocator Functions
// ============================================================================

// Allocate a buffer with n float elements
// @[extern "tg4_metal_alloc"]
lean_obj_res tg4_metal_alloc(b_lean_obj_arg n_obj, lean_object* world) {
    @autoreleasepool {
        ensure_metal_init();

        size_t n = lean_usize_of_nat(n_obj);
        size_t bytes = n * sizeof(float);

        TG4MetalBuffer* buf = malloc(sizeof(TG4MetalBuffer));
        buf->buffer = [g_device newBufferWithLength:bytes
                                            options:MTLResourceStorageModeShared];
        buf->numel = n;

        // Zero-initialize
        memset([buf->buffer contents], 0, bytes);

        return lean_io_result_mk_ok(metal_buffer_box(buf));
    }
}

// Free a buffer
// @[extern "tg4_metal_free"]
lean_obj_res tg4_metal_free(b_lean_obj_arg buf_obj, lean_object* world) {
    // Buffer will be freed by finalizer when ref count drops
    (void)buf_obj;
    return lean_io_result_mk_ok(lean_box(0));
}

// Copy data from FloatArray to Metal buffer
// @[extern "tg4_metal_copy_in"]
lean_obj_res tg4_metal_copy_in(b_lean_obj_arg buf_obj, b_lean_obj_arg arr_obj, lean_object* world) {
    @autoreleasepool {
        TG4MetalBuffer* buf = metal_buffer_unbox(buf_obj);

        size_t n = lean_sarray_size(arr_obj);
        size_t copy_n = n < buf->numel ? n : buf->numel;

        double* src = (double*)lean_sarray_cptr(arr_obj);
        float* dst = (float*)[buf->buffer contents];

        // Convert from double (Lean Float) to float (Metal)
        for (size_t i = 0; i < copy_n; i++) {
            dst[i] = (float)src[i];
        }

        return lean_io_result_mk_ok(lean_box(0));
    }
}

// Copy data from Metal buffer to FloatArray
// @[extern "tg4_metal_copy_out"]
lean_obj_res tg4_metal_copy_out(b_lean_obj_arg buf_obj, lean_object* world) {
    @autoreleasepool {
        // Ensure GPU work is done before reading
        if (g_last_cmd_buf != nil) {
            [g_last_cmd_buf waitUntilCompleted];
            g_last_cmd_buf = nil;
        }

        TG4MetalBuffer* buf = metal_buffer_unbox(buf_obj);
        size_t n = buf->numel;

        // Allocate FloatArray (array of doubles)
        lean_object* arr = lean_alloc_sarray(sizeof(double), n, n);
        double* dst = (double*)lean_sarray_cptr(arr);
        float* src = (float*)[buf->buffer contents];

        // Convert from float (Metal) to double (Lean Float)
        for (size_t i = 0; i < n; i++) {
            dst[i] = (double)src[i];
        }

        return lean_io_result_mk_ok(arr);
    }
}

// Get buffer size
// @[extern "tg4_metal_size"]
lean_obj_res tg4_metal_size(b_lean_obj_arg buf_obj, lean_object* world) {
    TG4MetalBuffer* buf = metal_buffer_unbox(buf_obj);
    return lean_io_result_mk_ok(lean_usize_to_nat(buf->numel));
}

// ============================================================================
// Compiler Functions
// ============================================================================

// Compile shader source to pipeline (with caching)
// @[extern "tg4_metal_compile"]
lean_obj_res tg4_metal_compile(b_lean_obj_arg name_obj, b_lean_obj_arg source_obj, lean_object* world) {
    @autoreleasepool {
        ensure_metal_init();

        const char* name_cstr = lean_string_cstr(name_obj);
        const char* source_cstr = lean_string_cstr(source_obj);

        NSString* name = [NSString stringWithUTF8String:name_cstr];
        NSString* source = [NSString stringWithUTF8String:source_cstr];

        // Create cache key: name + source concatenated
        NSString* cacheKey = [NSString stringWithFormat:@"%@\n%@", name, source];

        // Check cache first
        id<MTLComputePipelineState> cachedPipeline = g_pipeline_cache[cacheKey];
        if (cachedPipeline != nil) {
            g_cache_hits++;
            TG4MetalProgram* prog = malloc(sizeof(TG4MetalProgram));
            prog->pipeline = cachedPipeline;
            prog->name = name;
            return lean_io_result_mk_ok(metal_program_box(prog));
        }

        // Cache miss - compile the shader
        g_cache_misses++;

        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.mathMode = MTLMathModeFast;

        id<MTLLibrary> library = [g_device newLibraryWithSource:source
                                                        options:options
                                                          error:&error];
        if (error) {
            NSString* errMsg = [NSString stringWithFormat:@"Metal compile error: %@", [error localizedDescription]];
            return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string([errMsg UTF8String])));
        }

        id<MTLFunction> function = [library newFunctionWithName:name];
        if (!function) {
            NSString* errMsg = [NSString stringWithFormat:@"Kernel '%@' not found in shader", name];
            return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string([errMsg UTF8String])));
        }

        id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:function
                                                                                       error:&error];
        if (error) {
            NSString* errMsg = [NSString stringWithFormat:@"Pipeline creation error: %@", [error localizedDescription]];
            return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string([errMsg UTF8String])));
        }

        // Store in cache
        g_pipeline_cache[cacheKey] = pipeline;

        TG4MetalProgram* prog = malloc(sizeof(TG4MetalProgram));
        prog->pipeline = pipeline;
        prog->name = name;

        return lean_io_result_mk_ok(metal_program_box(prog));
    }
}

// ============================================================================
// Runtime Functions
// ============================================================================

// Launch a kernel
// @[extern "tg4_metal_launch"]
lean_obj_res tg4_metal_launch(
    b_lean_obj_arg prog_obj,
    b_lean_obj_arg bufs_obj,
    b_lean_obj_arg global_x, b_lean_obj_arg global_y, b_lean_obj_arg global_z,
    b_lean_obj_arg local_x, b_lean_obj_arg local_y, b_lean_obj_arg local_z,
    lean_object* world
) {
    @autoreleasepool {
        ensure_metal_init();

        TG4MetalProgram* prog = metal_program_unbox(prog_obj);

        size_t gx = lean_usize_of_nat(global_x);
        size_t gy = lean_usize_of_nat(global_y);
        size_t gz = lean_usize_of_nat(global_z);
        size_t lx = lean_usize_of_nat(local_x);
        size_t ly = lean_usize_of_nat(local_y);
        size_t lz = lean_usize_of_nat(local_z);

        // Clamp local size to max
        NSUInteger maxThreads = prog->pipeline.maxTotalThreadsPerThreadgroup;
        if (lx * ly * lz > maxThreads) {
            lx = maxThreads < 256 ? maxThreads : 256;
            ly = 1;
            lz = 1;
        }

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:prog->pipeline];

        // Bind buffers
        size_t num_bufs = lean_array_size(bufs_obj);
        for (size_t i = 0; i < num_bufs; i++) {
            lean_object* buf_lean = lean_array_get_core(bufs_obj, i);
            TG4MetalBuffer* buf = metal_buffer_unbox(buf_lean);
            [encoder setBuffer:buf->buffer offset:0 atIndex:i];
        }

        MTLSize threads = MTLSizeMake(gx, gy, gz);
        MTLSize threadgroup = MTLSizeMake(lx, ly, lz);

        // Use dispatchThreads for exact thread count (Metal 2+)
        [encoder dispatchThreads:threads threadsPerThreadgroup:threadgroup];

        [encoder endEncoding];
        [cmdBuf commit];

        // Track this command buffer for sync
        g_last_cmd_buf = cmdBuf;

        return lean_io_result_mk_ok(lean_box(0));
    }
}

// Sync (wait for all GPU work)
// @[extern "tg4_metal_sync"]
lean_obj_res tg4_metal_sync(lean_object* world) {
    @autoreleasepool {
        if (g_last_cmd_buf != nil) {
            [g_last_cmd_buf waitUntilCompleted];
            g_last_cmd_buf = nil;
        }
        return lean_io_result_mk_ok(lean_box(0));
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

// Get device name
// @[extern "tg4_metal_device_name"]
lean_obj_res tg4_metal_device_name(lean_object* world) {
    @autoreleasepool {
        ensure_metal_init();
        const char* name = [[g_device name] UTF8String];
        return lean_io_result_mk_ok(lean_mk_string(name));
    }
}

// Get pipeline cache hit count
// @[extern "tg4_metal_cache_hits"]
lean_obj_res tg4_metal_cache_hits(lean_object* world) {
    @autoreleasepool {
        ensure_metal_init();
        return lean_io_result_mk_ok(lean_uint64_to_nat(g_cache_hits));
    }
}

// Get pipeline cache miss count
// @[extern "tg4_metal_cache_misses"]
lean_obj_res tg4_metal_cache_misses(lean_object* world) {
    @autoreleasepool {
        ensure_metal_init();
        return lean_io_result_mk_ok(lean_uint64_to_nat(g_cache_misses));
    }
}

// Get pipeline cache size
// @[extern "tg4_metal_cache_size"]
lean_obj_res tg4_metal_cache_size(lean_object* world) {
    @autoreleasepool {
        ensure_metal_init();
        return lean_io_result_mk_ok(lean_usize_to_nat([g_pipeline_cache count]));
    }
}

// Clear the pipeline cache
// @[extern "tg4_metal_cache_clear"]
lean_obj_res tg4_metal_cache_clear(lean_object* world) {
    @autoreleasepool {
        ensure_metal_init();
        [g_pipeline_cache removeAllObjects];
        g_cache_hits = 0;
        g_cache_misses = 0;
        return lean_io_result_mk_ok(lean_box(0));
    }
}

// ============================================================================
// Synchronous Matmul (for pure evaluator)
// ============================================================================

// Compile, launch, and return matmul result synchronously
// This is the main entry point for GPU matmul from the pure evaluator
// @[extern "tg4_metal_matmul_sync"]
// Note: Returns ByteArray directly (not IO) for use in pure evaluator
LEAN_EXPORT lean_obj_res tg4_metal_matmul_sync(
    b_lean_obj_arg a_bytes, b_lean_obj_arg b_bytes,
    b_lean_obj_arg m_obj, b_lean_obj_arg k_obj, b_lean_obj_arg n_obj
) {
    @autoreleasepool {
        ensure_metal_init();

        size_t m = lean_usize_of_nat(m_obj);
        size_t k = lean_usize_of_nat(k_obj);
        size_t n = lean_usize_of_nat(n_obj);

        size_t a_size = m * k * sizeof(float);
        size_t b_size = k * n * sizeof(float);
        size_t c_size = m * n * sizeof(float);

        // Allocate Metal buffers
        id<MTLBuffer> aBuf = [g_device newBufferWithLength:a_size
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [g_device newBufferWithLength:b_size
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> cBuf = [g_device newBufferWithLength:c_size
                                                   options:MTLResourceStorageModeShared];

        // Copy input data
        memcpy([aBuf contents], lean_sarray_cptr(a_bytes), a_size);
        memcpy([bBuf contents], lean_sarray_cptr(b_bytes), b_size);

        // Generate and compile shader
        // For simplicity, use a tiled GEMM with fixed config
        NSMutableString* shader = [NSMutableString string];
        [shader appendString:@"#include <metal_stdlib>\n"];
        [shader appendString:@"using namespace metal;\n\n"];
        [shader appendFormat:@"constant uint M = %zu;\n", m];
        [shader appendFormat:@"constant uint K = %zu;\n", k];
        [shader appendFormat:@"constant uint N = %zu;\n", n];
        [shader appendString:@"constant uint TILE = 8;\n\n"];
        [shader appendString:@"kernel void matmul(\n"];
        [shader appendString:@"    device const float* A [[buffer(0)]],\n"];
        [shader appendString:@"    device const float* B [[buffer(1)]],\n"];
        [shader appendString:@"    device float* C [[buffer(2)]],\n"];
        [shader appendString:@"    uint2 gid [[thread_position_in_grid]]\n"];
        [shader appendString:@") {\n"];
        [shader appendString:@"    uint row = gid.y;\n"];
        [shader appendString:@"    uint col = gid.x;\n"];
        [shader appendString:@"    if (row >= M || col >= N) return;\n"];
        [shader appendString:@"    float sum = 0.0f;\n"];
        [shader appendString:@"    for (uint t = 0; t < K; t++) {\n"];
        [shader appendString:@"        sum += A[row * K + t] * B[t * N + col];\n"];
        [shader appendString:@"    }\n"];
        [shader appendString:@"    C[row * N + col] = sum;\n"];
        [shader appendString:@"}\n"];

        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.mathMode = MTLMathModeFast;

        id<MTLLibrary> library = [g_device newLibraryWithSource:shader
                                                        options:options
                                                          error:&error];
        if (error) {
            // Fall back to returning zeros
            lean_object* arr = lean_alloc_sarray(1, c_size, c_size);
            memset(lean_sarray_cptr(arr), 0, c_size);
            return arr;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"matmul"];
        id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:function
                                                                                       error:&error];
        if (error) {
            lean_object* arr = lean_alloc_sarray(1, c_size, c_size);
            memset(lean_sarray_cptr(arr), 0, c_size);
            return arr;
        }

        // Launch kernel
        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:aBuf offset:0 atIndex:0];
        [encoder setBuffer:bBuf offset:0 atIndex:1];
        [encoder setBuffer:cBuf offset:0 atIndex:2];

        MTLSize threads = MTLSizeMake(n, m, 1);
        NSUInteger w = pipeline.threadExecutionWidth;
        NSUInteger h = pipeline.maxTotalThreadsPerThreadgroup / w;
        MTLSize threadgroup = MTLSizeMake(w, h > 0 ? h : 1, 1);

        [encoder dispatchThreads:threads threadsPerThreadgroup:threadgroup];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        // Copy result to ByteArray
        lean_object* arr = lean_alloc_sarray(1, c_size, c_size);
        memcpy(lean_sarray_cptr(arr), [cBuf contents], c_size);

        return arr;
    }
}

// ============================================================================
// Byte-Based FFI (dtype-generic)
// ============================================================================

// Allocate a buffer with n bytes
// @[extern "tg4_metal_alloc_bytes"]
lean_obj_res tg4_metal_alloc_bytes(b_lean_obj_arg nbytes_obj, lean_object* world) {
    @autoreleasepool {
        ensure_metal_init();

        size_t nbytes = lean_usize_of_nat(nbytes_obj);

        TG4MetalBuffer* buf = malloc(sizeof(TG4MetalBuffer));
        buf->buffer = [g_device newBufferWithLength:nbytes
                                            options:MTLResourceStorageModeShared];
        buf->numel = nbytes;  // Store byte count in numel field

        // Zero-initialize
        memset([buf->buffer contents], 0, nbytes);

        return lean_io_result_mk_ok(metal_buffer_box(buf));
    }
}

// Copy raw bytes from ByteArray to Metal buffer (no conversion)
// @[extern "tg4_metal_copy_in_bytes"]
lean_obj_res tg4_metal_copy_in_bytes(b_lean_obj_arg buf_obj, b_lean_obj_arg bytes_obj, lean_object* world) {
    @autoreleasepool {
        TG4MetalBuffer* buf = metal_buffer_unbox(buf_obj);

        size_t len = lean_sarray_size(bytes_obj);
        size_t copy_len = len < buf->numel ? len : buf->numel;

        uint8_t* src = lean_sarray_cptr(bytes_obj);
        uint8_t* dst = (uint8_t*)[buf->buffer contents];

        memcpy(dst, src, copy_len);

        return lean_io_result_mk_ok(lean_box(0));
    }
}

// Copy raw bytes from Metal buffer to ByteArray (no conversion)
// @[extern "tg4_metal_copy_out_bytes"]
lean_obj_res tg4_metal_copy_out_bytes(b_lean_obj_arg buf_obj, b_lean_obj_arg nbytes_obj, lean_object* world) {
    @autoreleasepool {
        // Ensure GPU work is done before reading
        if (g_last_cmd_buf != nil) {
            [g_last_cmd_buf waitUntilCompleted];
            g_last_cmd_buf = nil;
        }

        TG4MetalBuffer* buf = metal_buffer_unbox(buf_obj);
        size_t nbytes = lean_usize_of_nat(nbytes_obj);
        size_t copy_len = nbytes < buf->numel ? nbytes : buf->numel;

        // Allocate ByteArray
        lean_object* arr = lean_alloc_sarray(1, copy_len, copy_len);
        uint8_t* dst = lean_sarray_cptr(arr);
        uint8_t* src = (uint8_t*)[buf->buffer contents];

        memcpy(dst, src, copy_len);

        return lean_io_result_mk_ok(arr);
    }
}

// Launch a kernel with 2D grid (for matmul)
// @[extern "tg4_metal_launch_2d"]
lean_obj_res tg4_metal_launch_2d(
    b_lean_obj_arg prog_obj,
    b_lean_obj_arg bufs_obj,
    b_lean_obj_arg grid_x, b_lean_obj_arg grid_y,
    b_lean_obj_arg tg_x, b_lean_obj_arg tg_y,
    lean_object* world
) {
    @autoreleasepool {
        ensure_metal_init();

        TG4MetalProgram* prog = metal_program_unbox(prog_obj);

        size_t gx = lean_usize_of_nat(grid_x);
        size_t gy = lean_usize_of_nat(grid_y);
        size_t tx = lean_usize_of_nat(tg_x);
        size_t ty = lean_usize_of_nat(tg_y);

        // Clamp threadgroup size to max
        NSUInteger maxThreads = prog->pipeline.maxTotalThreadsPerThreadgroup;
        if (tx * ty > maxThreads) {
            tx = 8;
            ty = 8;
        }

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:prog->pipeline];

        // Bind buffers
        size_t num_bufs = lean_array_size(bufs_obj);
        for (size_t i = 0; i < num_bufs; i++) {
            lean_object* buf_lean = lean_array_get_core(bufs_obj, i);
            TG4MetalBuffer* buf = metal_buffer_unbox(buf_lean);
            [encoder setBuffer:buf->buffer offset:0 atIndex:i];
        }

        MTLSize threadgroups = MTLSizeMake(gx, gy, 1);
        MTLSize threadgroup = MTLSizeMake(tx, ty, 1);

        // Use dispatchThreadgroups for tiled matmul
        [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroup];

        [encoder endEncoding];
        [cmdBuf commit];

        // Track this command buffer for sync
        g_last_cmd_buf = cmdBuf;

        return lean_io_result_mk_ok(lean_box(0));
    }
}

// ============================================================================
// Benchmark Instrumentation
// ============================================================================

#include <mach/mach_time.h>

static mach_timebase_info_data_t g_timebase = {0, 0};

static uint64_t get_nanos(void) {
    if (g_timebase.denom == 0) {
        mach_timebase_info(&g_timebase);
    }
    uint64_t ticks = mach_absolute_time();
    return ticks * g_timebase.numer / g_timebase.denom;
}

// Compile shader with timing - returns (program, compile_time_ns)
// @[extern "tg4_metal_compile_timed"]
lean_obj_res tg4_metal_compile_timed(b_lean_obj_arg name_obj, b_lean_obj_arg source_obj, lean_object* world) {
    @autoreleasepool {
        ensure_metal_init();

        uint64_t start = get_nanos();

        const char* name_cstr = lean_string_cstr(name_obj);
        const char* source_cstr = lean_string_cstr(source_obj);

        NSString* name = [NSString stringWithUTF8String:name_cstr];
        NSString* source = [NSString stringWithUTF8String:source_cstr];

        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.mathMode = MTLMathModeFast;

        id<MTLLibrary> library = [g_device newLibraryWithSource:source
                                                        options:options
                                                          error:&error];
        if (error) {
            NSString* errMsg = [NSString stringWithFormat:@"Metal compile error: %@", [error localizedDescription]];
            return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string([errMsg UTF8String])));
        }

        id<MTLFunction> function = [library newFunctionWithName:name];
        if (!function) {
            NSString* errMsg = [NSString stringWithFormat:@"Kernel '%@' not found in shader", name];
            return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string([errMsg UTF8String])));
        }

        id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:function
                                                                                       error:&error];
        if (error) {
            NSString* errMsg = [NSString stringWithFormat:@"Pipeline creation error: %@", [error localizedDescription]];
            return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string([errMsg UTF8String])));
        }

        uint64_t end = get_nanos();
        uint64_t compile_ns = end - start;

        TG4MetalProgram* prog = malloc(sizeof(TG4MetalProgram));
        prog->pipeline = pipeline;
        prog->name = name;

        // Return tuple (program, compile_time_ns)
        lean_object* tuple = lean_alloc_ctor(0, 2, 0);
        lean_ctor_set(tuple, 0, metal_program_box(prog));
        lean_ctor_set(tuple, 1, lean_uint64_to_nat(compile_ns));

        return lean_io_result_mk_ok(tuple);
    }
}

// Launch kernel with timing - returns (dispatch_ns, kernel_ns, sync_ns)
// @[extern "tg4_metal_launch_timed"]
lean_obj_res tg4_metal_launch_timed(
    b_lean_obj_arg prog_obj,
    b_lean_obj_arg bufs_obj,
    b_lean_obj_arg global_x, b_lean_obj_arg global_y, b_lean_obj_arg global_z,
    b_lean_obj_arg local_x, b_lean_obj_arg local_y, b_lean_obj_arg local_z,
    lean_object* world
) {
    @autoreleasepool {
        ensure_metal_init();

        uint64_t dispatch_start = get_nanos();

        TG4MetalProgram* prog = metal_program_unbox(prog_obj);

        size_t gx = lean_usize_of_nat(global_x);
        size_t gy = lean_usize_of_nat(global_y);
        size_t gz = lean_usize_of_nat(global_z);
        size_t lx = lean_usize_of_nat(local_x);
        size_t ly = lean_usize_of_nat(local_y);
        size_t lz = lean_usize_of_nat(local_z);

        NSUInteger maxThreads = prog->pipeline.maxTotalThreadsPerThreadgroup;
        if (lx * ly * lz > maxThreads) {
            lx = maxThreads < 256 ? maxThreads : 256;
            ly = 1;
            lz = 1;
        }

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:prog->pipeline];

        size_t num_bufs = lean_array_size(bufs_obj);
        for (size_t i = 0; i < num_bufs; i++) {
            lean_object* buf_lean = lean_array_get_core(bufs_obj, i);
            TG4MetalBuffer* buf = metal_buffer_unbox(buf_lean);
            [encoder setBuffer:buf->buffer offset:0 atIndex:i];
        }

        MTLSize threads = MTLSizeMake(gx, gy, gz);
        MTLSize threadgroup = MTLSizeMake(lx, ly, lz);

        [encoder dispatchThreads:threads threadsPerThreadgroup:threadgroup];
        [encoder endEncoding];

        uint64_t dispatch_end = get_nanos();
        uint64_t dispatch_ns = dispatch_end - dispatch_start;

        uint64_t commit_start = get_nanos();
        [cmdBuf commit];

        uint64_t sync_start = get_nanos();
        [cmdBuf waitUntilCompleted];
        uint64_t sync_end = get_nanos();

        uint64_t kernel_ns = sync_end - commit_start;
        uint64_t sync_ns = sync_end - sync_start;

        // Return tuple (dispatch_ns, kernel_ns, sync_ns)
        lean_object* tuple = lean_alloc_ctor(0, 3, 0);
        lean_ctor_set(tuple, 0, lean_uint64_to_nat(dispatch_ns));
        lean_ctor_set(tuple, 1, lean_uint64_to_nat(kernel_ns));
        lean_ctor_set(tuple, 2, lean_uint64_to_nat(sync_ns));

        return lean_io_result_mk_ok(tuple);
    }
}

// Get current time in nanoseconds
// @[extern "tg4_metal_get_time_ns"]
lean_obj_res tg4_metal_get_time_ns(lean_object* world) {
    uint64_t ns = get_nanos();
    return lean_io_result_mk_ok(lean_uint64_to_nat(ns));
}

// Sync with timing - returns sync time in nanoseconds
// @[extern "tg4_metal_sync_timed"]
lean_obj_res tg4_metal_sync_timed(lean_object* world) {
    @autoreleasepool {
        uint64_t start = get_nanos();
        if (g_last_cmd_buf != nil) {
            [g_last_cmd_buf waitUntilCompleted];
            g_last_cmd_buf = nil;
        }
        uint64_t end = get_nanos();
        return lean_io_result_mk_ok(lean_uint64_to_nat(end - start));
    }
}

// ============================================================================
// Zero-Copy Buffer Support
// ============================================================================

// Wrap ByteArray data in a Metal buffer without copying.
// IMPORTANT: The ByteArray must outlive the Metal buffer!
// Uses Apple Silicon unified memory - GPU reads directly from CPU memory.
// @[extern "tg4_metal_wrap_bytes_nocopy"]
lean_obj_res tg4_metal_wrap_bytes_nocopy(b_lean_obj_arg ba_obj, b_lean_obj_arg offset_obj, b_lean_obj_arg len_obj, lean_object* world) {
    @autoreleasepool {
        ensure_metal_init();

        uint8_t* data = lean_sarray_cptr(ba_obj);
        size_t ba_size = lean_sarray_size(ba_obj);
        size_t offset = lean_usize_of_nat(offset_obj);
        size_t len = lean_usize_of_nat(len_obj);

        // Bounds check
        if (offset + len > ba_size) {
            return lean_io_result_mk_error(lean_mk_io_user_error(
                lean_mk_string("tg4_metal_wrap_bytes_nocopy: offset+len exceeds ByteArray size")));
        }

        // Create Metal buffer wrapping existing memory (no copy!)
        // The deallocator is nil because Lean owns the memory
        id<MTLBuffer> mtlBuf = [g_device newBufferWithBytesNoCopy:data + offset
                                                           length:len
                                                          options:MTLResourceStorageModeShared
                                                      deallocator:nil];

        if (mtlBuf == nil) {
            // Fall back to regular copy if no-copy fails (e.g., alignment issues)
            mtlBuf = [g_device newBufferWithBytes:data + offset
                                           length:len
                                          options:MTLResourceStorageModeShared];
        }

        TG4MetalBuffer* buf = malloc(sizeof(TG4MetalBuffer));
        buf->buffer = mtlBuf;
        buf->numel = len / sizeof(float);  // Assume float32 for numel

        return lean_io_result_mk_ok(metal_buffer_box(buf));
    }
}

// Check if a pointer is page-aligned (required for zero-copy)
// @[extern "tg4_metal_is_aligned"]
lean_obj_res tg4_metal_is_aligned(b_lean_obj_arg ba_obj, b_lean_obj_arg offset_obj, lean_object* world) {
    uint8_t* data = lean_sarray_cptr(ba_obj);
    size_t offset = lean_usize_of_nat(offset_obj);
    uintptr_t addr = (uintptr_t)(data + offset);
    // Metal requires 4KB page alignment for no-copy buffers
    int aligned = (addr % 4096) == 0;
    return lean_io_result_mk_ok(lean_box(aligned));
}

// ============================================================================
// Shared Memory Support (POSIX shm for multi-process)
// ============================================================================

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// Create a shared memory region and return its file descriptor
// @[extern "tg4_shm_create"]
lean_obj_res tg4_shm_create(b_lean_obj_arg name_obj, size_t size, lean_object* world) {
    const char* name = lean_string_cstr(name_obj);

    // Remove any existing shm with this name
    shm_unlink(name);

    int fd = shm_open(name, O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("tg4_shm_create: shm_open failed")));
    }

    if (ftruncate(fd, size) < 0) {
        close(fd);
        shm_unlink(name);
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("tg4_shm_create: ftruncate failed")));
    }

    // Return fd as UInt32
    return lean_io_result_mk_ok(lean_box_uint32(fd));
}

// Open an existing shared memory region
// @[extern "tg4_shm_open"]
lean_obj_res tg4_shm_open(b_lean_obj_arg name_obj, lean_object* world) {
    const char* name = lean_string_cstr(name_obj);

    int fd = shm_open(name, O_RDWR, 0666);
    if (fd < 0) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("tg4_shm_open: shm_open failed")));
    }

    return lean_io_result_mk_ok(lean_box_uint32(fd));
}

// Map shared memory to a ByteArray (returns new ByteArray backed by shm)
// @[extern "tg4_shm_map"]
lean_obj_res tg4_shm_map(uint32_t fd, size_t size, lean_object* world) {
    void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("tg4_shm_map: mmap failed")));
    }

    // Create a ByteArray that references this memory
    // Note: This creates a copy because Lean ByteArray owns its memory
    // For true zero-copy, we'd need a custom external type
    lean_object* ba = lean_alloc_sarray(1, size, size);
    memcpy(lean_sarray_cptr(ba), ptr, size);

    // Unmap (data is now copied into Lean ByteArray)
    munmap(ptr, size);

    return lean_io_result_mk_ok(ba);
}

// Write ByteArray to shared memory
// @[extern "tg4_shm_write"]
lean_obj_res tg4_shm_write(uint32_t fd, b_lean_obj_arg ba_obj, size_t offset, lean_object* world) {
    uint8_t* data = lean_sarray_cptr(ba_obj);
    size_t size = lean_sarray_size(ba_obj);

    // Get shm size
    struct stat st;
    if (fstat(fd, &st) < 0) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("tg4_shm_write: fstat failed")));
    }

    void* ptr = mmap(NULL, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("tg4_shm_write: mmap failed")));
    }

    // Write data at offset
    size_t write_size = size;
    if (offset + write_size > (size_t)st.st_size) {
        write_size = st.st_size - offset;
    }
    memcpy((uint8_t*)ptr + offset, data, write_size);

    munmap(ptr, st.st_size);

    return lean_io_result_mk_ok(lean_box(0));
}

// Read from shared memory into ByteArray
// @[extern "tg4_shm_read"]
lean_obj_res tg4_shm_read(uint32_t fd, size_t offset, size_t len, lean_object* world) {
    // Get shm size
    struct stat st;
    if (fstat(fd, &st) < 0) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("tg4_shm_read: fstat failed")));
    }

    void* ptr = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        return lean_io_result_mk_error(lean_mk_io_user_error(
            lean_mk_string("tg4_shm_read: mmap failed")));
    }

    // Bounds check
    if (offset + len > (size_t)st.st_size) {
        len = st.st_size - offset;
    }

    // Create ByteArray and copy data
    lean_object* ba = lean_alloc_sarray(1, len, len);
    memcpy(lean_sarray_cptr(ba), (uint8_t*)ptr + offset, len);

    munmap(ptr, st.st_size);

    return lean_io_result_mk_ok(ba);
}

// Close shared memory fd
// @[extern "tg4_shm_close"]
lean_obj_res tg4_shm_close(uint32_t fd, lean_object* world) {
    close(fd);
    return lean_io_result_mk_ok(lean_box(0));
}

// Unlink (delete) shared memory region
// @[extern "tg4_shm_unlink"]
lean_obj_res tg4_shm_unlink(b_lean_obj_arg name_obj, lean_object* world) {
    const char* name = lean_string_cstr(name_obj);
    shm_unlink(name);
    return lean_io_result_mk_ok(lean_box(0));
}
