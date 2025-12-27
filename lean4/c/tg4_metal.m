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

static void ensure_metal_init(void) {
    if (g_device == nil) {
        g_device = MTLCreateSystemDefaultDevice();
        g_queue = [g_device newCommandQueue];
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

// Compile shader source to pipeline
// @[extern "tg4_metal_compile"]
lean_obj_res tg4_metal_compile(b_lean_obj_arg name_obj, b_lean_obj_arg source_obj, lean_object* world) {
    @autoreleasepool {
        ensure_metal_init();

        const char* name_cstr = lean_string_cstr(name_obj);
        const char* source_cstr = lean_string_cstr(source_obj);

        NSString* name = [NSString stringWithUTF8String:name_cstr];
        NSString* source = [NSString stringWithUTF8String:source_cstr];

        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;

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
