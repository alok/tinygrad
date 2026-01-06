// tg4_cuda.cu - CUDA backend FFI for Lean TinyGrad4
// Compile with: nvcc -c -O3 tg4_cuda.cu -o tg4_cuda.o

#include <cuda.h>
#include <nvrtc.h>
#include <lean/lean.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <limits.h>

// ============================================================================
// Global CUDA State
// ============================================================================

#define TG4_MAX_CUDA_DEVICES 16

static CUdevice g_device = 0;
static CUcontext g_context = NULL;
static CUstream g_stream = NULL;
static CUdevice g_devices[TG4_MAX_CUDA_DEVICES];
static CUcontext g_contexts[TG4_MAX_CUDA_DEVICES];
static CUstream g_streams[TG4_MAX_CUDA_DEVICES];
static int g_device_initialized[TG4_MAX_CUDA_DEVICES];
static int g_device_count = -1;
static int g_initialized = 0;
static __thread int g_current_device = 0;

#define CHECK_CU(x) do { \
    CUresult err = (x); \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, errStr); \
    } \
} while(0)

#define CHECK_NVRTC(x) do { \
    nvrtcResult err = (x); \
    if (err != NVRTC_SUCCESS) { \
        fprintf(stderr, "NVRTC error at %s:%d: %s\n", __FILE__, __LINE__, nvrtcGetErrorString(err)); \
    } \
} while(0)

static int g_nvrtc_env_set = 0;

static void ensure_nvrtc_env(void) {
    if (g_nvrtc_env_set) return;
    g_nvrtc_env_set = 1;

    Dl_info info;
    if (!dladdr((void*)nvrtcCreateProgram, &info) || info.dli_fname == NULL) return;

    char path[PATH_MAX];
    strncpy(path, info.dli_fname, sizeof(path) - 1);
    path[sizeof(path) - 1] = '\0';
    char* last = strrchr(path, '/');
    if (last == NULL) return;
    *last = '\0';

    const char* ld = getenv("LD_LIBRARY_PATH");
    if (ld == NULL || strstr(ld, path) == NULL) {
        char buf[PATH_MAX * 2];
        if (ld != NULL && ld[0] != '\0') {
            snprintf(buf, sizeof(buf), "%s:%s", path, ld);
        } else {
            snprintf(buf, sizeof(buf), "%s", path);
        }
        setenv("LD_LIBRARY_PATH", buf, 1);
    }

    char builtins[PATH_MAX];
    snprintf(builtins, sizeof(builtins), "%s/libnvrtc-builtins.so", path);
    void* builtins_handle = dlopen(builtins, RTLD_NOW | RTLD_GLOBAL);
    if (builtins_handle == NULL) {
        char builtins_ver[PATH_MAX];
        snprintf(builtins_ver, sizeof(builtins_ver), "%s/libnvrtc-builtins.so.12.4", path);
        (void)dlopen(builtins_ver, RTLD_NOW | RTLD_GLOBAL);
    }

    const char* cuda_home = getenv("CUDA_HOME");
    const char* cuda_path = getenv("CUDA_PATH");
    if ((cuda_home == NULL || cuda_home[0] == '\0') &&
        (cuda_path == NULL || cuda_path[0] == '\0')) {
        char home[PATH_MAX];
        strncpy(home, path, sizeof(home) - 1);
        home[sizeof(home) - 1] = '\0';
        char* tail = strrchr(home, '/');
        if (tail != NULL && (!strcmp(tail + 1, "lib64") || !strcmp(tail + 1, "lib"))) {
            *tail = '\0';
            setenv("CUDA_HOME", home, 0);
            setenv("CUDA_PATH", home, 0);
        }
    }
}

static int ensure_cuda_runtime(void) {
    if (!g_initialized) {
        ensure_nvrtc_env();
        CHECK_CU(cuInit(0));
        g_initialized = 1;
    }
    if (g_device_count < 0) {
        int count = 0;
        CUresult err = cuDeviceGetCount(&count);
        if (err != CUDA_SUCCESS) {
            const char* errStr = NULL;
            cuGetErrorString(err, &errStr);
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, errStr ? errStr : "unknown");
            g_device_count = 0;
        } else {
            g_device_count = count;
        }
    }
    return g_device_count;
}

static int ensure_cuda_device(int device_idx) {
    int count = ensure_cuda_runtime();
    if (count <= 0) return -1;
    if (device_idx < 0 || device_idx >= count) device_idx = 0;

    if (!g_device_initialized[device_idx]) {
        CHECK_CU(cuDeviceGet(&g_devices[device_idx], device_idx));
        CHECK_CU(cuCtxCreate(&g_contexts[device_idx], 0, g_devices[device_idx]));
        CHECK_CU(cuStreamCreate(&g_streams[device_idx], CU_STREAM_DEFAULT));
        g_device_initialized[device_idx] = 1;
    }

    g_current_device = device_idx;
    g_device = g_devices[device_idx];
    g_context = g_contexts[device_idx];
    g_stream = g_streams[device_idx];
    CHECK_CU(cuCtxSetCurrent(g_context));
    return device_idx;
}

static void ensure_cuda_init(void) {
    (void)ensure_cuda_device(g_current_device);
}

// ============================================================================
// Buffer Type (opaque to Lean)
// ============================================================================

typedef struct {
    CUdeviceptr gpu_mem;
    size_t numel;
    int device_idx;
} TG4CUDABuffer;

static void tg4_cuda_buffer_finalize(void* ptr) {
    TG4CUDABuffer* buf = (TG4CUDABuffer*)ptr;
    if (buf) {
        if (buf->gpu_mem) {
            ensure_cuda_device(buf->device_idx);
            CHECK_CU(cuMemFree(buf->gpu_mem));
        }
        free(buf);
    }
}

static void tg4_cuda_buffer_foreach(void* ptr, b_lean_obj_arg fn) {
    (void)ptr;
    (void)fn;
}

static lean_external_class* g_cuda_buffer_class = NULL;

static lean_external_class* get_cuda_buffer_class(void) {
    if (g_cuda_buffer_class == NULL) {
        g_cuda_buffer_class = lean_register_external_class(
            tg4_cuda_buffer_finalize,
            tg4_cuda_buffer_foreach
        );
    }
    return g_cuda_buffer_class;
}

static inline TG4CUDABuffer* cuda_buffer_unbox(lean_object* obj) {
    return (TG4CUDABuffer*)lean_get_external_data(obj);
}

static inline lean_object* cuda_buffer_box(TG4CUDABuffer* buf) {
    return lean_alloc_external(get_cuda_buffer_class(), buf);
}

// ============================================================================
// Program Type (opaque to Lean)
// ============================================================================

typedef struct {
    CUmodule module;
    CUfunction kernel;
    char* name;
} TG4CUDAProgram;

static void tg4_cuda_program_finalize(void* ptr) {
    TG4CUDAProgram* prog = (TG4CUDAProgram*)ptr;
    if (prog) {
        if (prog->module) {
            cuModuleUnload(prog->module);
        }
        if (prog->name) {
            free(prog->name);
        }
        free(prog);
    }
}

static void tg4_cuda_program_foreach(void* ptr, b_lean_obj_arg fn) {
    (void)ptr;
    (void)fn;
}

static lean_external_class* g_cuda_program_class = NULL;

static lean_external_class* get_cuda_program_class(void) {
    if (g_cuda_program_class == NULL) {
        g_cuda_program_class = lean_register_external_class(
            tg4_cuda_program_finalize,
            tg4_cuda_program_foreach
        );
    }
    return g_cuda_program_class;
}

static inline TG4CUDAProgram* cuda_program_unbox(lean_object* obj) {
    return (TG4CUDAProgram*)lean_get_external_data(obj);
}

static inline lean_object* cuda_program_box(TG4CUDAProgram* prog) {
    return lean_alloc_external(get_cuda_program_class(), prog);
}

// ============================================================================
// Allocator Functions
// ============================================================================

// Allocate a buffer with n float elements
// @[extern "tg4_cuda_alloc"]
extern "C" lean_obj_res tg4_cuda_alloc(b_lean_obj_arg n_obj, lean_object* world) {
    ensure_cuda_init();

    size_t n = lean_usize_of_nat(n_obj);
    size_t bytes = n * sizeof(float);

    TG4CUDABuffer* buf = (TG4CUDABuffer*)malloc(sizeof(TG4CUDABuffer));
    buf->numel = n;
    buf->device_idx = g_current_device;

    CHECK_CU(cuMemAlloc(&buf->gpu_mem, bytes));
    CHECK_CU(cuMemsetD8(buf->gpu_mem, 0, bytes));

    return lean_io_result_mk_ok(cuda_buffer_box(buf));
}

// Free a buffer
// @[extern "tg4_cuda_free"]
extern "C" lean_obj_res tg4_cuda_free(b_lean_obj_arg buf_obj, lean_object* world) {
    // Buffer will be freed by finalizer when ref count drops
    (void)buf_obj;
    return lean_io_result_mk_ok(lean_box(0));
}

// Copy data from FloatArray to CUDA buffer
// @[extern "tg4_cuda_copy_in"]
extern "C" lean_obj_res tg4_cuda_copy_in(b_lean_obj_arg buf_obj, b_lean_obj_arg arr_obj, lean_object* world) {
    TG4CUDABuffer* buf = cuda_buffer_unbox(buf_obj);
    ensure_cuda_device(buf->device_idx);

    size_t n = lean_sarray_size(arr_obj);
    size_t copy_n = n < buf->numel ? n : buf->numel;

    double* src = (double*)lean_sarray_cptr(arr_obj);

    // Allocate temp host buffer for float conversion
    float* host_buf = (float*)malloc(copy_n * sizeof(float));
    for (size_t i = 0; i < copy_n; i++) {
        host_buf[i] = (float)src[i];
    }

    // Copy to GPU
    CHECK_CU(cuMemcpyHtoD(buf->gpu_mem, host_buf, copy_n * sizeof(float)));

    free(host_buf);
    return lean_io_result_mk_ok(lean_box(0));
}

// Copy data from CUDA buffer to FloatArray
// @[extern "tg4_cuda_copy_out"]
extern "C" lean_obj_res tg4_cuda_copy_out(b_lean_obj_arg buf_obj, lean_object* world) {
    TG4CUDABuffer* buf = cuda_buffer_unbox(buf_obj);
    ensure_cuda_device(buf->device_idx);
    // Sync before reading
    CHECK_CU(cuStreamSynchronize(g_stream));
    size_t n = buf->numel;

    // Allocate temp host buffer
    float* host_buf = (float*)malloc(n * sizeof(float));
    CHECK_CU(cuMemcpyDtoH(host_buf, buf->gpu_mem, n * sizeof(float)));

    // Allocate FloatArray (array of doubles)
    lean_object* arr = lean_alloc_sarray(sizeof(double), n, n);
    double* dst = (double*)lean_sarray_cptr(arr);

    // Convert from float to double
    for (size_t i = 0; i < n; i++) {
        dst[i] = (double)host_buf[i];
    }

    free(host_buf);
    return lean_io_result_mk_ok(arr);
}

// Get buffer size
// @[extern "tg4_cuda_size"]
extern "C" lean_obj_res tg4_cuda_size(b_lean_obj_arg buf_obj, lean_object* world) {
    TG4CUDABuffer* buf = cuda_buffer_unbox(buf_obj);
    return lean_io_result_mk_ok(lean_usize_to_nat(buf->numel));
}

// ============================================================================
// Compiler Functions
// ============================================================================

// Compile CUDA kernel source to module/function via NVRTC
// @[extern "tg4_cuda_compile"]
extern "C" lean_obj_res tg4_cuda_compile(b_lean_obj_arg name_obj, b_lean_obj_arg source_obj, lean_object* world) {
    ensure_cuda_init();

    const char* name_cstr = lean_string_cstr(name_obj);
    const char* source_cstr = lean_string_cstr(source_obj);

    // Create NVRTC program
    nvrtcProgram prog;
    CHECK_NVRTC(nvrtcCreateProgram(&prog, source_cstr, "kernel.cu", 0, NULL, NULL));

    // Get compute capability
    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, g_device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, g_device);

    char arch_opt[32];
    snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d%d", major, minor);

    const char* opts[] = {
        arch_opt,
        "--use_fast_math",
        "-default-device"
    };

    nvrtcResult compileResult = nvrtcCompileProgram(prog, 3, opts);

    if (compileResult != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char* log = (char*)malloc(logSize);
        nvrtcGetProgramLog(prog, log);

        char errMsg[4096];
        snprintf(errMsg, sizeof(errMsg), "NVRTC compile error: %s", log);
        free(log);
        nvrtcDestroyProgram(&prog);

        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(errMsg)));
    }

    // Get PTX
    size_t ptxSize;
    CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptxSize));
    char* ptx = (char*)malloc(ptxSize);
    CHECK_NVRTC(nvrtcGetPTX(prog, ptx));
    nvrtcDestroyProgram(&prog);

    // Load module from PTX
    CUmodule module;
    CUresult loadResult = cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL);
    free(ptx);

    if (loadResult != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(loadResult, &errStr);
        char errMsg[256];
        snprintf(errMsg, sizeof(errMsg), "cuModuleLoadData error: %s", errStr);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(errMsg)));
    }

    // Get function
    CUfunction kernel;
    CUresult funcResult = cuModuleGetFunction(&kernel, module, name_cstr);
    if (funcResult != CUDA_SUCCESS) {
        cuModuleUnload(module);
        char errMsg[256];
        snprintf(errMsg, sizeof(errMsg), "Kernel '%s' not found in module", name_cstr);
        return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(errMsg)));
    }

    TG4CUDAProgram* cuda_prog = (TG4CUDAProgram*)malloc(sizeof(TG4CUDAProgram));
    cuda_prog->module = module;
    cuda_prog->kernel = kernel;
    cuda_prog->name = strdup(name_cstr);

    return lean_io_result_mk_ok(cuda_program_box(cuda_prog));
}

// ============================================================================
// Runtime Functions
// ============================================================================

// Launch a kernel
// @[extern "tg4_cuda_launch"]
extern "C" lean_obj_res tg4_cuda_launch(
    b_lean_obj_arg prog_obj,
    b_lean_obj_arg bufs_obj,
    b_lean_obj_arg global_x, b_lean_obj_arg global_y, b_lean_obj_arg global_z,
    b_lean_obj_arg local_x, b_lean_obj_arg local_y, b_lean_obj_arg local_z,
    lean_object* world
) {
    ensure_cuda_init();

    TG4CUDAProgram* prog = cuda_program_unbox(prog_obj);

    size_t gx = lean_usize_of_nat(global_x);
    size_t gy = lean_usize_of_nat(global_y);
    size_t gz = lean_usize_of_nat(global_z);
    size_t lx = lean_usize_of_nat(local_x);
    size_t ly = lean_usize_of_nat(local_y);
    size_t lz = lean_usize_of_nat(local_z);

    // Clamp block size to device max
    int maxThreads;
    cuDeviceGetAttribute(&maxThreads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, g_device);
    if (lx * ly * lz > (size_t)maxThreads) {
        lx = 256;
        ly = 1;
        lz = 1;
    }

    // Calculate grid dimensions
    size_t grid_x = (gx + lx - 1) / lx;
    size_t grid_y = (gy + ly - 1) / ly;
    size_t grid_z = (gz + lz - 1) / lz;

    // Pack buffer pointers for kernel args
    size_t num_bufs = lean_array_size(bufs_obj);
    void** kernel_args = (void**)malloc(num_bufs * sizeof(void*));
    CUdeviceptr* buf_ptrs = (CUdeviceptr*)malloc(num_bufs * sizeof(CUdeviceptr));

    for (size_t i = 0; i < num_bufs; i++) {
        lean_object* buf_lean = lean_array_get_core(bufs_obj, i);
        TG4CUDABuffer* buf = cuda_buffer_unbox(buf_lean);
        buf_ptrs[i] = buf->gpu_mem;
        kernel_args[i] = &buf_ptrs[i];
    }

    // Launch kernel
    CHECK_CU(cuLaunchKernel(
        prog->kernel,
        grid_x, grid_y, grid_z,   // Grid dims
        lx, ly, lz,               // Block dims
        0,                        // Shared memory
        g_stream,                 // Stream
        kernel_args,              // Kernel args
        NULL                      // Extra
    ));

    free(buf_ptrs);
    free(kernel_args);

    return lean_io_result_mk_ok(lean_box(0));
}

// Sync (wait for all GPU work)
// @[extern "tg4_cuda_sync"]
extern "C" lean_obj_res tg4_cuda_sync(lean_object* world) {
    ensure_cuda_device(g_current_device);
    CHECK_CU(cuStreamSynchronize(g_stream));
    return lean_io_result_mk_ok(lean_box(0));
}

// ============================================================================
// Utility Functions
// ============================================================================

// Get device name
// @[extern "tg4_cuda_device_name"]
extern "C" lean_obj_res tg4_cuda_device_name(lean_object* world) {
    ensure_cuda_device(g_current_device);

    char name[256];
    CHECK_CU(cuDeviceGetName(name, sizeof(name), g_device));

    return lean_io_result_mk_ok(lean_mk_string(name));
}

// Get device count
// @[extern "tg4_cuda_device_count"]
extern "C" lean_obj_res tg4_cuda_device_count(lean_object* world) {
    (void)world;
    int count = ensure_cuda_runtime();
    if (count < 0) count = 0;
    return lean_io_result_mk_ok(lean_box(count));
}

// Set current device for this thread
// @[extern "tg4_cuda_set_device"]
extern "C" lean_obj_res tg4_cuda_set_device(b_lean_obj_arg idx_obj, lean_object* world) {
    (void)world;
    int idx = (int)lean_usize_of_nat(idx_obj);
    ensure_cuda_device(idx);
    return lean_io_result_mk_ok(lean_box(0));
}

// ============================================================================
// Byte-Based API (for GPULoader compatibility)
// ============================================================================

// Allocate a buffer with n bytes
// @[extern "tg4_cuda_alloc_bytes"]
extern "C" lean_obj_res tg4_cuda_alloc_bytes(b_lean_obj_arg nbytes_obj, lean_object* world) {
    ensure_cuda_init();

    size_t nbytes = lean_usize_of_nat(nbytes_obj);

    TG4CUDABuffer* buf = (TG4CUDABuffer*)malloc(sizeof(TG4CUDABuffer));
    buf->numel = nbytes;  // Store byte count (reusing numel field)
    buf->device_idx = g_current_device;

    CHECK_CU(cuMemAlloc(&buf->gpu_mem, nbytes));
    CHECK_CU(cuMemsetD8(buf->gpu_mem, 0, nbytes));

    return lean_io_result_mk_ok(cuda_buffer_box(buf));
}

// Copy raw bytes from ByteArray to CUDA buffer
// @[extern "tg4_cuda_copy_in_bytes"]
extern "C" lean_obj_res tg4_cuda_copy_in_bytes(b_lean_obj_arg buf_obj, b_lean_obj_arg data_obj, lean_object* world) {
    TG4CUDABuffer* buf = cuda_buffer_unbox(buf_obj);
    ensure_cuda_device(buf->device_idx);

    size_t data_size = lean_sarray_size(data_obj);
    size_t copy_bytes = data_size < buf->numel ? data_size : buf->numel;

    uint8_t* src = lean_sarray_cptr(data_obj);

    CHECK_CU(cuMemcpyHtoD(buf->gpu_mem, src, copy_bytes));

    return lean_io_result_mk_ok(lean_box(0));
}

// Copy raw bytes from CUDA buffer to ByteArray
// @[extern "tg4_cuda_copy_out_bytes"]
extern "C" lean_obj_res tg4_cuda_copy_out_bytes(b_lean_obj_arg buf_obj, b_lean_obj_arg nbytes_obj, lean_object* world) {
    TG4CUDABuffer* buf = cuda_buffer_unbox(buf_obj);
    ensure_cuda_device(buf->device_idx);
    // Sync before reading
    CHECK_CU(cuStreamSynchronize(g_stream));
    size_t nbytes = lean_usize_of_nat(nbytes_obj);
    size_t copy_bytes = nbytes < buf->numel ? nbytes : buf->numel;

    // Allocate ByteArray
    lean_object* arr = lean_alloc_sarray(1, copy_bytes, copy_bytes);
    uint8_t* dst = lean_sarray_cptr(arr);

    CHECK_CU(cuMemcpyDtoH(dst, buf->gpu_mem, copy_bytes));

    return lean_io_result_mk_ok(arr);
}

// Launch a kernel with 2D grid
// @[extern "tg4_cuda_launch_2d"]
extern "C" lean_obj_res tg4_cuda_launch_2d(
    b_lean_obj_arg prog_obj,
    b_lean_obj_arg bufs_obj,
    b_lean_obj_arg global_x, b_lean_obj_arg global_y,
    b_lean_obj_arg local_x, b_lean_obj_arg local_y,
    lean_object* world
) {
    ensure_cuda_init();

    TG4CUDAProgram* prog = cuda_program_unbox(prog_obj);

    size_t gx = lean_usize_of_nat(global_x);
    size_t gy = lean_usize_of_nat(global_y);
    size_t lx = lean_usize_of_nat(local_x);
    size_t ly = lean_usize_of_nat(local_y);

    // Clamp block size to device max
    int maxThreads;
    cuDeviceGetAttribute(&maxThreads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, g_device);
    if (lx * ly > (size_t)maxThreads) {
        lx = 256;
        ly = 1;
    }

    // Calculate grid dimensions
    size_t grid_x = (gx + lx - 1) / lx;
    size_t grid_y = (gy + ly - 1) / ly;

    // Pack buffer pointers for kernel args
    size_t num_bufs = lean_array_size(bufs_obj);
    void** kernel_args = (void**)malloc(num_bufs * sizeof(void*));
    CUdeviceptr* buf_ptrs = (CUdeviceptr*)malloc(num_bufs * sizeof(CUdeviceptr));

    for (size_t i = 0; i < num_bufs; i++) {
        lean_object* buf_lean = lean_array_get_core(bufs_obj, i);
        TG4CUDABuffer* buf = cuda_buffer_unbox(buf_lean);
        buf_ptrs[i] = buf->gpu_mem;
        kernel_args[i] = &buf_ptrs[i];
    }

    // Launch kernel (2D grid, z=1)
    CHECK_CU(cuLaunchKernel(
        prog->kernel,
        grid_x, grid_y, 1,        // Grid dims
        lx, ly, 1,                // Block dims
        0,                        // Shared memory
        g_stream,                 // Stream
        kernel_args,              // Kernel args
        NULL                      // Extra
    ));

    free(buf_ptrs);
    free(kernel_args);

    return lean_io_result_mk_ok(lean_box(0));
}

// ============================================================================
// Synchronous Matmul (Pure API for fallback)
// ============================================================================

// Simple SGEMM kernel source
static const char* g_sgemm_source =
"extern \"C\" __global__ void sgemm_naive(\n"
"    const float* A, const float* B, float* C,\n"
"    int M, int K, int N\n"
") {\n"
"    int row = blockIdx.y * blockDim.y + threadIdx.y;\n"
"    int col = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (row < M && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < K; k++) {\n"
"            sum += A[row * K + k] * B[k * N + col];\n"
"        }\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n";

static CUmodule g_sgemm_module = NULL;
static CUfunction g_sgemm_kernel = NULL;

static void ensure_sgemm_compiled(void) {
    if (g_sgemm_kernel != NULL) return;

    ensure_cuda_init();

    // Compile SGEMM kernel via NVRTC
    nvrtcProgram prog;
    CHECK_NVRTC(nvrtcCreateProgram(&prog, g_sgemm_source, "sgemm.cu", 0, NULL, NULL));

    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, g_device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, g_device);

    char arch_opt[32];
    snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d%d", major, minor);

    const char* opts[] = { arch_opt, "--use_fast_math", "-default-device" };
    nvrtcResult compileResult = nvrtcCompileProgram(prog, 3, opts);

    if (compileResult != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char* log = (char*)malloc(logSize);
        nvrtcGetProgramLog(prog, log);
        fprintf(stderr, "SGEMM compile error: %s\n", log);
        free(log);
        nvrtcDestroyProgram(&prog);
        return;
    }

    size_t ptxSize;
    CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptxSize));
    char* ptx = (char*)malloc(ptxSize);
    CHECK_NVRTC(nvrtcGetPTX(prog, ptx));
    nvrtcDestroyProgram(&prog);

    CHECK_CU(cuModuleLoadDataEx(&g_sgemm_module, ptx, 0, NULL, NULL));
    free(ptx);

    CHECK_CU(cuModuleGetFunction(&g_sgemm_kernel, g_sgemm_module, "sgemm_naive"));
}

// Synchronous matmul: C[m,n] = A[m,k] @ B[k,n]
// Pure function - returns zeros on error
// @[extern "tg4_cuda_matmul_sync"]
extern "C" lean_obj_res tg4_cuda_matmul_sync(
    b_lean_obj_arg a_obj, b_lean_obj_arg b_obj,
    b_lean_obj_arg m_obj, b_lean_obj_arg k_obj, b_lean_obj_arg n_obj
) {
    ensure_cuda_init();
    size_t M = lean_usize_of_nat(m_obj);
    size_t K = lean_usize_of_nat(k_obj);
    size_t N = lean_usize_of_nat(n_obj);

    size_t out_bytes = M * N * sizeof(float);

    // Allocate output ByteArray
    lean_object* out_arr = lean_alloc_sarray(1, out_bytes, out_bytes);
    uint8_t* out_ptr = lean_sarray_cptr(out_arr);
    memset(out_ptr, 0, out_bytes);  // Default to zeros

    // Try GPU path
    try {
        ensure_sgemm_compiled();
        if (g_sgemm_kernel == NULL) {
            return out_arr;  // Return zeros
        }

        // Get input data
        size_t a_bytes = M * K * sizeof(float);
        size_t b_bytes = K * N * sizeof(float);

        const uint8_t* a_ptr = lean_sarray_cptr(a_obj);
        const uint8_t* b_ptr = lean_sarray_cptr(b_obj);

        // Allocate GPU buffers
        CUdeviceptr d_A, d_B, d_C;
        CHECK_CU(cuMemAlloc(&d_A, a_bytes));
        CHECK_CU(cuMemAlloc(&d_B, b_bytes));
        CHECK_CU(cuMemAlloc(&d_C, out_bytes));

        // Copy inputs to GPU
        CHECK_CU(cuMemcpyHtoD(d_A, a_ptr, a_bytes));
        CHECK_CU(cuMemcpyHtoD(d_B, b_ptr, b_bytes));

        // Launch kernel
        int m = (int)M, k = (int)K, n = (int)N;
        void* args[] = { &d_A, &d_B, &d_C, &m, &k, &n };

        int block_x = 16, block_y = 16;
        int grid_x = (N + block_x - 1) / block_x;
        int grid_y = (M + block_y - 1) / block_y;

        CHECK_CU(cuLaunchKernel(
            g_sgemm_kernel,
            grid_x, grid_y, 1,
            block_x, block_y, 1,
            0, g_stream,
            args, NULL
        ));

        // Sync and copy back
        CHECK_CU(cuStreamSynchronize(g_stream));
        CHECK_CU(cuMemcpyDtoH(out_ptr, d_C, out_bytes));

        // Free GPU buffers
        cuMemFree(d_A);
        cuMemFree(d_B);
        cuMemFree(d_C);

    } catch (...) {
        // Return zeros on any error
    }

    return out_arr;
}
