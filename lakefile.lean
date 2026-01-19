import Lake

open Lake DSL System

package TinyGrad4 where
  version := v!"0.1.0"
  srcDir := "lean4"
  -- Global linter options - these apply to all modules in the project
  -- DO NOT disable these without good reason - they catch common mistakes
  -- Use weak.* prefix since options are defined in the project itself
  leanOptions := #[
    ⟨`weak.linter.floatExplicit, true⟩,  -- Warn about Float vs Float64 confusion
    ⟨`weak.linter.useRawBuffer, true⟩    -- Warn about FloatArray/FlatArray, use RawBuffer
  ]

require batteries from git "https://github.com/leanprover-community/batteries" @ "main"
require Cli from git "https://github.com/leanprover/lean4-cli" @ "main"
require LeanBench from "../LeanBench"

def cFlags : Array String :=
  if System.Platform.isWindows then
    #["-O3", "-DNDEBUG", "-DLEAN_EXPORTING"]
  else if System.Platform.isOSX then
    -- macOS: need SDK paths for system headers (math.h, etc.)
    #["-O3", "-DNDEBUG", "-DLEAN_EXPORTING", "-fPIC",
      "-isystem", "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include"]
  else
    -- Linux: need system headers (math.h, etc.)
    #["-O3", "-DNDEBUG", "-DLEAN_EXPORTING", "-fPIC",
      "-isystem", "/usr/include",
      "-isystem", "/usr/include/x86_64-linux-gnu"]

-- Objective-C flags for Metal FFI (macOS only)
-- Note: Lake's buildLeanO adds --sysroot to Lean toolchain, so we use -iframework for Metal
def objcFlags : Array String :=
  #["-O3", "-DNDEBUG", "-DLEAN_EXPORTING", "-fPIC", "-fobjc-arc",
    "-isystem", "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include",
    "-iframework", "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks"]

def dropSuffix (s suf : String) : String :=
  if s.endsWith suf then s.dropRight suf.length else s

def configCuda? : Option Bool :=
  match get_config? cuda with
  | none => none
  | some v =>
    let v := v.toLower
    if v == "1" || v == "true" || v == "yes" then
      some true
    else if v == "0" || v == "false" || v == "no" then
      some false
    else
      none

private def anyPathExists (paths : Array String) : IO Bool := do
  for path in paths do
    if ← FilePath.pathExists path then
      return true
  return false

def checkCudaAvailable : IO Bool := do
  if System.Platform.isOSX then
    return false
  let home := (← IO.getEnv "HOME").getD ""
  let nvccCandidates := #[
    "/usr/local/cuda/bin/nvcc",
    "/usr/local/cuda-12.4/bin/nvcc",
    home ++ "/cuda/bin/nvcc",
    home ++ "/cuda-12.4/bin/nvcc"
  ]
  let hasNvcc ← anyPathExists nvccCandidates
  if hasNvcc then
    return true
  let includeCandidates := #[
    "/usr/local/cuda/include/cuda.h",
    "/usr/local/cuda-12.4/include/cuda.h",
    home ++ "/cuda/include/cuda.h",
    home ++ "/cuda-12.4/include/cuda.h",
    "/usr/include/cuda.h"
  ]
  let hasInclude ← anyPathExists includeCandidates
  if !hasInclude then
    return false
  let cudaLibCandidates := #[
    "/usr/lib/x86_64-linux-gnu/libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/local/cuda/lib64/libcuda.so",
    "/usr/local/cuda/lib64/libcuda.so.1",
    "/usr/local/cuda-12.4/lib64/libcuda.so",
    "/usr/local/cuda-12.4/lib64/libcuda.so.1",
    home ++ "/cuda-12.4/lib64/libcuda.so",
    home ++ "/cuda-12.4/lib64/libcuda.so.1"
  ]
  let nvrtcLibCandidates := #[
    "/usr/lib/x86_64-linux-gnu/libnvrtc.so",
    "/usr/lib/x86_64-linux-gnu/libnvrtc.so.1",
    "/usr/local/cuda/lib64/libnvrtc.so",
    "/usr/local/cuda/lib64/libnvrtc.so.1",
    "/usr/local/cuda-12.4/lib64/libnvrtc.so",
    "/usr/local/cuda-12.4/lib64/libnvrtc.so.1",
    home ++ "/cuda-12.4/lib64/libnvrtc.so",
    home ++ "/cuda-12.4/lib64/libnvrtc.so.1"
  ]
  let hasCudaLib ← anyPathExists cudaLibCandidates
  let hasNvrtcLib ← anyPathExists nvrtcLibCandidates
  return hasCudaLib && hasNvrtcLib

def cudaLinkArgs : Array String :=
  #["/usr/lib/x86_64-linux-gnu/libcuda.so",
    "-L/usr/local/cuda/lib64",
    "-L/home/alok/cuda-12.4/lib64",
    "-Wl,-rpath,/usr/local/cuda/lib64",
    "-Wl,-rpath,/home/alok/cuda-12.4/lib64",
    "-lnvrtc", "-lstdc++"]

-- Linker args for Metal + Accelerate FFI (macOS only)
-- Required for any executable that links libtg4c.a
-- On non-macOS platforms, these are empty (no Metal/Accelerate support)
def cudaLinkEnabled : Bool :=
  match configCuda? with
  | some v => v && !System.Platform.isOSX
  | none => false

def metalLinkArgs : Array String :=
  if System.Platform.isOSX then
    #["-Wl,-syslibroot,/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
      "-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks",
      "-framework", "Metal",
      "-framework", "Foundation",
      "-framework", "Accelerate",
      "-lobjc"]
  else if cudaLinkEnabled then
    cudaLinkArgs
  else
    #[]

-- C files to compile (add new files here)
def cSourceFiles : Array String := #["tg4c_stub.c", "tg4_accel.c"]

extern_lib tg4c pkg := do
  pkg.afterBuildCacheAsync do
    let mut oFiles : Array (Job FilePath) := #[]
    let cDir := pkg.dir / "lean4" / "c"
    -- Check for CUDA availability first (affects stub compilation)
    let hasCuda ←
      if System.Platform.isOSX then
        pure false
      else
        match configCuda? with
        | some true =>
          let available ← checkCudaAvailable
          if !available then
            IO.eprintln "Warning: CUDA requested but not detected; continuing anyway"
          pure true
        | _ => pure false
    let cFlagsWithCuda := if hasCuda then cFlags ++ #["-DTG4_HAS_CUDA"] else cFlags
    -- Build C files with Lake's clang
    for file in (← cDir.readDir) do
      if file.path.extension == some "c" && cSourceFiles.contains file.fileName then
        -- Skip tg4_accel.c on macOS (needs system clang for Accelerate framework)
        if file.fileName == "tg4_accel.c" && System.Platform.isOSX then continue
        let oFile := pkg.buildDir / "c" / (dropSuffix file.fileName ".c" ++ ".o")
        let srcJob ← inputTextFile file.path
        oFiles := oFiles.push (← buildLeanO oFile srcJob #[] cFlagsWithCuda)
    -- Build tg4_metal.m (Objective-C Metal FFI) on macOS
    if System.Platform.isOSX then
      for file in (← cDir.readDir) do
        if file.path.extension == some "m" && file.fileName == "tg4_metal.m" then
          let oFile := pkg.buildDir / "c" / (dropSuffix file.fileName ".m" ++ ".o")
          let srcJob ← inputTextFile file.path
          oFiles := oFiles.push (← buildLeanO oFile srcJob #[] objcFlags)
    -- Build tg4_accel.c with system clang (Accelerate framework needs arm_neon.h)
    if System.Platform.isOSX then
      let accelSrc := cDir / "tg4_accel.c"
      if ← accelSrc.pathExists then
        let oFile := pkg.buildDir / "c" / "tg4_accel.o"
        IO.FS.createDirAll (pkg.buildDir / "c")
        let leanInclude := (← getLeanSysroot) / "include"
        let args := #["clang", "-c", "-o", oFile.toString, accelSrc.toString,
          "-I", leanInclude.toString,
          "-O3", "-DNDEBUG", "-DLEAN_EXPORTING", "-fPIC",
          "-DACCELERATE_NEW_LAPACK"]
        let out ← IO.Process.output { cmd := "xcrun", args := args }
        if out.exitCode != 0 then
          IO.eprintln s!"Failed to compile tg4_accel.c: {out.stderr}"
        else
          oFiles := oFiles.push (← inputBinFile oFile)
    -- Build tg4_cuda.cu with g++ (Linux with CUDA only)
    -- On macOS/non-CUDA systems, stubs in tg4c_stub.c provide dummy implementations
    if hasCuda then
      let cudaSrc := cDir / "tg4_cuda.cu"
      if ← cudaSrc.pathExists then
        let oFile := pkg.buildDir / "c" / "tg4_cuda.o"
        IO.FS.createDirAll (pkg.buildDir / "c")
        let leanInclude := (← getLeanSysroot) / "include"
        -- Find CUDA include path (check common locations)
        let cudaInclude ← do
          if ← FilePath.pathExists "/usr/local/cuda/include" then
            pure "/usr/local/cuda/include"
          else if ← FilePath.pathExists "/usr/local/cuda-12.4/include" then
            pure "/usr/local/cuda-12.4/include"
          else
            let home := (← IO.getEnv "HOME").getD ""
            if ← FilePath.pathExists (home ++ "/cuda/include") then
              pure (home ++ "/cuda/include")
            else if ← FilePath.pathExists (home ++ "/cuda-12.4/include") then
              pure (home ++ "/cuda-12.4/include")
            else
              pure "/usr/include"
        -- Compile with g++ as C++ (Driver API only, no nvcc needed)
        let args := #[
          "-c", "-x", "c++", "-O3", "-fPIC",
          "-DTG4_HAS_CUDA",
          "-I", leanInclude.toString,
          "-I", cudaInclude,
          "-o", oFile.toString,
          cudaSrc.toString
        ]
        let out ← IO.Process.output { cmd := "g++", args := args }
        if out.exitCode != 0 then
          IO.eprintln s!"Failed to compile tg4_cuda.cu: {out.stderr}"
        else
          IO.println s!"Compiled CUDA support: {oFile}"
          oFiles := oFiles.push (← inputBinFile oFile)
    let name := nameToStaticLib "tg4c"
    buildStaticLib (pkg.staticLibDir / name) oFiles

@[default_target]
lean_lib TinyGrad4 where
  precompileModules := true

lean_lib Float64

lean_lib LeanBenchNew where
  globs := #[`LeanBenchNew.*]

lean_lib Wandb where
  globs := #[`Wandb.*]

lean_lib LeanBenchWandb where
  globs := #[`LeanBenchWandb.*]

lean_lib TinyGrad4Bench where
  globs := #[`TinyGrad4Bench.*]

lean_exe mnist_fusion_bench where
  root := `TinyGrad4Bench.MNISTFusionBenchMain

lean_exe tg4_leanbench where
  root := `TinyGrad4Bench.LeanBenchMain
  moreLinkArgs := metalLinkArgs

lean_exe benchmark where
  root := `TinyGrad4.Test.BenchmarkMain
  moreLinkArgs := metalLinkArgs

lean_exe mathops_smoke where
  root := `TinyGrad4.Test.MathOpsSmokeMain

lean_exe vec_test where
  root := `TinyGrad4.Test.VectorizationTest

lean_exe renderer_test where
  root := `TinyGrad4.Test.MetalRendererTest

lean_exe io_eval_test where
  root := `TinyGrad4.Test.IOEvalSmoke
  moreLinkArgs := metalLinkArgs

lean_exe float16_cast_smoke where
  root := `TinyGrad4.Test.Float16CastSmoke

lean_exe triton_matmul_smoke where
  root := `TinyGrad4.Test.CUDATritonMatmulSmoke
  moreLinkArgs := metalLinkArgs

lean_exe linear_triton_smoke where
  root := `TinyGrad4.Test.LinearTritonSmoke
  moreLinkArgs := metalLinkArgs

lean_exe linear_triton_bias_smoke where
  root := `TinyGrad4.Test.LinearTritonBiasSmoke
  moreLinkArgs := metalLinkArgs

lean_exe linear_triton_bias_batched_smoke where
  root := `TinyGrad4.Test.LinearTritonBiasBatchedSmoke
  moreLinkArgs := metalLinkArgs

lean_exe emit_triton_ptx where
  root := `TinyGrad4.Test.EmitTritonPTXMain

lean_exe metal_matmul_test where
  root := `TinyGrad4.Test.MetalMatmulSmoke
  moreLinkArgs := metalLinkArgs

lean_exe metal_ewise_test where
  root := `TinyGrad4.Test.MetalEwiseTest
  moreLinkArgs := metalLinkArgs

-- Metal test executable (requires scripts/build_metal.sh first)
-- NOTE: Lake's bundled clang doesn't support macOS frameworks properly.
-- To build this executable, use the manual build script instead:
--   ./scripts/build_metal_test.sh
lean_exe metal_test where
  root := `TinyGrad4.Test.MetalTestMain
  -- These flags don't work with Lake's clang due to sysroot issues
  -- Kept here for documentation purposes
  moreLinkArgs := #[]

-- Benchmark library (reusable infrastructure)
lean_lib TinyGrad4Benchmark where
  globs := #[`TinyGrad4.Benchmark, `TinyGrad4.Benchmark.*]

-- GPU benchmark CLI with lean4-cli
lean_exe tg4_bench where
  root := `TinyGrad4.Benchmark.Main
  moreLinkArgs := metalLinkArgs

-- Comprehensive Metal benchmark (requires scripts/build_metal.sh first)
lean_exe comprehensive_bench where
  root := `TinyGrad4.Test.ComprehensiveBench

-- Gradient verification tests
lean_exe gradcheck where
  root := `TinyGrad4.Test.GradientCheck

lean_exe mnist_gpu_bench where
  root := `TinyGrad4.Test.MNISTGPUBench
  moreLinkArgs := metalLinkArgs

lean_exe matmul_nan_debug where
  root := `TinyGrad4.Test.MatmulNaNDebug
  moreLinkArgs := metalLinkArgs

lean_exe mnist_nan_debug where
  root := `TinyGrad4.Test.MNISTNaNDebug
  moreLinkArgs := metalLinkArgs

lean_exe transpose_matmul_debug where
  root := `TinyGrad4.Test.TransposeMatmulDebug
  moreLinkArgs := metalLinkArgs

lean_exe backward_step_debug where
  root := `TinyGrad4.Test.BackwardStepDebug
  moreLinkArgs := metalLinkArgs

lean_exe relu_gpu_debug where
  root := `TinyGrad4.Test.ReluGPUDebug
  moreLinkArgs := metalLinkArgs

lean_exe relu_gpu_debug2 where
  root := `TinyGrad4.Test.ReluGPUDebug2
  moreLinkArgs := metalLinkArgs

lean_exe gpu_reduce_test where
  root := `TinyGrad4.Test.GPUReduceTest
  moreLinkArgs := metalLinkArgs

lean_exe gpu_matmul_test where
  root := `TinyGrad4.Test.GPUMatmulTest
  moreLinkArgs := metalLinkArgs

lean_exe matmul_debug32 where
  root := `TinyGrad4.Test.MatmulDebug32
  moreLinkArgs := metalLinkArgs

lean_exe conv_test where
  root := `TinyGrad4.Test.Conv2dSmoke
  moreLinkArgs := metalLinkArgs

lean_exe batchnorm_test where
  root := `TinyGrad4.Test.BatchNormSmoke
  moreLinkArgs := metalLinkArgs

lean_exe dropout_test where
  root := `TinyGrad4.Test.DropoutSmoke
  moreLinkArgs := metalLinkArgs

lean_exe instrumentation_test where
  root := `TinyGrad4.Test.InstrumentationSmoke
  moreLinkArgs := metalLinkArgs

-- View stack regression tests
lean_exe viewstack_test where
  root := `TinyGrad4.Test.ViewStackTest
  moreLinkArgs := metalLinkArgs

-- Direct Metal FFI benchmark (requires scripts/build_metal.sh first)
lean_exe metal_direct_bench where
  root := `TinyGrad4.Benchmark.MetalDirectMain
  moreLinkArgs := metalLinkArgs

lean_exe jit_test where
  root := `TinyGrad4.Test.JITSmoke
  moreLinkArgs := metalLinkArgs

lean_exe dataset_test where
  root := `TinyGrad4.Test.DatasetSmoke
  moreLinkArgs := metalLinkArgs

lean_exe checkpoint_resume_test where
  root := `TinyGrad4.Test.CheckpointResumeSmoke
  moreLinkArgs := metalLinkArgs

lean_exe data_loader_bench where
  root := `TinyGrad4.Test.DataLoaderBench
  moreLinkArgs := metalLinkArgs

lean_exe zero_copy_bench where
  root := `TinyGrad4.Test.ZeroCopyBench
  moreLinkArgs := metalLinkArgs

lean_exe raw_file_bench where
  root := `TinyGrad4.Test.RawFileBench
  moreLinkArgs := metalLinkArgs

lean_exe gpu_loader_bench where
  root := `TinyGrad4.Test.GPULoaderBench
  moreLinkArgs := metalLinkArgs

lean_exe gpu_loader_smoke where
  root := `TinyGrad4.Test.GPULoaderSmokeTest
  moreLinkArgs := metalLinkArgs

lean_exe tpu_loader_smoke where
  root := `TinyGrad4.Test.TPULoaderSmokeTest
  moreLinkArgs := metalLinkArgs

lean_exe cuda_smoke where
  root := `TinyGrad4.Test.CUDASmoke
  moreLinkArgs := metalLinkArgs

lean_exe cuda_bench where
  root := `TinyGrad4.Test.CUDABench
  moreLinkArgs := metalLinkArgs

lean_exe cuda_compile_test where
  root := `TinyGrad4.Test.CUDACompileTest
  moreLinkArgs := metalLinkArgs

lean_exe cuda_minimal where
  root := `TinyGrad4.Test.CUDAMinimal
  moreLinkArgs := metalLinkArgs

lean_exe buffer_bench where
  root := `TinyGrad4Bench.BufferProtocolBench
  moreLinkArgs := metalLinkArgs
