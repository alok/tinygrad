import Lake

open Lake DSL System

package TinyGrad4 where
  version := v!"0.1.0"
  srcDir := "lean4"
  testDriver := "tg4_test"
  -- Global linter options - these apply to all modules in the project
  -- Disable Float→Float64 linter warnings globally (intentional use of Float alias)
  leanOptions := #[
    ⟨`weak.linter.floatExplicit, false⟩,
    ⟨`weak.linter.useRawBuffer, true⟩,
    ⟨`doc.verso, false⟩,
    ⟨`linter.missingDocs, true⟩,
  ]

require batteries from git "https://github.com/leanprover-community/batteries" @ "main"
require strata from git "https://github.com/strata-org/Strata" @ "main"
require Cli from git "https://github.com/leanprover/lean4-cli" @ "main"
require mathlib from git "https://github.com/leanprover-community/mathlib4" @ "v4.27.0-rc1"
require LeanBench from git "https://github.com/alok/leanbench" @ "ae0820b"
require scilean from "../scilean"

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

-- Linker args for Metal + Accelerate FFI (macOS only)
-- Required for any executable that links libtg4c.a
-- On non-macOS platforms, these are empty (no Metal/Accelerate support)

def metalLinkArgs : Array String :=
  if System.Platform.isOSX then
    #["-Wl,-syslibroot,/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
      "-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks",
      "-framework", "Metal",
      "-framework", "Foundation",
      "-framework", "Accelerate",
      "-lobjc"]
  else
    -- Linux: CUDA libs - search multiple common installation paths
    -- libcuda.so is always in /usr/lib, libnvrtc location varies by install
    #["-L/usr/lib/x86_64-linux-gnu",
      "-L/usr/local/cuda/lib64",
      "-L/home/alok/cuda-12.4/lib64",  -- ww (freja.mit.edu) local install
      "-Wl,-rpath,/usr/lib/x86_64-linux-gnu",
      "-Wl,-rpath,/usr/local/cuda/lib64",
      "-Wl,-rpath,/home/alok/cuda-12.4/lib64",
      "-Wl,-rpath,$ORIGIN/../lib",
      "-lcuda", "-lnvrtc", "-lcudart", "-lstdc++"]

-- C files to compile (add new files here)

def cSourceFiles : Array String := #["tg4c_stub.c", "tg4_accel.c"]

-- Accelerate framework flags for vDSP/BLAS (macOS only)
-- Needs: system clang headers for arm_neon.h, framework path for Accelerate.h

def accelFlags : Array String :=
  if System.Platform.isOSX then
    cFlags ++ #[
      "-isystem", "/Library/Developer/CommandLineTools/usr/lib/clang/17/include",
      "-iframework", "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks"
    ]
  else
    cFlags

-- Find CUDA include path (returns none if headers not found)

def findCudaInclude : IO (Option String) := do
  -- Respect CUDA_HOME/CUDA_PATH when set
  let envHome ← IO.getEnv "CUDA_HOME"
  let envPath ← IO.getEnv "CUDA_PATH"
  let envs := #[envHome, envPath]
  for env? in envs do
    match env? with
    | some base =>
        let inc := base ++ "/include"
        if ← FilePath.pathExists (inc ++ "/cuda.h") then
          return some inc
    | none => pure ()
  if ← FilePath.pathExists "/usr/local/cuda/include/cuda.h" then
    return some "/usr/local/cuda/include"
  -- Check $HOME/cuda-12.4 for local installs
  let home := (← IO.getEnv "HOME").getD ""
  if !home.isEmpty && (← FilePath.pathExists (home ++ "/cuda-12.4/include/cuda.h")) then
    return some (home ++ "/cuda-12.4/include")
  if ← FilePath.pathExists "/usr/include/cuda.h" then
    return some "/usr/include"
  return none

-- Find nvcc (returns none if not found)

def findNvcc : IO (Option String) := do
  -- Respect CUDA_HOME/CUDA_PATH when set
  let envHome ← IO.getEnv "CUDA_HOME"
  let envPath ← IO.getEnv "CUDA_PATH"
  let home := (← IO.getEnv "HOME").getD ""
  let candidates : Array (Option String) := #[
    envHome.map (· ++ "/bin/nvcc"),
    envPath.map (· ++ "/bin/nvcc"),
    some "/usr/local/cuda/bin/nvcc",
    some "/opt/cuda/bin/nvcc",
    if !home.isEmpty then some (home ++ "/cuda-12.4/bin/nvcc") else none
  ]
  for path? in candidates do
    match path? with
    | some path =>
        if ← FilePath.pathExists path then
          return some path
    | none => pure ()
  return none

/-!
FFI build notes:
- `extern_lib` builds a static library that satisfies `@[extern]` declarations.
- CUDA support is enabled only when headers + nvcc are found; then we compile `tg4_cuda.cu`
  and define `TG4_HAS_CUDA` to disable the stub symbols in `tg4c_stub.c`.
- When CUDA is enabled, link `-lcudart` (see `metalLinkArgs`) to resolve nvcc fatbin registration.
See the Lake build tool docs and the Lean FFI reference for details.
-/

extern_lib tg4c pkg := do
  pkg.afterBuildCacheAsync do
    let cDir := pkg.dir / "lean4" / "c"
    let mut oFiles : Array (Job FilePath) := #[]
    -- Check for CUDA availability first (affects stub compilation)
    let cudaInclude? ← if System.Platform.isOSX then pure none else findCudaInclude
    let nvcc? ← if System.Platform.isOSX then pure none else findNvcc
    let mut enableCuda := cudaInclude?.isSome && nvcc?.isSome
    if enableCuda then
      let cudaSrc := cDir / "tg4_cuda.cu"
      if ← cudaSrc.pathExists then
        let oFile := pkg.buildDir / "c" / "tg4_cuda.o"
        IO.FS.createDirAll (pkg.buildDir / "c")
        let leanInclude := (← getLeanSysroot) / "include"
        let cudaInclude := cudaInclude?.getD "/usr/include"
        let nvcc := nvcc?.getD "nvcc"
        let args := #[
          "-c", "-O3", "-std=c++17", "-lineinfo",
          "-Xcompiler", "-fPIC",
          "-I", leanInclude.toString,
          "-I", cudaInclude,
          "-o", oFile.toString,
          cudaSrc.toString
        ]
        let out ← IO.Process.output { cmd := nvcc, args := args }
        if out.exitCode != 0 then
          enableCuda := false
          IO.eprintln s!"Failed to compile tg4_cuda.cu with nvcc ({nvcc}): {out.stderr}"
        else
          IO.println s!"Compiled CUDA support: {oFile}"
          oFiles := oFiles.push (← inputBinFile oFile)
    let cFlagsWithCuda := if enableCuda then cFlags ++ #["-DTG4_HAS_CUDA"] else cFlags
    -- Build C files with Lake's clang
    for file in (← cDir.readDir) do
      if file.path.extension == some "c" && cSourceFiles.contains file.fileName then
        -- Skip tg4_accel.c on macOS (needs system clang for Accelerate framework)
        if file.fileName == "tg4_accel.c" && System.Platform.isOSX then continue
        let oFile := pkg.buildDir / "c" / ((file.fileName.dropSuffix ".c").toString ++ ".o")
        let srcJob ← inputTextFile file.path
        oFiles := oFiles.push (← buildLeanO oFile srcJob #[] cFlagsWithCuda)
    -- Build tg4_metal.m (Objective-C Metal FFI) on macOS
    if System.Platform.isOSX then
      for file in (← cDir.readDir) do
        if file.path.extension == some "m" && file.fileName == "tg4_metal.m" then
          let oFile := pkg.buildDir / "c" / ((file.fileName.dropSuffix ".m").toString ++ ".o")
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
    if !System.Platform.isOSX && !System.Platform.isWindows then
      let buildLib := pkg.buildDir / "lib"
      IO.FS.createDirAll buildLib
      let leanLib := (← getLeanSysroot) / "lib"
      let candidates := #["libc++.so.1", "libc++abi.so.1"]
      for name in candidates do
        let src := leanLib / name
        if ← src.pathExists then
          let dst := buildLib / name
          let out ← IO.Process.output { cmd := "ln", args := #["-sf", src.toString, dst.toString] }
          if out.exitCode != 0 then
            IO.eprintln s!"Failed to link {name} into {buildLib}: {out.stderr}"
    let name := nameToStaticLib "tg4c"
    buildStaticLib (pkg.staticLibDir / name) oFiles

-- Default build compiles core libraries and their submodules.
-- Use `.andSubmodules` to include the root module plus every submodule in the namespace.
-- TinyGrad4 excludes TinyGrad4.Test.* and TinyGrad4.Backend.Engine to keep `lake build` green; build those explicitly when needed.
-- Data loaders, benches, and experiments are opt-in to avoid work-in-progress build failures.
@[default_target]
lean_lib TinyGrad4 where
  roots := #[`TinyGrad4]
  globs := #[
    .one `TinyGrad4,
    .one `TinyGrad4.Basic,
    .one `TinyGrad4.DType,
    .one `TinyGrad4.Shape,
    .one `TinyGrad4.Tags,
    .one `TinyGrad4.Ops,
    .submodules `TinyGrad4.UOp,
    .submodules `TinyGrad4.Tensor,
    .one `TinyGrad4.Debug,
    .one `TinyGrad4.Pretty,
    .one `TinyGrad4.SimpSets,
    .submodules `TinyGrad4.Linter,
    .one `TinyGrad4.Backend.Accelerate,
    .one `TinyGrad4.Backend.Buffer,
    .one `TinyGrad4.Backend.Cost,
    .one `TinyGrad4.Backend.CostExpr,
    .one `TinyGrad4.Backend.CostExprMeta,
    .one `TinyGrad4.Backend.Cuda,
    .one `TinyGrad4.Backend.Device,
    .one `TinyGrad4.Backend.DeviceBuffer,
    .one `TinyGrad4.Backend.FusedContract,
    .one `TinyGrad4.Backend.FusedEwise,
    .one `TinyGrad4.Backend.FusedEwiseExpr,
    .one `TinyGrad4.Backend.FusedGELU,
    .one `TinyGrad4.Backend.FusedLayerNorm,
    .one `TinyGrad4.Backend.FusedMatmul,
    .one `TinyGrad4.Backend.FusedMatmulExpr,
    .one `TinyGrad4.Backend.FusedReduce,
    .one `TinyGrad4.Backend.FusedReduceExpr,
    .one `TinyGrad4.Backend.FusedSGD,
    .one `TinyGrad4.Backend.FusedSoftmax,
    .one `TinyGrad4.Backend.FusedSoftmaxExpr,
    .one `TinyGrad4.Backend.Fusion,
    .one `TinyGrad4.Backend.Interpreter,
    .one `TinyGrad4.Backend.JIT,
    .one `TinyGrad4.Backend.Memory,
    .one `TinyGrad4.Backend.Metal,
    .one `TinyGrad4.Backend.MetalEwise,
    .one `TinyGrad4.Backend.MetalMatmul,
    .one `TinyGrad4.Backend.MetalRenderer,
    .one `TinyGrad4.Backend.Native,
    .one `TinyGrad4.Backend.PassManager,
    .one `TinyGrad4.Backend.Pattern,
    .one `TinyGrad4.Backend.Rangeify,
    .one `TinyGrad4.Backend.Schedule,
    .one `TinyGrad4.Backend.ShapeTracker,
    .one `TinyGrad4.Backend.TimeM,
    .one `TinyGrad4.Backend.Vectorization,
    .one `TinyGrad4.Backend.View,
    .one `TinyGrad4.Kernel.Spec,
    .one `TinyGrad4.Kernel.Trusted,
    .submodules `TinyGrad4.Gradient,
    .submodules `TinyGrad4.Optim,
    .andSubmodules `TinyGrad4.NN,
    .one `TinyGrad4.Benchmark.Framework,
    .one `TinyGrad4.Benchmark.Runner,
    .one `TinyGrad4.Benchmark.Kernels,
    .one `TinyGrad4.Benchmark.Instrumentation,
    .one `TinyGrad4.Benchmark.CudaBenchmark,
    .one `TinyGrad4.Benchmark.MetalBenchmark,
    .one `TinyGrad4.Benchmark.MetalDirect,
    .andSubmodules `TinyGrad4.Spec
  ]
  needs := #[tg4c]
  precompileModules := true

lean_lib TinyGrad4Data where
  globs := #[.andSubmodules `TinyGrad4.Data]
  needs := #[tg4c]

lean_lib TinyGrad4Test where
  globs := #[.andSubmodules `TinyGrad4.Test]
  needs := #[tg4c]

@[default_target]
lean_lib Tqdm where
  globs := #[.andSubmodules `Tqdm]

@[default_target]
lean_lib LeanBenchNew where
  globs := #[.andSubmodules `LeanBenchNew]

lean_lib Wandb where
  roots := #[`Wandb]
  precompileModules := false

@[default_target]
lean_lib LeanBenchWandb where
  globs := #[.andSubmodules `LeanBenchWandb]

lean_lib TinyGrad4Bench where
  globs := #[.andSubmodules `TinyGrad4Bench]
  needs := #[tg4c]

lean_exe tg4_test where
  root := `TinyGrad4.Test.SmokeAllMain
  needs := #[tg4c]
  moreLinkArgs := metalLinkArgs

lean_exe mnist_fusion_bench where
  root := `TinyGrad4Bench.MNISTFusionBenchMain

lean_exe tg4_leanbench where
  root := `TinyGrad4Bench.LeanBenchMain
  moreLinkArgs := metalLinkArgs

lean_lib StrataExperiments where
  globs := #[.andSubmodules `StrataExperiments]

lean_exe cpu_bench where
  root := `TinyGrad4.Test.BenchmarkMain
  moreLinkArgs := metalLinkArgs

lean_exe benchmark where
  root := `TinyGrad4.Test.BenchmarkMain
  moreLinkArgs := metalLinkArgs

lean_exe tg4_bench where
  root := `TinyGrad4.Benchmark.Main
  moreLinkArgs := metalLinkArgs

lean_exe metal_direct_bench where
  root := `TinyGrad4.Benchmark.MetalDirectMain
  moreLinkArgs := metalLinkArgs

lean_exe io_eval_test where
  root := `TinyGrad4.Test.IOEvalSmoke
  moreLinkArgs := metalLinkArgs

lean_exe metal_matmul_test where
  root := `TinyGrad4.Test.MetalMatmulSmoke
  moreLinkArgs := metalLinkArgs

lean_exe metal_ewise_test where
  root := `TinyGrad4.Test.MetalEwiseTest
  moreLinkArgs := metalLinkArgs

lean_exe mnist_gpu_bench where
  root := `TinyGrad4.Test.MNISTGPUBench
  moreLinkArgs := metalLinkArgs

lean_exe mnist_compiled_train where
  root := `TinyGrad4.Test.MNISTCompiledTrain
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

lean_exe viewstack_test where
  root := `TinyGrad4.Test.ViewStackTest
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

lean_exe multi_prefetch_best_effort_smoke where
  root := `TinyGrad4.Test.MultiPrefetchBestEffortSmoke
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

lean_exe device_loader_resume_smoke where
  root := `TinyGrad4.Test.DeviceLoaderResumeSmoke
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

lean_exe profiler_bench where
  root := `TinyGrad4Bench.ProfilerBench
  moreLinkArgs := metalLinkArgs

lean_exe mathops_smoke where
  root := `TinyGrad4.Test.MathOpsSmokeMain
  moreLinkArgs := metalLinkArgs

lean_exe vec_test where
  root := `TinyGrad4.Test.VectorizationTest
  moreLinkArgs := metalLinkArgs

lean_exe renderer_test where
  root := `TinyGrad4.Test.MetalRendererTest
  moreLinkArgs := metalLinkArgs

-- Metal test executable (requires scripts/build_metal.sh first)
-- NOTE: Lake's bundled clang doesn't support macOS frameworks properly.
-- To build this executable, use the manual build script instead:
--   ./lean4/scripts/build_metal_test.sh
lean_exe metal_test where
  root := `TinyGrad4.Test.MetalTestMain
  -- These flags don't work with Lake's clang due to sysroot issues
  -- Kept here for documentation purposes
  moreLinkArgs := #[]

-- Comprehensive Metal benchmark (requires scripts/build_metal.sh first)
lean_exe comprehensive_bench where
  root := `TinyGrad4.Test.ComprehensiveBench
  moreLinkArgs := metalLinkArgs

-- Gradient verification tests
lean_exe gradcheck where
  root := `TinyGrad4.Test.GradientCheck
  moreLinkArgs := metalLinkArgs
