import Lake

open Lake DSL System

package TinyGrad4 where
  version := v!"0.1.0"
  -- Disable Float→Float64 linter warnings globally (intentional use of Float alias)
  leanOptions := #[⟨`weak.linter.floatExplicit, false⟩]

require batteries from git "https://github.com/leanprover-community/batteries" @ "main"
require strata from git "https://github.com/strata-org/Strata" @ "main"
require Cli from git "https://github.com/leanprover/lean4-cli" @ "main"

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
      "-lcuda", "-lnvrtc", "-lstdc++"]

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

-- Check if nvcc (CUDA compiler) is available
def checkNvcc : IO Bool := do
  let out ← IO.Process.output { cmd := "which", args := #["nvcc"] }
  return out.exitCode == 0

extern_lib tg4c pkg := do
  pkg.afterBuildCacheAsync do
    let mut oFiles : Array (Job FilePath) := #[]
    -- Check for CUDA availability first (affects stub compilation)
    let hasCuda ← if System.Platform.isOSX then pure false else checkNvcc
    let cFlagsWithCuda := if hasCuda then cFlags ++ #["-DTG4_HAS_CUDA"] else cFlags
    -- Build C files with Lake's clang
    for file in (← (pkg.dir / "c").readDir) do
      if file.path.extension == some "c" && cSourceFiles.contains file.fileName then
        -- Skip tg4_accel.c - it needs system clang for Accelerate framework
        if file.fileName == "tg4_accel.c" then continue
        let oFile := pkg.buildDir / "c" / ((file.fileName.dropSuffix ".c").toString ++ ".o")
        let srcJob ← inputTextFile file.path
        oFiles := oFiles.push (← buildLeanO oFile srcJob #[] cFlagsWithCuda)
    -- Build tg4_metal.m (Objective-C Metal FFI) on macOS
    if System.Platform.isOSX then
      for file in (← (pkg.dir / "c").readDir) do
        if file.path.extension == some "m" && file.fileName == "tg4_metal.m" then
          let oFile := pkg.buildDir / "c" / ((file.fileName.dropSuffix ".m").toString ++ ".o")
          let srcJob ← inputTextFile file.path
          oFiles := oFiles.push (← buildLeanO oFile srcJob #[] objcFlags)
    -- Build tg4_accel.c with system clang (Accelerate framework needs arm_neon.h)
    if System.Platform.isOSX then
      let accelSrc := pkg.dir / "c" / "tg4_accel.c"
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
    -- Build tg4_cuda.cu with nvcc (Linux with CUDA only)
    -- On macOS/non-CUDA systems, stubs in tg4c_stub.c provide dummy implementations
    if hasCuda then
        let cudaSrc := pkg.dir / "c" / "tg4_cuda.cu"
        if ← cudaSrc.pathExists then
          let oFile := pkg.buildDir / "c" / "tg4_cuda.o"
          IO.FS.createDirAll (pkg.buildDir / "c")
          let leanInclude := (← getLeanSysroot) / "include"
          -- Find CUDA include path (check common locations)
          let cudaInclude ← do
            if ← FilePath.pathExists "/usr/local/cuda/include" then
              pure "/usr/local/cuda/include"
            else
              -- Check $HOME/cuda-12.4 for local installs
              let home := (← IO.getEnv "HOME").getD ""
              if ← FilePath.pathExists (home ++ "/cuda-12.4/include") then
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

lean_lib LeanBench where
  globs := #[`LeanBench.*]

lean_lib Wandb where
  globs := #[`Wandb.*]

lean_lib LeanBenchWandb where
  globs := #[`LeanBenchWandb.*]

lean_lib TinyGrad4Bench where
  globs := #[`TinyGrad4Bench.*]

lean_exe mnist_fusion_bench where
  root := `TinyGrad4Bench.MNISTFusionBenchMain
  moreLinkArgs := metalLinkArgs

lean_lib StrataExperiments

lean_exe cpu_bench where
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

lean_exe data_loader_bench where
  root := `TinyGrad4.Test.DataLoaderBench
  moreLinkArgs := metalLinkArgs

lean_exe zero_copy_bench where
  root := `TinyGrad4.Test.ZeroCopyBench
  moreLinkArgs := metalLinkArgs

lean_exe gpu_loader_bench where
  root := `TinyGrad4.Test.GPULoaderBench
  moreLinkArgs := metalLinkArgs

lean_exe gpu_loader_smoke where
  root := `TinyGrad4.Test.GPULoaderSmokeTest
  moreLinkArgs := metalLinkArgs

lean_exe cuda_smoke where
  root := `TinyGrad4.Test.CUDASmoke
  moreLinkArgs := metalLinkArgs
