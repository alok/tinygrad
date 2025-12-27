import Lake

open Lake DSL System

package TinyGrad4 where
  version := v!"0.1.0"
  -- Global linter options - these apply to all modules in the project
  -- DO NOT disable these without good reason - they catch common mistakes
  -- Use weak.* prefix since options are defined in the project itself
  leanOptions := #[
    ⟨`weak.linter.floatExplicit, true⟩  -- Warn about Float vs Float64 confusion
  ]

require batteries from git "https://github.com/leanprover-community/batteries" @ "main"
require Cli from git "https://github.com/leanprover/lean4-cli" @ "main"

def cFlags : Array String :=
  if System.Platform.isWindows then
    #["-O3", "-DNDEBUG", "-DLEAN_EXPORTING"]
  else if System.Platform.isOSX then
    -- macOS needs SDK headers for math.h etc
    #["-O3", "-DNDEBUG", "-DLEAN_EXPORTING", "-fPIC",
      "-isystem", "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include"]
  else
    #["-O3", "-DNDEBUG", "-DLEAN_EXPORTING", "-fPIC"]

-- Objective-C flags for Metal FFI (macOS only)
-- Need to use system SDK for Metal framework
def objcFlags : Array String :=
  #["-O3", "-DNDEBUG", "-DLEAN_EXPORTING", "-fPIC", "-fobjc-arc",
    "-isysroot", "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
    "-iframework", "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks"]

extern_lib tg4c pkg := do
  pkg.afterBuildCacheAsync do
    let mut oFiles : Array (Job FilePath) := #[]
    for file in (← (pkg.dir / "c").readDir) do
      -- Compile .c files only (skip .m files - they need system clang)
      if file.path.extension == some "c" then
        let oFile := pkg.buildDir / "c" / ((file.fileName.dropSuffix ".c").toString ++ ".o")
        let srcJob ← inputTextFile file.path
        oFiles := oFiles.push (← buildLeanO oFile srcJob #[] cFlags)
      -- Note: .m files (Metal FFI) must be compiled separately with system clang:
      --   clang -framework Metal -framework Foundation -c tg4_metal.m -o tg4_metal.o
    let name := nameToStaticLib "tg4c"
    buildStaticLib (pkg.staticLibDir / name) oFiles

@[default_target]
lean_lib TinyGrad4 where
  precompileModules := false

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

lean_exe benchmark where
  root := `TinyGrad4.Test.BenchmarkMain

lean_exe mathops_smoke where
  root := `TinyGrad4.Test.MathOpsSmokeMain

lean_exe vec_test where
  root := `TinyGrad4.Test.VectorizationTest

lean_exe renderer_test where
  root := `TinyGrad4.Test.MetalRendererTest

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

-- Comprehensive Metal benchmark (requires scripts/build_metal.sh first)
lean_exe comprehensive_bench where
  root := `TinyGrad4.Test.ComprehensiveBench
