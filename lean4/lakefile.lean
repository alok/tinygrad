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
    #["-O3", "-DNDEBUG", "-DLEAN_EXPORTING", "-fPIC"]

extern_lib tg4c pkg := do
  pkg.afterBuildCacheAsync do
    let mut oFiles : Array (Job FilePath) := #[]
    -- Only build tg4c_stub.c; tg4_gemm.c has duplicate symbols
    for file in (← (pkg.dir / "c").readDir) do
      if file.path.extension == some "c" && file.fileName == "tg4c_stub.c" then
        let oFile := pkg.buildDir / "c" / ((file.fileName.dropSuffix ".c").toString ++ ".o")
        let srcJob ← inputTextFile file.path
        oFiles := oFiles.push (← buildLeanO oFile srcJob #[] cFlags)
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

lean_lib StrataExperiments

-- Benchmark CLI (uses subprocess-based Metal runner)
lean_exe tg4_bench where
  root := `TinyGrad4.Benchmark.Main

-- Direct Metal FFI benchmark
-- Requires: ./scripts/build_metal_ffi.sh to be run first
-- Uses moreLinkArgs to link the pre-built Metal FFI and frameworks
lean_exe metal_direct_bench where
  root := `TinyGrad4.Benchmark.MetalDirectMain
  -- Override sysroot and link Metal frameworks
  moreLinkArgs := #[
    -- Use macOS SDK for framework linking
    "-Wl,-syslibroot,/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
    "-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks",
    "-framework", "Metal",
    "-framework", "Foundation",
    "-lobjc",
    ".lake/build/metal/tg4_metal.o"
  ]
