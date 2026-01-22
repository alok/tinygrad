import Float64
import TinyGrad4.Backend.Cuda
import TinyGrad4.Backend.Buffer
import TinyGrad4.Backend.Native
import TinyGrad4.Backend.LeanPtxEmit

/-!
# Triton-Generated Matmul (CUDA PTX)

Loads a precompiled PTX kernel and runs a fixed-shape matmul.
By default, PTX is emitted by Lean for correctness; Triton PTX can override via env.
This mirrors `extra/gemm/triton_nv_matmul.py` at the interface level and is intentionally narrow:
- float16 inputs/outputs
- fixed M/N/K and block sizes from the compiled kernel

Environment config (used by `getConfigFromEnv`):
- TG4_TRITON_PTX (path to PTX)
- TG4_TRITON_KERNEL (default: matmul_kernel)
- TG4_TRITON_BLOCK_M / _N / _K
- TG4_TRITON_NUM_WARPS
- TG4_TRITON_SHARED_BYTES (default: 0)
- TG4_TRITON_M / _N / _K (expected shapes)
- TG4_TRITON_PTX_DIR (optional cache dir for auto-emitted PTX)
- TG4_TRITON_LEAN_VARIANT (basic|tiled|smem, optional for Lean PTX emit)
- TG4_TRITON_DUMP (set to dump PTX to {ptxPath}.dump when emitting)
-/

namespace TinyGrad4.Backend.CudaTritonMatmul

open TinyGrad4
open TinyGrad4.Backend.Cuda

inductive TritonPreset where
  | linearSmall
  | linear
  | linearLarge
  | leanBasic
  deriving Repr, DecidableEq, Inhabited

structure TritonPresetParams where
  blockM : Nat
  blockN : Nat
  blockK : Nat
  numWarps : Nat
  numStages : Nat

private def presetParams : TritonPreset → TritonPresetParams
  | .linearSmall => { blockM := 64, blockN := 64, blockK := 32, numWarps := 4, numStages := 2 }
  | .linear => { blockM := 64, blockN := 128, blockK := 64, numWarps := 4, numStages := 2 }
  | .linearLarge => { blockM := 128, blockN := 128, blockK := 32, numWarps := 8, numStages := 3 }
  | .leanBasic => { blockM := 1, blockN := 1, blockK := 1, numWarps := 1, numStages := 1 }

private structure CudaTarget where
  sm : Nat
  ptxVersion : Nat

private def ptxVersionFromDriver (driver : Nat) : Nat :=
  let cudaMajor := driver / 1000
  let cudaMinor := (driver % 1000) / 10
  if cudaMajor < 4 then
    50
  else
    (cudaMajor - 4) * 10 + cudaMinor

private def getCudaTarget : IO CudaTarget := do
  let driver ← TinyGrad4.Backend.Cuda.cudaDriverVersion
  let (major, minor) ← TinyGrad4.Backend.Cuda.cudaComputeCapability
  let sm := major * 10 + minor
  let ptxVersion := ptxVersionFromDriver driver
  pure { sm, ptxVersion }

private def ptxDir : IO System.FilePath := do
  match ← IO.getEnv "TG4_TRITON_PTX_DIR" with
  | some dir => pure (System.FilePath.mk dir)
  | none => pure (System.FilePath.mk "tmp")

private def leanKernelName (withBias : Bool) : String :=
  if withBias then "tg4_matmul_bias" else "tg4_matmul_basic"

private def leanVariantTag (variant : TinyGrad4.Backend.LeanPtxEmit.PtxVariant) : String :=
  match variant with
  | .basic => "basic"
  | .tiled => "tiled"
  | .smem => "smem"

private def hexDigit (n : Nat) : Char :=
  if n < 10 then
    Char.ofNat (n + 48)
  else
    Char.ofNat (n - 10 + 97)

private def hex8 (v : UInt32) : String :=
  let rec go (n : Nat) (count : Nat) (acc : List Char) : List Char :=
    match count with
    | 0 => acc
    | count + 1 =>
      let digit := n % 16
      let ch := hexDigit digit
      go (n / 16) count (ch :: acc)
  String.ofList (go v.toNat 8 [])

private def scaleTag (scaleBits : Option UInt32) : String :=
  match scaleBits with
  | some bits => s!"_s{hex8 bits}"
  | none => ""

private def reluTag (relu : Bool) : String :=
  if relu then "_relu" else ""

private def leanPtxPath (dir : System.FilePath) (target : CudaTarget)
    (m n k blockM blockN blockK numWarps : Nat) (variantTag : String)
    (withBias : Bool) (bias2 : Bool) (scaleBits : Option UInt32) (relu : Bool) : System.FilePath :=
  let baseTag := if bias2 then "lean_bias2" else if withBias then "lean_bias" else "lean"
  let tag := s!"{baseTag}{scaleTag scaleBits}{reluTag relu}"
  dir / s!"{tag}_{variantTag}_sm{target.sm}_ptx{target.ptxVersion}_{m}x{n}x{k}_bm{blockM}bn{blockN}bk{blockK}w{numWarps}.ptx"

-- Triton PTX emit helpers removed; Lean PTX generator is the default path now.

private def divisible (x y : Nat) : Bool :=
  x % y == 0

/-- Select a Triton preset for a matmul shape/dtype. -/
def choosePreset (dtype : DType) (m n k : Nat) : Option TritonPreset :=
  if dtype != .float16 && dtype != .float32 then
    none
  else if divisible m 128 && divisible n 128 && divisible k 32 then
    some TritonPreset.linearLarge
  else if divisible m 64 && divisible n 128 && divisible k 64 then
    some TritonPreset.linear
  else if divisible m 64 && divisible n 64 && divisible k 32 then
    some TritonPreset.linearSmall
  else
    some TritonPreset.leanBasic

/-- Configuration for a Triton-generated matmul kernel. -/
structure TritonMatmulConfig where
  kernelName : String := "matmul_kernel"
  ptxPath : System.FilePath
  blockM : Nat
  blockN : Nat
  blockK : Nat
  numWarps : Nat
  sharedBytes : Nat
  paramCount : Nat
  expectedM : Nat
  expectedN : Nat
  expectedK : Nat
  scaleBits : Option UInt32 := none
  bias2 : Bool := false
  relu : Bool := false

private def ensure (cond : Bool) (msg : String) : IO Unit :=
  if cond then pure () else throw (IO.userError msg)

private def configMatches (cfg : TritonMatmulConfig) (m n k : Nat)
    (scaleBits : Option UInt32) (bias2 : Bool) (relu : Bool) : Bool :=
  cfg.expectedM == m && cfg.expectedN == n && cfg.expectedK == k &&
    cfg.scaleBits == scaleBits && cfg.bias2 == bias2 && cfg.relu == relu

private def ceilDiv (a b : Nat) : Nat :=
  (a + b - 1) / b

private def ltrim (s : String) : String :=
  let rec drop (cs : List Char) : List Char :=
    match cs with
    | [] => []
    | c :: rest =>
      if c == ' ' || c == '\t' then
        drop rest
      else
        cs
  String.ofList (drop s.toList)

private def countKernelParams (ptx : String) (kernel : String) : Nat :=
  let entry := s!".entry {kernel}("
  match ptx.splitOn entry with
  | _ :: after :: _ =>
    let lines := after.splitOn "\n"
    let rec go (rest : List String) (acc : Nat) : Nat :=
      match rest with
      | [] => acc
      | line :: tail =>
        let trimmed := ltrim line
        if trimmed.startsWith ")" then
          acc
        else if trimmed.startsWith ".param" then
          go tail (acc + 1)
        else
          go tail acc
    go lines 0
  | _ => 0

private def kernelParamCount (ptxPath : System.FilePath) (kernel : String) : IO Nat := do
  let ptx ← IO.FS.readFile ptxPath
  let count := countKernelParams ptx kernel
  if count == 0 then
    throw (IO.userError s!"CudaTritonMatmul: kernel {kernel} not found in PTX")
  if count < 3 then
    throw (IO.userError s!"CudaTritonMatmul: kernel {kernel} param count {count} is too small")
  return count

/-- Cache for optional Triton config loaded from environment. -/
initialize tritonConfigCache : IO.Ref (Option (Option TritonMatmulConfig)) ← IO.mkRef none

/-- Cache for optional Triton bias config (auto-emitted). -/
initialize tritonBiasConfigCache : IO.Ref (Option (Option TritonMatmulConfig)) ← IO.mkRef none

/-- Optional default Triton preset (enables auto-emit without env). -/
initialize tritonPresetCache : IO.Ref (Option TritonPreset) ← IO.mkRef none

private def envNat? (name : String) : IO (Option Nat) := do
  match ← IO.getEnv name with
  | none => pure none
  | some v =>
    match v.toNat? with
    | some n => pure (some n)
    | none => throw (IO.userError s!"CudaTritonMatmul: {name} must be Nat, got '{v}'")

private def envFlag (name : String) : IO Bool := do
  match ← IO.getEnv name with
  | none => pure false
  | some v =>
    let v := v.toLower
    pure (v == "1" || v == "true" || v == "yes")

private def requireEnvNat (name : String) : IO Nat := do
  match ← envNat? name with
  | some n => pure n
  | none => throw (IO.userError s!"CudaTritonMatmul: missing {name}")

private def envNatDefault (name : String) (default : Nat) : IO Nat := do
  match ← envNat? name with
  | some n => pure n
  | none => pure default

/-- Load Triton config from environment variables.
    Required:
    - TG4_TRITON_PTX
    - TG4_TRITON_BLOCK_M / _N / _K
    - TG4_TRITON_NUM_WARPS
    - TG4_TRITON_M / _N / _K
    Optional:
    - TG4_TRITON_KERNEL (default: matmul_kernel)
    - TG4_TRITON_SHARED_BYTES (default: 0)
    - TG4_TRITON_PTX_DIR (used when TG4_TRITON_PTX is unset)
    -/
def loadConfigFromEnv : IO (Option TritonMatmulConfig) := do
  let ptxStr? ← IO.getEnv "TG4_TRITON_PTX"
  let useDefault := ptxStr?.isNone
  let ptxPath? ←
    match ptxStr? with
    | some ptxStr => pure (some (System.FilePath.mk ptxStr))
    | none =>
      let dir ← ptxDir
      let defaultPath := dir / "triton_matmul.ptx"
      if ← defaultPath.pathExists then
        pure (some defaultPath)
      else
        pure none
  match ptxPath? with
  | none => return none
  | some ptxPath =>
    if !(← ptxPath.pathExists) then
      throw (IO.userError s!"CudaTritonMatmul: TG4_TRITON_PTX not found: {ptxPath}")
    let blockM? ← envNat? "TG4_TRITON_BLOCK_M"
    let blockN? ← envNat? "TG4_TRITON_BLOCK_N"
    let blockK? ← envNat? "TG4_TRITON_BLOCK_K"
    let numWarps? ← envNat? "TG4_TRITON_NUM_WARPS"
    let expectedM? ← envNat? "TG4_TRITON_M"
    let expectedN? ← envNat? "TG4_TRITON_N"
    let expectedK? ← envNat? "TG4_TRITON_K"
    if useDefault && (blockM?.isNone || blockN?.isNone || blockK?.isNone || numWarps?.isNone ||
        expectedM?.isNone || expectedN?.isNone || expectedK?.isNone) then
      return none
    let kernelName := (← IO.getEnv "TG4_TRITON_KERNEL").getD "matmul_kernel"
    let blockM ← requireEnvNat "TG4_TRITON_BLOCK_M"
    let blockN ← requireEnvNat "TG4_TRITON_BLOCK_N"
    let blockK ← requireEnvNat "TG4_TRITON_BLOCK_K"
    let numWarps ← requireEnvNat "TG4_TRITON_NUM_WARPS"
    let sharedBytes ← envNatDefault "TG4_TRITON_SHARED_BYTES" 0
    let expectedM ← requireEnvNat "TG4_TRITON_M"
    let expectedN ← requireEnvNat "TG4_TRITON_N"
    let expectedK ← requireEnvNat "TG4_TRITON_K"
    let paramCount ← kernelParamCount ptxPath kernelName
    return some {
      kernelName,
      ptxPath,
      blockM,
      blockN,
      blockK,
      numWarps,
      sharedBytes,
      paramCount,
      expectedM,
      expectedN,
      expectedK,
      scaleBits := none,
      bias2 := false,
      relu := false
    }

/-- Load Triton bias config from environment variables (requires TG4_TRITON_WITH_BIAS=1, optional TG4_TRITON_WITH_BIAS2=1). -/
def loadBiasConfigFromEnv : IO (Option TritonMatmulConfig) := do
  if !(← envFlag "TG4_TRITON_WITH_BIAS") then
    return none
  let ptxStr? ← IO.getEnv "TG4_TRITON_PTX"
  let ptxPath ←
    match ptxStr? with
    | some ptxStr => pure (System.FilePath.mk ptxStr)
    | none => throw (IO.userError "CudaTritonMatmul: TG4_TRITON_PTX required for bias kernel")
  if !(← ptxPath.pathExists) then
    throw (IO.userError s!"CudaTritonMatmul: TG4_TRITON_PTX not found: {ptxPath}")
  let kernelName := (← IO.getEnv "TG4_TRITON_KERNEL").getD "linear_kernel"
  let blockM ← requireEnvNat "TG4_TRITON_BLOCK_M"
  let blockN ← requireEnvNat "TG4_TRITON_BLOCK_N"
  let blockK ← requireEnvNat "TG4_TRITON_BLOCK_K"
  let numWarps ← requireEnvNat "TG4_TRITON_NUM_WARPS"
  let sharedBytes ← envNatDefault "TG4_TRITON_SHARED_BYTES" 0
  let expectedM ← requireEnvNat "TG4_TRITON_M"
  let expectedN ← requireEnvNat "TG4_TRITON_N"
  let expectedK ← requireEnvNat "TG4_TRITON_K"
  let bias2 ← envFlag "TG4_TRITON_WITH_BIAS2"
  let paramCount ← kernelParamCount ptxPath kernelName
  let minParams := if bias2 then 5 else 4
  if paramCount < minParams then
    throw (IO.userError s!"CudaTritonMatmul: bias kernel {kernelName} must have at least {minParams} params")
  return some {
    kernelName,
    ptxPath,
    blockM,
    blockN,
    blockK,
    numWarps,
    sharedBytes,
    paramCount,
    expectedM,
    expectedN,
    expectedK,
    scaleBits := none,
    bias2,
    relu := false
  }

/-- Load (and memoize) Triton config from the environment. -/
def getConfigFromEnv : IO (Option TritonMatmulConfig) := do
  match ← tritonConfigCache.get with
  | some cfg => return cfg
  | none =>
    let cfg ← loadConfigFromEnv
    tritonConfigCache.set (some cfg)
    return cfg

/-- Load (and memoize) Triton bias config from the environment. -/
def getBiasConfigFromEnv : IO (Option TritonMatmulConfig) := do
  match ← tritonBiasConfigCache.get with
  | some cfg => return cfg
  | none =>
    let cfg ← loadBiasConfigFromEnv
    tritonBiasConfigCache.set (some cfg)
    return cfg

/-- Clear cached Triton config (useful when env vars change). -/
def clearConfigCache : IO Unit :=
  tritonConfigCache.set none

/-- Override cached Triton config (useful for tests or programmatic setup). -/
def setConfig (cfg? : Option TritonMatmulConfig) : IO Unit :=
  tritonConfigCache.set (some cfg?)

/-- Set a default preset for auto-emitting PTX when env config is missing. -/
def setDefaultPreset (preset? : Option TritonPreset) : IO Unit :=
  tritonPresetCache.set preset?

/-- Get the default preset for auto-emitting PTX. -/
def getDefaultPreset : IO (Option TritonPreset) :=
  tritonPresetCache.get

/-- Emit PTX + configure Triton based on a preset when env config is missing. -/
def ensureConfig (preset : TritonPreset) (m n k : Nat) (scaleBits : Option UInt32 := none) (relu : Bool := false) :
    IO (Option TritonMatmulConfig) := do
  match ← tritonConfigCache.get with
  | some (some cfg) =>
    if configMatches cfg m n k scaleBits false relu then
      return some cfg
  | _ => pure ()

  let envCfg? ← loadConfigFromEnv
  let cfgFromEnv? :=
    match envCfg? with
    | some cfg =>
      if configMatches cfg m n k scaleBits false relu then
        some cfg
      else
        none
    | none => none
  match cfgFromEnv? with
  | some cfg =>
    setConfig (some cfg)
    return some cfg
  | none => do
    let target ← getCudaTarget
    let kernelName := leanKernelName false
    let params0 := presetParams preset
    let useLeanBasic :=
      match TinyGrad4.Backend.LeanPtxEmit.tileShape params0.blockM params0.blockN params0.numWarps with
      | some _ => false
      | none => true
    let params := if useLeanBasic then presetParams .leanBasic else params0
    let defaultVariant :=
      if useLeanBasic then
        TinyGrad4.Backend.LeanPtxEmit.PtxVariant.basic
      else
        TinyGrad4.Backend.LeanPtxEmit.PtxVariant.tiled
    let variant ←
      match ← TinyGrad4.Backend.LeanPtxEmit.variantFromEnv with
      | some v => pure v
      | none => pure defaultVariant
    let variantTag := leanVariantTag variant
    let dir ← ptxDir
    let ptxPath :=
      leanPtxPath dir target m n k params.blockM params.blockN params.blockK params.numWarps variantTag false false scaleBits relu
    if !(← ptxPath.pathExists) || (← envFlag "TG4_TRITON_FORCE") then
      let emitCfg : TinyGrad4.Backend.LeanPtxEmit.EmitConfig := {
        ptxPath,
        kernelName,
        m,
        n,
        k,
        strideAm := k,
        strideAk := 1,
        strideBk := n,
        strideBn := 1,
        strideCm := n,
        strideCn := 1,
        aOffset := 0,
        bOffset := 0,
        cOffset := 0,
        biasOffset := 0,
        bias2Offset := 0,
        maskMStart := 0,
        maskMEnd := m,
        maskNStart := 0,
        maskNEnd := n,
        maskKStart := 0,
        maskKEnd := k,
        blockM := params.blockM,
        blockN := params.blockN,
        blockK := params.blockK,
        numWarps := params.numWarps,
        ptxVersion := target.ptxVersion,
        sm := target.sm,
        withBias := false,
        scaleBits,
        relu,
        variant
      }
      let rc ← TinyGrad4.Backend.LeanPtxEmit.emit emitCfg
      if rc != 0 then
        return none
    if !(← ptxPath.pathExists) then
      return none
    let paramCount ← kernelParamCount ptxPath kernelName
    let cfg : TritonMatmulConfig := {
      kernelName,
      ptxPath,
      blockM := params.blockM,
      blockN := params.blockN,
      blockK := params.blockK,
      numWarps := params.numWarps,
      sharedBytes := 0,
      paramCount,
      expectedM := m,
      expectedN := n,
      expectedK := k,
      scaleBits,
      bias2 := false,
      relu
    }
    setConfig (some cfg)
    return some cfg

/-- Emit PTX + configure Triton for a fused bias kernel. -/
def ensureConfigBias (preset : TritonPreset) (m n k : Nat)
    (scaleBits : Option UInt32 := none) (bias2 : Bool := false) (relu : Bool := false) :
    IO (Option TritonMatmulConfig) := do
  match ← tritonBiasConfigCache.get with
  | some (some cfg) =>
    if configMatches cfg m n k scaleBits bias2 relu then
      return some cfg
  | _ => pure ()

  let envCfg? ← loadBiasConfigFromEnv
  let cfgFromEnv? :=
    match envCfg? with
    | some cfg =>
      if configMatches cfg m n k scaleBits bias2 relu then
        some cfg
      else
        none
    | none => none
  match cfgFromEnv? with
  | some cfg =>
    tritonBiasConfigCache.set (some (some cfg))
    return some cfg
  | none => do
    let target ← getCudaTarget
    let kernelName := leanKernelName true
    let params0 := presetParams preset
    let useLeanBasic :=
      match TinyGrad4.Backend.LeanPtxEmit.tileShape params0.blockM params0.blockN params0.numWarps with
      | some _ => false
      | none => true
    let params := if useLeanBasic then presetParams .leanBasic else params0
    let defaultVariant :=
      if useLeanBasic then
        TinyGrad4.Backend.LeanPtxEmit.PtxVariant.basic
      else
        TinyGrad4.Backend.LeanPtxEmit.PtxVariant.tiled
    let variant ←
      match ← TinyGrad4.Backend.LeanPtxEmit.variantFromEnv with
      | some v => pure v
      | none => pure defaultVariant
    let variantTag := leanVariantTag variant
    let dir ← ptxDir
    let ptxPath :=
      leanPtxPath dir target m n k params.blockM params.blockN params.blockK params.numWarps variantTag true bias2 scaleBits relu
    if !(← ptxPath.pathExists) || (← envFlag "TG4_TRITON_FORCE") then
      let emitCfg : TinyGrad4.Backend.LeanPtxEmit.EmitConfig := {
        ptxPath,
        kernelName,
        m,
        n,
        k,
        strideAm := k,
        strideAk := 1,
        strideBk := n,
        strideBn := 1,
        strideCm := n,
        strideCn := 1,
        aOffset := 0,
        bOffset := 0,
        cOffset := 0,
        biasOffset := 0,
        bias2Offset := 0,
        maskMStart := 0,
        maskMEnd := m,
        maskNStart := 0,
        maskNEnd := n,
        maskKStart := 0,
        maskKEnd := k,
        blockM := params.blockM,
        blockN := params.blockN,
        blockK := params.blockK,
        numWarps := params.numWarps,
        ptxVersion := target.ptxVersion,
        sm := target.sm,
        withBias := true,
        withBias2 := bias2,
        scaleBits,
        relu,
        variant
      }
      let rc ← TinyGrad4.Backend.LeanPtxEmit.emit emitCfg
      if rc != 0 then
        return none
    if !(← ptxPath.pathExists) then
      return none
    let paramCount ← kernelParamCount ptxPath kernelName
    let minParams := if bias2 then 5 else 4
    if paramCount < minParams then
      return none
    let cfg : TritonMatmulConfig := {
      kernelName,
      ptxPath,
      blockM := params.blockM,
      blockN := params.blockN,
      blockK := params.blockK,
      numWarps := params.numWarps,
      sharedBytes := 0,
      paramCount,
      expectedM := m,
      expectedN := n,
      expectedK := k,
      scaleBits,
      bias2,
      relu
    }
    tritonBiasConfigCache.set (some (some cfg))
    return some cfg

/-- Load a PTX module from disk and bind the kernel. -/
private def loadKernel (cfg : TritonMatmulConfig) : IO CUDAProgram := do
  let ptx ← IO.FS.readFile cfg.ptxPath
  cudaLoadPTX cfg.kernelName ptx

private def padKernelArgs (bufs : Array CUDABuffer) (count : Nat) (fill : CUDABuffer) : Array CUDABuffer :=
  if count <= bufs.size then
    bufs
  else
    let extra := Array.replicate (count - bufs.size) fill
    bufs ++ extra

/-- Execute Triton matmul (float16) with fixed sizes. -/
@[inline] def matmulF16 (cfg : TritonMatmulConfig)
    (a b : RawBuffer) (m k n : Nat) : IO RawBuffer := do
  ensure (a.dtype == .float16) "CudaTritonMatmul: A must be float16"
  ensure (b.dtype == .float16) "CudaTritonMatmul: B must be float16"
  ensure (m == cfg.expectedM) "CudaTritonMatmul: M does not match compiled kernel"
  ensure (n == cfg.expectedN) "CudaTritonMatmul: N does not match compiled kernel"
  ensure (k == cfg.expectedK) "CudaTritonMatmul: K does not match compiled kernel"
  ensure (m % cfg.blockM == 0) "CudaTritonMatmul: M must be multiple of blockM"
  ensure (n % cfg.blockN == 0) "CudaTritonMatmul: N must be multiple of blockN"
  ensure (k % cfg.blockK == 0) "CudaTritonMatmul: K must be multiple of blockK"

  let aBytes := m * k * 2
  let bBytes := k * n * 2
  let cBytes := m * n * 2
  ensure (a.data.size == aBytes) "CudaTritonMatmul: A buffer size mismatch"
  ensure (b.data.size == bBytes) "CudaTritonMatmul: B buffer size mismatch"

  let prog ← loadKernel cfg

  let aBuf ← cudaAllocBytes aBytes
  let bBuf ← cudaAllocBytes bBytes
  let cBuf ← cudaAllocBytes cBytes

  cudaCopyInBytes aBuf a.data
  cudaCopyInBytes bBuf b.data

  let gridX := m / cfg.blockM
  let gridY := n / cfg.blockN
  let blockX := cfg.numWarps * 32
  let blockY := 1

  let args := padKernelArgs #[cBuf, aBuf, bBuf] cfg.paramCount cBuf
  cudaLaunchGrid2D prog args gridX gridY blockX blockY cfg.sharedBytes

  let outBytes ← cudaCopyOutBytes cBuf cBytes

  cudaFree aBuf
  cudaFree bBuf
  cudaFree cBuf

  pure { dtype := .float16, data := outBytes }

/-- Execute Triton matmul (float16) with bias, fixed sizes. -/
@[inline] def matmulF16Bias (cfg : TritonMatmulConfig)
    (a b bias : RawBuffer) (m k n : Nat) : IO RawBuffer := do
  ensure (a.dtype == .float16) "CudaTritonMatmul: A must be float16"
  ensure (b.dtype == .float16) "CudaTritonMatmul: B must be float16"
  ensure (bias.dtype == .float16) "CudaTritonMatmul: bias must be float16"
  ensure (m == cfg.expectedM) "CudaTritonMatmul: M does not match compiled kernel"
  ensure (n == cfg.expectedN) "CudaTritonMatmul: N does not match compiled kernel"
  ensure (k == cfg.expectedK) "CudaTritonMatmul: K does not match compiled kernel"
  ensure (m % cfg.blockM == 0) "CudaTritonMatmul: M must be multiple of blockM"
  ensure (n % cfg.blockN == 0) "CudaTritonMatmul: N must be multiple of blockN"
  ensure (k % cfg.blockK == 0) "CudaTritonMatmul: K must be multiple of blockK"

  let aBytes := m * k * 2
  let bBytes := k * n * 2
  let biasBytes := n * 2
  let cBytes := m * n * 2
  ensure (a.data.size == aBytes) "CudaTritonMatmul: A buffer size mismatch"
  ensure (b.data.size == bBytes) "CudaTritonMatmul: B buffer size mismatch"
  ensure (bias.data.size == biasBytes) "CudaTritonMatmul: bias buffer size mismatch"

  let prog ← loadKernel cfg

  let aBuf ← cudaAllocBytes aBytes
  let bBuf ← cudaAllocBytes bBytes
  let biasBuf ← cudaAllocBytes biasBytes
  let cBuf ← cudaAllocBytes cBytes

  cudaCopyInBytes aBuf a.data
  cudaCopyInBytes bBuf b.data
  cudaCopyInBytes biasBuf bias.data

  let gridX := m / cfg.blockM
  let gridY := n / cfg.blockN
  let blockX := cfg.numWarps * 32
  let blockY := 1

  let args := padKernelArgs #[cBuf, aBuf, bBuf, biasBuf] cfg.paramCount cBuf
  cudaLaunchGrid2D prog args gridX gridY blockX blockY cfg.sharedBytes

  let outBytes ← cudaCopyOutBytes cBuf cBytes

  cudaFree aBuf
  cudaFree bBuf
  cudaFree biasBuf
  cudaFree cBuf

  pure { dtype := .float16, data := outBytes }


/-- Execute Triton matmul (float16) with bias2, fixed sizes. -/
@[inline] def matmulF16Bias2 (cfg : TritonMatmulConfig)
    (a b bias bias2 : RawBuffer) (m k n : Nat) : IO RawBuffer := do
  ensure (a.dtype == .float16) "CudaTritonMatmul: A must be float16"
  ensure (b.dtype == .float16) "CudaTritonMatmul: B must be float16"
  ensure (bias.dtype == .float16) "CudaTritonMatmul: bias must be float16"
  ensure (bias2.dtype == .float16) "CudaTritonMatmul: bias2 must be float16"
  ensure (m == cfg.expectedM) "CudaTritonMatmul: M does not match compiled kernel"
  ensure (n == cfg.expectedN) "CudaTritonMatmul: N does not match compiled kernel"
  ensure (k == cfg.expectedK) "CudaTritonMatmul: K does not match compiled kernel"
  ensure (m % cfg.blockM == 0) "CudaTritonMatmul: M must be multiple of blockM"
  ensure (n % cfg.blockN == 0) "CudaTritonMatmul: N must be multiple of blockN"
  ensure (k % cfg.blockK == 0) "CudaTritonMatmul: K must be multiple of blockK"

  let aBytes := m * k * 2
  let bBytes := k * n * 2
  let biasBytes := n * 2
  let bias2Bytes := n * 2
  let cBytes := m * n * 2
  ensure (a.data.size == aBytes) "CudaTritonMatmul: A buffer size mismatch"
  ensure (b.data.size == bBytes) "CudaTritonMatmul: B buffer size mismatch"
  ensure (bias.data.size == biasBytes) "CudaTritonMatmul: bias buffer size mismatch"
  ensure (bias2.data.size == bias2Bytes) "CudaTritonMatmul: bias2 buffer size mismatch"

  let prog ← loadKernel cfg

  let aBuf ← cudaAllocBytes aBytes
  let bBuf ← cudaAllocBytes bBytes
  let biasBuf ← cudaAllocBytes biasBytes
  let bias2Buf ← cudaAllocBytes bias2Bytes
  let cBuf ← cudaAllocBytes cBytes

  cudaCopyInBytes aBuf a.data
  cudaCopyInBytes bBuf b.data
  cudaCopyInBytes biasBuf bias.data
  cudaCopyInBytes bias2Buf bias2.data

  let gridX := m / cfg.blockM
  let gridY := n / cfg.blockN
  let blockX := cfg.numWarps * 32
  let blockY := 1

  let args := padKernelArgs #[cBuf, aBuf, bBuf, biasBuf, bias2Buf] cfg.paramCount cBuf
  cudaLaunchGrid2D prog args gridX gridY blockX blockY cfg.sharedBytes

  let outBytes ← cudaCopyOutBytes cBuf cBytes

  cudaFree aBuf
  cudaFree bBuf
  cudaFree biasBuf
  cudaFree bias2Buf
  cudaFree cBuf

  pure { dtype := .float16, data := outBytes }

/-- Execute Triton matmul for float32 inputs by converting to float16 and back. -/
@[inline] def matmulF32ViaF16 (cfg : TritonMatmulConfig)
    (a b : RawBuffer) (m k n : Nat) : IO RawBuffer := do
  ensure (a.dtype == .float32) "CudaTritonMatmul: A must be float32"
  ensure (b.dtype == .float32) "CudaTritonMatmul: B must be float32"
  ensure (m == cfg.expectedM) "CudaTritonMatmul: M does not match compiled kernel"
  ensure (n == cfg.expectedN) "CudaTritonMatmul: N does not match compiled kernel"
  ensure (k == cfg.expectedK) "CudaTritonMatmul: K does not match compiled kernel"
  ensure (m % cfg.blockM == 0) "CudaTritonMatmul: M must be multiple of blockM"
  ensure (n % cfg.blockN == 0) "CudaTritonMatmul: N must be multiple of blockN"
  ensure (k % cfg.blockK == 0) "CudaTritonMatmul: K must be multiple of blockK"

  let aBytes := m * k * 4
  let bBytes := k * n * 4
  ensure (a.data.size == aBytes) "CudaTritonMatmul: A buffer size mismatch"
  ensure (b.data.size == bBytes) "CudaTritonMatmul: B buffer size mismatch"

  let aF16 := Native.f32ToF16 a.data
  let bF16 := Native.f32ToF16 b.data
  let aF16Bytes := m * k * 2
  let bF16Bytes := k * n * 2
  ensure (aF16.size == aF16Bytes) "CudaTritonMatmul: A float16 conversion failed"
  ensure (bF16.size == bF16Bytes) "CudaTritonMatmul: B float16 conversion failed"

  let outF16 ← matmulF16 cfg { dtype := .float16, data := aF16 } { dtype := .float16, data := bF16 } m k n
  let outF32 := Native.f16ToF32 outF16.data
  pure { dtype := .float32, data := outF32 }

/-- Execute Triton matmul with bias for float32 inputs by converting to float16 and back. -/
@[inline] def matmulF32ViaF16Bias (cfg : TritonMatmulConfig)
    (a b bias : RawBuffer) (m k n : Nat) : IO RawBuffer := do
  ensure (a.dtype == .float32) "CudaTritonMatmul: A must be float32"
  ensure (b.dtype == .float32) "CudaTritonMatmul: B must be float32"
  ensure (bias.dtype == .float32) "CudaTritonMatmul: bias must be float32"
  ensure (m == cfg.expectedM) "CudaTritonMatmul: M does not match compiled kernel"
  ensure (n == cfg.expectedN) "CudaTritonMatmul: N does not match compiled kernel"
  ensure (k == cfg.expectedK) "CudaTritonMatmul: K does not match compiled kernel"
  ensure (m % cfg.blockM == 0) "CudaTritonMatmul: M must be multiple of blockM"
  ensure (n % cfg.blockN == 0) "CudaTritonMatmul: N must be multiple of blockN"
  ensure (k % cfg.blockK == 0) "CudaTritonMatmul: K must be multiple of blockK"

  let aBytes := m * k * 4
  let bBytes := k * n * 4
  let biasBytes := n * 4
  ensure (a.data.size == aBytes) "CudaTritonMatmul: A buffer size mismatch"
  ensure (b.data.size == bBytes) "CudaTritonMatmul: B buffer size mismatch"
  ensure (bias.data.size == biasBytes) "CudaTritonMatmul: bias buffer size mismatch"

  let aF16 := Native.f32ToF16 a.data
  let bF16 := Native.f32ToF16 b.data
  let biasF16 := Native.f32ToF16 bias.data
  let aF16Bytes := m * k * 2
  let bF16Bytes := k * n * 2
  let biasF16Bytes := n * 2
  ensure (aF16.size == aF16Bytes) "CudaTritonMatmul: A float16 conversion failed"
  ensure (bF16.size == bF16Bytes) "CudaTritonMatmul: B float16 conversion failed"
  ensure (biasF16.size == biasF16Bytes) "CudaTritonMatmul: bias float16 conversion failed"

  let outF16 ← matmulF16Bias cfg
    { dtype := .float16, data := aF16 }
    { dtype := .float16, data := bF16 }
    { dtype := .float16, data := biasF16 } m k n
  let outF32 := Native.f16ToF32 outF16.data
  pure { dtype := .float32, data := outF32 }


/-- Execute Triton matmul with bias2 for float32 inputs by converting to float16 and back. -/
@[inline] def matmulF32ViaF16Bias2 (cfg : TritonMatmulConfig)
    (a b bias bias2 : RawBuffer) (m k n : Nat) : IO RawBuffer := do
  ensure (a.dtype == .float32) "CudaTritonMatmul: A must be float32"
  ensure (b.dtype == .float32) "CudaTritonMatmul: B must be float32"
  ensure (bias.dtype == .float32) "CudaTritonMatmul: bias must be float32"
  ensure (bias2.dtype == .float32) "CudaTritonMatmul: bias2 must be float32"
  ensure (m == cfg.expectedM) "CudaTritonMatmul: M does not match compiled kernel"
  ensure (n == cfg.expectedN) "CudaTritonMatmul: N does not match compiled kernel"
  ensure (k == cfg.expectedK) "CudaTritonMatmul: K does not match compiled kernel"
  ensure (m % cfg.blockM == 0) "CudaTritonMatmul: M must be multiple of blockM"
  ensure (n % cfg.blockN == 0) "CudaTritonMatmul: N must be multiple of blockN"
  ensure (k % cfg.blockK == 0) "CudaTritonMatmul: K must be multiple of blockK"

  let aBytes := m * k * 4
  let bBytes := k * n * 4
  let biasBytes := n * 4
  let bias2Bytes := n * 4
  ensure (a.data.size == aBytes) "CudaTritonMatmul: A buffer size mismatch"
  ensure (b.data.size == bBytes) "CudaTritonMatmul: B buffer size mismatch"
  ensure (bias.data.size == biasBytes) "CudaTritonMatmul: bias buffer size mismatch"
  ensure (bias2.data.size == bias2Bytes) "CudaTritonMatmul: bias2 buffer size mismatch"

  let aF16 := Native.f32ToF16 a.data
  let bF16 := Native.f32ToF16 b.data
  let biasF16 := Native.f32ToF16 bias.data
  let bias2F16 := Native.f32ToF16 bias2.data
  let aF16Bytes := m * k * 2
  let bF16Bytes := k * n * 2
  let biasF16Bytes := n * 2
  let bias2F16Bytes := n * 2
  ensure (aF16.size == aF16Bytes) "CudaTritonMatmul: A float16 conversion failed"
  ensure (bF16.size == bF16Bytes) "CudaTritonMatmul: B float16 conversion failed"
  ensure (biasF16.size == biasF16Bytes) "CudaTritonMatmul: bias float16 conversion failed"
  ensure (bias2F16.size == bias2F16Bytes) "CudaTritonMatmul: bias2 float16 conversion failed"

  let outF16 ← matmulF16Bias2 cfg
    { dtype := .float16, data := aF16 }
    { dtype := .float16, data := bF16 }
    { dtype := .float16, data := biasF16 }
    { dtype := .float16, data := bias2F16 } m k n
  let outF32 := Native.f16ToF32 outF16.data
  pure { dtype := .float32, data := outF32 }

/-- Execute batched Triton matmul for float32 inputs via host loop. -/
@[inline] def matmulBatchedF32ViaF16 (cfg : TritonMatmulConfig)
    (a b : RawBuffer) (m k n : Nat) (aStarts bStarts : Array Nat) : IO RawBuffer := do
  let numBatches := aStarts.size
  ensure (bStarts.size == numBatches) "CudaTritonMatmul: batch sizes mismatch"
  if numBatches == 0 then
    return { dtype := .float32, data := ByteArray.empty }
  let aBytes := m * k * 4
  let bBytes := k * n * 4
  let mut out := ByteArray.emptyWithCapacity (numBatches * m * n * 4)
  for i in [:numBatches] do
    let aStart := aStarts[i]!
    let bStart := bStarts[i]!
    let aSlice := a.data.extract aStart (aStart + aBytes)
    let bSlice := b.data.extract bStart (bStart + bBytes)
    let aBuf : RawBuffer := { dtype := .float32, data := aSlice }
    let bBuf : RawBuffer := { dtype := .float32, data := bSlice }
    let batchOut ← matmulF32ViaF16 cfg aBuf bBuf m k n
    out := out ++ batchOut.data
  pure { dtype := .float32, data := out }

/-- Execute batched Triton matmul with bias for float32 inputs via host loop. -/
@[inline] def matmulBatchedF32ViaF16Bias (cfg : TritonMatmulConfig)
    (a b bias : RawBuffer) (m k n : Nat)
    (aStarts bStarts biasStarts : Array Nat) : IO RawBuffer := do
  let numBatches := aStarts.size
  ensure (bStarts.size == numBatches) "CudaTritonMatmul: batch sizes mismatch"
  ensure (biasStarts.size == numBatches) "CudaTritonMatmul: bias batch sizes mismatch"
  if numBatches == 0 then
    return { dtype := .float32, data := ByteArray.empty }
  let aBytes := m * k * 4
  let bBytes := k * n * 4
  let biasBytes := n * 4
  let mut out := ByteArray.emptyWithCapacity (numBatches * m * n * 4)
  for i in [:numBatches] do
    let aStart := aStarts[i]!
    let bStart := bStarts[i]!
    let biasStart := biasStarts[i]!
    let aSlice := a.data.extract aStart (aStart + aBytes)
    let bSlice := b.data.extract bStart (bStart + bBytes)
    let biasSlice := bias.data.extract biasStart (biasStart + biasBytes)
    let aBuf : RawBuffer := { dtype := .float32, data := aSlice }
    let bBuf : RawBuffer := { dtype := .float32, data := bSlice }
    let biasBuf : RawBuffer := { dtype := .float32, data := biasSlice }
    let batchOut ← matmulF32ViaF16Bias cfg aBuf bBuf biasBuf m k n
    out := out ++ batchOut.data
  pure { dtype := .float32, data := out }

/-- Attempt Triton matmul for float32 inputs if configured and CUDA is available. -/
def tryMatmulF32ViaF16 (a b : RawBuffer) (m k n : Nat) : IO (Option RawBuffer) := do
  if a.dtype != .float32 || b.dtype != .float32 then
    return none
  let available ← TinyGrad4.Backend.Cuda.isAvailable
  if !available then
    return none
  let cfg? ← getConfigFromEnv
  let cfg? ←
    match cfg? with
    | some cfg => pure (some cfg)
    | none =>
      match ← getDefaultPreset with
      | some preset => ensureConfig preset m n k
      | none =>
        match choosePreset .float32 m n k with
        | none => pure none
        | some preset => ensureConfig preset m n k
  match cfg? with
  | none => return none
  | some cfg =>
    if !configMatches cfg m n k none false false then
      return none
    if m % cfg.blockM != 0 || n % cfg.blockN != 0 || k % cfg.blockK != 0 then
      return none
    try
      let out ← matmulF32ViaF16 cfg a b m k n
      return some out
    catch _ =>
      return none

/-- Attempt Triton matmul with bias + optional scale/relu for float32 inputs if CUDA is available. -/
def tryMatmulF32ViaF16BiasScaleRelu (a b bias : RawBuffer) (m k n : Nat)
    (scaleBits : Option UInt32) (relu : Bool) : IO (Option RawBuffer) := do
  if a.dtype != .float32 || b.dtype != .float32 || bias.dtype != .float32 then
    return none
  let available ← TinyGrad4.Backend.Cuda.isAvailable
  if !available then
    return none
  let cfg? ←
    match ← getDefaultPreset with
    | some preset => ensureConfigBias preset m n k scaleBits false relu
    | none =>
      match choosePreset .float32 m n k with
      | none => pure none
      | some preset => ensureConfigBias preset m n k scaleBits false relu
  match cfg? with
  | none => return none
  | some cfg =>
    if !configMatches cfg m n k scaleBits false relu then
      return none
    if m % cfg.blockM != 0 || n % cfg.blockN != 0 || k % cfg.blockK != 0 then
      return none
    try
      let out ← matmulF32ViaF16Bias cfg a b bias m k n
      return some out
    catch _ =>
      return none

/-- Attempt Triton matmul with bias for float32 inputs if CUDA is available. -/
def tryMatmulF32ViaF16Bias (a b bias : RawBuffer) (m k n : Nat) : IO (Option RawBuffer) :=
  tryMatmulF32ViaF16BiasScaleRelu a b bias m k n none false

/-- Attempt Triton matmul with bias2 + optional scale/relu for float32 inputs if CUDA is available. -/
def tryMatmulF32ViaF16Bias2ScaleRelu (a b bias bias2 : RawBuffer) (m k n : Nat)
    (scaleBits : Option UInt32) (relu : Bool) : IO (Option RawBuffer) := do
  if a.dtype != .float32 || b.dtype != .float32 || bias.dtype != .float32 || bias2.dtype != .float32 then
    return none
  let available ← TinyGrad4.Backend.Cuda.isAvailable
  if !available then
    return none
  let cfg? ←
    match ← getDefaultPreset with
    | some preset => ensureConfigBias preset m n k scaleBits true relu
    | none =>
      match choosePreset .float32 m n k with
      | none => pure none
      | some preset => ensureConfigBias preset m n k scaleBits true relu
  match cfg? with
  | none => return none
  | some cfg =>
    if !configMatches cfg m n k scaleBits true relu then
      return none
    if m % cfg.blockM != 0 || n % cfg.blockN != 0 || k % cfg.blockK != 0 then
      return none
    try
      let out ← matmulF32ViaF16Bias2 cfg a b bias bias2 m k n
      return some out
    catch _ =>
      return none

/-- Attempt batched Triton matmul for float32 inputs if configured and CUDA is available. -/
def tryMatmulBatchedF32ViaF16 (a b : RawBuffer) (m k n : Nat)
    (aStarts bStarts : Array Nat) : IO (Option RawBuffer) := do
  if a.dtype != .float32 || b.dtype != .float32 then
    return none
  let available ← TinyGrad4.Backend.Cuda.isAvailable
  if !available then
    return none
  let cfg? ← getConfigFromEnv
  let cfg? ←
    match cfg? with
    | some cfg => pure (some cfg)
    | none =>
      match ← getDefaultPreset with
      | some preset => ensureConfig preset m n k
      | none =>
        match choosePreset .float32 m n k with
        | none => pure none
        | some preset => ensureConfig preset m n k
  match cfg? with
  | none => return none
  | some cfg =>
    if !configMatches cfg m n k none false false then
      return none
    if m != cfg.expectedM || n != cfg.expectedN || k != cfg.expectedK then
      return none
    if m % cfg.blockM != 0 || n % cfg.blockN != 0 || k % cfg.blockK != 0 then
      return none
    try
      let out ← matmulBatchedF32ViaF16 cfg a b m k n aStarts bStarts
      return some out
    catch _ =>
      return none

/-- Attempt batched Triton matmul with bias for float32 inputs if CUDA is available. -/
def tryMatmulBatchedF32ViaF16Bias (a b bias : RawBuffer) (m k n : Nat)
    (aStarts bStarts biasStarts : Array Nat) : IO (Option RawBuffer) := do
  if a.dtype != .float32 || b.dtype != .float32 || bias.dtype != .float32 then
    return none
  let available ← TinyGrad4.Backend.Cuda.isAvailable
  if !available then
    return none
  let cfgEnv? ← getBiasConfigFromEnv
  let cfg? ←
    match cfgEnv? with
    | some cfg => pure (some cfg)
    | none =>
      match ← getDefaultPreset with
      | some preset => ensureConfigBias preset m n k
      | none =>
        match choosePreset .float32 m n k with
        | none => pure none
        | some preset => ensureConfigBias preset m n k
  match cfg? with
  | none => return none
  | some cfg =>
    if m != cfg.expectedM || n != cfg.expectedN || k != cfg.expectedK then
      return none
    if m % cfg.blockM != 0 || n % cfg.blockN != 0 || k % cfg.blockK != 0 then
      return none
    try
      let out ← matmulBatchedF32ViaF16Bias cfg a b bias m k n aStarts bStarts biasStarts
      return some out
    catch _ =>
      return none

end TinyGrad4.Backend.CudaTritonMatmul
