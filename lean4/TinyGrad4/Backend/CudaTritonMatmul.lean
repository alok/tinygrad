import Float64
import TinyGrad4.Backend.Cuda
import TinyGrad4.Backend.Buffer
import TinyGrad4.Backend.Native
import TinyGrad4.Backend.TritonEmit

/-!
# Triton-Generated Matmul (CUDA PTX)

Loads a precompiled PTX kernel (from Triton) and runs a fixed-shape matmul.
This mirrors `extra/gemm/triton_nv_matmul.py` and is intentionally narrow:
- float16 inputs/outputs
- fixed M/N/K and block sizes from the compiled kernel

Environment config (used by `getConfigFromEnv`):
- TG4_TRITON_PTX (path to PTX)
- TG4_TRITON_KERNEL (default: matmul_kernel)
- TG4_TRITON_BLOCK_M / _N / _K
- TG4_TRITON_NUM_WARPS
- TG4_TRITON_SHARED_BYTES (default: 0)
- TG4_TRITON_M / _N / _K (expected shapes)
-/

namespace TinyGrad4.Backend.CudaTritonMatmul

open TinyGrad4
open TinyGrad4.Backend.Cuda

inductive TritonPreset where
  | linearSmall
  | linear
  | linearLarge
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

private def presetTag : TritonPreset → String
  | .linearSmall => "linear_small"
  | .linear => "linear"
  | .linearLarge => "linear_large"

private def defaultPtxPath (preset : TritonPreset) (m n k : Nat) : System.FilePath :=
  System.FilePath.mk "tmp" / s!"triton_{presetTag preset}_{m}x{n}x{k}.ptx"

private def emitConfigForPreset (preset : TritonPreset) (m n k : Nat) : TinyGrad4.Backend.TritonEmit.EmitConfig :=
  let params := presetParams preset
  { ptxPath := defaultPtxPath preset m n k,
    blockM := params.blockM,
    blockN := params.blockN,
    blockK := params.blockK,
    numWarps := params.numWarps,
    numStages := params.numStages,
    m := m,
    n := n,
    k := k }

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
    none

/-- Configuration for a Triton-generated matmul kernel. -/
structure TritonMatmulConfig where
  kernelName : String := "matmul_kernel"
  ptxPath : System.FilePath
  blockM : Nat
  blockN : Nat
  blockK : Nat
  numWarps : Nat
  sharedBytes : Nat
  expectedM : Nat
  expectedN : Nat
  expectedK : Nat

private def ensure (cond : Bool) (msg : String) : IO Unit :=
  if cond then pure () else throw (IO.userError msg)

private def ceilDiv (a b : Nat) : Nat :=
  (a + b - 1) / b

/-- Cache for optional Triton config loaded from environment. -/
initialize tritonConfigCache : IO.Ref (Option (Option TritonMatmulConfig)) ← IO.mkRef none

/-- Optional default Triton preset (enables auto-emit without env). -/
initialize tritonPresetCache : IO.Ref (Option TritonPreset) ← IO.mkRef none

private def envNat? (name : String) : IO (Option Nat) := do
  match ← IO.getEnv name with
  | none => pure none
  | some v =>
    match v.toNat? with
    | some n => pure (some n)
    | none => throw (IO.userError s!"CudaTritonMatmul: {name} must be Nat, got '{v}'")

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
    -/
def loadConfigFromEnv : IO (Option TritonMatmulConfig) := do
  let ptxStr? ← IO.getEnv "TG4_TRITON_PTX"
  let useDefault := ptxStr?.isNone
  let ptxPath? ←
    match ptxStr? with
    | some ptxStr => pure (some (System.FilePath.mk ptxStr))
    | none =>
      let defaultPath := System.FilePath.mk "tmp" / "triton_matmul.ptx"
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
    return some {
      kernelName,
      ptxPath,
      blockM,
      blockN,
      blockK,
      numWarps,
      sharedBytes,
      expectedM,
      expectedN,
      expectedK
    }

/-- Load (and memoize) Triton config from the environment. -/
def getConfigFromEnv : IO (Option TritonMatmulConfig) := do
  match ← tritonConfigCache.get with
  | some cfg => return cfg
  | none =>
    let cfg ← loadConfigFromEnv
    tritonConfigCache.set (some cfg)
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
def ensureConfig (preset : TritonPreset) (m n k : Nat) : IO (Option TritonMatmulConfig) := do
  match ← tritonConfigCache.get with
  | some (some cfg) => return some cfg
  | _ => pure ()

  let envCfg? ← loadConfigFromEnv
  match envCfg? with
  | some cfg =>
    setConfig (some cfg)
    return some cfg
  | none =>
    let params := presetParams preset
    if m % params.blockM != 0 || n % params.blockN != 0 || k % params.blockK != 0 then
      return none
    let emitCfg := emitConfigForPreset preset m n k
    if !(← emitCfg.ptxPath.pathExists) then
      let rc ← TinyGrad4.Backend.TritonEmit.emit emitCfg
      if rc != 0 then
        return none
    if !(← emitCfg.ptxPath.pathExists) then
      return none
    let cfg : TritonMatmulConfig := {
      kernelName := "matmul_kernel",
      ptxPath := emitCfg.ptxPath,
      blockM := params.blockM,
      blockN := params.blockN,
      blockK := params.blockK,
      numWarps := params.numWarps,
      sharedBytes := 0,
      expectedM := m,
      expectedN := n,
      expectedK := k
    }
    setConfig (some cfg)
    return some cfg

/-- Load a PTX module from disk and bind the kernel. -/
private def loadKernel (cfg : TritonMatmulConfig) : IO CUDAProgram := do
  let ptx ← IO.FS.readFile cfg.ptxPath
  cudaLoadPTX cfg.kernelName ptx

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

  cudaLaunchGrid2D prog #[cBuf, aBuf, bBuf] gridX gridY blockX blockY cfg.sharedBytes

  let outBytes ← cudaCopyOutBytes cBuf cBytes

  cudaFree aBuf
  cudaFree bBuf
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

/-- Attempt Triton matmul for float32 inputs if configured and CUDA is available. -/
def tryMatmulF32ViaF16 (a b : RawBuffer) (m k n : Nat) : IO (Option RawBuffer) := do
  if a.dtype != .float32 || b.dtype != .float32 then
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
    let available ← TinyGrad4.Backend.Cuda.isAvailable
    if !available then
      return none
    if m != cfg.expectedM || n != cfg.expectedN || k != cfg.expectedK then
      return none
    if m % cfg.blockM != 0 || n % cfg.blockN != 0 || k % cfg.blockK != 0 then
      return none
    try
      let out ← matmulF32ViaF16 cfg a b m k n
      return some out
    catch _ =>
      return none

end TinyGrad4.Backend.CudaTritonMatmul
