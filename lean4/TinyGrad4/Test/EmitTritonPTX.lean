import Float64

/-!
# Emit Triton PTX

Generates a PTX file for a fixed-shape Triton matmul kernel using `uv`.
- Default output: tmp/triton_matmul.ptx
- Override with TG4_TRITON_PTX
- Block sizes via TG4_TRITON_BLOCK_M/_N/_K
- Warp/stage via TG4_TRITON_NUM_WARPS, TG4_TRITON_NUM_STAGES
- Shapes via TG4_TRITON_M/_N/_K (compile-time constants in the kernel)
-/

namespace TinyGrad4.Test.EmitTritonPTX

structure EmitConfig where
  ptxPath : String := "tmp/triton_matmul.ptx"
  blockM : Nat := 64
  blockN : Nat := 64
  blockK : Nat := 32
  numWarps : Nat := 4
  numStages : Nat := 2
  m : Nat := 256
  n : Nat := 256
  k : Nat := 256

private def envNat (name : String) (default : Nat) : IO Nat := do
  match ← IO.getEnv name with
  | none => pure default
  | some v =>
    match v.toNat? with
    | some n => pure n
    | none => throw (IO.userError s!"EmitTritonPTX: {name} must be Nat, got '{v}'")

private def findUv : IO String := do
  let out ← IO.Process.output { cmd := "which", args := #["uv"] }
  if out.exitCode == 0 then
    return "uv"
  let home := (← IO.getEnv "HOME").getD ""
  let fallback := home ++ "/.local/bin/uv"
  if ← System.FilePath.pathExists fallback then
    return fallback
  throw (IO.userError "EmitTritonPTX: uv not found in PATH or ~/.local/bin")

private def pythonSource (ptxPath : String) (m n k : Nat) (blockM blockN blockK numWarps numStages : Nat) : String :=
  String.intercalate "\n" [
    "import os",
    "import triton",
    "import triton.language as tl",
    "from triton.compiler import ASTSource, compile as triton_compile",
    "",
    "@triton.jit",
    "def matmul_kernel(c_ptr, a_ptr, b_ptr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):",
    s!"  M, N, K = {m}, {n}, {k}",
    "  stride_am = K",
    "  stride_ak = 1",
    "  stride_bk = N",
    "  stride_bn = 1",
    "  stride_cm = N",
    "  stride_cn = 1",
    "",
    "  pid_m = tl.program_id(axis=0)",
    "  pid_n = tl.program_id(axis=1)",
    "  offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M",
    "  offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N",
    "  offs_k = tl.arange(0, BLOCK_SIZE_K)",
    "",
    "  a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)",
    "  b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)",
    "",
    "  acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)",
    "  for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):",
    "    a = tl.load(a_ptrs)",
    "    b = tl.load(b_ptrs)",
    "    acc = tl.dot(a, b, acc)",
    "    a_ptrs += BLOCK_SIZE_K * stride_ak",
    "    b_ptrs += BLOCK_SIZE_K * stride_bk",
    "",
    "  c = tl.cast(acc, tl.float16)",
    "  offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)",
    "  offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)",
    "  c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]",
    "  tl.store(c_ptrs, c)",
    "",
    "",
    s!"def emit_ptx(path, block_m={blockM}, block_n={blockN}, block_k={blockK}, num_warps={numWarps}, num_stages={numStages}):",
    "  signature = {\"c_ptr\": \"*fp16\", \"a_ptr\": \"*fp16\", \"b_ptr\": \"*fp16\"}",
    "  src = ASTSource(",
    "    matmul_kernel,",
    "    signature,",
    "    constexprs={\"BLOCK_SIZE_M\": block_m, \"BLOCK_SIZE_N\": block_n, \"BLOCK_SIZE_K\": block_k},",
    "  )",
    "  try:",
    "    compiled = triton_compile(src, options={\"num_warps\": num_warps, \"num_stages\": num_stages})",
    "  except TypeError:",
    "    compiled = triton_compile(src)",
    "  asm = compiled.asm.get(\"ptx\")",
    "  if asm is None:",
    "    raise RuntimeError(\"PTX not found in compiled asm\")",
    "  asm = asm.replace(\".extern .shared .align 16 .b8 global_smem[];\",",
    "                    f\".shared .align 16 .b8 global_smem[{compiled.metadata.shared}];\")",
    "  asm = asm.replace(\"\\t// begin inline asm\\n\", \"\")",
    "  asm = asm.replace(\"\\t// end inline asm\\n\", \"\")",
    "  asm = asm.split(\"\\t.file\")[0]",
    "  os.makedirs(os.path.dirname(path), exist_ok=True)",
    "  with open(path, \"w\") as f:",
    "    f.write(asm)",
    "  print(f\"shared={compiled.metadata.shared} num_warps={compiled.metadata.num_warps}\")",
    "",
    "",
    "if __name__ == \"__main__\":",
    s!"  emit_ptx(\"{ptxPath}\")"
  ]

/-- Emit Triton PTX for an explicit config. -/
def emit (cfg : EmitConfig) : IO UInt32 := do
  let scriptPath := System.FilePath.mk "tmp" / "emit_triton_ptx.py"
  IO.FS.createDirAll (scriptPath.parent.getD (System.FilePath.mk "."))
  let source :=
    pythonSource cfg.ptxPath cfg.m cfg.n cfg.k
      cfg.blockM cfg.blockN cfg.blockK cfg.numWarps cfg.numStages
  IO.FS.writeFile scriptPath source

  let uv ← findUv
  let args := #[
    "run", "--no-project",
    "--with", "triton",
    "--with", "torch",
    "--index", "https://download.pytorch.org/whl/cu121",
    "python3", scriptPath.toString
  ]
  let out ← IO.Process.output { cmd := uv, args := args }
  if out.exitCode != 0 then
    IO.eprintln out.stdout
    IO.eprintln out.stderr
    return 1

  IO.println s!"Wrote PTX to {cfg.ptxPath}"
  return 0

/-- Emit Triton PTX using uv + python. -/
def emitMain : IO UInt32 := do
  let ptxPath := (← IO.getEnv "TG4_TRITON_PTX").getD "tmp/triton_matmul.ptx"
  let blockM ← envNat "TG4_TRITON_BLOCK_M" 64
  let blockN ← envNat "TG4_TRITON_BLOCK_N" 64
  let blockK ← envNat "TG4_TRITON_BLOCK_K" 32
  let numWarps ← envNat "TG4_TRITON_NUM_WARPS" 4
  let numStages ← envNat "TG4_TRITON_NUM_STAGES" 2
  let m ← envNat "TG4_TRITON_M" 256
  let n ← envNat "TG4_TRITON_N" 256
  let k ← envNat "TG4_TRITON_K" 256
  emit {
    ptxPath,
    blockM,
    blockN,
    blockK,
    numWarps,
    numStages,
    m,
    n,
    k
  }

private def envFlag (name : String) : IO Bool := do
  match ← IO.getEnv name with
  | none => pure false
  | some v =>
    let v := v.toLower
    pure (v == "1" || v == "true" || v == "yes")

/-- Auto-generate PTX if TG4_TRITON_AUTOGEN is set and the PTX is missing. -/
def autogenIfNeeded : IO Unit := do
  if !(← envFlag "TG4_TRITON_AUTOGEN") then
    return
  let ptxPath := (← IO.getEnv "TG4_TRITON_PTX").getD "tmp/triton_matmul.ptx"
  let path := System.FilePath.mk ptxPath
  if ← path.pathExists then
    return
  let rc ← emitMain
  if rc != 0 then
    throw (IO.userError "EmitTritonPTX: autogen failed")

end TinyGrad4.Test.EmitTritonPTX
