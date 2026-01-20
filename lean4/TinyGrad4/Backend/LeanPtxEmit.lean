import Float64

/-!
# Lean PTX Emitter

Emits a minimal PTX matmul kernel from Lean (no Python/Triton runtime).
The kernel is intentionally simple: one output element per block, one thread
per block (thread 0) does the full K loop in FP32 and stores FP16.
- Supports optional FP16 bias add.
- Uses compile-time constants for M/N/K baked into the PTX.
- Intended as a correctness-first fallback.
-/

namespace TinyGrad4.Backend.LeanPtxEmit

structure EmitConfig where
  ptxPath : System.FilePath
  kernelName : String
  m : Nat
  n : Nat
  k : Nat
  ptxVersion : Nat
  sm : Nat
  withBias : Bool := false

private def envFlag (name : String) : IO Bool := do
  match ← IO.getEnv name with
  | none => pure false
  | some v =>
    let v := v.toLower
    pure (v == "1" || v == "true" || v == "yes")

private def ptxVersionString (v : Nat) : String :=
  let major := v / 10
  let minor := v % 10
  s!"{major}.{minor}"

private def paramLines (withBias : Bool) : List String :=
  let params : List String :=
    if withBias then
      ["c_ptr", "a_ptr", "b_ptr", "bias_ptr"]
    else
      ["c_ptr", "a_ptr", "b_ptr"]
  let len := params.length
  let rec go (ps : List String) (idx : Nat) : List String :=
    match ps with
    | [] => []
    | name :: rest =>
      let suffix := if idx + 1 < len then "," else ""
      s!"  .param .u64 {name}{suffix}" :: go rest (idx + 1)
  go params 0

private def biasParamLine (cfg : EmitConfig) : List String :=
  if cfg.withBias then
    ["  ld.param.u64 %rd3, [bias_ptr];"]
  else
    []

private def biasBodyLines (cfg : EmitConfig) : List String :=
  if cfg.withBias then
    [ s!"  mul.wide.u32 %rd8, %r1, 2"
    , "  add.u64 %rd9, %rd3, %rd8"
    , "  ld.global.u16 %h2, [%rd9]"
    , "  cvt.f32.f16 %f1, %h2"
    , "  add.f32 %f0, %f0, %f1"
    ]
  else
    []

private def ptxLines (cfg : EmitConfig) : List String :=
  [ s!".version {ptxVersionString cfg.ptxVersion}"
  , s!".target sm_{cfg.sm}"
  , ".address_size 64"
  , ""
  , s!".visible .entry {cfg.kernelName}(" ]
  ++ paramLines cfg.withBias ++
  [ ")"
  , "{"
  , "  .reg .pred %p<2>;"
  , "  .reg .b32 %r<10>;"
  , "  .reg .b64 %rd<10>;"
  , "  .reg .f32 %f<3>;"
  , "  .reg .b16 %h<3>;"
  , ""
  , "  ld.param.u64 %rd0, [c_ptr];"
  , "  ld.param.u64 %rd1, [a_ptr];"
  , "  ld.param.u64 %rd2, [b_ptr];"
  ]
  ++ biasParamLine cfg ++
  [ "  mov.u32 %r0, %ctaid.x;"
  , "  mov.u32 %r1, %ctaid.y;"
  , "  mov.u32 %r2, %tid.x;"
  , "  setp.ne.u32 %p0, %r2, 0;"
  , "  @%p0 bra DONE;"
  , ""
  , "  mov.f32 %f0, 0f00000000;"
  , "  mov.u32 %r3, 0;"
  , "LOOP:"
  , s!"  setp.ge.u32 %p1, %r3, {cfg.k};"
  , "  @%p1 bra LOOP_END;"
  , s!"  mul.lo.u32 %r4, %r0, {cfg.k};"
  , "  add.u32 %r5, %r4, %r3;"
  , s!"  mul.lo.u32 %r6, %r3, {cfg.n};"
  , "  add.u32 %r7, %r6, %r1;"
  , "  mul.wide.u32 %rd4, %r5, 2;"
  , "  add.u64 %rd5, %rd1, %rd4;"
  , "  ld.global.u16 %h0, [%rd5];"
  , "  cvt.f32.f16 %f1, %h0;"
  , "  mul.wide.u32 %rd6, %r7, 2;"
  , "  add.u64 %rd7, %rd2, %rd6;"
  , "  ld.global.u16 %h1, [%rd7];"
  , "  cvt.f32.f16 %f2, %h1;"
  , "  fma.rn.f32 %f0, %f1, %f2, %f0;"
  , "  add.u32 %r3, %r3, 1;"
  , "  bra LOOP;"
  , "LOOP_END:"
  ]
  ++ biasBodyLines cfg ++
  [ s!"  mul.lo.u32 %r8, %r0, {cfg.n};"
  , "  add.u32 %r9, %r8, %r1;"
  , "  mul.wide.u32 %rd4, %r9, 2;"
  , "  add.u64 %rd5, %rd0, %rd4;"
  , "  cvt.rn.f16.f32 %h0, %f0;"
  , "  st.global.u16 [%rd5], %h0;"
  , "DONE:"
  , "  ret;"
  , "}"
  ]

private def ptxSource (cfg : EmitConfig) : String :=
  String.intercalate "\n" (ptxLines cfg)

/-- Emit PTX to the configured path. -/
def emit (cfg : EmitConfig) : IO UInt32 := do
  IO.FS.createDirAll (cfg.ptxPath.parent.getD (System.FilePath.mk "."))
  if (← cfg.ptxPath.pathExists) && !(← envFlag "TG4_TRITON_FORCE") then
    return 0
  IO.FS.writeFile cfg.ptxPath (ptxSource cfg)
  return 0

end TinyGrad4.Backend.LeanPtxEmit
