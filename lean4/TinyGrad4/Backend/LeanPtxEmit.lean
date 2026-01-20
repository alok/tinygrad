import Float64

/-!
# Lean PTX Emitter

Emits a minimal PTX matmul kernel from Lean (no Python/Triton runtime).
Variants:
- basic: one output element per block, one thread does the full K loop.
- tiled: register-tiled microtiles per thread.
- smem: shared-memory staging + register-tiled microtiles per thread.
- Supports optional FP16 bias add.
- Uses compile-time constants for M/N/K baked into the PTX.
- Intended as a correctness-first fallback.
-/

namespace TinyGrad4.Backend.LeanPtxEmit

inductive PtxVariant where
  | basic
  | tiled
  | smem
  deriving Repr, DecidableEq

structure EmitConfig where
  ptxPath : System.FilePath
  kernelName : String
  m : Nat
  n : Nat
  k : Nat
  blockM : Nat := 1
  blockN : Nat := 1
  blockK : Nat := 1
  numWarps : Nat := 1
  ptxVersion : Nat
  sm : Nat
  withBias : Bool := false
  variant : PtxVariant := .tiled

def parseVariant (value : String) : Option PtxVariant :=
  match value.toLower with
  | "basic" => some .basic
  | "tiled" => some .tiled
  | "smem" => some .smem
  | _ => none

def variantFromEnv : IO (Option PtxVariant) := do
  match ← IO.getEnv "TG4_TRITON_LEAN_VARIANT" with
  | none => pure none
  | some value =>
    match parseVariant value with
    | some v => pure (some v)
    | none =>
      throw (IO.userError s!"LeanPtxEmit: TG4_TRITON_LEAN_VARIANT must be basic|tiled|smem, got '{value}'")

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

def tileShape (blockM blockN numWarps : Nat) : Option (Nat × Nat) :=
  if blockM == 64 && blockN == 64 && numWarps == 4 then
    some (8, 4)
  else if blockM == 64 && blockN == 128 && numWarps == 4 then
    some (8, 8)
  else if blockM == 128 && blockN == 128 && numWarps == 8 then
    some (8, 8)
  else
    none

private def tileShapeFor (cfg : EmitConfig) : Option (Nat × Nat) :=
  tileShape cfg.blockM cfg.blockN cfg.numWarps

private def fReg (idx : Nat) : String := s!"%f{idx}"
private def rReg (idx : Nat) : String := s!"%r{idx}"
private def rdReg (idx : Nat) : String := s!"%rd{idx}"
private def hReg (idx : Nat) : String := s!"%h{idx}"

private def ptxLinesBasic (cfg : EmitConfig) : List String :=
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

private def ptxLinesTiled (cfg : EmitConfig) (tileM tileN : Nat) : List String := Id.run do
  let tilesPerRow := cfg.blockN / tileN
  let tilesPerCol := cfg.blockM / tileM
  let accCount := tileM * tileN
  let aBase := accCount
  let bBase := accCount + tileM
  let tempBase := bBase + tileN
  let fCount := tempBase + 1
  let rCount := 16
  let rdCount := if cfg.withBias then 7 else 6
  let hCount := if cfg.withBias then 3 else 2
  let accReg := fun i j => fReg (aBase - accCount + i * tileN + j)
  let aReg := fun i => fReg (aBase + i)
  let bReg := fun j => fReg (bBase + j)
  let tempReg := fReg tempBase
  let mut lines : Array String := #[]
  for line in
      [ s!".version {ptxVersionString cfg.ptxVersion}"
      , s!".target sm_{cfg.sm}"
      , ".address_size 64"
      , ""
      , s!".visible .entry {cfg.kernelName}(" ] do
    lines := lines.push line
  for line in paramLines cfg.withBias do
    lines := lines.push line
  for line in
      [ ")"
      , "{"
      , "  .reg .pred %p<2>;"
      , s!"  .reg .b32 %r<{rCount}>;"
      , s!"  .reg .b64 %rd<{rdCount}>;"
      , s!"  .reg .f32 %f<{fCount}>;"
      , s!"  .reg .b16 %h<{hCount}>;"
      , ""
      , "  ld.param.u64 %rd0, [c_ptr];"
      , "  ld.param.u64 %rd1, [a_ptr];"
      , "  ld.param.u64 %rd2, [b_ptr];"
      ] do
    lines := lines.push line
  for line in biasParamLine cfg do
    lines := lines.push line
  for line in
      [ "  mov.u32 %r0, %ctaid.x;"
      , "  mov.u32 %r1, %ctaid.y;"
      , "  mov.u32 %r2, %tid.x;"
      , s!"  mov.u32 %r3, {tilesPerRow};"
      , s!"  mov.u32 %r4, {tilesPerCol};"
      , "  div.u32 %r5, %r2, %r3;"
      , "  rem.u32 %r6, %r2, %r3;"
      , "  setp.ge.u32 %p0, %r5, %r4;"
      , "  @%p0 bra DONE;"
      , s!"  mul.lo.u32 %r7, %r0, {cfg.blockM};"
      , s!"  mul.lo.u32 %r15, %r5, {tileM};"
      , "  add.u32 %r7, %r7, %r15;"
      , s!"  mul.lo.u32 %r8, %r1, {cfg.blockN};"
      , s!"  mul.lo.u32 %r15, %r6, {tileN};"
      , "  add.u32 %r8, %r8, %r15;"
      , ""
      ] do
    lines := lines.push line

  for i in List.range accCount do
    lines := lines.push s!"  mov.f32 {fReg i}, 0f00000000;"

  for line in
      [ "  mov.u32 %r9, 0;"
      , "K_LOOP:"
      , s!"  setp.ge.u32 %p1, %r9, {cfg.k};"
      , "  @%p1 bra K_DONE;"
      , s!"  mul.lo.u32 %r13, %r9, {cfg.n};"
      ] do
    lines := lines.push line

  for i in List.range tileM do
    for line in
        [ s!"  add.u32 %r10, %r7, {i};"
        , s!"  mul.lo.u32 %r12, %r10, {cfg.k};"
        , "  add.u32 %r14, %r12, %r9;"
        , "  mul.wide.u32 %rd4, %r14, 2;"
        , "  add.u64 %rd5, %rd1, %rd4;"
        , "  ld.global.u16 %h0, [%rd5];"
        , s!"  cvt.f32.f16 {aReg i}, %h0;"
        ] do
      lines := lines.push line

  for j in List.range tileN do
    for line in
        [ s!"  add.u32 %r11, %r8, {j};"
        , "  add.u32 %r14, %r13, %r11;"
        , "  mul.wide.u32 %rd4, %r14, 2;"
        , "  add.u64 %rd5, %rd2, %rd4;"
        , "  ld.global.u16 %h1, [%rd5];"
        , s!"  cvt.f32.f16 {bReg j}, %h1;"
        ] do
      lines := lines.push line

  for i in List.range tileM do
    for j in List.range tileN do
      lines := lines.push s!"  fma.rn.f32 {accReg i j}, {aReg i}, {bReg j}, {accReg i j};"

  for line in
      [ "  add.u32 %r9, %r9, 1;"
      , "  bra K_LOOP;"
      , "K_DONE:"
      ] do
    lines := lines.push line

  if cfg.withBias then
    for j in List.range tileN do
      for line in
          [ s!"  add.u32 %r11, %r8, {j};"
          , "  mul.wide.u32 %rd4, %r11, 2;"
          , "  add.u64 %rd5, %rd3, %rd4;"
          , "  ld.global.u16 %h2, [%rd5];"
          , s!"  cvt.f32.f16 {tempReg}, %h2;"
          ] do
        lines := lines.push line
      for i in List.range tileM do
        lines := lines.push s!"  add.f32 {accReg i j}, {accReg i j}, {tempReg};"

  for i in List.range tileM do
    for j in List.range tileN do
      for line in
          [ s!"  add.u32 %r10, %r7, {i};"
          , s!"  add.u32 %r11, %r8, {j};"
          , s!"  mul.lo.u32 %r12, %r10, {cfg.n};"
          , "  add.u32 %r14, %r12, %r11;"
          , "  mul.wide.u32 %rd4, %r14, 2;"
          , "  add.u64 %rd5, %rd0, %rd4;"
          , s!"  cvt.rn.f16.f32 %h0, {accReg i j};"
          , "  st.global.u16 [%rd5], %h0;"
          ] do
        lines := lines.push line

  for line in
      [ "DONE:"
      , "  ret;"
      , "}"
      ] do
    lines := lines.push line
  return lines.toList

private def ptxLinesSmem (cfg : EmitConfig) (tileM tileN : Nat) : List String := Id.run do
  let tilesPerRow := cfg.blockN / tileN
  let tilesPerCol := cfg.blockM / tileM
  let accCount := tileM * tileN
  let aBase := accCount
  let bBase := accCount + tileM
  let tempBase := bBase + tileN
  let fCount := tempBase + 1
  let rCount := 20
  let rdCount := if cfg.withBias then 8 else 7
  let hCount := if cfg.withBias then 3 else 2
  let threads := cfg.numWarps * 32
  let totalA := cfg.blockM * cfg.blockK
  let totalB := cfg.blockK * cfg.blockN
  let accReg := fun i j => fReg (aBase - accCount + i * tileN + j)
  let aReg := fun i => fReg (aBase + i)
  let bReg := fun j => fReg (bBase + j)
  let tempReg := fReg tempBase
  let mut lines : Array String := #[]

  for line in
      [ s!".version {ptxVersionString cfg.ptxVersion}"
      , s!".target sm_{cfg.sm}"
      , ".address_size 64"
      , ""
      , s!".visible .entry {cfg.kernelName}(" ] do
    lines := lines.push line
  for line in paramLines cfg.withBias do
    lines := lines.push line
  for line in
      [ ")"
      , "{"
      , "  .reg .pred %p<2>;"
      , s!"  .reg .b32 %r<{rCount}>;"
      , s!"  .reg .b64 %rd<{rdCount}>;"
      , s!"  .reg .f32 %f<{fCount}>;"
      , s!"  .reg .b16 %h<{hCount}>;"
      , s!"  .shared .align 2 .b16 sA[{totalA}];"
      , s!"  .shared .align 2 .b16 sB[{totalB}];"
      , ""
      , "  ld.param.u64 %rd0, [c_ptr];"
      , "  ld.param.u64 %rd1, [a_ptr];"
      , "  ld.param.u64 %rd2, [b_ptr];"
      ] do
    lines := lines.push line
  for line in biasParamLine cfg do
    lines := lines.push line
  for line in
      [ "  cvta.to.shared.u64 %rd6, sA;"
      , "  cvta.to.shared.u64 %rd7, sB;"
      , "  mov.u32 %r0, %ctaid.x;"
      , "  mov.u32 %r1, %ctaid.y;"
      , "  mov.u32 %r2, %tid.x;"
      , s!"  mov.u32 %r3, {tilesPerRow};"
      , s!"  mov.u32 %r4, {tilesPerCol};"
      , "  div.u32 %r5, %r2, %r3;"
      , "  rem.u32 %r6, %r2, %r3;"
      , "  setp.ge.u32 %p0, %r5, %r4;"
      , "  @%p0 bra DONE;"
      , s!"  mul.lo.u32 %r7, %r0, {cfg.blockM};"
      , s!"  mul.lo.u32 %r8, %r1, {cfg.blockN};"
      , s!"  mul.lo.u32 %r12, %r5, {tileM};"
      , "  add.u32 %r12, %r7, %r12;"
      , s!"  mul.lo.u32 %r13, %r6, {tileN};"
      , "  add.u32 %r13, %r8, %r13;"
      , s!"  mul.lo.u32 %r16, %r5, {tileM * cfg.blockK};"
      , s!"  mul.lo.u32 %r17, %r6, {tileN};"
      , ""
      ] do
    lines := lines.push line

  for i in List.range accCount do
    lines := lines.push s!"  mov.f32 {fReg i}, 0f00000000;"

  for line in
      [ "  mov.u32 %r9, 0;"
      , "K_OUTER:"
      , s!"  setp.ge.u32 %p1, %r9, {cfg.k};"
      , "  @%p1 bra K_DONE;"
      , ""
      , "  mov.u32 %r10, %r2;"
      , "ALOAD_LOOP:"
      , s!"  setp.ge.u32 %p1, %r10, {totalA};"
      , "  @%p1 bra ALOAD_DONE;"
      , s!"  div.u32 %r11, %r10, {cfg.blockK};"
      , s!"  rem.u32 %r14, %r10, {cfg.blockK};"
      , "  add.u32 %r15, %r7, %r11;"
      , "  add.u32 %r18, %r9, %r14;"
      , s!"  mul.lo.u32 %r19, %r15, {cfg.k};"
      , "  add.u32 %r19, %r19, %r18;"
      , "  mul.wide.u32 %rd4, %r19, 2;"
      , "  add.u64 %rd5, %rd1, %rd4;"
      , "  ld.global.u16 %h0, [%rd5];"
      , "  mul.wide.u32 %rd4, %r10, 2;"
      , "  add.u64 %rd5, %rd6, %rd4;"
      , "  st.shared.u16 [%rd5], %h0;"
      , s!"  add.u32 %r10, %r10, {threads};"
      , "  bra ALOAD_LOOP;"
      , "ALOAD_DONE:"
      , ""
      , "  mov.u32 %r10, %r2;"
      , "BLOAD_LOOP:"
      , s!"  setp.ge.u32 %p1, %r10, {totalB};"
      , "  @%p1 bra BLOAD_DONE;"
      , s!"  div.u32 %r11, %r10, {cfg.blockN};"
      , s!"  rem.u32 %r14, %r10, {cfg.blockN};"
      , "  add.u32 %r15, %r9, %r11;"
      , "  add.u32 %r18, %r8, %r14;"
      , s!"  mul.lo.u32 %r19, %r15, {cfg.n};"
      , "  add.u32 %r19, %r19, %r18;"
      , "  mul.wide.u32 %rd4, %r19, 2;"
      , "  add.u64 %rd5, %rd2, %rd4;"
      , "  ld.global.u16 %h1, [%rd5];"
      , "  mul.wide.u32 %rd4, %r10, 2;"
      , "  add.u64 %rd5, %rd7, %rd4;"
      , "  st.shared.u16 [%rd5], %h1;"
      , s!"  add.u32 %r10, %r10, {threads};"
      , "  bra BLOAD_LOOP;"
      , "BLOAD_DONE:"
      , "  bar.sync 0;"
      ] do
    lines := lines.push line

  for line in
      [ "  mov.u32 %r10, 0;"
      , "K_INNER:"
      , s!"  setp.ge.u32 %p1, %r10, {cfg.blockK};"
      , "  @%p1 bra K_INNER_DONE;"
      ] do
    lines := lines.push line

  for i in List.range tileM do
    for line in
        [ s!"  add.u32 %r11, %r16, {i * cfg.blockK};"
        , "  add.u32 %r11, %r11, %r10;"
        , "  mul.wide.u32 %rd4, %r11, 2;"
        , "  add.u64 %rd5, %rd6, %rd4;"
        , "  ld.shared.u16 %h0, [%rd5];"
        , s!"  cvt.f32.f16 {aReg i}, %h0;"
        ] do
      lines := lines.push line

  for j in List.range tileN do
    for line in
        [ s!"  mul.lo.u32 %r14, %r10, {cfg.blockN};"
        , "  add.u32 %r14, %r14, %r17;"
        , s!"  add.u32 %r14, %r14, {j};"
        , "  mul.wide.u32 %rd4, %r14, 2;"
        , "  add.u64 %rd5, %rd7, %rd4;"
        , "  ld.shared.u16 %h1, [%rd5];"
        , s!"  cvt.f32.f16 {bReg j}, %h1;"
        ] do
      lines := lines.push line

  for i in List.range tileM do
    for j in List.range tileN do
      lines := lines.push s!"  fma.rn.f32 {accReg i j}, {aReg i}, {bReg j}, {accReg i j};"

  for line in
      [ "  add.u32 %r10, %r10, 1;"
      , "  bra K_INNER;"
      , "K_INNER_DONE:"
      , "  bar.sync 0;"
      , s!"  add.u32 %r9, %r9, {cfg.blockK};"
      , "  bra K_OUTER;"
      , "K_DONE:"
      ] do
    lines := lines.push line

  if cfg.withBias then
    for j in List.range tileN do
      for line in
          [ s!"  add.u32 %r11, %r13, {j};"
          , "  mul.wide.u32 %rd4, %r11, 2;"
          , "  add.u64 %rd5, %rd3, %rd4;"
          , "  ld.global.u16 %h2, [%rd5];"
          , s!"  cvt.f32.f16 {tempReg}, %h2;"
          ] do
        lines := lines.push line
      for i in List.range tileM do
        lines := lines.push s!"  add.f32 {accReg i j}, {accReg i j}, {tempReg};"

  for i in List.range tileM do
    for j in List.range tileN do
      for line in
          [ s!"  add.u32 %r11, %r12, {i};"
          , s!"  add.u32 %r14, %r13, {j};"
          , s!"  mul.lo.u32 %r15, %r11, {cfg.n};"
          , "  add.u32 %r15, %r15, %r14;"
          , "  mul.wide.u32 %rd4, %r15, 2;"
          , "  add.u64 %rd5, %rd0, %rd4;"
          , s!"  cvt.rn.f16.f32 %h0, {accReg i j};"
          , "  st.global.u16 [%rd5], %h0;"
          ] do
        lines := lines.push line

  for line in
      [ "DONE:"
      , "  ret;"
      , "}"
      ] do
    lines := lines.push line
  return lines.toList

private def ptxLines (cfg : EmitConfig) : List String :=
  match cfg.variant, tileShapeFor cfg with
  | .basic, _ => ptxLinesBasic cfg
  | .tiled, some (tileM, tileN) => ptxLinesTiled cfg tileM tileN
  | .smem, some (tileM, tileN) => ptxLinesSmem cfg tileM tileN
  | _, _ => ptxLinesBasic cfg

def ptxSource (cfg : EmitConfig) : String :=
  String.intercalate "\n" (ptxLines cfg)

/-- Write a debug dump of the PTX source. Defaults to `{ptxPath}.dump`. -/
def dump (cfg : EmitConfig) (path? : Option System.FilePath := none) : IO Unit := do
  let outPath :=
    match path? with
    | some p => p
    | none => System.FilePath.mk (cfg.ptxPath.toString ++ ".dump")
  IO.FS.writeFile outPath (ptxSource cfg)

/-- Emit PTX to the configured path. -/
def emit (cfg : EmitConfig) : IO UInt32 := do
  IO.FS.createDirAll (cfg.ptxPath.parent.getD (System.FilePath.mk "."))
  if (← cfg.ptxPath.pathExists) && !(← envFlag "TG4_TRITON_FORCE") then
    return 0
  let cfg ←
    match ← variantFromEnv with
    | some variant => pure { cfg with variant }
    | none => pure cfg
  IO.FS.writeFile cfg.ptxPath (ptxSource cfg)
  if (← envFlag "TG4_TRITON_DUMP") then
    dump cfg none
  return 0

end TinyGrad4.Backend.LeanPtxEmit
