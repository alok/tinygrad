import Float64
import TinyGrad4.Backend.Native

private def pushF32LE (out : ByteArray) (v : Float32) : ByteArray :=
  let bits := v.toBits
  let b0 := (bits &&& 0xFF).toUInt8
  let b1 := ((bits >>> 8) &&& 0xFF).toUInt8
  let b2 := ((bits >>> 16) &&& 0xFF).toUInt8
  let b3 := ((bits >>> 24) &&& 0xFF).toUInt8
  out.push b0 |>.push b1 |>.push b2 |>.push b3

private def packF32 (vals : Array Float32) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity (vals.size * 4)
  for v in vals do
    out := pushF32LE out v
  return out

private def readU32LE (b : ByteArray) (offset : Nat) : UInt32 :=
  let b0 := b.get! offset
  let b1 := b.get! (offset + 1)
  let b2 := b.get! (offset + 2)
  let b3 := b.get! (offset + 3)
  (UInt32.ofNat b0.toNat) |||
    ((UInt32.ofNat b1.toNat) <<< 8) |||
    ((UInt32.ofNat b2.toNat) <<< 16) |||
    ((UInt32.ofNat b3.toNat) <<< 24)

private def readF32At (b : ByteArray) (idx : Nat) : Float64 :=
  let base := idx * 4
  let bits := readU32LE b base
  (Float32.ofBits bits).toFloat

private def approxEq (a b : Float64) (eps : Float64 := 1.0e-2) : Bool :=
  Float64.abs (a - b) <= eps

def main : IO UInt32 := do
  let vals : Array Float32 := #[(1.0 : Float32), (-2.0 : Float32), (0.5 : Float32), (123.0 : Float32)]
  let bytes := packF32 vals
  let f16 := TinyGrad4.Backend.Native.f32ToF16 bytes
  let roundtrip := TinyGrad4.Backend.Native.f16ToF32 f16

  let mut ok := true
  for i in [:vals.size] do
    let orig := vals[i]!.toFloat
    let got := readF32At roundtrip i
    if !approxEq orig got then
      ok := false
      IO.println s!"Mismatch at {i}: {orig} vs {got}"

  if ok then
    IO.println "Float16 cast smoke: ok"
  else
    IO.println "Float16 cast smoke: FAIL"

  return if ok then 0 else 1
