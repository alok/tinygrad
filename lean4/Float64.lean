import Lean

abbrev Float64 := Float

namespace Float64

abbrev ofNat := Float.ofNat
abbrev ofInt := Float.ofInt
abbrev ofScientific := Float.ofScientific
abbrev toFloat32 (x : Float64) : Float32 := Float.toFloat32 x
abbrev toInt64 (x : Float64) : Int64 := Float.toInt64 x
abbrev toUInt32 (x : Float64) : UInt32 := Float.toUInt32 x
abbrev abs (x : Float64) : Float64 := Float.abs x
abbrev sqrt (x : Float64) : Float64 := Float.sqrt x
abbrev log (x : Float64) : Float64 := Float.log x
abbrev exp (x : Float64) : Float64 := Float.exp x
abbrev sin (x : Float64) : Float64 := Float.sin x
abbrev cos (x : Float64) : Float64 := Float.cos x
abbrev tan (x : Float64) : Float64 := Float.tan x
abbrev round (x : Float64) : Float64 := Float.round x

end Float64
