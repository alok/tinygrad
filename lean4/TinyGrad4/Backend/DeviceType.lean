import Std

namespace TinyGrad4.Backend

inductive DeviceType where
  | CPU
  | METAL
  | CUDA
  deriving Repr, DecidableEq, Inhabited

/-- Canonical device name used in UOp buffers and tags. -/
def deviceTypeName : DeviceType → String
  | .CPU => "CPU"
  | .METAL => "METAL"
  | .CUDA => "CUDA"

/-- Parse device string like "CPU", "METAL", "CUDA:0" -/
def parseDeviceType (s : String) : DeviceType :=
  if s.startsWith "METAL" then .METAL
  else if s.startsWith "CUDA" then .CUDA
  else .CPU

end TinyGrad4.Backend
