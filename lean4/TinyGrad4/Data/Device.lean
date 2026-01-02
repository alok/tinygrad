/-!
# Device - Memory Location Abstraction

Represents where a buffer resides: CPU, GPU, NVMe, or remote storage.
Inspired by DLPack's device model and JAX's device mesh.

## Usage
```lean
let dev := Device.gpu 0        -- GPU device 0
let remote := Device.remote "s3://bucket/path"
```
-/

namespace TinyGrad4.Data

/-- Device identifier representing where a buffer resides.
    Follows DLPack device model with extensions for distributed training. -/
inductive Device where
  | cpu                          -- CPU memory (host)
  | gpu (deviceId : Nat := 0)    -- GPU memory (CUDA, Metal, etc.)
  | tpu (deviceId : Nat := 0)    -- TPU memory (XLA/PJRT-backed)
  | nvme (path : String)         -- NVMe/SSD storage (for ZeRO-Infinity style offloading)
  | remote (url : String)        -- Remote storage (S3, GCS, HTTP)
  deriving BEq, Hashable, Repr, Inhabited

namespace Device

/-- Check if device is host-accessible (CPU or NVMe) -/
def isHost : Device → Bool
  | .cpu => true
  | .nvme _ => true
  | _ => false

/-- Check if device is GPU -/
def isGPU : Device → Bool
  | .gpu _ => true
  | _ => false

/-- Check if device is TPU -/
def isTPU : Device → Bool
  | .tpu _ => true
  | _ => false

/-- Check if device is remote (requires network I/O) -/
def isRemote : Device → Bool
  | .remote _ => true
  | _ => false

/-- Get device name for display -/
def name : Device → String
  | .cpu => "cpu"
  | .gpu id => s!"gpu:{id}"
  | .tpu id => s!"tpu:{id}"
  | .nvme path => s!"nvme:{path}"
  | .remote url => s!"remote:{url}"

instance : ToString Device where
  toString := name

/-- Default CPU device -/
def default : Device := .cpu

/-- Default GPU device (device 0) -/
def defaultGPU : Device := .gpu 0

end Device

/-- Device type for DLPack compatibility (values match DLPack spec) -/
inductive DLDeviceType where
  | cpu
  | cuda
  | cudaHost
  | openCL
  | vulkan
  | metal
  | vpi
  | rocm
  | oneAPI
  deriving BEq, Repr

namespace DLDeviceType

/-- Get DLPack device type ID (for FFI) -/
def toInt : DLDeviceType → Nat
  | .cpu => 1
  | .cuda => 2
  | .cudaHost => 3
  | .openCL => 4
  | .vulkan => 7
  | .metal => 8
  | .vpi => 9
  | .rocm => 10
  | .oneAPI => 14

end DLDeviceType

/-- Convert Device to DLPack device type (for FFI) -/
def Device.toDLDeviceType : Device → DLDeviceType
  | .cpu => .cpu
  | .gpu _ => .metal  -- On macOS; would be .cuda on NVIDIA
  | .tpu _ => .cpu    -- TPU buffers are staged via host before device transfer
  | .nvme _ => .cpu   -- NVMe is accessed via CPU
  | .remote _ => .cpu -- Remote fetched to CPU first

end TinyGrad4.Data
