import Std.Data.HashMap
import TinyGrad4.Backend.Buffer
import TinyGrad4.UOp.Typed
import TinyGrad4.Data.ArrayN
import TinyGrad4.Data.GPULoader
import TinyGrad4.Data.TPULoader

namespace TinyGrad4.Data

open TinyGrad4

/-- Environment map for feeding UOp buffer inputs. -/
abbrev UOpEnv := Std.HashMap UOpId RawBuffer

/-- Typed UOp buffer input with device annotation. -/
structure TUOpInput (shape : Shape) (dtype : DType) where
  node : TUOp .BUFFER shape (Shape.rank shape) dtype
  device : String
  deriving Repr

namespace TUOpInput

/-- Create a typed buffer input node. -/
def create (dtype : DType) (shape : Shape) (device : String := "CPU") : TUOpM (TUOpInput shape dtype) := do
  let u ← TUOp.buffer dtype shape device
  pure { node := u, device }

/-- Build a fresh env with this buffer filled by a RawBuffer. -/
def envOfRaw (input : TUOpInput shape dtype) (buf : RawBuffer) : UOpEnv :=
  ({} : UOpEnv).insert input.node.raw.uid buf

/-- Insert a RawBuffer into an existing env for this input. -/
def insertRaw (env : UOpEnv) (input : TUOpInput shape dtype) (buf : RawBuffer) : UOpEnv :=
  env.insert input.node.raw.uid buf

/-- Build env from a typed DataArrayN. -/
def envOfArrayN (input : TUOpInput shape dtype) (arr : DataArrayN shape dtype) : UOpEnv :=
  envOfRaw input arr.toRawBuffer

/-- Insert a typed DataArrayN into an existing env. -/
def insertArrayN (env : UOpEnv) (input : TUOpInput shape dtype) (arr : DataArrayN shape dtype) : UOpEnv :=
  insertRaw env input arr.toRawBuffer

/-- Build env from a TPU buffer (host-staged bytes). -/
def envOfTPU (input : TUOpInput shape dtype) (buf : TPULoader.TPUBuffer shape dtype) : UOpEnv :=
  envOfRaw input { dtype := dtype, data := buf.data }

/-- Insert a TPU buffer into an existing env. -/
def insertTPU (env : UOpEnv) (input : TUOpInput shape dtype) (buf : TPULoader.TPUBuffer shape dtype) : UOpEnv :=
  insertRaw env input { dtype := dtype, data := buf.data }

/-- Build env from a GPU lease by copying data back to host. -/
def envOfGPULease (input : TUOpInput shape dtype)
    (lease : GPULoader.GPULease shape dtype) : IO UOpEnv := do
  let bytes ← lease.value.copyOut
  pure (envOfRaw input { dtype := dtype, data := bytes })

/-- Insert a GPU lease into an existing env by copying data back to host. -/
def insertGPULease (env : UOpEnv) (input : TUOpInput shape dtype)
    (lease : GPULoader.GPULease shape dtype) : IO UOpEnv := do
  let bytes ← lease.value.copyOut
  pure (insertRaw env input { dtype := dtype, data := bytes })

end TUOpInput

end TinyGrad4.Data
