import TinyGrad4.Data.Dataset
import TinyGrad4.Data.TPULoader
import TinyGrad4.Data.GPULoader

/-! Simple smoke test for TPULoader. -/

open TinyGrad4.Data
open TinyGrad4.Data.TPULoader
open TinyGrad4.Data.GPULoader

def assert (cond : Bool) (msg : String) : IO Unit := do
  if !cond then
    throw (IO.userError s!"Assertion failed: {msg}")

def main : IO Unit := do
  IO.println "Starting TPU loader smoke test..."

  let device : DeviceId := .tpu 0
  let sample : Array ByteArray := #[ByteArray.mk (Array.replicate 8 7)]
  let ds := ofArray sample

  let loader ← TPUDataLoader.create ds device (batchSize := 1) (itemShape := [8]) (bufferSize := 2) (dtype := .uint8)
  match ← loader.next with
  | some buf => assert (buf.device == device) "TPU buffer on wrong device"
  | none => throw (IO.userError "TPUDataLoader produced no batches")

  loader.stop
  IO.println "✓ TPU loader smoke test passed"
