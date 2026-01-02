import TinyGrad4.Data.Dataset
import TinyGrad4.Data.GPULoader

/-! Simple smoke test for GPULoader -/

open TinyGrad4.Data.GPULoader

def assert (cond : Bool) (msg : String) : IO Unit := do
  if !cond then
    throw (IO.userError s!"Assertion failed: {msg}")

def main : IO Unit := do
  IO.println "Starting GPU loader smoke test..."

  -- Test device discovery
  IO.println "Discovering devices..."
  let devices ← discoverDevices
  IO.println s!"Found {devices.size} GPU device(s)"

  if devices.isEmpty then
    IO.println "No GPU devices - skipping GPU tests"
    return

  for dev in devices do
    let name ← dev.name
    IO.println s!"  - {name} ({dev}, {repr dev})"

  IO.println "Testing buffer allocation..."
  let target := devices[0]!
  IO.println s!"  Allocating on {target} ({repr target})"
  let buf ← try
    GPUBuffer.alloc target [1024] .uint8
  catch e =>
    IO.println s!"  ✗ Allocation failed on {target}: {e}"
    throw e
  IO.println s!"Allocated {buf.bytes} bytes"

  let testData := ByteArray.mk (Array.replicate 1024 42)
  buf.copyIn testData
  IO.println "Copied data in"

  buf.free
  IO.println "Buffer freed"

  IO.println "Testing GPUDataLoader device placement..."
  let sample : Array ByteArray := #[ByteArray.mk (Array.replicate 16 1), ByteArray.mk (Array.replicate 16 2)]
  let ds := TinyGrad4.Data.ofArray sample
  let loader ← GPUDataLoader.create ds devices[0]! (batchSize := 1) (itemShape := [16])
    (dtype := .uint8) (bufferSize := 2)
  for _ in [:2] do
    match ← loader.next with
    | some batch =>
        assert (batch.value.device == devices[0]!) "Batch on wrong device"
        batch.release
    | none => break
  loader.stop
  IO.println "GPUDataLoader device placement ok"

  if devices.size >= 2 then
    IO.println "Testing MultiGPULoader device placement..."
    let pool ← MultiGPULoader.create ds devices (batchSize := 1) (itemShape := [16])
      (dtype := .uint8) (bufferSize := 2)
    let batches ← pool.nextAll
    for (device, buf?) in batches do
      match buf? with
      | some buf =>
          assert (buf.value.device == device) "MultiGPULoader batch on wrong device"
          buf.release
      | none => pure ()
    pool.stop
    IO.println "MultiGPULoader device placement ok"

  IO.println "✓ GPU loader smoke test passed"
