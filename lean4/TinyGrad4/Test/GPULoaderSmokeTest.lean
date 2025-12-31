import TinyGrad4.Data.GPULoader

/-! Simple smoke test for GPULoader -/

open TinyGrad4.Data.GPULoader

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
    IO.println s!"  - {name}"

  IO.println "Testing buffer allocation..."
  let buf ← GPUBuffer.alloc .metal 1024 .uint8
  IO.println s!"Allocated {buf.byteSize} bytes"

  let testData := ByteArray.mk (Array.replicate 1024 42)
  buf.copyIn testData
  IO.println "Copied data in"

  buf.free
  IO.println "Buffer freed"

  IO.println "✓ GPU loader smoke test passed"
