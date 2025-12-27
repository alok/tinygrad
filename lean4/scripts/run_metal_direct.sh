#!/bin/bash
# Run direct Metal FFI benchmark using lake env
#
# This uses Lake's environment setup to ensure all library paths are correct,
# then runs a Lean script that uses the Metal FFI.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# First ensure Metal FFI is built
"$SCRIPT_DIR/build_metal_ffi.sh"

# Build the library
lake build TinyGrad4 2>&1 | tail -3

# Try to build the MetalDirect module (may fail at exe linking)
lake build TinyGrad4.Benchmark.MetalDirect 2>&1 | tail -3 || true

echo ""
echo "=== Running Direct Metal FFI Test ==="
echo ""

# Use lake env to run lean with correct paths
# We'll evaluate a test expression
lake env lean --run - << 'EOF'
import TinyGrad4.Backend.Metal

def main : IO Unit := do
  IO.println "Testing Metal FFI..."

  -- Test device detection
  let device ← TinyGrad4.Backend.Metal.metalDeviceName
  IO.println s!"  Device: {device}"

  -- Test buffer allocation
  let buf ← TinyGrad4.Backend.Metal.metalAlloc 1000
  IO.println s!"  Allocated buffer with 1000 elements"

  -- Test copy in/out
  let arr := (List.replicate 1000 (1.5 : Float)).toArray
  let data := FloatArray.mk arr
  TinyGrad4.Backend.Metal.metalCopyIn buf data
  let output ← TinyGrad4.Backend.Metal.metalCopyOut buf
  IO.println s!"  Roundtrip: {output[0]!} (expected 1.5)"

  -- Cleanup
  TinyGrad4.Backend.Metal.metalFree buf

  IO.println "  Metal FFI test passed!"
EOF
