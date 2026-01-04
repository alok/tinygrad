#!/bin/bash
# Build Metal FFI for TinyGrad4
# Compiles tg4_metal.m with system clang (supports frameworks)
# and produces object file for Lake to link

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LEAN4_DIR="$ROOT_DIR/lean4"
C_DIR="$LEAN4_DIR/c"
BUILD_DIR="$ROOT_DIR/.lake/build/metal"

# Get Lean installation path from elan toolchain
# The `lean` binary is a shim; get real path via elan
LEAN_BIN="$(elan which lean)"
LEAN_HOME="$(dirname "$(dirname "$LEAN_BIN")")"
LEAN_INCLUDE="$LEAN_HOME/include"

# Verify Lean headers exist
if [ ! -f "$LEAN_INCLUDE/lean/lean.h" ]; then
    echo "Error: Lean headers not found at $LEAN_INCLUDE"
    echo "Set LEAN_HOME to your Lean installation directory"
    exit 1
fi

mkdir -p "$BUILD_DIR"

echo "Building Metal FFI..."
echo "  Lean include: $LEAN_INCLUDE"
echo "  C source: $C_DIR"
echo "  Output: $BUILD_DIR"

# Compile tg4_metal.m to object file
clang -c \
    -I"$LEAN_INCLUDE" \
    -framework Metal \
    -framework Foundation \
    -fPIC \
    -O2 \
    -o "$BUILD_DIR/tg4_metal.o" \
    "$C_DIR/tg4_metal.m"

echo "Built: $BUILD_DIR/tg4_metal.o"

# Create static library for easier linking
ar rcs "$BUILD_DIR/libtg4_metal.a" "$BUILD_DIR/tg4_metal.o"
echo "Built: $BUILD_DIR/libtg4_metal.a"

# Also build standalone metal_runner for testing
clang \
    -framework Metal \
    -framework Foundation \
    -O2 \
    -o "$BUILD_DIR/metal_runner" \
    "$C_DIR/metal_runner.m"

echo "Built: $BUILD_DIR/metal_runner"
echo "Done."
