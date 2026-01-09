#!/bin/bash
# Build Metal FFI with system clang and create static library

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LEAN4_DIR="$ROOT_DIR/lean4"
C_DIR="$LEAN4_DIR/c"
BUILD_DIR="$ROOT_DIR/.lake/build/metal"

# Find Lean toolchain include directory
# Convert leanprover/lean4:v4.27.0-rc1 to leanprover--lean4---v4.27.0-rc1
LEAN_TOOLCHAIN_RAW=$(cat "$ROOT_DIR/lean-toolchain" | tr -d '\n')
LEAN_TOOLCHAIN=$(echo "$LEAN_TOOLCHAIN_RAW" | sed 's|/|--|g' | sed 's|:|---|g')
LEAN_INCLUDE="$HOME/.elan/toolchains/$LEAN_TOOLCHAIN/include"

echo "=== Building Metal FFI ==="
echo "Project: $LEAN4_DIR"
echo "Lean include: $LEAN_INCLUDE"

mkdir -p "$BUILD_DIR"

# Compile tg4_metal.m
echo "Compiling tg4_metal.m..."
clang -c "$C_DIR/tg4_metal.m" \
    -o "$BUILD_DIR/tg4_metal.o" \
    -I "$LEAN_INCLUDE" \
    -fPIC -O3 -fobjc-arc \
    -Wno-deprecated-declarations

# Create static library
echo "Creating libtg4metal.a..."
ar rcs "$BUILD_DIR/libtg4metal.a" "$BUILD_DIR/tg4_metal.o"

echo "=== Metal FFI Build Complete ==="
echo "Library: $BUILD_DIR/libtg4metal.a"
echo ""
echo "To link with a Lean executable, add:"
echo "  -L$BUILD_DIR -ltg4metal -framework Metal -framework Foundation"
