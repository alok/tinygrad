#!/bin/bash
# Build and run Metal test executable using system clang
# This bypasses Lake's linker which doesn't support macOS frameworks

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LEAN4_DIR="$ROOT_DIR/lean4"
PROJECT_DIR="$LEAN4_DIR"
BUILD_DIR="$ROOT_DIR/.lake/build"

echo "=== Building Metal Test Executable ==="

# Get Lean toolchain
LEAN_TOOLCHAIN_RAW=$(cat "$ROOT_DIR/lean-toolchain" | tr -d '\n')
LEAN_TOOLCHAIN=$(echo "$LEAN_TOOLCHAIN_RAW" | sed 's|/|--|g' | sed 's|:|---|g')
LEAN_ROOT="$HOME/.elan/toolchains/$LEAN_TOOLCHAIN"
CACHE_DIR="$LEAN_ROOT/lake/cache/artifacts"

echo "Lean toolchain: $LEAN_TOOLCHAIN"

# Step 1: Ensure Metal FFI is built
echo ""
echo "Step 1: Building Metal FFI..."
"$SCRIPT_DIR/build_metal.sh"

# Step 2: Try to build with Lake (will fail at link but compiles .o files)
echo ""
echo "Step 2: Building Lean components (link will fail, that's expected)..."
cd "$ROOT_DIR"
lake build metal_test 2>&1 | tee /tmp/lake_build.log || true

# Step 3: Extract the .o files Lake would use from the error message
echo ""
echo "Step 3: Extracting object files from Lake build..."

# Parse the clang command from Lake's output to get the .o files
O_FILES=$(grep -oE "$CACHE_DIR/[a-f0-9]+\.o" /tmp/lake_build.log | sort -u | tr '\n' ' ')

if [ -z "$O_FILES" ]; then
    echo "Error: Could not find object files in Lake output"
    exit 1
fi

echo "Found $(echo $O_FILES | wc -w | tr -d ' ') object files"

# Step 4: Link with system clang
echo ""
echo "Step 4: Linking with system clang..."

mkdir -p "$BUILD_DIR/bin"
OUTPUT="$BUILD_DIR/bin/metal_test"

echo "Linking to $OUTPUT..."

# The Init library provides the main entry point
# We need to link with all the Lean runtime components
/usr/bin/clang++ -o "$OUTPUT" \
    $O_FILES \
    "$BUILD_DIR/lib/libtg4c.a" \
    "$BUILD_DIR/metal/libtg4metal.a" \
    -L"$LEAN_ROOT/lib/lean" \
    -L"$LEAN_ROOT/lib" \
    -lleancpp -lInit -lStd -lLean -lleanrt \
    -lgmp -luv -lLake \
    -framework Metal \
    -framework Foundation \
    -framework CoreFoundation \
    -lc++ \
    -Wl,-rpath,"$LEAN_ROOT/lib/lean" \
    -Wl,-rpath,"$LEAN_ROOT/lib"

echo ""
echo "=== Build Complete ==="
echo "Executable: $OUTPUT"
echo ""
echo "Running tests..."
echo ""

"$OUTPUT"
