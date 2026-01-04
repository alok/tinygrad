#!/bin/bash
# Build Metal benchmark with direct FFI linking
# This bypasses Lake's clang limitations with frameworks

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LEAN4_DIR="$ROOT_DIR/lean4"
PROJECT_DIR="$LEAN4_DIR"
WORKSPACE_ROOT="$ROOT_DIR"

# Ensure Metal FFI is built
"$SCRIPT_DIR/build_metal_ffi.sh"

# Get Lean toolchain paths
LEAN_BIN="$(elan which lean)"
LEAN_HOME="$(dirname "$(dirname "$LEAN_BIN")")"
LEAN_LIB="$LEAN_HOME/lib/lean"

# Build all Lean objects first
cd "$WORKSPACE_ROOT"
echo "Building Lean objects..."
lake build metal_test:c.o

# Paths
BUILD_DIR="$WORKSPACE_ROOT/.lake/build"
METAL_OBJ="$BUILD_DIR/metal/tg4_metal.o"
OUTPUT="$BUILD_DIR/metal/metal_bench"

# Collect all object files for the executable
# Lake puts them in .lake/build/ir/
echo "Collecting object files..."

# The metal_test executable needs these object files
# Get them from Lake's build directory
OBJ_FILES=(
    "$BUILD_DIR/ir/TinyGrad4/Test/MetalTestMain.c.o"
)

# Also need the static library Lake built (has all TinyGrad4 dependencies)
STATIC_LIB="$BUILD_DIR/lib/libTinyGrad4.a"

echo "Metal FFI: $METAL_OBJ"
echo "Static lib: $STATIC_LIB"

# Link with system clang (supports frameworks)
echo "Linking metal_bench executable..."
clang++ \
    -o "$OUTPUT" \
    "${OBJ_FILES[@]}" \
    "$METAL_OBJ" \
    "$STATIC_LIB" \
    -L"$LEAN_LIB" \
    -lleanshared \
    -L"$BUILD_DIR/lib" \
    -ltg4c \
    -framework Metal \
    -framework Foundation \
    -Wl,-rpath,"$LEAN_LIB" \
    2>&1 || {
        echo "Direct linking failed. Trying alternative approach..."

        # Alternative: link via Lake's native output and just add frameworks
        LAKE_EXE="$BUILD_DIR/bin/metal_test"
        if [ -f "$LAKE_EXE" ]; then
            echo "Lake built partial executable, relinking with frameworks..."
            # Extract objects from Lake's partial build
        else
            echo "Run 'lake build metal_test' first, then manually add frameworks"
        fi
        exit 1
    }

echo "Built: $OUTPUT"
echo ""
echo "Run with: $OUTPUT"
