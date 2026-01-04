#!/bin/bash
# Link metal_direct_bench with Metal frameworks
#
# This script manually links the Lake-compiled objects with the Metal FFI
# and macOS frameworks that Lake cannot handle.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Get paths
LEAN_BIN="$(elan which lean)"
LEAN_HOME="$(dirname "$(dirname "$LEAN_BIN")")"
LEAN_LIB="$LEAN_HOME/lib/lean"
BUILD_DIR="$ROOT_DIR/.lake/build"
METAL_DIR="$BUILD_DIR/metal"

mkdir -p "$METAL_DIR"

echo "Building Metal FFI..."
"$SCRIPT_DIR/build_metal_ffi.sh"

echo ""
echo "Linking metal_direct executable..."

# The main object file for metal_direct_bench
MAIN_OBJ="$BUILD_DIR/ir/TinyGrad4/Benchmark/MetalDirectMain.c.o"

if [ ! -f "$MAIN_OBJ" ]; then
    echo "Error: Main object file not found: $MAIN_OBJ"
    echo "Run 'lake build metal_direct_bench' first (it will fail at linking, that's OK)"
    exit 1
fi

# Metal FFI object
METAL_OBJ="$METAL_DIR/tg4_metal.o"

# Collect all library paths and files
LIBS=(
    # Lean runtime
    "-L$LEAN_LIB"
    "-lleanshared"

    # TinyGrad4 library
    "-L$BUILD_DIR/lib"
    "-lTinyGrad4"
    "-ltg4c"

    # macOS frameworks
    "-framework" "Metal"
    "-framework" "Foundation"
)

# Add dependency libraries
for pkg in strata batteries Cli plausible; do
    DEPLIB="$ROOT_DIR/.lake/packages/$pkg/.lake/build/lib"
    if [ -d "$DEPLIB" ]; then
        LIBS+=("-L$DEPLIB")
        for lib in $pkg; do
            if [ -f "$DEPLIB/lib${lib}.a" ]; then
                LIBS+=("-l${lib}")
            fi
        done
    fi
done

# All Benchmark module object files needed
OBJS=(
    "$MAIN_OBJ"
    "$BUILD_DIR/ir/TinyGrad4/Benchmark/MetalDirect.c.o"
)

# Link with system clang
clang++ \
    -o "$METAL_DIR/metal_direct" \
    "${OBJS[@]}" \
    "$METAL_OBJ" \
    "${LIBS[@]}" \
    -Wl,-rpath,"$LEAN_LIB" \
    2>&1

echo ""
echo "Success! Built: $METAL_DIR/metal_direct"
echo ""
echo "Run with:"
echo "  $METAL_DIR/metal_direct"
