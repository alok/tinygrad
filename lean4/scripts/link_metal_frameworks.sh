#!/bin/bash
# Re-link a Lake-built executable with Metal frameworks
# Usage: ./link_metal_frameworks.sh <lake_exe_name> <output_name>
#
# This works by extracting object files and re-linking with system clang

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

# Ensure Metal FFI is built
"$SCRIPT_DIR/build_metal_ffi.sh"

# Get the Metal object file
METAL_OBJ="$METAL_DIR/tg4_metal.o"

# Extract all dependency libraries
LIBS=(
    "-L$LEAN_LIB"
    "-lleanshared"
    "-L$BUILD_DIR/lib"
    "-lTinyGrad4"
    "-ltg4c"
)

# Check for strata, batteries, Cli libraries
for lib in strata batteries Cli plausible; do
    DEPLIB="$ROOT_DIR/.lake/packages/$lib/.lake/build/lib"
    if [ -d "$DEPLIB" ]; then
        LIBS+=("-L$DEPLIB")
        # Find any .a files and add them
        for a in "$DEPLIB"/*.a; do
            if [ -f "$a" ]; then
                LIBS+=("$a")
            fi
        done
    fi
done

EXE_NAME="${1:-tg4_bench}"
OUTPUT_NAME="${2:-${EXE_NAME}_metal}"

# Find the main object file for the executable
MAIN_OBJ=$(find "$BUILD_DIR/ir" -name "*.c.o" -path "*Main*" 2>/dev/null | head -1)

if [ -z "$MAIN_OBJ" ]; then
    echo "Could not find main object file for $EXE_NAME"
    echo "Make sure to run 'lake build $EXE_NAME' first"
    exit 1
fi

echo "Linking $OUTPUT_NAME with Metal frameworks..."
echo "  Main object: $MAIN_OBJ"
echo "  Metal FFI: $METAL_OBJ"

# Link with system clang
clang++ \
    -o "$METAL_DIR/$OUTPUT_NAME" \
    "$MAIN_OBJ" \
    "$METAL_OBJ" \
    "${LIBS[@]}" \
    -framework Metal \
    -framework Foundation \
    -Wl,-rpath,"$LEAN_LIB" \
    2>&1

echo ""
echo "Built: $METAL_DIR/$OUTPUT_NAME"
echo "Run with: $METAL_DIR/$OUTPUT_NAME"
