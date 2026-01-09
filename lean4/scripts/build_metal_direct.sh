#!/bin/bash
# Build direct Metal FFI benchmark using C driver approach
#
# This compiles a C main() that initializes Lean and calls the benchmark,
# bypassing Lake's linking limitations with Metal frameworks.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LEAN4_DIR="$ROOT_DIR/lean4"

cd "$ROOT_DIR"

# Get Lean paths
LEAN_BIN="$(elan which lean)"
LEAN_HOME="$(dirname "$(dirname "$LEAN_BIN")")"
LEAN_INCLUDE="$LEAN_HOME/include"
LEAN_LIB="$LEAN_HOME/lib/lean"

BUILD_DIR="$ROOT_DIR/.lake/build"
METAL_DIR="$BUILD_DIR/metal"
OUTPUT="$METAL_DIR/metal_direct"

mkdir -p "$METAL_DIR"

echo "=== Building Direct Metal FFI Benchmark ==="
echo ""

# Step 1: Build Metal FFI
echo "Step 1: Building Metal FFI..."
"$SCRIPT_DIR/build_metal_ffi.sh"
echo ""

# Step 2: Build TinyGrad4 library with Lake
echo "Step 2: Building TinyGrad4 library..."
lake build TinyGrad4 2>&1 | tail -5
echo ""

# Step 3: Try to build MetalDirect module (will fail at exe linking, but compiles the module)
echo "Step 3: Compiling MetalDirect module..."
lake build TinyGrad4.Benchmark.MetalDirect 2>&1 | grep -v "^error:" | tail -5 || true
echo ""

# Step 4: Compile C driver
echo "Step 4: Compiling C driver..."
C_DRIVER="$LEAN4_DIR/c/metal_bench_main.c"
C_OBJ="$METAL_DIR/metal_bench_main.o"

clang -c \
    -o "$C_OBJ" \
    "$C_DRIVER" \
    -I"$LEAN_INCLUDE" \
    -O2 \
    2>&1

echo "  Compiled: $C_OBJ"
echo ""

# Step 5: Link everything together
echo "Step 5: Linking..."

# Find the TinyGrad4 shared library
TG4_SHLIB="$BUILD_DIR/lib/libTinyGrad4.dylib"
if [ ! -f "$TG4_SHLIB" ]; then
    TG4_SHLIB="$BUILD_DIR/lib/libTinyGrad4.so"
fi

# Metal FFI object
METAL_OBJ="$METAL_DIR/tg4_metal.o"

# Dependency libraries
DEP_LIBS=""
for pkg in strata batteries Cli plausible; do
    PKGLIB="$ROOT_DIR/.lake/packages/$pkg/.lake/build/lib"
    if [ -d "$PKGLIB" ]; then
        DEP_LIBS="$DEP_LIBS -L$PKGLIB"
        # Add the shared library if it exists
        for dylib in "$PKGLIB"/*.dylib; do
            if [ -f "$dylib" ]; then
                DEP_LIBS="$DEP_LIBS -l$(basename "$dylib" .dylib | sed 's/^lib//')"
            fi
        done
    fi
done

echo "  TinyGrad4 lib: $TG4_SHLIB"
echo "  Metal FFI: $METAL_OBJ"
echo "  Lean lib: $LEAN_LIB"

clang++ \
    -o "$OUTPUT" \
    "$C_OBJ" \
    "$METAL_OBJ" \
    -L"$BUILD_DIR/lib" \
    -lTinyGrad4 \
    -ltg4c \
    $DEP_LIBS \
    -L"$LEAN_LIB" \
    -lleanshared \
    -framework Metal \
    -framework Foundation \
    -Wl,-rpath,"$LEAN_LIB" \
    -Wl,-rpath,"$BUILD_DIR/lib" \
    2>&1

echo ""
echo "=== Build Complete ==="
echo ""
echo "Output: $OUTPUT"
echo ""
echo "Run with:"
echo "  $OUTPUT"
