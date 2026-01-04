#!/bin/bash
# Build CUDA test binary for TinyGrad4
# Usage: ./scripts/build_cuda_test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LEAN4_DIR="$ROOT_DIR/lean4"
BUILD_DIR="$ROOT_DIR/.lake/build"
C_DIR="$LEAN4_DIR/c"
OUTPUT_DIR="$LEAN4_DIR/build"

echo "=== Building TinyGrad4 CUDA Test ==="
echo "LEAN4_DIR: $LEAN4_DIR"

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA toolkit."
    exit 1
fi

# Get CUDA paths
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
if [ ! -d "$CUDA_PATH" ]; then
    CUDA_PATH=$(dirname $(dirname $(which nvcc)))
fi

echo "CUDA_PATH: $CUDA_PATH"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Build Lean project (compiles .lean files)
echo "Step 1: Building Lean project..."
cd "$ROOT_DIR"
lake build TinyGrad4 || echo "Lake build may fail due to missing FFI - continuing..."

# Step 2: Compile CUDA FFI
echo ""
echo "Step 2: Compiling CUDA FFI..."
nvcc -c -O3 \
    -I"$CUDA_PATH/include" \
    -I"$(lean --print-prefix)/include" \
    "$C_DIR/tg4_cuda.cu" \
    -o "$OUTPUT_DIR/tg4_cuda.o"

echo "Compiled: $OUTPUT_DIR/tg4_cuda.o"

# Step 3: Find Lean object files
echo ""
echo "Step 3: Locating Lean object files..."

# Find the CudaTestMain object file
CUDA_TEST_O=$(find "$BUILD_DIR" -name "CudaTestMain.o" 2>/dev/null | head -1)

if [ -z "$CUDA_TEST_O" ]; then
    echo "CudaTestMain.o not found. Building CudaTestMain module..."
    lake build +TinyGrad4.Test.CudaTestMain || true
    CUDA_TEST_O=$(find "$BUILD_DIR" -name "CudaTestMain.o" 2>/dev/null | head -1)
fi

if [ -z "$CUDA_TEST_O" ]; then
    echo "ERROR: Could not find CudaTestMain.o"
    echo "Available .o files:"
    find "$BUILD_DIR" -name "*.o" | head -20
    exit 1
fi

echo "Found: $CUDA_TEST_O"

# Find all required Lean object files
echo ""
echo "Step 4: Collecting all Lean object files..."

# Core modules needed
LEAN_O_FILES=""
for module in Cuda Device CudaTestMain; do
    MODULE_O=$(find "$BUILD_DIR" -name "$module.o" 2>/dev/null | head -1)
    if [ -n "$MODULE_O" ]; then
        LEAN_O_FILES="$LEAN_O_FILES $MODULE_O"
        echo "  Found: $MODULE_O"
    fi
done

# Find initialization objects
INIT_O=$(find "$BUILD_DIR" -name "*.init.o" 2>/dev/null | head -5)
echo "Init objects:"
echo "$INIT_O" | head -5

# Step 5: Link everything
echo ""
echo "Step 5: Linking..."

# Get Lean library paths
LEAN_PREFIX=$(lean --print-prefix)
LEAN_LIB="$LEAN_PREFIX/lib/lean"

# Find leanc for linking
LEANC=$(which leanc 2>/dev/null || echo "$LEAN_PREFIX/bin/leanc")

if [ -x "$LEANC" ]; then
    echo "Using leanc for linking..."

    # Use leanc with explicit CUDA libraries
    $LEANC -o "$OUTPUT_DIR/cuda_test" \
        $LEAN_O_FILES \
        "$OUTPUT_DIR/tg4_cuda.o" \
        -L"$CUDA_PATH/lib64" -L"$CUDA_PATH/lib" \
        -lcuda -lnvrtc \
        || echo "leanc linking failed, trying manual link..."
fi

# Fallback: manual linking with g++
if [ ! -f "$OUTPUT_DIR/cuda_test" ]; then
    echo "Attempting manual linking with g++..."

    g++ -o "$OUTPUT_DIR/cuda_test" \
        $LEAN_O_FILES \
        "$OUTPUT_DIR/tg4_cuda.o" \
        -L"$LEAN_LIB" \
        -L"$CUDA_PATH/lib64" -L"$CUDA_PATH/lib" \
        -lleanshared -lcuda -lnvrtc -lpthread -ldl \
        -Wl,-rpath,"$LEAN_LIB" \
        -Wl,-rpath,"$CUDA_PATH/lib64" \
        2>&1 || echo "Manual linking may need adjustments"
fi

if [ -f "$OUTPUT_DIR/cuda_test" ]; then
    echo ""
    echo "=== Build Successful ==="
    echo "Binary: $OUTPUT_DIR/cuda_test"
    echo ""
    echo "To run: $OUTPUT_DIR/cuda_test"
    echo "To profile: nsys profile --stats=true $OUTPUT_DIR/cuda_test"
else
    echo ""
    echo "=== Build Output ==="
    echo "CUDA object file: $OUTPUT_DIR/tg4_cuda.o"
    echo ""
    echo "To manually link on your target system:"
    echo "  g++ -o cuda_test *.o -lleanshared -lcuda -lnvrtc"
fi
