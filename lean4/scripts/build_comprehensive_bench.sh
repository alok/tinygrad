#!/bin/bash
# Build and run comprehensive Metal benchmark
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=== Building Comprehensive Benchmark ==="

# 1. Build Metal FFI
./scripts/build_metal.sh

# 2. Build Lean components (will fail at link stage due to Metal symbols)
echo ""
echo "Step 2: Building Lean components..."
lake build comprehensive_bench 2>&1 | grep -v "undefined symbol" | tail -20 || true

# 3. Find object files from Lake build
echo ""
echo "Step 3: Extracting object files..."
OBJ_DIR=".lake/build/ir"
LAKE_CACHE="$HOME/.elan/toolchains/leanprover--lean4---v4.27.0-rc1/lake/cache/artifacts"

# Get the list of object files that Lake tried to link
OBJS=$(find "$LAKE_CACHE" -name "*.o" -newer .lake/build/lib/lean/TinyGrad4/Test/ComprehensiveBench.olean 2>/dev/null | head -20)
echo "Found $(echo "$OBJS" | wc -l | tr -d ' ') object files"

# 4. Link with system clang (has Metal framework support)
echo ""
echo "Step 4: Linking with system clang..."

LEAN_ROOT="$HOME/.elan/toolchains/leanprover--lean4---v4.27.0-rc1"
OUTPUT=".lake/build/bin/comprehensive_bench"

# Get ALL object files from the lake cache that are newer than the olean
ALL_OBJS=$(ls -t "$LAKE_CACHE"/*.o 2>/dev/null | head -20)

mkdir -p .lake/build/bin

/usr/bin/clang++ \
    $ALL_OBJS \
    .lake/build/lib/libtg4c.a \
    .lake/build/metal/libtg4metal.a \
    -L"$LEAN_ROOT/lib/lean" \
    -L"$LEAN_ROOT/lib" \
    -lleancpp -lInit -lStd -lLean -lleanrt -lLake -lgmp -luv \
    -framework Metal -framework Foundation \
    -lc++ \
    -o "$OUTPUT" 2>&1 || echo "(Link may show warnings, checking if executable was created...)"

if [ -f "$OUTPUT" ]; then
    echo ""
    echo "=== Build Complete ==="
    echo "Executable: $OUTPUT"
    echo ""
    echo "Running benchmarks..."
    echo ""
    "$OUTPUT"
else
    echo "Build failed - executable not created"
    exit 1
fi
