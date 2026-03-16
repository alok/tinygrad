# Cost Model Analysis - M4 Max (Dec 2024)

## Key Findings

### 1. Interpreter Overhead Dominates
- Memory bandwidth on M4 Max: **2.3 GB/s** (vs 400 GB/s theoretical)
- This is ~170x slower than native, indicating interpreter overhead dominates
- Kernel launch overhead: ~2.3 µs (negligible compared to interpreter)

### 2. Fusion Not Beneficial for Small Tensors
- MNIST MLP with batch=32, hidden=128:
  - **Fused: 3.17 ms/iter** (18 kernels)
  - **Node: 2.43 ms/iter** (no fusion)
  - **Speedup: 0.77x** (fusion hurts!)

### 3. Root Cause
The Lean interpreter executes per-UOp with overhead. Fusion:
1. Adds complexity to pattern matching and code generation
2. Creates KERNEL nodes with extra transformations
3. Doesn't actually dispatch to GPU (yet) - still interpreted

Fusion will become beneficial when:
- GPU dispatch is implemented (Metal FFI for real kernels)
- Larger batch sizes increase arithmetic intensity
- Interpreter overhead is amortized over more compute

### 4. Cost Model Updates

**Original defaults:**
```lean
kernelOverhead := 1000
elem := 1
memReadByte := 1
```

**Updated `shouldFuse` logic:**
- Single-node patterns: only fuse if compute >= 2 * kernelOverhead
- Multi-node patterns: fuse if launchSavings > fusionOverhead/10

**Recommended cost models by backend:**

| Parameter        | Interpreter | Metal GPU (future) |
|------------------|-------------|-------------------|
| kernelOverhead   | 50000       | 500               |
| elem             | 1           | 1                 |
| memReadByte      | 1           | 1                 |
| memReadViewByte  | 3           | 2                 |
| reduceElem       | 2           | 1                 |

### 5. Files Created
- `m4_max_cost_model.json` - GPU-tuned (for future)
- `conservative_cost_model.json` - Interpreter-friendly
- `very_conservative_cost_model.json` - Minimal fusion

### 6. Next Steps
1. Implement actual Metal GPU dispatch (not just FFI stubs)
2. Profile GPU execution to get real kernel overhead
3. Add adaptive cost model that measures at runtime
