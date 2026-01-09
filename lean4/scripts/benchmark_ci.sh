#!/usr/bin/env bash
# CI-friendly benchmark script for TinyGrad4
#
# Usage:
#   ./scripts/benchmark_ci.sh              # Run all benchmarks
#   ./scripts/benchmark_ci.sh --quick      # Quick run (fewer iterations)
#   ./scripts/benchmark_ci.sh --json       # Output JSON only
#   ./scripts/benchmark_ci.sh --compare    # Compare with baseline
#
# Exit codes:
#   0 - Success
#   1 - Build failure
#   2 - Benchmark failure
#   3 - Regression detected

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LEAN4_DIR="$ROOT_DIR/lean4"
BUILD_DIR="$ROOT_DIR/.lake/build"
RESULTS_DIR="$LEAN4_DIR/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors (disabled in CI environments without TTY)
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[0;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    GREEN=''
    RED=''
    YELLOW=''
    CYAN=''
    BOLD=''
    NC=''
fi

log_info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${BOLD}==> $*${NC}"; }

# Parse args
QUICK=false
JSON_ONLY=false
COMPARE=false
BASELINE_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK=true
            shift
            ;;
        --json)
            JSON_ONLY=true
            shift
            ;;
        --compare)
            COMPARE=true
            shift
            ;;
        --baseline)
            BASELINE_FILE="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup results directory
mkdir -p "$RESULTS_DIR"

# Build
log_step "Building tg4_bench..."
cd "$ROOT_DIR"

if ! lake build tg4_bench 2>&1; then
    log_error "Build failed"
    exit 1
fi
log_ok "Build complete"

# Get system info
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
MACHINE_INFO=$(uname -s -r -m 2>/dev/null || echo "unknown")

# Construct CLI args
CLI_ARGS=()
if [ "$QUICK" = true ]; then
    CLI_ARGS+=(--quick)
fi

RESULT_JSON="$RESULTS_DIR/benchmark_${TIMESTAMP}.json"
CLI_ARGS+=(--json "$RESULT_JSON")

# Run benchmarks
log_step "Running benchmarks..."
echo "  Machine: $MACHINE_INFO"
echo "  Git: $GIT_COMMIT"
echo "  Output: $RESULT_JSON"
echo ""

if ! "$BUILD_DIR/bin/tg4_bench" run "${CLI_ARGS[@]}"; then
    log_error "Benchmark run failed"
    exit 2
fi

log_ok "Benchmarks complete"

# Output results
if [ "$JSON_ONLY" = true ]; then
    cat "$RESULT_JSON"
else
    echo ""
    log_step "Results Summary"

    # Parse JSON and format (if jq available)
    if command -v jq &> /dev/null && [ -f "$RESULT_JSON" ]; then
        jq -r '
            .results_by_backend | to_entries[] |
            "\(.key):" as $backend |
            .value[] |
            "  \(.spec.name): \(.stats.mean_us) μs (min: \(.stats.min_us) μs)"
        ' "$RESULT_JSON" 2>/dev/null || cat "$RESULT_JSON"
    else
        cat "$RESULT_JSON"
    fi
fi

# Comparison with baseline
if [ "$COMPARE" = true ]; then
    log_step "Comparing with baseline..."

    if [ -z "$BASELINE_FILE" ]; then
        # Find most recent baseline
        BASELINE_FILE=$(ls -t "$RESULTS_DIR"/benchmark_*.json 2>/dev/null | head -2 | tail -1)
    fi

    if [ -z "$BASELINE_FILE" ] || [ ! -f "$BASELINE_FILE" ]; then
        log_warn "No baseline file found for comparison"
    else
        echo "  Baseline: $BASELINE_FILE"
        echo "  Current:  $RESULT_JSON"

        if command -v jq &> /dev/null; then
            # Compare mean times (simplified - could be more sophisticated)
            BASELINE_MEAN=$(jq '[.. | .mean_us? // empty] | add / length' "$BASELINE_FILE" 2>/dev/null || echo "0")
            CURRENT_MEAN=$(jq '[.. | .mean_us? // empty] | add / length' "$RESULT_JSON" 2>/dev/null || echo "0")

            if [ "$BASELINE_MEAN" != "0" ] && [ "$CURRENT_MEAN" != "0" ]; then
                RATIO=$(echo "scale=2; $CURRENT_MEAN / $BASELINE_MEAN" | bc 2>/dev/null || echo "1.0")
                echo ""
                echo "  Performance ratio: ${RATIO}x (1.0 = same, <1.0 = faster, >1.0 = slower)"

                # Check for regression (>10% slower)
                if [ "$(echo "$RATIO > 1.10" | bc 2>/dev/null)" = "1" ]; then
                    log_error "Performance regression detected! (${RATIO}x slower)"
                    exit 3
                elif [ "$(echo "$RATIO < 0.90" | bc 2>/dev/null)" = "1" ]; then
                    log_ok "Performance improvement! (${RATIO}x faster)"
                else
                    log_ok "Performance within tolerance"
                fi
            fi
        else
            log_warn "jq not available - skipping detailed comparison"
        fi
    fi
fi

# Create symlink to latest results
ln -sf "benchmark_${TIMESTAMP}.json" "$RESULTS_DIR/latest.json"

log_ok "Done. Results saved to $RESULT_JSON"
