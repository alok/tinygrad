#!/usr/bin/env bash
# Helper wrapper for running Lean benches on machines with constrained /tmp.
# Usage: ./scripts/bench_env.sh <command...>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

METRICS_OUT="${TG4_METRICS_OUT:-}"
if [ -n "$METRICS_OUT" ]; then
  METRICS_BASE="${METRICS_OUT%.json}"
  "$SCRIPT_DIR/bench_metrics.sh" > "${METRICS_BASE}.pre.json" || true
fi

if [ -d /dev/shm ]; then
  export TMPDIR="${TMPDIR:-/dev/shm/tg4_tmp}"
  mkdir -p "$TMPDIR"
fi

if [ "${TG4_SHM_LAKE:-0}" = "1" ]; then
  SHM_LAKE="${TG4_SHM_LAKE_DIR:-/dev/shm/tg4_lake}"
  mkdir -p "$SHM_LAKE"
  LAKE_DIR="$ROOT_DIR/.lake"
  if [ -d "$LAKE_DIR" ] && [ ! -L "$LAKE_DIR" ]; then
    mv "$LAKE_DIR" "$SHM_LAKE"
    ln -s "$SHM_LAKE" "$LAKE_DIR"
  fi
fi

LEAN_LIB="${HOME}/.elan/toolchains/leanprover--lean4---v4.27.0-rc1/lib"
if [ -d "$LEAN_LIB" ]; then
export LD_LIBRARY_PATH="${LEAN_LIB}:${LD_LIBRARY_PATH:-}"
fi

if [ -n "$METRICS_OUT" ]; then
  "$@"
  exit_code=$?
  "$SCRIPT_DIR/bench_metrics.sh" > "${METRICS_BASE}.post.json" || true
  exit $exit_code
fi

exec "$@"
