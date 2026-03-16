#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if command -v lake >/dev/null 2>&1; then
  LAKE_BIN="lake"
else
  LAKE_BIN="${HOME}/.elan/bin/lake"
fi

cd "${REPO_ROOT}"
exec "${LAKE_BIN}" exe tg4_autoresearch "$@"
