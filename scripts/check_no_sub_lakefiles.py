#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "lakefile.lean"
IGNORE_DIRS = {
  ".git",
  ".lake",
  ".venv",
  ".mypy_cache",
  ".ruff_cache",
  ".pytest_cache",
  ".hypothesis",
  "data",
  "LeanBench",
  "wandb",
  "other_projects",
  "teenygrad",
}


def find_lakefiles() -> list[Path]:
  found: list[Path] = []
  for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
    if "lakefile.lean" in filenames:
      found.append(Path(dirpath) / "lakefile.lean")
  return found


def main() -> int:
  if not EXPECTED.exists():
    print(f"error: missing root lakefile.lean at {EXPECTED}")
    return 1
  found = find_lakefiles()
  bad = [p for p in found if p.resolve() != EXPECTED.resolve()]
  if bad:
    print("error: nested lakefile.lean is not allowed (top-level only)")
    for p in bad:
      print(f"  - {p.relative_to(ROOT)}")
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
