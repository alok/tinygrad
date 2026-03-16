#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PARITY_PATH = ROOT / "lean4/PARITY.md"
PARITY_PLAN_PATH = ROOT / "lean4/PARITY_TEST_PLAN.md"
FIXTURE_PATH = ROOT / "lean4/testdata/parity/core_ops.json"
BASELINE_PATH = ROOT / "lean4/testdata/parity/scorecard_baseline.json"
FIXTURE_GENERATOR_PATH = ROOT / "lean4/scripts/generate_parity_fixtures.py"

TENSOR_SURFACE_FILES = [
  ROOT / "lean4/TinyGrad4/Tensor/Math.lean",
  ROOT / "lean4/TinyGrad4/Tensor/Movement.lean",
  ROOT / "lean4/TinyGrad4/Tensor/Notation.lean",
]

LEGACY_MARKER_RE = re.compile(
  r"conv2dPlaceholder|maxPool2dPlaceholder|avgPool2dPlaceholder|\bplaceholder\b",
  re.IGNORECASE,
)

PROFILE_THRESHOLDS = {"fast": 8, "medium": 10, "slow": 12}
MISSING_DECREASE_THRESHOLD = 6


@dataclass
class GateResult:
  name: str
  passed: bool
  reason: str


def run(cmd: list[str], cwd: Path = ROOT) -> str:
  proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
  if proc.returncode != 0:
    details = (proc.stdout + "\n" + proc.stderr).strip()
    raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{details}")
  return proc.stdout + proc.stderr


def parse_parity_counts(text: str) -> dict[str, int]:
  counts = {"implemented": 0, "partial": 0, "missing": 0}
  for line in text.splitlines():
    m = re.match(r"^- \[(x|~| )\] ", line)
    if not m:
      continue
    token = m.group(1)
    if token == "x":
      counts["implemented"] += 1
    elif token == "~":
      counts["partial"] += 1
    else:
      counts["missing"] += 1
  return counts


def parse_selection_counts_from_plan(text: str) -> dict[str, int]:
  out: dict[str, int] = {}
  for profile in ("fast", "medium", "slow"):
    m = re.search(rf"^- {profile}:\s*(\d+)\s*$", text, flags=re.MULTILINE)
    if m:
      out[profile] = int(m.group(1))
  return out


def extract_last_json_object(text: str) -> dict[str, Any]:
  decoder = json.JSONDecoder()
  last_obj: dict[str, Any] | None = None
  idx = 0
  while idx < len(text):
    if text[idx] != "{":
      idx += 1
      continue
    try:
      obj, end = decoder.raw_decode(text, idx)
    except json.JSONDecodeError:
      idx += 1
      continue
    if isinstance(obj, dict) and "selected_count" in obj:
      last_obj = obj
    idx = end
  if last_obj is None:
    raise RuntimeError("failed to parse JSON test-list payload from `lake test -- --json --list` output")
  return last_obj


def collect_selection_counts() -> dict[str, int]:
  counts: dict[str, int] = {}
  for profile in ("fast", "medium", "slow"):
    raw = run(["lake", "test", "--", "--json", "--list", "--profile", profile], cwd=ROOT)
    payload = extract_last_json_object(raw)
    counts[profile] = int(payload["selected_count"])
  return counts


def scan_legacy_markers_from_texts(texts: dict[str, str]) -> dict[str, Any]:
  hits: list[dict[str, Any]] = []
  for rel, text in texts.items():
    for lineno, line in enumerate(text.splitlines(), start=1):
      if LEGACY_MARKER_RE.search(line):
        hits.append({"file": rel, "line": lineno, "line_text": line.strip()})
  return {"count": len(hits), "hits": hits}


def scan_current_legacy_markers() -> dict[str, Any]:
  texts = {str(path.relative_to(ROOT)): path.read_text() for path in TENSOR_SURFACE_FILES}
  return scan_legacy_markers_from_texts(texts)


def fixture_freshness() -> tuple[bool, str]:
  sys.path.insert(0, str(ROOT))
  spec = importlib.util.spec_from_file_location("generate_parity_fixtures", FIXTURE_GENERATOR_PATH)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"unable to load fixture generator at {FIXTURE_GENERATOR_PATH}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  payload = module.build_fixtures()
  expected = json.dumps(payload, indent=2, sort_keys=True) + "\n"
  actual = FIXTURE_PATH.read_text()
  if expected == actual:
    return True, "fixture file matches generator output"
  return False, "fixture file differs from generator output (run generate_parity_fixtures.py)"


def git_show_text(ref: str, rel_path: str) -> str | None:
  proc = subprocess.run(["git", "show", f"{ref}:{rel_path}"], cwd=ROOT, capture_output=True, text=True)
  if proc.returncode != 0:
    return None
  return proc.stdout


def collect_baseline_from_ref(ref: str) -> dict[str, Any]:
  parity_text = git_show_text(ref, "lean4/PARITY.md")
  fixture_text = git_show_text(ref, "lean4/testdata/parity/core_ops.json")
  plan_text = git_show_text(ref, "lean4/PARITY_TEST_PLAN.md")
  if parity_text is None or fixture_text is None or plan_text is None:
    raise RuntimeError(f"unable to read baseline files from git ref `{ref}`")

  legacy_texts: dict[str, str] = {}
  for path in TENSOR_SURFACE_FILES:
    rel = str(path.relative_to(ROOT))
    text = git_show_text(ref, rel)
    if text is None:
      raise RuntimeError(f"unable to read `{rel}` from git ref `{ref}`")
    legacy_texts[rel] = text

  fixture_payload = json.loads(fixture_text)
  selection_counts = parse_selection_counts_from_plan(plan_text)
  if not all(profile in selection_counts for profile in ("fast", "medium", "slow")):
    raise RuntimeError(f"baseline PARITY_TEST_PLAN.md at `{ref}` is missing profile selection counts")

  legacy = scan_legacy_markers_from_texts(legacy_texts)
  return {
    "source_ref": ref,
    "parity_counts": parse_parity_counts(parity_text),
    "fixture_case_count": len(fixture_payload.get("cases", [])),
    "selection_counts": selection_counts,
    "legacy_marker_count": legacy["count"],
  }


def load_baseline(baseline_json: Path, fallback_ref: str) -> dict[str, Any]:
  if baseline_json.exists():
    raw = json.loads(baseline_json.read_text())
    metrics = raw.get("metrics", raw)
    required = {"parity_counts", "fixture_case_count", "selection_counts", "legacy_marker_count"}
    missing = required - set(metrics.keys())
    if missing:
      raise RuntimeError(f"baseline file `{baseline_json}` missing keys: {sorted(missing)}")
    metrics["source_ref"] = raw.get("source_ref", metrics.get("source_ref", "baseline-json"))
    return metrics
  return collect_baseline_from_ref(fallback_ref)


def collect_current_metrics() -> dict[str, Any]:
  parity_text = PARITY_PATH.read_text()
  fixture_payload = json.loads(FIXTURE_PATH.read_text())
  selections = collect_selection_counts()
  legacy = scan_current_legacy_markers()
  return {
    "parity_counts": parse_parity_counts(parity_text),
    "fixture_case_count": len(fixture_payload.get("cases", [])),
    "selection_counts": selections,
    "legacy_marker_count": legacy["count"],
    "legacy_hits": legacy["hits"],
  }


def evaluate_gates(current: dict[str, Any], baseline: dict[str, Any]) -> list[GateResult]:
  gates: list[GateResult] = []

  missing_delta = baseline["parity_counts"]["missing"] - current["parity_counts"]["missing"]
  gates.append(
    GateResult(
      name="parity_missing_decrease",
      passed=missing_delta >= MISSING_DECREASE_THRESHOLD,
      reason=(
        f"missing delta = {missing_delta} (baseline {baseline['parity_counts']['missing']} -> "
        f"current {current['parity_counts']['missing']}, requires >= {MISSING_DECREASE_THRESHOLD})"
      ),
    )
  )

  for profile, threshold in PROFILE_THRESHOLDS.items():
    delta = current["selection_counts"][profile] - baseline["selection_counts"][profile]
    gates.append(
      GateResult(
        name=f"driver_selection_{profile}",
        passed=delta >= threshold,
        reason=(
          f"{profile} delta = {delta} (baseline {baseline['selection_counts'][profile]} -> "
          f"current {current['selection_counts'][profile]}, requires >= {threshold})"
        ),
      )
    )

  gates.append(
    GateResult(
      name="legacy_placeholder_markers_zero",
      passed=current["legacy_marker_count"] == 0,
      reason=f"legacy marker count = {current['legacy_marker_count']} (requires 0)",
    )
  )

  fresh_ok, fresh_reason = fixture_freshness()
  gates.append(GateResult(name="fixture_freshness", passed=fresh_ok, reason=fresh_reason))
  return gates


def render_markdown(report: dict[str, Any]) -> str:
  cur = report["current"]
  base = report["baseline"]
  deltas = report["deltas"]
  lines: list[str] = []
  lines.append("# Parity Scorecard")
  lines.append("")
  lines.append(f"- generated_at: {report['generated_at']}")
  lines.append(f"- baseline_source: {base.get('source_ref', 'unknown')}")
  lines.append(f"- check_passed: {report['check_passed']}")
  lines.append("")
  lines.append("## Current")
  lines.append(
    f"- parity: implemented={cur['parity_counts']['implemented']}, "
    f"partial={cur['parity_counts']['partial']}, missing={cur['parity_counts']['missing']}"
  )
  lines.append(f"- fixtures: {cur['fixture_case_count']}")
  lines.append(
    f"- selections: fast={cur['selection_counts']['fast']}, medium={cur['selection_counts']['medium']}, slow={cur['selection_counts']['slow']}"
  )
  lines.append(f"- legacy markers: {cur['legacy_marker_count']}")
  lines.append("")
  lines.append("## Baseline")
  lines.append(f"- parity missing: {base['parity_counts']['missing']}")
  lines.append(f"- fixtures: {base['fixture_case_count']}")
  lines.append(
    f"- selections: fast={base['selection_counts']['fast']}, medium={base['selection_counts']['medium']}, slow={base['selection_counts']['slow']}"
  )
  lines.append(f"- legacy markers: {base['legacy_marker_count']}")
  lines.append("")
  lines.append("## Deltas")
  lines.append(f"- parity missing decrease: {deltas['parity_missing_decrease']}")
  lines.append(f"- selection fast delta: {deltas['selection_fast_delta']}")
  lines.append(f"- selection medium delta: {deltas['selection_medium_delta']}")
  lines.append(f"- selection slow delta: {deltas['selection_slow_delta']}")
  lines.append("")
  lines.append("## Gates")
  for gate in report["gates"]:
    status = "PASS" if gate["passed"] else "FAIL"
    lines.append(f"- [{status}] {gate['name']}: {gate['reason']}")
  return "\n".join(lines) + "\n"


def capture_baseline(path: Path, ref: str) -> None:
  metrics = collect_baseline_from_ref(ref)
  payload = {
    "captured_at": datetime.now(timezone.utc).isoformat(),
    "source_ref": ref,
    "metrics": metrics,
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(f"Wrote baseline metrics to {path}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Compute and enforce TinyGrad4 Lean parity scorecard gates.")
  parser.add_argument("--json-out", type=Path, required=True)
  parser.add_argument("--md-out", type=Path, required=True)
  parser.add_argument("--check", action="store_true", help="Exit non-zero when any gate fails")
  parser.add_argument("--baseline-json", type=Path, default=BASELINE_PATH)
  parser.add_argument("--baseline-ref", type=str, default="HEAD")
  parser.add_argument("--capture-baseline", action="store_true", help="Write baseline metrics and exit")
  args = parser.parse_args()

  if args.capture_baseline:
    capture_baseline(args.baseline_json, args.baseline_ref)
    return

  baseline = load_baseline(args.baseline_json, args.baseline_ref)
  current = collect_current_metrics()
  gates = evaluate_gates(current, baseline)
  check_passed = all(g.passed for g in gates)

  deltas = {
    "parity_missing_decrease": baseline["parity_counts"]["missing"] - current["parity_counts"]["missing"],
    "selection_fast_delta": current["selection_counts"]["fast"] - baseline["selection_counts"]["fast"],
    "selection_medium_delta": current["selection_counts"]["medium"] - baseline["selection_counts"]["medium"],
    "selection_slow_delta": current["selection_counts"]["slow"] - baseline["selection_counts"]["slow"],
  }

  report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "current": current,
    "baseline": baseline,
    "deltas": deltas,
    "gates": [asdict(g) for g in gates],
    "check_passed": check_passed,
  }

  args.json_out.parent.mkdir(parents=True, exist_ok=True)
  args.md_out.parent.mkdir(parents=True, exist_ok=True)
  args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  args.md_out.write_text(render_markdown(report))
  print(f"Wrote JSON scorecard to {args.json_out}")
  print(f"Wrote Markdown scorecard to {args.md_out}")

  if args.check and not check_passed:
    for gate in gates:
      if not gate.passed:
        print(f"FAIL: {gate.name} - {gate.reason}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
  main()
