# Parity Scorecard

- generated_at: 2026-02-26T03:12:56.829054+00:00
- baseline_source: 505b70a0506d72db10ada998280bf4ed10bbf5da
- check_passed: True

## Current
- parity: implemented=38, partial=14, missing=0
- fixtures: 56
- selections: fast=35, medium=49, slow=54
- legacy markers: 0

## Baseline
- parity missing: 15
- fixtures: 19
- selections: fast=26, medium=34, slow=35
- legacy markers: 8

## Deltas
- parity missing decrease: 15
- selection fast delta: 9
- selection medium delta: 15
- selection slow delta: 19

## Gates
- [PASS] parity_missing_decrease: missing delta = 15 (baseline 15 -> current 0, requires >= 6)
- [PASS] driver_selection_fast: fast delta = 9 (baseline 26 -> current 35, requires >= 8)
- [PASS] driver_selection_medium: medium delta = 15 (baseline 34 -> current 49, requires >= 10)
- [PASS] driver_selection_slow: slow delta = 19 (baseline 35 -> current 54, requires >= 12)
- [PASS] legacy_placeholder_markers_zero: legacy marker count = 0 (requires 0)
- [PASS] fixture_freshness: fixture file matches generator output
