# tinygrad Ralph PRD

## Goal
Keep tinygrad stable and minimal via small, tested, incremental tasks.

## Guardrails
- Follow `AGENTS.md` + `CLAUDE.md` (2-space indent, 150 char lines, no whitespace-only diffs).
- One task per iteration; keep diffs small and readable.
- Run relevant tests for any functional change.
- Use `uv` for Python package management.
- Do not commit unless explicitly asked.

## Task Backlog
T1: Test triage
- Run a quick smoke test (pick a focused test file or case).
- If failures appear, fix the root cause and add a regression test.
- If clean, mark done and move on.

T2: UOp/Schedule correctness
- Look for any obvious correctness gaps in schedule cache or graph rewrites.
- If you change behavior, add or update a test in `test/` to lock it in.

T3: Pattern matcher cost vs benefit
- Use `TRACK_MATCH_STATS=2` on a relevant benchmark.
- If a pattern has 0% match rate and non-trivial cost for this workload, consider a safe tweak or guard.
- Keep changes minimal and verify with the same benchmark.

T4: Docs + comments for behavior changes
- If behavior changes, update the closest doc or add a short comment.

## Done Criteria
- All tasks are complete or explicitly marked "skipped" with rationale.
- Progress log ends with `<promise>COMPLETE</promise>`.
