# TinyGrad4 AutoResearch

This is the TinyGrad4 adaptation of the `~/autoresearch` idea.

- `ralph/PRD.md` plays the role of `program.md`: it is the research brief for the agent loop.
- `ralph/progress.txt` is the running log that accumulates iteration notes.
- `lake exe tg4_autoresearch` is the Lean-native runner that drives Codex or Claude against those files.

The runner is intentionally small. It does not try to own experiment state or rewrite your PRD. It just:

1. Resolves a prompt from `PRD.md` and `progress.txt`.
2. Invokes `codex` or `claude`.
3. Repeats for `N` iterations.
4. Stops early if the runner emits `<promise>COMPLETE</promise>`.

## Quick start

From the repo root:

```bash
lake exe tg4_autoresearch --prompt-only
lake exe tg4_autoresearch --runner codex --iterations 1
lake exe tg4_autoresearch --runner codex --iterations 12 --sleep 10
```

If you prefer the wrapper:

```bash
./scripts/tg4-autoresearch.sh --runner codex --iterations 12 --sleep 10
```

## Useful flags

- `--runner claude` switches to Claude.
- `--runner-arg ...` appends extra runner args and can be repeated.
- `--resume` or `--resume-id ...` resumes a Codex exec session.
- `--no-linear` removes the default Linear guidance from the generated prompt.
- `--prompt-template path/to/template.md` uses a custom template with `{{PRD}}` and `{{PROGRESS}}`.
- `--prompt-flag ""` omits the prompt flag for non-Codex runners that expect a positional prompt argument.

## Notes

- The runner defaults to `codex`.
- It executes from the repo root when launched through `scripts/tg4-autoresearch.sh`, so `@ralph/PRD.md` style prompt references resolve correctly.
- Keep `ralph/PRD.md` focused on one current research arc. The runner is generic, but the loop quality depends on the PRD staying current.
