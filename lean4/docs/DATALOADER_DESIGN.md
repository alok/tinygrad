# TinyGrad4 DataLoader Design (Lean)

This note defines the ordering and resume semantics for the multi-worker data pipeline.
It is intentionally operational and test-backed rather than proof-heavy.

## Goals

- Deterministic ordering given a fixed dataset + RNG key.
- Deterministic resume after interruption.
- Explicit ordering semantics (no "maybe" concurrency effects).
- Clear knobs for performance (buffer sizes, worker count).

## Terms

- **Base iterator**: `IteratorConfig` over a dataset (includes key/epoch/resume state).
- **Shard**: `ShardConfig` applied per worker (interleaved or contiguous).
- **Worker iterator**: `IteratorConfig` after sharding.
- **Multi prefetcher**: `MultiIteratorPrefetcher` over N worker prefetchers.

## Ordering Semantics (MultiIteratorPrefetcher)

MultiIteratorPrefetcher supports explicit ordering policies.

### Strict (default)

Strict ordering is **round-robin** across workers:

1. Visit workers in index order `0..N-1`.
2. Yield the next item from the current worker.
3. Move to the next worker (wrap around).
4. If a worker is exhausted, mark it done and skip it on future rounds.

This is deterministic even with uneven shard lengths (`dropRemainder=false`).
It also defines a clear global order for comparisons and tests.

### Best-effort (optional)

Best-effort ordering trades determinism for throughput. The policy:

- Scans workers non-blocking and yields the first available item.
- Enforces a bounded lead (`maxLead`) relative to the slowest active worker.
- Caps empty scans per round (`maxSkipsPerRound`) before blocking.

This mode is explicitly **not** deterministic across schedules.

## Resume Semantics

Checkpoint state is:

- `nextWorker`: the next worker index to attempt.
- `workerStates`: per-worker `IteratorState` after the last **consumed** item from that worker.
- `produced`: per-worker count of consumed items (used by best-effort policy).

On resume:

1. Rebuild each worker iterator from `workerConfig`.
2. Restore each iterator with its saved state.
3. Set `nextWorker`.
4. Discard any prefetched-but-unconsumed items (they will be regenerated).

This yields a **prefix property**: all items consumed before checkpoint are preserved,
and the resumed stream matches the baseline order from that point onward.

## Tests

- `MultiPrefetchResumeSmoke` compares multi-prefetch to a sequential
  round-robin baseline (no prefetch), then checks resume correctness at
  multiple split points.

## Performance Knobs (Current)

- `bufferSize`: per-worker prefetch queue size.
- `numWorkers`: number of shard workers.
- `ShardMode`: interleaved (better balance) or contiguous (better locality).
- `dropRemainder`: equal shard sizes vs full coverage.

## Known Tradeoffs

- Strict ordering can block if one worker is slow.
- We prefer determinism over throughput by default.
- Prefetched but unconsumed items are intentionally dropped on resume to keep correctness simple.

## Future Extensions

- Separate control of worker queue size vs consumer queue size.
- Backpressure propagation across stages (IO, transform, transfer).
