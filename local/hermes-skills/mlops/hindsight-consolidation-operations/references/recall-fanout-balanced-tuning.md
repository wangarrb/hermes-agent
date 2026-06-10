# Recall fanout and balanced consolidation tuning

Session-derived lesson from a Hindsight offline consolidation tail drain.

## Symptom

After provider switch and parallel consolidation patches, backlog still moved slowly and consolidation ops failed with:

- `Failed to search memories`
- `asyncpg TimeoutError`
- PostgreSQL/API pool near saturation (`pool size=60`, high `in_use`/`waiters`)

The user correctly suspected the code/parameters might be wrong: pure serial scope passes were stable but too slow; high parallelism was fast in theory but unstable.

## Source-code finding

Upstream `consolidator.py` already parallelizes recall inside a batch:

```python
recall_tasks = [
    _find_related_observations(...)
    for m in memories
]
per_fact_recalls = await asyncio.gather(*recall_tasks)
```

So a single `_process_memory_batch(memories=N)` launches roughly `N` concurrent recall/search calls before the LLM.

If an outer patch also parallelizes observation-scope passes or same-level LLM batches, effective DB/search fanout is approximately:

```text
recall_fanout ~= consolidation_llm_batch_size * active_scope_or_batch_parallelism
```

Examples from the session:

- `50 * 6 ~= 300` recall/search calls: overloaded DB pool 60 and caused search timeouts.
- `50 * 1 ~= 50`: stable but too serial; scope passes became slow.
- Balanced proposal: `25 * 2 ~= 50`: same DB fanout ceiling as stable mode, but two LLM/scope passes can overlap.
- Possible aggressive test: `20 * 3 ~= 60`: near DB pool limit; monitor carefully.

## Practical tuning rule

Do not tune `parallel_batches` in isolation. Keep estimated recall fanout at or below the effective DB pool headroom.

Recommended progression:

1. Stabilize: `consolidation_batch_size=50`, `consolidation_llm_batch_size=50`, `max_memories_per_round=50`, external `parallel_batches=1`.
2. If stable but too slow, use conservative balanced mode: `25/25/max_round=50/parallel_batches=2`.
3. Current user-selected default for offline/session consolidation drains: `20/20/max_round=60/parallel_batches=3` (recall fanout target ~60). This is the normal speed profile unless failures repeat.
4. `25/25/max_round=75/parallel_batches=3` with API DB pool raised to 80 is an aggressive tail-drain profile, not a safe default. In the 2026-05-12 run it eventually cleared the remaining backlog, but produced one `Failed to search memories` / `asyncpg TimeoutError` retry, high CPU (~1100%), and large pool waiters before succeeding. Use only for short tail drains with active monitoring; revert to `20x3`, `25x2`, or lower if failures repeat.
5. Avoid `50/50/parallel_batches>=2` unless DB pool and search latency are proven safe.
6. Never return blindly to `parallel_batches=8`; source-level fanout makes this much larger than it looks.

## Runtime / restart notes

- Tuning JSON is read at consolidation job start; an already-running op may keep old runtime behavior.
- If the Python module code changed, container restart is required.
- Safe pattern:
  1. copy/compile patched file in container;
  2. write tuning JSON;
  3. stop external `hindsight-obs-drain` tmux so it cannot queue another old-runtime job;
  4. wait until `processing:consolidation=0`;
  5. `docker restart -t 30 hindsight`;
  6. health check;
  7. restart drain tmux.

## Verification log signatures

Expected stable/balanced signatures:

- Balanced mode: logs should show `job_limit=2, local_limit=2`.
- Stable mode: `job_limit=1, local_limit=1`.
- Backlog (`unconsolidated_base`) should drop after job commit.
- `failed_base` should remain 0.
- If `Failed to search memories` repeats, reduce estimated recall fanout or inspect DB search queries/indexes before increasing LLM concurrency.

## Reporting guidance

When the user asks why it is slow, be explicit:

- Hindsight already has inner recall parallelism.
- Outer LLM/scope parallelism multiplies DB/search fanout.
- Provider speed only helps after recall/search completes.
- Use a fanout budget, not a raw concurrency number.
