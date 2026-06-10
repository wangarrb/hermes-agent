# Pre-fetch Pipeline: Overlapping DB Fetch with LLM Waves

Added 2026-05-14 to the patched consolidator (`consolidator.py`).

## Problem

Without pre-fetch, every consolidation round follows this cycle:
1. Fetch unconsolidated memories from DB (~3-5s)
2. Group into LLM batches
3. Run all batches concurrently via `asyncio.gather` (~200s for 8-way)
4. Wait for all to finish
5. Next round fetches again

The fetch at the start of each round is **blocked** until the previous round's
LLM work is 100% complete. This idle gap adds up across many rounds.

## Solution: Background Pre-fetch

Replace the in-loop `fetch → batch → gather` pattern with a pipeline:

```
Round N starts:
  fetch + batch + launch 8 tasks (instant — prefetched in Round N-1)

Round N LLM runs:
  asyncio.create_task(_prepare_llm_batches(...))  ← background
  asyncio.gather(*tasks)                           ← blocking

Round N+1 starts:
  collect prefetched batches via `await next_prefetch`
  launch immediately — no DB wait
```

## Implementation

In `_run_consolidation_for_bank()`:

1. `_prepare_llm_batches(pool, fetch_limit)` — extracted helper that fetches
   from DB and groups by tags. Returns `(memories, batches)` tuple.

2. `next_prefetch: asyncio.Task | None` — tracks the background fetch task.

3. At the top of each iteration, if `next_prefetch is not None`, collect its
   result; otherwise do a synchronous fetch.

4. After launching the current round's batch tasks, immediately kick off the
   next round's fetch in the background.

5. Cancel `next_prefetch` on early-exit (cancelled, round limit hit).

## Observed Impact

From logs (2026-05-14):

| Without pre-fetch | With pre-fetch |
|---|---|
| recall=24~29s every round | first wave:~24s, second wave: ~2-4s |
| round gap ~3-5s | round gap <1s |

The second wave's recall time dropped from 24s to 2s because the DB fetch
happened during the first wave's LLM work. This saves ~22s per round × 30
rounds = ~11 minutes for a 1754-memory backlog.
