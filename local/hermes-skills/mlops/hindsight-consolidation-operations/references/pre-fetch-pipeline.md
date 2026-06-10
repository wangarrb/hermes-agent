# Pre-Fetch Pipeline Overlap

## What It Does

The patched consolidator (`consolidator.py`, 2026-05-14+) includes a
background DB pre-fetch mechanism.  While the current round's LLM batches
are running, a `asyncio.create_task` fetches and groups the *next* round's
unconsolidated memories.  When the current wave of tasks finishes, the next
round's data is already prepared — the initial `fetch_memories` DB call
(for the next round) is effectively zero-cost.

## Why It Matters

Without pre-fetch:

```
Round N:  [fetch 64 rows → 300ms] → [8 LLM batches → 220s] → gather returns
Round N+1: [fetch 64 rows → 300ms] → [8 LLM batches → 220s] → gather returns
```

With pre-fetch:

```
Round N:  [fetch 64 rows → 300ms] → [8 LLM batches → 220s]
                                          ↓ background fetch Round N+1 (300ms)
                                     gather returns → Round N+1 data ready
Round N+1: [8 LLM batches → 220s] → gather returns
```

The recall timing also drops significantly because the DB connection and
index lookups are already warm.

## Real-World Benchmark (2026-05-14)

| Metric | Without Pre-fetch | With Pre-fetch |
|--------|-------------------|----------------|
| Round 1 recall | 27.2s | 27.2s (cold DB) |
| Round 2 recall | 28.8s (fresh fetch) | **2.0s** (pre-fetched) |
| Round 2 total | 205s | **23s** (batch #9-#10 tail) |

Aggregate effect: ~15s saved per round on DB fetch+recall overhead,
compounding over 28+ rounds.

## Log Signature

Pre-fetch is active when the startup log includes `rate_limit_backoff=300s`:

```
[CONSOLIDATION] bank=hermes total_unconsolidated=1754 parallel_batches=8
llm_limit=8 recall_limit=60 llm_batch_size=8 rate_limit_backoff=300s
```

The `rate_limit_backoff=300s` line confirms the patched consolidator with
`AdaptiveLLMConcurrencyLimiter` + pre-fetch pipeline is loaded.

## Concurrent Safety

- The background fetch uses `acquire_with_retry(pool)` to get its own
  connection — it does not compete with in-flight LLM batch connections.
- `next_prefetch` is only kicked off when `round_remaining > len(memories)`,
  i.e., there will be at least one more round.
- If the current round hits the round limit, `next_prefetch.cancel()` is
  called and the pre-fetched data is discarded (cheap).
