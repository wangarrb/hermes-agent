# Consolidation Throughput Benchmarks

Validated 2026-05-14 on full pipeline verification run.

## Hardware / Setup

- Hindsight v0.6.1 Docker (ghcr.io/vectorize-io/hindsight:0.6.1)
- Postgres pg0 backend, local BGE-M3 embeddings
- MiniMax-M2.7 provider, 8 concurrent LLM slots
- Patched consolidator with pre-fetch pipeline and AdaptiveLLMConcurrencyLimiter

## Full-pipeline Drain (64x8 default profile)

```
CONSOLIDATION_BATCH_SIZE=64
CONSOLIDATION_LLM_BATCH_SIZE=8
CONSOLIDATION_MAX_MEMORIES_PER_ROUND=64
CONSOLIDATION_PARALLEL_BATCHES=8
CONSOLIDATION_RECALL_MAX_CONCURRENT=60
CONSOLIDATION_LLM_MAX_CONCURRENT=8
```

| Metric | Value |
|--------|-------|
| Input memories | 1,754 |
| Output observations | 14,350 |
| Wall time | 2h 20min |
| Throughput | ~0.21 memories/sec |
| LLM calls (estimated) | ~440 |
| Total input tokens (estimated) | ~11M |
| MiniMax cost (estimated) | ~$6 |
| 429 rate limit events | 0 |
| Serial equivalent | ~16h |
| Speedup | ~8x |

## Per-round timing

| Phase | Duration | Notes |
|-------|----------|-------|
| DB fetch (round 1) | ~24s | Sequential; pre-fetch not yet active |
| LLM wave (8 batches) | ~170-220s | Determined by slowest batch |
| DB fetch (round 2+) | ~2s | Pre-fetch overlap eliminates latency |
| Observation write | <2s | Serialized, negligible |
| Re-queue gap | <1s | Sub-second; poll-waiting infeasible |

## Per-batch sample (single round)

```
batch_parallel=8 llm=8/8 recall=60
llm_batch #1: 8 memories, 3 LLM calls, 205s
llm_batch #2: 8 memories, 3 LLM calls, 186s
llm_batch #3: 8 memories, 3 LLM calls, 191s
llm_batch #4: 8 memories, 3 LLM calls, 214s  ← slowest
llm_batch #5: 8 memories, 3 LLM calls, 221s  ← wall-time anchor
llm_batch #6: 8 memories, 3 LLM calls, 169s
llm_batch #7: 8 memories, 3 LLM calls, 205s
llm_batch #8: 2 memories, 3 LLM calls, 130s
Total: 58 memories processed
```

Input tokens range: 4K-35K per batch (depends on tag × observation_scopes complexity).
LLM calls per batch: 1-3 (depends on observation_scopes configuration).

## Small-batch drain (3-way, post-recreate)

```
CONSOLIDATION_PARALLEL_BATCHES=3
CONSOLIDATION_BATCH_SIZE=20
CONSOLIDATION_LLM_BATCH_SIZE=20
```

| Metric | Value |
|--------|-------|
| Input memories | 67 |
| Wall time | ~6min |
| 429 events | 0 |

## Pre-fetch effectiveness

In the first round, recall takes ~24s for each batch (DB query runs at batch start). After pre-fetch kicks in, recall drops to ~2s per batch (DB query completed during previous LLM wave). Log signature:

```
rate_limit_backoff=300s  ← indicates pre-fetch enabled
```

## Cost Model

Parallelism reduces wall time but NOT LLM call count. The same number of LLM calls are made regardless of concurrency. Cost per drain is determined by total memory count × tag complexity, not by parallelism level. Key insight:

```
Cost = f(memories, observation_scopes)  # identical at 3-way and 8-way
Wall time = Cost / parallelism          # linear speedup
```
