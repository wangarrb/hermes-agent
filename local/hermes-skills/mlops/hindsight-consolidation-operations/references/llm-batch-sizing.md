# Hindsight consolidation LLM batch sizing

Session-derived rule from 2026-05-12: increasing `HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE` is not a free speedup.

## What was observed

- The live container had:
  - `HINDSIGHT_API_CONSOLIDATION_BATCH_SIZE=50`
  - `HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE=50`
  - `HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND=50` (official in Hindsight >=0.5.3 / 0.6.x; older local wrappers may still call this `MAX_MEMORIES_PER_JOB`)
- The earlier slowdown was caused by a combination of:
  - scope-level serial passes inside a batch,
  - missing source-marking on success,
  - and limited real batch cardinality.
- Once scope-level parallelism was fixed, `LLM_BATCH_SIZE=50` was already high enough; increasing it further would have had diminishing or negative returns.

## Risks of increasing it further

1. Longer prompt construction and LLM latency.
2. Higher token spend.
3. Bigger failure radius when a single call fails.
4. More chance of hitting context/output limits.
5. More mixed semantics per prompt, which can degrade observation quality.
6. Fewer batches overall, so concurrency can be underutilized.

## Tuning order

1. Verify source rows are marked consolidated/failed correctly.
2. Verify scope-level and batch-level parallelism are both actually active.
3. Verify the per-job fetch window is not too small to produce enough parallel work.
4. Only then try small `LLM_BATCH_SIZE` changes with a short A/B run.

## Rule of thumb

- 50 is a quality-first ceiling, not a universal optimum.
- If the goal is throughput, first tune job/window size and concurrency topology.
- If the goal is quality, keep batch size conservative and reduce prompt complexity instead of inflating batch size.
