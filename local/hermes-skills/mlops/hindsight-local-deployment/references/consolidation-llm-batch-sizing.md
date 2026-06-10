# Hindsight consolidation LLM batch sizing note

Session takeaway from 2026-05-12: increasing `HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE` is not a generic speed win.

## Current observed state

- Container env had:
  - `HINDSIGHT_API_CONSOLIDATION_BATCH_SIZE=50`
  - `HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE=50`
  - `HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND=50` (official in Hindsight >=0.5.3 / 0.6.x; older local wrappers may say `MAX_MEMORIES_PER_JOB`)
- The true earlier bottleneck was not just batch size; it also included scope-level serial execution and later the missing source-marking bug.
- After scope-level parallelization, `LLM_BATCH_SIZE=50` was already large enough that further increase would likely hurt more than help.

## Why not keep increasing it

1. Longer prompts → slower LLM calls.
2. Higher token cost.
3. Bigger failure radius when a single call errors.
4. Higher risk of hitting context/output limits.
5. More mixed semantics inside one prompt, which can reduce observation quality.
6. Fewer batches overall, which can reduce effective parallelism and leave workers underutilized.

## Practical tuning order

1. Verify source rows are being marked consolidated or failed.
2. Verify batch-level and scope-level parallelism are both actually active.
3. Verify the number of simultaneously runnable batches is not being capped by too-small job fetch windows.
4. Only then consider batch-size changes, and prefer small A/B steps.

## Parallelism-preserving tuning

`CONSOLIDATION_BATCH_SIZE` and `CONSOLIDATION_LLM_BATCH_SIZE` are different knobs:

- `HINDSIGHT_API_CONSOLIDATION_BATCH_SIZE` controls how many source facts are fetched into one consolidation round before splitting into LLM batches.
- `HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE` controls how many facts go into one LLM call.
- To increase effective parallelism without changing per-call prompt density, raise `CONSOLIDATION_BATCH_SIZE` while keeping `LLM_BATCH_SIZE` fixed. Example: with `LLM_BATCH_SIZE=50` and `PARALLEL_BATCHES=8`, a fetch window around 400 can expose up to 8 chunks of 50 facts to run concurrently, subject to tag-group distribution and scope ordering.
- Raising only `MAX_MEMORIES_PER_ROUND` increases total work in one operation, but if `CONSOLIDATION_BATCH_SIZE` remains 50 each fetch round may still expose only one 50-fact LLM batch and therefore not fill external parallel slots.
- Raising `LLM_BATCH_SIZE` reduces the number of runnable chunks and can lower effective parallelism; use it mainly when reducing LLM call count matters more than wall-clock parallelism.

## Rule of thumb

- 50 is a quality-first ceiling, not a universal optimum.
- If the goal is throughput, first tune fetch/job window size and concurrency topology.
- If the goal is better observation quality, keep batch size conservative and reduce prompt complexity instead of inflating batch size.
