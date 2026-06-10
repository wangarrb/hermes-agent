# Hindsight retain throughput and queue diagnostics

Session note: 2026-05-10 paid session-manifest retain backfill on bank `hermes`.

## What was observed
- Queue was progressing normally but slowly: `completed` kept rising, `pending` kept falling, `failed=0`.
- Worker slots were saturated: `processing=4`, with the worker reporting `slots=4/4`.
- Real bottleneck was not only queue length; individual retain tasks were heavy and took long wall-clock time.
- Recent completed retain/batch_retain tasks in a 3-hour window averaged about 78–82 minutes wall-clock per task.
- The worker logs showed frequent LLM calls taking tens of seconds each, plus occasional JSON parse retry warnings that recovered without failing the task.
- There were many `batch_retain` parent rows left pending with `task_payload IS NULL`; these are bookkeeping rows and not equivalent to claimable work.

## Useful checks
- Queue counts:
  - `SELECT operation_type, status, count(*) FROM async_operations GROUP BY 1,2 ORDER BY 1,2;`
- Claimable retain backlog:
  - `SELECT count(*) FROM async_operations WHERE status='pending' AND operation_type='retain' AND task_payload IS NOT NULL AND (next_retry_at IS NULL OR next_retry_at <= now());`
- Non-claimable parent rows:
  - `SELECT count(*) FROM async_operations WHERE status='pending' AND operation_type='batch_retain' AND task_payload IS NULL;`
- Processing age:
  - `SELECT operation_id, operation_type, created_at, updated_at FROM async_operations WHERE status='processing' ORDER BY created_at ASC;`
- Recent throughput:
  - measure `completed` delta over 10 minutes to estimate drain ETA.

## Interpretation
- If `processing` is at the slot limit and `completed` keeps increasing, the system is working, just saturated.
- If `pending` is large but mostly `batch_retain` rows with `task_payload IS NULL`, do not treat them as the true backlog.
- If `processing` rows stay alive but `updated_at` advances, the worker is still making progress even when single tasks are slow.
- A single JSON parse warning from the LLM is not necessarily a failure if the task retries and completes.
- Host port checks can be misleading: the worker may be healthy even when `127.0.0.1:8000` is not mapped on the host.

## Session-specific numbers
- Initial snapshot: `completed=263`, `pending=780`, `processing=4`.
- Later snapshot: `completed=437`, `pending=606`, `processing=4`.
- Documents: `144 -> 304`.
- Memory units: `992 -> 2506`.
- Estimated remaining time at one snapshot: about 4.1 hours at ~2.5 completed ops/min.
