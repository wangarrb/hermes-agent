# Parallel consolidation drain reference

Session-derived detail for Hindsight offline/session consolidation when a large backlog is LLM-bound.

## Symptoms observed

- Drain had stopped earlier because of an 8h hard timeout and wrote a final report; this was not a crash.
- After restart, Hindsight showed one long-running `processing:consolidation`, no failed base units, and a large unconsolidated base backlog.
- LLM calls inside consolidation were the bottleneck; individual jobs took hundreds of seconds.

## Chosen design

- Do not interrupt active consolidation in most cases.
- **In production, direct restart is the standard practice** (`docker restart hindsight` without waiting). The `patch_hindsight_container_and_restart()` function does exactly this. The re-queue gap between consolidation rounds is sub-second and unreliable to catch with polling-based watchdogs.
- After restart, recover stuck `processing` operations by marking them `failed` via asyncpg or the `/consolidation/recover` API endpoint.
- The idle-watcher approach (poll for `processing==0` then restart) is only used when the user explicitly chooses conservative handling.

Parallelization shape:

1. Select/claim a work set once.
2. Split into LLM batches.
3. Run LLM batches concurrently.
4. Serialize all observation writes behind a lock.
5. Protect write actions:
   - create: exact text + tags/scope dedupe before insertion.
   - update/delete: lock the target row where possible.
6. Keep a conservative default concurrency (`3` in this session) configurable by env var.

## Important cautions

- Keep mental model refresh disabled during large drains.
- Avoid global proxy env in Hindsight unless required and verified.
- Do not edit `.env` or add credentials automatically.
- Container-layer patches survive simple restart but can be lost on container recreation; keep a host-side source/patch path for reapply.

## Minimal verification checklist

- Host `py_compile` passes.
- Container `py_compile` passes.
- API health returns healthy after restart.
- `async_operations` transitions normally.
- `unconsolidated_base` decreases.
- `failed_base` remains zero or is explained.
- Observations increase without duplicate explosion.
