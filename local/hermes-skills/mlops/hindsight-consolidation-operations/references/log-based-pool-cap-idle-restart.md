# Log-based pool-cap idle restart pattern

Session-derived detail from the 2026-05-12 Hindsight offline consolidation drain.

## When to use

Use this when:

- heavy consolidation is still active,
- external `psql`/tmux monitor repeatedly reports `FATAL: sorry, too many clients already`,
- container logs show Hindsight API pool near Postgres capacity (`pool: size=75..100 limits=5-100 ... waiters=...`),
- you need a DB pool cap/restart but must not interrupt the active consolidation job.

## Pattern

1. Stage the cap first, do not restart immediately.
   - In this session the immediate hotfix was changing `DEFAULT_DB_POOL_MAX_SIZE = 100` to `60` in the container source and host-side source copy.
   - Run `python3 -m py_compile` on both copies.
   - Confirm the container file contains the new cap.

2. Identify the current active consolidation op from logs, not DB:
   - `docker logs --since 2h hindsight | grep 'type=consolidation'`
   - Capture the latest `op=<uuid> type=consolidation`.

3. Stop only external trigger/drain drivers:
   - terminate `hindsight_observations_drain.py` and any completion watcher that may queue the next consolidation op.
   - Do not kill the API worker or container while the active op is still running.

4. Wait for the active op to finish using logs:
   - look for `Marked async operation as completed: <op>` or `Marked async operation as failed: <op>`.
   - Avoid polling Postgres while it is saturated.

5. After current op finishes:
   - `docker restart -t 30 hindsight`
   - wait for `/health` to report healthy/database connected,
   - restart the external drain driver in tmux.

## Why this works

The active consolidation job lives inside the API worker. Killing the external drain script does not stop the current job; it only prevents automatic queuing of a new job. This creates a safe idle window for applying staged config changes without wasting the LLM work already in progress.

## Pitfalls

- Do not confuse monitor starvation with drain failure. Validate via container logs before restarting.
- Do not leave the drain trigger running while waiting for idle; it can enqueue a new op immediately after the current one finishes, eliminating the safe restart window.
- A source patch inside the container writable layer is not durable across container recreation; preserve it in the host-side patch/wrapper path if needed.
- Lowering the pool cap trades fewer external monitor failures for potentially slower DB phases. Start with 60; consider 40 only if starvation persists.
