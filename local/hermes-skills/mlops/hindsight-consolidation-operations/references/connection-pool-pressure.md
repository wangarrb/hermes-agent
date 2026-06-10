# Hindsight consolidation DB connection pressure

Session-derived note from 2026-05-12 offline drain.

## Symptom

During high-parallel consolidation, external status queries failed with:

```text
psql: FATAL: sorry, too many clients already
```

At the same time, the API remained healthy (`/health` database connected), but `lsof -iTCP:5432` showed the Hindsight API process holding about 100 established PostgreSQL connections. Poller logs showed pool lines such as:

```text
pool: size=100 limits=5-100 idle=98 in_use=2 waiters=98
```

So the failure was not just a bad psql command; the service had saturated Postgres connection capacity while consolidation was still alive.

## Interpretation

- A flat or failing monitor is not proof the drain is dead.
- Use API health, container logs, and observation timestamps to distinguish liveness from external monitor starvation.
- If `observations` continue increasing and `failed_base` stays 0, prefer waiting over killing the job.
- If the job naturally reaches idle, a restart can shrink the pool back down; in this session it returned briefly to size 5 after restart.

## Operational pattern

1. Check process state and container health first.
2. Check `docker logs hindsight` for active `CONSOLIDATION` / `WORKER_TASK` lines.
3. Use `lsof -nP -iTCP:5432` or `ss` to confirm whether API-owned connections dominate.
4. Make watcher/monitor scripts tolerate psql failure and retry instead of exiting.
5. When safe idle is reached, restart the container to reset a bloated pool before starting the next drain.
6. If monitor starvation recurs, stage a lower API DB pool cap before the next restart:
   - patch host/container source default such as `DEFAULT_DB_POOL_MAX_SIZE = 60` (or equivalent env/config if available),
   - compile-check host and container copies,
   - stop the external drain trigger process so no new consolidation op is queued,
   - wait for the active op id from container logs to be marked completed/failed,
   - then restart the container and resume drain.
7. Report this as “connection-pool pressure while still progressing”, not as a DB outage, unless `/health` or the worker actually fails.
