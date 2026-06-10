# Log-based drain watcher under DB pool pressure

Use this pattern when Hindsight consolidation is actively draining but external monitor/psql probes intermittently fail with:

```text
FATAL: sorry, too many clients already
```

## Symptom

- `hindsight_observations_monitor.py` or direct `psql` queries stop updating or fail.
- Container logs still show active consolidation work, e.g.:
  - `[WORKER_TASK] ... type=consolidation ... stage=llm.openai.consolidation...`
  - `[CONSOLIDATION] ... llm_batch #N running ... observation-scope passes in parallel`
  - slow LLM call logs for `scope=consolidation`.
- API DB pool may show `pool: size=100 ... waiters=...`.
- Backlog counters move in steps only after a job/batch commits.

## Interpretation

Do not immediately restart the container. In heavy LLM-parallel consolidation, the API can occupy the PostgreSQL client limit, causing external probes to fail while the drain is healthy. Restarting can waste a long-running job.

A flat `unconsolidated_base` over short windows is not enough to declare a stall. Check container logs for active LLM/consolidation progress first.

## Safe watcher pattern

Prefer a non-DB watcher that tails the existing monitor JSONL and checks the drain process, rather than repeatedly opening new DB connections.

Recommended checks:

1. Read the latest valid JSON object from `$HERMES_HOME/logs/hindsight-observations/*monitor-live.jsonl`.
2. Track:
   - `unconsolidated_base`
   - `observations`
   - `failed_base`
   - `ops.processing:consolidation`
   - `ops.completed:consolidation`
   - log file mtime / staleness
3. Check `pgrep -af 'hindsight_observations_drain\.py'` for drain liveness.
4. Alert only when:
   - `failed_base > 0`, or
   - drain process exits while backlog is non-zero, or
   - log remains stale beyond a generous threshold and container logs show no active LLM/consolidation progress.
5. Treat `unconsolidated_base=0` and `failed_base=0` as the clean completion condition.

## Example session finding

During the 2026-05-12 drain:

- Monitor hit `too many clients already` and paused around 13:02.
- Container logs still showed active LLM calls and consolidation batches.
- Backlog later resumed visible progress:
  - `1152 -> 1081 -> 1031`
- A job completed with:
  - `processed=200`
  - `CONSOLIDATION COMPLETE: 2991.701s total`
  - next job queued automatically with `total_unconsolidated=1031`.

## Reporting

Tell the user explicitly:

- whether the drain is alive,
- whether backlog is falling by job commits,
- whether failed count remains zero,
- whether monitor gaps are due to DB pool pressure rather than a dead drain.
