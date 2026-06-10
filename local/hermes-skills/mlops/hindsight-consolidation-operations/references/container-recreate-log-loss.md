# Container Recreate → Docker Log Loss

## The Problem

The Hindsight daily pipeline (`hindsight_daily_noagent.py` → `hindsight_memory_pipeline.py` → `hindsight_minimax_import.py::recreate_container()`) does **`docker rm -f` + `docker run`** to recreate the hindsight container with new patches/env. This destroys all docker logs from the previous container instance.

After recreate, `docker logs --since <pre-recreate-time>` returns **zero output** for timestamps before the new container's creation time.

## Impact on daily_stats.py

`daily_stats.py` collects Hindsight LLM token usage via:

```python
def docker_logs_since(start_dt: datetime) -> str:
    docker_cmd = f"docker logs --since {since} hindsight 2>&1"
```

If the container was recreated during the pipeline (which it always is, often 4+ times), `--since` for the daily_stats reporting window (08:30 ~ 08:30 previous day) covers timestamps before the last recreate. The docker daemon returns nothing for those timestamps because the container they belonged to no longer exists.

The daily_stats.py 2026-05-20 patch added `container_created_at()` clamping:

```python
def docker_logs_since(start_dt: datetime) -> str:
    created = container_created_at()
    effective_since = max(start_dt, created) if created else start_dt
```

This prevents requesting non-existent time ranges but cannot recover lost data.

## Observed Data Loss Example

- Pipeline ran 01:43~03:21 CST, produced 15 slow_llm_call events across reflect/consolidation/retain_extract_facts
- Container was recreated 4 times during pipeline (logs show `recreating Docker container: hindsight` x4)
- Final container created at 2026-05-20 02:39 CST
- daily_stats.py at 08:30 used `--since 2026-05-19T00:30:00+00:00`, but old container logs gone
- Result: only 5 reflect calls visible (those happening after last recreate), ~10 calls / ~500K tokens lost from daily report

## Mitigation Options

### Option A: Pipeline-level log dump (recommended for accuracy)

Before `recreate_container()` in `hindsight_minimax_import.py`, dump current docker logs to a file:

```python
import subprocess, json, time
if os.path.exists("/var/lib/docker/containers/..."):  # or just try
    dump_path = f"/tmp/hindsight-docker-log-dump-{int(time.time())}.json"
    subprocess.run(
        ["docker", "logs", "hindsight", "--tail", "all"],
        stdout=open(dump_path, "w"), stderr=subprocess.STDOUT, timeout=30
    )
```

But this doesn't help after the fact, and the dump path must be known to daily_stats.py.

### Option B: daily_stats.py reads from Hindsight DB instead

Hindsight DB has:
- `async_operations` table with type/status/timestamps (but not token counts)
- `memory_units` table for fact counts by type

Token-level precision is only available from docker logs. DB gives operation counts but not token costs.

### Option C: Accept the gap

The LLM token data is lost from the daily report but nothing else is affected (Hindsight DB state is complete). Accept that the "Hindsight LLM 用量" section in the daily report undercounts on days with container recreate.

## Detection

Check whether container was recreated in the stats window:

```bash
# Current container creation time
docker inspect hindsight --format '{{.Created}}'

# Pipeline log shows recreations
grep 'recreating Docker container' /home/wyr/.hermes/logs/hindsight-offline-pipeline/*.log

# Compare pipeline first-recreate time vs daily_stats window start
# If recreate_time > window_start → data loss occurred
```