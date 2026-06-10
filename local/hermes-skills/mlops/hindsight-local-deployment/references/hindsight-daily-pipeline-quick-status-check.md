# Hindsight Daily Pipeline Quick Status Check

## When to use

User asks "did yesterday's offline pipeline run successfully?" or "is the daily hindsight pipeline working?"

## Step-by-step diagnostic path

### 1. Check the concise summary log

```bash
ls -la /home/wyr/.hermes/logs/hindsight_daily_YYYYMMDD_*.log
cat /home/wyr/.hermes/logs/hindsight_daily_YYYYMMDD_*.log
```

This is the outer wrapper output. It shows:
- Exit code (0 = success)
- Elapsed time
- Changes: docs/obs/nodes deltas
- "Pipeline completed successfully" or error message
- Path to the full log

### 2. Check the full offline pipeline log

```bash
ls -la /home/wyr/.hermes/logs/hindsight-offline-pipeline/YYYYMMDD-*.log
```

The naming pattern is `YYYYMMDD-HHMMSS-daily-noagent.log`. There are typically two per day:
- `0001xx` — the cron-triggered run (11:45 AM cron → wrapper → pipeline)
- `1145xx` — same day's second run if configured

### 3. Check for errors in the full log

```bash
grep -iE "error|fail|exception|traceback" /home/wyr/.hermes/logs/hindsight-offline-pipeline/YYYYMMDD-HHMMSS-daily-noagent.log | head -30
```

### 4. Check the cron schedule

```bash
crontab -l | grep hindsight
```

Current production cron: `45 11 * * * /home/wyr/.hermes/scripts/run_hindsight_daily_detached.sh`

### 5. Check live container for real-time issues

```bash
# Container status
docker ps -a --filter "name=hindsight" --format "{{.Names}}\t{{.Status}}\t{{.CreatedAt}}"

# Recent errors in the running container
docker logs hindsight 2>&1 | grep -iE "error|fail|Auth err" | tail -20
```

## Key distinctions to make

1. **Historical `failed_operations` ≠ today's pipeline failure.** The `/stats` endpoint shows cumulative `failed_operations` (e.g. 30). These are historical and do NOT mean the current pipeline failed. Check the daily log exit code instead.

2. **Live container LLM auth errors ≠ offline pipeline failure.** The running Hindsight container may show 401 errors from `openai/deepseek-v4-flash` (topenrouter/ChinaDataPay) during real-time retain/reflect calls. The offline pipeline uses its own LLM config (MiniMax or whatever `pipeline_config.json` specifies) and may succeed independently.

3. **`hindsight-old-*` containers are previous-version backups.** They show `Exited (0)` and are not the current pipeline. Don't confuse their logs with the active container.

4. **`Hindsight JSON/backoff patch not applicable` is expected on v0.6.1.** This message means the old v0.5.2-era patch was correctly skipped because the upstream code changed. Not an error.

5. **Container env vs. pipeline config.** The container's `HINDSIGHT_API_RETAIN_LLM_*` env vars (visible via `docker exec hindsight env`) may differ from the offline pipeline's `pipeline_config.json` LLM profile. The offline pipeline can use a different provider/model than the live container.

## Quick one-liner status check

```bash
# Today's pipeline result
cat /home/wyr/.hermes/logs/hindsight_daily_$(date +%Y%m%d)_*.log 2>/dev/null || echo "No daily log for today"

# Yesterday's pipeline result
cat /home/wyr/.hermes/logs/hindsight_daily_$(date -d yesterday +%Y%m%d)_*.log 2>/dev/null || echo "No daily log for yesterday"
```
