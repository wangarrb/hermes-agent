# Hindsight Offline Full-Cycle Verification

This note captures the safest way to verify a full paid/offline cycle after starting a manual trigger or cron job.

## What to run

Start the full cycle via the cron runner:

```bash
python3 ~/.hermes/scripts/hindsight_offline_cron_runner.py both \
  --llm-profile minimax \
  --date-mode auto \
  --week-mode current \
  --prefilter safe \
  --poll 60 \
  --timeout 0 \
  --lock-timeout 21600
```

The runner should:
- import SQLite incrementally with the paid LLM profile
- run daily offline reflect/consolidation
- publish V2 canonical output with explicit confirmation
- run weekly all-history consolidation
- publish again if the gate still passes
- restore normal-local mode at the end

## Verification checklist

1. Confirm the process started and is still running or exited cleanly.
2. Inspect the latest pipeline log under:
   - `~/.hermes/logs/hindsight-offline-pipeline/`
3. Check the final summary JSON:
   - `latest-both.json`
   - `summaries/<timestamp>-both.json`
4. Confirm the summary has:
   - `status: ok`
   - `anomaly_count: 0`
   - a valid `log_path`
5. Confirm Hindsight returned to normal-local:
   - provider `ollama`
   - `HINDSIGHT_API_ENABLE_OBSERVATIONS=false`
   - queue drained
6. Confirm the latest V2 rebuild state shows:
   - `published: True`
   - `decision: eligible_for_local_proposal`
   - `errors: []`

## Expected log phases

A healthy run usually looks like:

1. `DAILY target_date=... profile=minimax`
2. SQLite import phase
3. `queue drained`
4. `restoring normal-local mode...`
5. `RUN ... offline-reflect-llm --scope daily ...`
6. `Submit complete: ok=... failed=0`
7. `queue drained`
8. `restoring normal-local mode...`
9. `RUN ... hindsight_offline_v2_rebuild.py --mode publish --confirm-publish publish-hindsight-v2-canonical`
10. `WEEKLY target_week=... profile=minimax`
11. `RUN ... wait-queue`
12. `weekly all-history` units and submissions
13. `queue drained`
14. `restoring normal-local mode...`
15. final `status: ok`

## Common pitfall

Do not confuse the temporary `normal-local` restoration between stages with failure. The import wrapper intentionally restores local mode after each paid-LLM segment so the next stage can re-enter cleanly.

## Anomaly signals

Escalate if the log shows:
- `JSON parse error`
- `STUCK`
- `Traceback`
- repeated `429`
- `failed > 0`
- non-zero `anomaly_count`
- queue not drained at the end

## Notes

- The V2 publish step now requires explicit confirmation: `--confirm-publish publish-hindsight-v2-canonical`.
- If the publish step is missing the confirmation string, the run may finish local validation but skip actual writeback.
- For full historical coverage, prefer `both` over only `daily`.
