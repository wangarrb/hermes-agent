# Offline daily + V2 publish runbook (2026-05-07)

This session validated the current end-to-end offline maintenance flow for Hindsight.

## Safe preflight

1. Check live state first:
```bash
python3 ~/.hermes/scripts/hindsight_minimax_import.py status
```

Expect:
- `health: {"status":"healthy","database":"connected"}`
- `pending_operations: 0`
- `processing_operations: 0`
- `failed_operations: 0`

2. Prefer background execution for the full daily pipeline:
```bash
python3 ~/.hermes/scripts/hindsight_offline_cron_runner.py daily \
  --date-mode today \
  --prefilter safe \
  --poll 30 \
  --timeout 0
```

## What the daily runner does

The runner executes, in order:
1. `sqlite-import-llm --mode submit --group-by day-topic`
2. `offline-reflect-llm --scope daily --daily-source facts --mode submit`
3. `hindsight_offline_v2_rebuild.py --mode publish --confirm-publish publish-hindsight-v2-canonical`
4. `hindsight_minimax_import.py status`
5. automatic restore to `normal-local`

## What to watch during a long run

- Use the log file, not the immediate command output, as the source of truth.
- Temporary front-end timeout does not imply failure; the process may still be running.
- The import window can stay in `import-minimax` for many minutes while retain/consolidation tasks drain.
- Verify progress by tailing the log and checking `pending_operations` / `processing_operations` in the status command.

## Success criteria

At completion, confirm:
- `status: ok`
- `anomaly_count: 0`
- `pending_operations: 0`
- `processing_operations: 0`
- provider restored to `ollama`
- `enable_observations: false`

Useful artifacts:
- daily log: `~/.hermes/logs/hindsight-offline-pipeline/<timestamp>-daily.log`
- daily summary: `~/.hermes/logs/hindsight-offline-pipeline/summaries/<timestamp>-daily.json`
- V2 rebuild summary: `~/.hermes/hindsight/offline_reflect/v2_rebuild/latest.json`

## Practical pitfall

If the run looks “stuck” but logs still show new async posts or queue drain progress, do not restart immediately. Let the runner finish the queue drain and the automatic `normal-local` restore. In this workflow, a foreground tool timeout is often just a monitoring limit, not a pipeline failure.
