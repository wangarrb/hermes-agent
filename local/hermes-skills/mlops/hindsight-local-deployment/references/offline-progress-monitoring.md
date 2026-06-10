# Hindsight offline progress monitoring notes

Session context: 2026-05-11 all-history offline reflect/consolidation using Bailian/DashScope GLM-5.

## When to use

Use this reference when a long Hindsight offline retain/reflect/consolidation job is running and the user asks for a progress bar, ETA-ish status, provider call-rate sanity check, or whether a restart/concurrency change harmed progress.

## Practical progress signal

For offline reflect/consolidation, the most reliable live progress comes from combining:

1. Latest `~/.hermes/logs/hindsight-offline-reflect/*submit*.log`
2. Existing markdown outputs under `~/.hermes/hindsight/offline_reflect/daily/<YYYY-MM-DD>/`
3. `~/.hermes/hindsight/offline_reflect/offline_reflect_progress.json`
4. Production Hindsight DB counts from `.hindsight-docker` pg0, not stale `~/.pg0`

The session-created helper script is:

```bash
python3 $HOME/.hermes/scripts/hindsight_progress_bar.py
```

It prints a compact terminal-safe report, e.g.:

```text
HINDSIGHT_PROGRESS 2026-05-11 15:32:29
状态: running
并发: start=16 min=8 429_backoff=300s
Daily backfill days: [████████████░░░░░░░░] 6/10 partial=0
Known unit outputs: [████████████████░░░░] 21/26
当前: daily 2026-04-22 facts=242 units_saved=0/5 started=5/5 chars=79469
429: 0
DB: batch_retain:completed=436, consolidation:completed=2, retain:completed=727, docs=992 units=3106
```

If the script is missing, recreate a minimal version that:
- reads the latest `*bailian-history-submit-c16.log` or `*bailian-history-submit*.log`
- parses the `backfilling missing/incomplete daily outputs before weekly: ...` line
- parses `Daily period=...` + `Units: N total_chars=M`
- counts actual `*.md` outputs per daily directory
- queries production DB via `$HOME/.hindsight-docker/installation/18.1.0/bin/psql` and instance config
- redacts/never prints secrets

## Concurrency switch pattern

To raise offline reflect concurrency without duplicate submission:

1. Identify the current wrapper and underlying offline script processes:
   ```bash
   pgrep -af 'hindsight_minimax_import|offline_hindsight_reflect_consolidate|offline-reflect-llm'
   ```
2. Stop only the local wrapper/offline script processes, not PostgreSQL or the Hindsight container/worker, unless intentionally restarting provider config.
3. Restart with the same scope and `--allow-existing-queue`, higher concurrency, and an adaptive floor:
   ```bash
   HINDSIGHT_PSQL=$HOME/.hindsight-docker/installation/18.1.0/bin/psql \
   HINDSIGHT_OFFLINE_LLM_CONCURRENCY=16 \
   python3 $HOME/.hermes/scripts/hindsight_minimax_import.py offline-reflect-llm \
     --llm-profile bailian --allow-existing-queue -- \
     --scope both --mode submit \
     --weekly-source daily --weekly-window all-history --backfill-missing-daily \
     --llm-label bailian --concurrency 16 --min-concurrency 8
   ```
4. Interpret old process exit code `143` as expected SIGTERM if you intentionally stopped it.
5. Check new logs for real 429 lines (`429 rate limited`, `429 encountered`, `Too Many Requests`) before declaring the provider saturated.

Important nuance: `--concurrency 16` is an upper bound; many daily batches contain only 1–7 units, so the process may not actually issue 16 simultaneous LLM calls until a larger weekly/history batch.

## Hermes display limitations

Hermes CLI can show a text progress bar when asked, by running the helper script and returning its output. It does not natively refresh one line like `tqdm` inside the current chat without modifying Hermes TUI/statusbar code.

Cron with `deliver=local` is not a reliable way to print periodic updates into the current interactive CLI session. For automatic notifications, prefer delivery to a messaging platform. For manual CLI checks, run the progress script on demand.

## Safety notes

- Do not print API keys or passwords from `instance.json`, env, or logs.
- Do not mutate production Hindsight data while monitoring.
- Do not resubmit the same manifest/reflect scope just to change provider/concurrency; stop/restart the local wrapper and preserve progress/queue state.
- DB `operation_type` counts may show retain/batch_retain increments after posting offline consolidation markdown because Hindsight indexes those posted documents through retain-like async operations; distinguish offline LLM progress from native Hindsight ingestion progress.
