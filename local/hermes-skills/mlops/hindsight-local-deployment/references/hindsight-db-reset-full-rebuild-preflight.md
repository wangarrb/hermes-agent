# Hindsight DB Reset + Full Rebuild Preflight

Use this reference when the user asks to reset Hindsight and rebuild from Hermes SQLite/session history.

## Why this exists

A Hindsight database reset deletes the active memory data plane: documents, memory units/facts, observations, links, entities, and async operation history. SQLite raw sessions and local offline artifacts are separate, but the active Hindsight bank becomes empty until retain/rebuild finishes.

Always treat this as a destructive operation even if the user sounds casual.

## Required sequence

1. Inspect current state before changing anything:
   - `python3 ~/.hermes/scripts/hindsight_minimax_import.py status`
   - `curl -s http://127.0.0.1:8888/health`
   - `curl -s http://127.0.0.1:8888/v1/default/banks/hermes/stats`
   - DB counts:
     ```bash
     $HOME/.pg0/installation/18.1.0/bin/psql -h /tmp -p 5432 -U hindsight -d hindsight -Atc \
       "SELECT 'documents='||COUNT(*) FROM documents WHERE bank_id='hermes';
        SELECT 'memory_units='||COUNT(*) FROM memory_units WHERE bank_id='hermes';
        SELECT 'async='||operation_type||':'||status||':'||COUNT(*)||':payload_null='||COUNT(*) FILTER (WHERE task_payload IS NULL)
        FROM async_operations GROUP BY operation_type,status ORDER BY operation_type,status;"
     ```

2. Determine whether current pipeline is incremental or full:
   - Read `~/.hermes/hindsight/sqlite_import_progress.json`.
   - If it has `last_imported_timestamp`, current submit path is incremental unless `--full` is passed.
   - **Critical full-rebuild invariant**: `--full` must ignore both the incremental cutoff and the historical `processed` document-id list. After a DB reset, reusing old `processed` ids will silently skip bundles that no longer exist in Hindsight. If the importer exposes `progress_for_run(full=True)`, verify it returns an empty progress object before reset; otherwise inspect/patch the importer first.
   - Run both dry-runs when deciding reset strategy:
     ```bash
     # remaining incremental work
     python3 ~/.hermes/scripts/import_sqlite_to_hindsight.py \
       --mode dry-run --group-by day-topic --prefilter safe --retain-chunk-size 8000 --sample-report 0

     # full rebuild size/cost estimate
     python3 ~/.hermes/scripts/import_sqlite_to_hindsight.py \
       --mode dry-run --full --group-by day-topic --prefilter safe --retain-chunk-size 8000 --sample-report 0
     ```
   - Report sessions, records, skipped counts, bundles, total chars, and estimated retain chunks. Do not summarize only bundle count; retain chunks drive LLM cost.

3. Backup before any destructive action:
   ```bash
   TS=$(date +%Y%m%d-%H%M%S)
   BACKUP_ROOT=$HOME/.hindsight-docker/backups/reset-pre-full-rebuild-$TS
   mkdir -p "$BACKUP_ROOT"

   $HOME/.pg0/installation/18.1.0/bin/pg_dump \
     -h /tmp -p 5432 -U hindsight -d hindsight -Fc \
     -f "$BACKUP_ROOT/hindsight-pre-reset.dump"
   sha256sum "$BACKUP_ROOT/hindsight-pre-reset.dump" > "$BACKUP_ROOT/hindsight-pre-reset.dump.sha256"
   sha256sum -c "$BACKUP_ROOT/hindsight-pre-reset.dump.sha256"
   ```
   Also capture metadata/config/progress files when useful:
   - `~/.hermes/hindsight/config.json`
   - `~/.hermes/hindsight/sqlite_import_progress.json`
   - latest v2 rebuild/gate summaries

4. Explain destructive scope and ask for explicit confirmation.
   Required confirmation phrasing should be unambiguous, e.g.:
   - `确认重置 Hindsight 数据库并开始 SQLite 全量重建`

5. Only after confirmation: reset DB and initialize service.
   Follow the main SKILL.md reset flow for the current architecture. On this user's current setup, Hindsight uses an external PostgreSQL under `~/.hindsight-docker/instances/hindsight/data` and Docker container `hindsight` with `--network host`.

6. After reset, full import must use explicit `--full` and should go through the controlled paid-provider wrapper when quality matters:
   ```bash
   python3 ~/.hermes/scripts/hindsight_minimax_import.py sqlite-import-minimax -- \
     --mode submit --full --group-by day-topic --prefilter safe --retain-chunk-size 8000
   ```
   Ensure observations/consolidation stay disabled unless explicitly requested.

7. Verify after start/import:
   - `/health` healthy
   - stats pending/processing/failed progression is expected
   - DB counts increase from zero as retain completes
   - final mode returns to `normal-local` / Ollama / `enable_observations=false`
   - if using an external PostgreSQL reset, verify migrations/tables exist before import; do not assume `docker start hindsight` runs migrations if the current container/env disables migration-on-start
   - if long paid retain is still running, monitor via API stats + DB `async_operations` + Docker logs; a single `STUCK?` line is not fatal while completed/docs/facts keep increasing

Session-specific lessons from the 2026-05-08 reset/full rebuild are captured in `references/hindsight-db-reset-full-rebuild-2026-05-08-lessons.md`.

## Session-specific 2026-05-07 reference numbers

Observed before planned reset:
- Active Hindsight: healthy; documents 273; memory_units/nodes 7264; observations 2390; pending/processing/failed 0.
- SQLite import progress showed incremental cutoff `2026-05-07T02:27:53.161356`, 87 bundles imported.
- Incremental dry-run: 13 sessions, 2 bundles, 109,824 chars, 15 estimated retain chunks.
- Full dry-run: 570 sessions, 494 records with content, skipped `too_short=42`, `prefiltered=34`, 111 bundles, 7,317,687 chars, 974 estimated retain chunks.
- Pre-reset pg_dump backup path: `$HOME/.hindsight-docker/backups/reset-pre-full-rebuild-20260507-125207/hindsight-pre-reset.dump`, sha256 verified.

These numbers are not reusable as live state; they are examples of what to report.

## Fast response pattern when user is impatient

If the user asks a bundled question like “增量加载实现了吗、并发/多消息合并设置好了吗、直接 reset 从头跑”, do not skip the destructive-operation gate. Give a short status answer first, then ask for the exact confirmation phrase.

Minimum facts to answer from live/script checks:
- SQLite path is incremental by default through `~/.hermes/hindsight/sqlite_import_progress.json`; `--full` bypasses the cutoff for full rebuild.
- Current progress fields worth reporting: `last_imported_timestamp` / `last_imported_iso`, `total_sessions_imported`, `total_bundles_imported`.
- SQLite multi-message merge is `--group-by day-topic`; wrapper injects `--prefilter safe` if absent; full rebuild command should include `--retain-chunk-size 8000` and `--retain-extraction-mode concise` when applicable.
- Paid wrapper concurrency defaults to `HINDSIGHT_OFFLINE_LLM_CONCURRENCY` or 4; normal-local daily mode stays concurrency 1.
- Session-manifest retain uses batch-size 5 by default and submit-state for idempotence; do not conflate it with SQLite day-topic bundling.
- Long paid runs should use background/no-short-timeout execution and generous health timeout (`--health-timeout-s 600` is reasonable for full runs), then monitor health/stats/logs.

Recommended wording:
```text
增量/参数基本就绪：SQLite 有 progress 水位，默认增量；全量用 --full；day-topic 合并、safe prefilter、8K retain chunk；paid 并发默认 4，日常 local 是 1。Reset 是破坏性操作，我会先备份/校验，再 reset，再后台 full import 并盯异常。请回复：确认重置 Hindsight 数据库并开始 SQLite 全量重建
```

If the user’s broader architecture discussion favored session JSON v3 manifests, mention the route distinction before reset: SQLite full rebuild is the current reset runbook; session JSON v3 full retain is a different source path and should not be silently substituted.

## Pitfalls

- Do not run `rm -rf` or drop DB tables before a verified backup and user confirmation.
- Do not mistake `stats.queue_status` for worker truth; use stats pending/processing plus DB operation counts and logs when needed.
- Do not reuse incremental progress after a DB reset unless intentionally doing a partial rebuild; full rebuild needs `--full` and the importer must ignore old `processed` ids as well as the cutoff.
- Do not put Python watchdog/probe scripts under `/tmp` when importing stdlib modules such as `urllib.request`; `/tmp/http.py` or similar files can shadow the Python standard library. Prefer `~/.hermes/scripts/` or a clean working directory.
- Do not assume provider credentials live in the default profile env. If a paid wrapper reports a missing key before writes, check profile-specific env files (e.g. `profiles/<name>/.env`) without printing secret values. In this environment, the default `$HOME/.hermes/.env` may not contain `MINIMAX_API_KEY`, while `$HOME/.hermes/profiles/<profile>/.env` can.
- If the DB is dropped/recreated while the container env has `HINDSIGHT_API_RUN_MIGRATIONS_ON_STARTUP=false`, `/health` can still return healthy while core tables (`documents`, `memory_units`, `async_operations`) do not exist and `/stats` returns 500. Run migrations manually inside the container before retain: `sg docker -c "docker exec hindsight bash -lc 'cd /app/api && /app/api/.venv/bin/python - <<\"PY\"\nfrom hindsight_api.config import get_config\nfrom hindsight_api.migrations import run_migrations\nconfig = get_config()\nrun_migrations(str(config.database_url), migration_database_url=str(config.migration_database_url) if getattr(config, \"migration_database_url\", None) else None)\nPY'"`.
- Do not treat `STUCK?` logs as immediate failure during MiniMax retain; first check whether `completed`, documents, and facts are still increasing and whether `failed_operations` remains 0.
- Do not answer “并发/合并都好了” from memory alone; verify wrapper/importer args or cite the live script defaults.
- Do not leave the main API in paid-provider mode after import; restore local mode and verify.
- Do not bulk-promote v2 local cards to directives/mental-models after rebuild. Treat canonical observations/documents as the primary source; only manually select stable high-level principles for mental models/directives.
