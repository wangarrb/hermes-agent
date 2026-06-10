# Hindsight DB Reset + SQLite Full Rebuild: 2026-05-08 Lessons

Use this as session-specific detail under the class-level reset/full rebuild runbook.

## What happened

During a confirmed destructive reset and SQLite `--full` rebuild:

1. Preflight/backup/reset were correct: status/stats/dry-runs, pg_dump backup + sha256, external PostgreSQL data dir reset, DB recreated, extensions installed.
2. The first reset/start attempt exposed an environment mismatch: Docker commands in this environment may require the `docker` group wrapper (`sg docker -c '...'`) rather than plain `docker`.
3. The API container did not auto-run migrations after fresh DB init because the container/env had migrations disabled. A one-off migration/recreate path was needed before `/health` could become useful.
4. The first full import attempt failed before writes because the default profile env lacked MiniMax credentials; credentials were present in `~/.hermes/profiles/<profile>/.env`. The fix was to load profile env without printing secret values.
5. A more serious full-rebuild bug surfaced: `import_sqlite_to_hindsight.py --full` ignored the time cutoff but still loaded the old `sqlite_import_progress.json` `processed` document ids. After DB reset this can skip historical bundles that no longer exist in Hindsight. Fix: for `--full`, start from an empty progress object; incremental mode still loads progress. Regression test added.
6. After correcting the progress bug, the database was reset again to a true empty state and the full rebuild was restarted.
7. Long MiniMax retain should run in a background wrapper with monitoring, not a short foreground timeout. During this run 118/118 bundles submitted successfully, then Hindsight retained them with 4 worker slots.
8. `STUCK?` warnings around 300s are not automatically fatal; in this run several long ops later emitted `STREAMING RETAIN COMPLETE`. Treat as an alert, not an immediate cancel condition, unless `failed_operations` grows or no progress occurs for a sustained window.
9. A custom watchdog placed in `/tmp` failed because `/tmp/http.py` shadowed Python stdlib `http.client` during `import urllib.request`. Put reusable watchdog/probe scripts outside `/tmp` (e.g. `~/.hermes/scripts/`) or run with a clean working directory.

## Concrete checks to add to future runs

Before destructive reset:

```bash
python3 ~/.hermes/scripts/import_sqlite_to_hindsight.py \
  --mode dry-run --full --group-by day-topic --prefilter safe \
  --retain-chunk-size 8000 --sample-report 0
```

Then verify `--full` is not using historical progress:

```bash
python3 - <<'PY'
import importlib.util, sys
from pathlib import Path
p = Path.home() / '.hermes/scripts/import_sqlite_to_hindsight.py'
spec = importlib.util.spec_from_file_location('import_sqlite_to_hindsight', p)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
progress = mod.progress_for_run(full=True)
assert progress.get('processed') == [], 'full rebuild must ignore old processed ids'
assert float(progress.get('last_imported_timestamp') or 0) == 0.0
print('FULL_PROGRESS_EMPTY_OK')
PY
```

If code lacks `progress_for_run(full=True)`, inspect/patch before resetting. `--full` must bypass both cutoff and processed ids.

## Provider/key handling

- Do not print API keys or connection strings.
- If default `~/.hermes/.env` has no provider key, check profile-specific env such as `~/.hermes/profiles/<profile>/.env` by key name only, then inject values into the subprocess environment without echoing them.
- If the wrapper says the key is missing before write, no Hindsight data was modified by that run; still verify stats/DB counts.

## Docker / migration handling

- In this environment, prefer `sg docker -c 'docker ...'` for container management.
- After a fresh external-PostgreSQL reset, do not assume `docker start hindsight` runs migrations. Verify real tables via DB counts and API health. If migrations are disabled, use Hindsight's own migration entry point or recreate via the controlled wrapper path that runs migrations.
- Avoid raw `rm -rf` without backup and explicit confirmation; if a partial/buggy import wrote to an empty DB, stop API first, snapshot the partial state under the backup root, then reset again.

## Long-run monitoring pattern

Monitor all of these, not only process status:

```bash
curl -s http://127.0.0.1:8888/health
curl -s http://127.0.0.1:8888/v1/default/banks/hermes/stats
$HOME/.pg0/installation/18.1.0/bin/psql -h /tmp -p 5432 -U hindsight -d hindsight -Atc \
  "SELECT operation_type,status,COUNT(*) FROM async_operations GROUP BY operation_type,status ORDER BY operation_type,status;"
sg docker -c "docker logs hindsight --since 5m 2>&1" | \
  grep -E 'PENDING_BREAKDOWN|WORKER_TASK|STUCK|ERROR|429|JSON parse error|STREAMING RETAIN COMPLETE|failed|retry|slow llm call' | tail -120
```

Use `failed_operations > 0` or sustained no-progress as intervention triggers. Do not cancel solely on one `STUCK?` line if documents/facts/completed counts are still increasing.

## Watchdog placement pitfall

Do not place Python watchdog scripts in `/tmp` if that directory may contain files named like stdlib modules (`http.py`, `json.py`, etc.). Python prepends the script directory to `sys.path`; `/tmp/http.py` can break `urllib.request` by shadowing stdlib `http`.

Prefer:

```bash
cd ~/.hermes
python3 ~/.hermes/scripts/hindsight_full_rebuild_watchdog.py --wrapper-pid <pid> --interval-s 300 --settle-s 180
```

## Final audit acceptance pattern

After a reset + SQLite `--full` import completes, do not stop at "import process exit 0". Run and save a final audit that captures:

- importer exit code and watchdog exit code
- Hindsight `/health` status and DB connection
- provider restored to normal-local/Ollama, not paid import mode
- `enable_observations=false` unless explicitly requested
- queue drained: pending=0, processing=0, failed=0
- DB/API counts: documents, memory_units by `fact_type`, async operations by type/status
- duplicate text groups, docs without units, broad/source tag distribution, and topic contamination samples
- secret-pattern scan results, with any credential-like material redacted in reports

Known-good result from this run:

- `118` documents imported
- `6298` facts/nodes: `4839 experience`, `1459 world`
- `236` async operations completed: `118 batch_retain`, `118 retain`
- `0` failed operations
- `0` observations, expected because observations/consolidation stayed disabled
- exact duplicate text groups: `0`
- docs without units: `3` low-signal/empty bundles

Quality caveat: a successful SQLite day-topic rebuild is not the same as a clean canonical knowledge base. Day-topic bundling may leave broad tags such as `sqlite/hermes/incremental` on all docs and can create topic-scope contamination, e.g. `egomotion4d` tagged documents containing patent/openclaw material from the same day-topic bundle. Treat this as a structural quality limitation of the import route, not a runtime failure. Before enabling native consolidation/observations broadly, either clean tag/scope or move to the cleaner session JSON v3 route.

Recommended report names/locations:

```bash
~/.hermes/hindsight/reports/YYYY-MM-DD-hermes-sqlite-full-rebuild-final-audit.json
~/.hermes/hindsight/reports/YYYY-MM-DD-hermes-sqlite-full-rebuild-final-audit.md
```

## Regression test added in this session

`~/.hermes/scripts/tests/test_import_sqlite_to_hindsight_progress.py` verifies:
- `progress_for_run(full=True)` ignores historical `processed` ids and timestamp.
- `progress_for_run(full=False)` preserves incremental progress.
