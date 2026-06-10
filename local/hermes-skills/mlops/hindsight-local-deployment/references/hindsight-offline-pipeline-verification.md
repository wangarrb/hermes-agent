# Hindsight Offline Pipeline Verification Notes

Session-derived checks for offline SQLite retain + daily/weekly consolidation runs.

## Background wrapper exit codes

A background process notification such as `exit code 130` or shell/PTTY noise like `tcsetattr: 对设备不适当的 ioctl 操作` does not by itself prove Hindsight failed. Treat it as a wrapper/interruption signal until verified against service state and persisted outputs.

Do not summarize failure from the process code alone. Verify:

```bash
python3 ~/.hermes/scripts/hindsight_minimax_import.py status
python3 ~/.hermes/scripts/hindsight_offline_audit.py
```

Expected post-run normal-local state:
- `/health` healthy
- `pending_operations=0`, `processing_operations=0`, `failed_operations=0`
- provider env back to `ollama`
- `HINDSIGHT_API_ENABLE_OBSERVATIONS=false`
- `HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS=0`

## Coverage checks

Audit should show all closed retained SQLite days have daily consolidation. Current/open day may be intentionally missing until tomorrow's daily cron.

Key audit interpretation:
- `sqlite_retained_days`: days represented by retained SQLite import
- `daily_consolidated_days`: days with offline daily documents
- `closed_sqlite_days_without_daily_consolidation=0`: pass condition
- `open_current_missing_daily_days=<today>`: usually OK; do not force partial daily unless explicitly requested

Weekly/global refresh should include `history-through-YYYY-Www` when using `--weekly-window all-history`.

## DB verification commands

This Hindsight schema uses `documents.id` for document IDs; there is no `documents.document_id` column. Use `id` when querying documents.

```bash
PSQL=$HOME/.pg0/installation/18.1.0/bin/psql
$PSQL -h /tmp -p 5432 -U hindsight -d hindsight -At -c "
SELECT substring(id from '::weekly::([^:]+)::') AS period, COUNT(*)
FROM documents
WHERE bank_id='hermes' AND id LIKE 'hermes-offline-consolidation::weekly::%'
GROUP BY 1 ORDER BY 1;

SELECT substring(id from '::cross-topic::([0-9]+)::') AS idx, COUNT(*)
FROM documents
WHERE bank_id='hermes'
  AND id LIKE 'hermes-offline-consolidation::weekly::history-through-2026-W19::cross-topic::%'
GROUP BY 1 ORDER BY 1;

SELECT operation_type, status, COUNT(*), COUNT(*) FILTER (WHERE task_payload IS NULL)
FROM async_operations
GROUP BY operation_type,status
ORDER BY operation_type,status;
"
```

Notes:
- `batch_retain|completed|...|payload_null=...` can appear for completed wrapper bookkeeping; it is not automatically an active-burn problem if pending/processing/failed are all zero.
- For a 20-unit all-history weekly refresh, require idx `00` through `19` each count 1.

## Transient ConnectionRefused during submit

During paid-provider runs the wrapper may restart/recreate the Docker container. A short `/memories` POST `ConnectionRefused` can happen in that window. Correct handling:
1. Keep local `.md/.json` outputs.
2. Resume/retry missing posts after service health returns.
3. Reconcile file outputs against DB document IDs.
4. Finalize only after status/audit/DB checks all pass and normal-local mode is restored.

Do not re-run the entire history blindly if only a few document posts failed; reconcile and submit only the missing document IDs where possible.
