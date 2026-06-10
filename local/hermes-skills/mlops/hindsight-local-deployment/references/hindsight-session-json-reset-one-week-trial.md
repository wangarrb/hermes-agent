# Hindsight session/json reset + one-week trial runbook

Use this reference when the user wants to delete/reset the active Hindsight DB but test the clean session/json-native route on a limited time window before broader backfill.

## When to use

- User asks to reset/delete Hindsight DB and "先跑一周数据" / observe call counts and errors.
- The current architecture decision is session/json manifest + native Hindsight retain API, not SQLite day-topic import.
- Final clean production bank should be named `hermes`; temporary v3/v4 banks are only smoke history.

## Core principles

1. Do not silently use SQLite day-topic reset/full-rebuild runbooks. Session/json is the production route.
2. Reset is destructive: preflight, backup, sha256 verification, scope explanation, explicit confirmation.
3. Use a fresh submit-state file for the reset trial; old submit-state can incorrectly skip documents after DB reset.
4. Trial retain should include only `action=production` manifest records. `manual_review` and `skip` never go to paid retain by default.
5. Keep `enable_observations=false` during retain. Run native consolidation/observations only after facts audit passes, in a bounded window.
6. Include V4 gates: query-level recall smoke and topic/tag co-occurrence drift, not only count/tag audits.
7. Paid runs need background or generous timeouts, and must restore `normal-local` afterwards.
8. Audit reports/stdout must redact credential-like strings.

## Recommended one-week flow

Default window: the latest complete 7-day local-time range, e.g. `2026-05-01T00:00:00+08:00 <= started_at < 2026-05-08T00:00:00+08:00`.

1. Preflight live status:
   - `python3 ~/.hermes/scripts/hindsight_minimax_import.py status`
   - `curl -s http://127.0.0.1:8888/health`
   - `curl -s http://127.0.0.1:8888/v1/default/banks/hermes/stats`
   - DB counts for documents/memory_units/async_operations including `payload_null`.
   - Fail closed if pending/processing/failed or payload_null are nonzero without explanation.

2. Rebuild a fresh manifest for final bank name:
   ```bash
   python3 ~/.hermes/scripts/hindsight_session_manifest.py \
     --bank-target hermes \
     --retain-chunk-size 8000 \
     --json
   ```

3. Filter manifest to the one-week window and production-only records:
   - time key priority: `metadata.started_at`, then `session_last_updated`, then `last_updated`.
   - normalize timestamps to Asia/Shanghai for the window comparison.
   - produce both all-records and production-only JSONL plus a JSON summary.
   - summary should include counts by action/reason/tag, content chars, `sum(estimated_retain_chunks)`, rough retain calls `chunks * 2..3`, and secret/manual-review counts.

4. Dry-run retain without provider switch:
   ```bash
   python3 ~/.hermes/scripts/hindsight_minimax_import.py session-manifest-retain-llm \
     --manifest <week-production.jsonl> \
     --bank hermes \
     --batch-size 5 \
     --submit-state <fresh-run-submit-state.json> \
     --wait-timeout-s 7200 \
     --poll-s 10
   ```
   Dry-run should report all selected records as would-submit; if skipped unchanged is nonzero, check submit-state choice.

5. Backup before reset:
   ```bash
   pg_dump -h /tmp -p 5432 -U hindsight -d hindsight -Fc -f <backup>/hindsight-pre-reset.dump
   sha256sum <backup>/hindsight-pre-reset.dump > <backup>/hindsight-pre-reset.dump.sha256
   sha256sum -c <backup>/hindsight-pre-reset.dump.sha256
   ```
   Also copy relevant config/manifests/reports into the backup root.

6. Ask for explicit confirmation before destructive work. Suggested phrase:
   `确认重置 Hindsight 数据库并开始一周 session/json 试跑`

7. Reset DB using move-aside, not immediate deletion:
   - stop Docker container and PostgreSQL;
   - move current data dir under the backup root;
   - `initdb`, add `listen_addresses='*'` and trust rules for local/docker networks;
   - start PostgreSQL, create DB/extensions, start Hindsight, wait for health/migrations.

8. Harden bank config before retain:
   ```bash
   curl -s -X PATCH http://127.0.0.1:8888/v1/default/banks/hermes/config \
     -H 'Content-Type: application/json' \
     -d '{"updates":{"retain_chunk_size":8000,"retain_extraction_mode":"concise","enable_observations":false}}'
   ```

9. Execute paid retain with a fresh submit-state and long timeouts:
   ```bash
   python3 ~/.hermes/scripts/hindsight_minimax_import.py session-manifest-retain-llm \
     --manifest <week-production.jsonl> \
     --bank hermes \
     --batch-size 5 \
     --submit-state <fresh-run-submit-state.json> \
     --execute \
     --confirm retain-hindsight-session-manifest \
     --health-timeout-s 600 \
     --wait-timeout-s 7200 \
     --poll-s 10
   ```

10. Monitor during run:
    - wrapper status;
    - stats pending/processing/failed;
    - DB async operation status + payload_null;
    - documents and memory_units growth;
    - Docker logs for `JSON parse error|STUCK|PENDING_BREAKDOWN|ERROR|429|payload_null`.

11. Restore/verify normal-local:
    - provider should be Ollama/local;
    - observations false;
    - worker consolidation slots 0;
    - pending/processing/failed 0.

12. Facts audit + recall smoke:
    ```bash
    python3 ~/.hermes/scripts/hindsight_bank_quality_audit.py \
      --bank hermes \
      --recall-smoke \
      --stem <run-id-retain-audit>
    ```
    Required checks: docs_without_units=0, payload_null=0, failed=0, no broad/system tags, secret redaction, no obvious query-level cross-topic contamination.

    Important pitfall from the 2026-05-08 one-week trial: completed `batch_retain`/`retain` operations and `failed_operations=0` do **not** prove every submitted document produced facts. Hindsight can record all `document_ids` in completed retain result_metadata while many documents have zero `memory_units` (e.g. concise extraction returned no facts). Always compute `docs_without_units` and block consolidation/scale-out if nonzero; then inspect zero-unit document samples, extraction mode/mission, candidate filtering, and recall-smoke drift before retrying or expanding.

13. Only if facts audit passes, run a bounded native consolidation/observations window, e.g. one job / 50 facts, then audit again. If the guard/runner CLI has changed, check `--help` first rather than guessing.

14. Final report should include:
    - window and manifest counts;
    - backup path and sha256;
    - estimated chunks and rough calls;
    - observed provider/dashboard delta or local call-counter result;
    - operation status/errors/retries;
    - docs/memory_units/observations;
    - recall-smoke and contamination results;
    - decision: expand, fix selector/taxonomy, add raw chunks/topic selector, or stop.

## Trial scale reference from 2026-05-08 planning

A prior manifest for `2026-05-01..2026-05-08` estimated:
- total records: 478
- production: 188
- manual_review: 277
- skip: 13
- production chars: 1,524,195
- production estimated retain chunks: 295
- rough retain calls: 590-885
- manual_review estimated chunks: 603 and should not be submitted automatically

These numbers are examples only; always regenerate and restat before execution.

## Stop conditions

- production estimated chunks exceeds the agreed budget (example hard gate from the plan: >350 chunks);
- secret/credential-like record appears in production manifest;
- payload_null appears;
- failed_operations > 0 without a clear transient explanation;
- provider restore fails;
- repeated JSON parse retry/STUCK with no docs/facts growth;
- V4-style recall contamination appears after retain, before consolidation.
