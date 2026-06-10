# Missing-weeks Hindsight session retain backfill

Session context: 2026-05-10 W14/W17/W18 backfill after discovering weekly gaps and incomplete prior retain coverage.

## Trigger
Use this when a Hindsight weekly/daily report has missing weeks, or when production has native session documents but weekly reflect/consolidation did not see them.

## Core lesson
For paid session retain backfill, content completeness is the first gate. Do not start by narrowing to `action=production` or other high-level filters if the user asked to recover missing history. First prove the raw user/assistant message content can be reconstructed, then filter only unsafe material.

## Backfill workflow
1. Identify exact missing week/date windows and raw sources.
   - Older gaps may require rebuilding synthetic session JSON from `~/.hermes/state.db` messages.
   - Newer gaps may already have native session JSON under the Hindsight/session ingestion directories.
2. Build a full candidate manifest from user inputs and Hermes assistant outputs only.
   - Exclude tool calls, commands, search traces, thinking traces, and process logs.
   - Keep project evolution and substantive assistant answers even if they are not already tagged production.
3. Run a secret/credential classifier before paid submit.
   - Route records containing `api key`, `token`, `secret`, `password`, `sk-...`, etc. to local manual-review/quarantine.
   - Do not send secret/credential material to paid retain.
4. Run a reconstruction/integrity audit before submit.
   - Compare raw user/assistant message counts per week/window against retained candidates.
   - Reassemble chunked records and verify body text is reconstructable.
   - Small boundary-only differences such as extra/missing blank lines across chunk joins are not content loss, but record them explicitly.
5. Preflight exact manifest before `--execute`.
   - Record manifest path, sha256, record counts, non-secret count, secret/manual-review count, date windows, and LLM profile.
   - Avoid `latest` aliases for approved paid jobs.
6. Submit once, then monitor DB queue rather than resubmitting.
   - Track `async_operations` by status and `documents`/`memory_units` growth.
   - `failed=0` and steadily increasing docs/nodes is the primary progress signal.
7. After queue drain, run facts/recall smoke and then rerun affected weekly dry-run/submit.

## Weekly script compatibility pitfall
Weekly/offline reflection code must recognize native session document IDs, not only legacy day-topic IDs.
Known accepted forms:
- `hermes-sqlite::day-topic::...`
- `hermes-session::YYYYMMDD...`
- `hermes-session::session_YYYYMMDD...`

If weekly runs show zero or missing observations despite successful native retain, inspect the document-id parser before blaming retain quality.

## SQL progress probes
```sql
SELECT status, count(*) FROM async_operations GROUP BY status ORDER BY status;
SELECT count(*) AS documents FROM documents WHERE bank_id='hermes';
SELECT count(*) AS memory_units FROM memory_units WHERE bank_id='hermes';
SELECT count(*) AS recent_failed
FROM async_operations
WHERE status IN ('failed','error','cancelled')
  AND created_at > now() - interval '2 hours';
```

## Crash / reboot recovery notes
- If the machine dies mid-retain, do **not** resubmit the same manifest first. Existing `async_operations` may already contain submitted child retain rows; resubmission can duplicate paid work and production documents.
- First stabilize Hindsight/API and query the existing queue. Treat `pending retain` + `processing retain` as the real unfinished work; `pending batch_retain` rows with `task_payload IS NULL` are often parent bookkeeping and should not trigger resubmission by themselves.
- Preserve and report the last monitor snapshot: completed/pending/processing/failed, documents, memory_units, manifest sha256, and provider/concurrency. After recovery, continue the existing queue with a provider-switch/import mode if needed; do not call `session-manifest-retain-llm --execute` again unless duplicate checks prove nothing was submitted.
- Be conservative with concurrency after recovery. A jump from low concurrency to 32 workers during a large bge-m3/session retain backfill is a resource/cost risk; prefer low concurrency until health, queue progress, and failure counts are stable.

## Safety notes
- This workflow can incur paid LLM cost. Use guarded small-batch/temp-bank smoke when changing provider/profile/prompt config.
- Do not mutate production derived observations or documents for cleanup until snapshots/quarantine and user confirmation exist.
- Do not edit `.env` to add/switch keys automatically.
