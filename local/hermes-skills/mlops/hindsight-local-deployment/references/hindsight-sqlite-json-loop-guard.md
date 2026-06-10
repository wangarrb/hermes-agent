# Hindsight SQLite Import: MiniMax JSON Parse / STUCK Loop Guard

## Trigger

Use this when a SQLite full/import run is active and Docker logs show any of:

- `JSON parse error from LLM response` where MiniMax returns prose or answers a historical user question instead of structured JSON
- `WORKER_TASK [STUCK?] ... stage=llm.minimax.retain_extract_facts+structured`
- Active/pending retain payloads are very large, commonly >90k chars, especially 100k–350k chars

This is not just cosmetic retry noise. It can amplify MiniMax calls because Hindsight retries JSON parsing up to 11 attempts per chunk.

## Root Cause

`import_sqlite_to_hindsight.py --max-bundle-chars` historically constrained multi-session bundles but did not hard-split a single oversized session. A single 250k–350k char transcript could still be submitted as one memory item. MiniMax may then be prompt-injected by historical conversation content and answer the embedded task rather than returning JSON.

Another observed MiniMax-specific quirk: responses may use uppercase fenced code blocks such as ```JSON, or reasoning-style `<think>...</think>` before the JSON body. Hindsight's `_strip_code_fences()` must strip fences case-insensitively, remove `<think>` blocks, and if needed extract the JSON object/array from surrounding prose; otherwise valid structured responses still fail at `json.loads` and enter costly retry loops.

## Prevention

The importer should hard-split single oversized sessions before bundling. The current known-good behavior:

- import `replace` from `dataclasses`
- add `split_oversized_record(record, max_bundle_chars)`
- in `build_bundles()`, expand records through that splitter before grouping
- target split size leaves header room: roughly `max_bundle_chars - 5000`, with a floor around 20k chars
- split on `\nUser:`, `\nAssistant:`, or blank-line boundaries when possible

Validate with:

```bash
python3 -m py_compile $HOME/.hermes/scripts/import_sqlite_to_hindsight.py
python3 $HOME/.hermes/scripts/import_sqlite_to_hindsight.py \
  --mode dry-run --full --group-by day-topic --no-main \
  --prefilter balanced --prefilter-threshold 15 \
  --retain-chunk-size 16000 --sample-report 0 \
  --max-bundle-chars 120000 | grep -E 'Bundles|Estimated retain chunks|Max bundle|WARNING|\[split\]'
```

Expected: oversized sessions print `[split] oversized session ... -> N parts`; max bundles are no longer 200k–350k chars.

## Live Queue Remediation

If large operations were already submitted, do not wait for all 11 JSON retries. Use a guard flow:

1. Inspect active/pending payload sizes:

```bash
PSQL=$HOME/.pg0/installation/18.1.0/bin/psql
$PSQL -h /tmp -p 5432 -U hindsight -d hindsight -F $'\t' -Atc "
WITH ops AS (
  SELECT operation_id, operation_type, status, claimed_at,
         task_payload->'contents'->0->>'document_id' AS document_id,
         task_payload->'contents'->0->>'content' AS content
  FROM async_operations
  WHERE bank_id='hermes' AND status IN ('pending','processing') AND task_payload ? 'contents'
)
SELECT operation_id, operation_type, status, COALESCE(document_id,''),
       length(content) AS content_chars, COALESCE(claimed_at::text,'')
FROM ops
ORDER BY content_chars DESC
LIMIT 20;"
```

2. For pending/processing retain ops above the guard threshold (90k chars was used on 2026-05-05), back up payloads to JSON before modification.

3. If one is actively processing and retrying JSON, stop the container to interrupt the active call before DB surgery:

```bash
newgrp docker <<'SH'
docker stop hindsight
SH
```

4. Mark original high-risk operations as completed/superseded, and delete the partial original document if it exists. Do not leave them as failed unless the wrapper's `wait-queue` ignores failed operations; failed status may keep dashboards noisy.

5. Restart MiniMax import mode:

```bash
python3 $HOME/.hermes/scripts/hindsight_minimax_import.py import-minimax --allow-existing-queue
```

6. Resubmit each original payload as smaller split items, target ~60k–70k chars each. Preserve metadata and add:

- `original_document_id`
- `original_operation_id`
- `split_part`
- `split_total`
- tag `json-loop-guard`

7. Start a fresh wait/restore monitor:

```bash
python3 $HOME/.hermes/scripts/hindsight_minimax_import.py wait-queue --poll 60 --timeout 43200
python3 $HOME/.hermes/scripts/hindsight_minimax_import.py normal-local
python3 $HOME/.hermes/scripts/hindsight_minimax_import.py status
```

Run in background when appropriate.

## Verification

Check all of these:

```bash
python3 $HOME/.hermes/scripts/hindsight_minimax_import.py status
newgrp docker <<'SH'
docker logs hindsight --since 5m 2>&1 | grep -E 'JSON parse error|STUCK|STREAMING RETAIN COMPLETE|Document:|Marked async operation|PENDING_BREAKDOWN|WORKER_TASK|429|rate limit|ERROR' | tail -120 || true
SH
$PSQL -h /tmp -p 5432 -U hindsight -d hindsight -Atc "SELECT COUNT(*) FROM memory_units WHERE bank_id='hermes' AND text ILIKE '%HEARTBEAT%';"
```

Good signs:

- pending decreases over time
- `STREAMING RETAIN COMPLETE` continues
- no repeated JSON parse attempts on the same operation
- no STUCK continuing past several minutes after splitting
- `failed_operations=0` after superseded originals are marked completed
- heartbeat facts remain 0

## Session Record

On 2026-05-05, a 267,911 char bundle and thirteen >90k char retain ops caused MiniMax JSON/stuck risk. They were backed up, marked superseded, split into ~27k–70k char items, and resubmitted. Details were logged in:

`$HOME/.hermes/hindsight/sqlite_import_skipped_bundles.md`
