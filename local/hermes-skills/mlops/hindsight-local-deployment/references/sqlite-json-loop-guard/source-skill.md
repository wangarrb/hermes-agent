---
name: hindsight-sqlite-json-loop-guard
description: Handle MiniMax JSON parse retry/STUCK loops during Hindsight SQLite import by splitting oversized bundles and patching JSON fence parsing.
tags: hindsight, sqlite, minimax, json-parse, docker, memory-import
---

# Hindsight SQLite JSON Loop Guard

Use when Hindsight SQLite import with MiniMax shows:
- `JSON parse error from LLM response`
- `Content preview: '```json\n{...}\n```'` but json.loads still fails
- `[STUCK?] stage=llm.minimax.retain_extract_facts...`
- MiniMax answers historical conversation content instead of structured JSON

## Quick triage

```bash
python3 ~/.hermes/scripts/hindsight_minimax_import.py status

sg docker -c "docker logs hindsight --since 10m 2>&1 | grep -E 'JSON parse error|Content preview|STUCK|STREAMING RETAIN COMPLETE|PENDING_BREAKDOWN|WORKER_TASK' | tail -200 || true"

PSQL=$HOME/.pg0/installation/18.1.0/bin/psql
$PSQL -h /tmp -p 5432 -U hindsight -d hindsight -F $'\t' -Atc "
WITH ops AS (
  SELECT operation_id, operation_type, status, claimed_at,
         task_payload->'contents'->0->>'document_id' AS document_id,
         task_payload->'contents'->0->>'content' AS content
  FROM async_operations
  WHERE bank_id='hermes' AND status IN ('pending','processing') AND task_payload ? 'contents'
)
SELECT operation_id, operation_type, status, COALESCE(document_id,''), length(content), COALESCE(claimed_at::text,'')
FROM ops ORDER BY length(content) DESC LIMIT 20;"
```

## Two different failure classes

### A. Content-injection / oversized transcript

Symptoms:
- MiniMax responds with natural language answering old conversation questions.
- Often from 90k+ or 200k+ char bundles.

Fix:
1. Stop Hindsight to interrupt active expensive retry loop.
2. Mark original pending/processing op as completed/skipped, delete partial document if any.
3. Resubmit content split into ~60k-70k char pieces.
4. Record in `~/.hermes/hindsight/sqlite_import_skipped_bundles.md`.

Current import script fix:
- `~/.hermes/scripts/import_sqlite_to_hindsight.py` now hard-splits a single oversized session too, not only multi-session bundles.
- Verify with dry-run; it should print `[split] oversized session ... -> N parts`.

### B. Fenced JSON parser bug

Symptoms:
- `Content preview` is valid-looking fenced JSON:
  ` ```json\n{ "facts": [...] }\n``` `
- Hindsight still logs `json.loads`: `Expecting value: line 1 column 1`.
- Usually the first attempt is actually usable; retries are waste.

Fix running container:

```bash
python3 ~/.hermes/scripts/patch_hindsight_minimax_json_parser.py
sg docker -c "docker exec hindsight bash -lc 'grep -n \"HERMES_MINIMAX_JSON_FENCE_FIX\\|HERMES_RATE_LIMIT_BACKOFF_FIX\\|HINDSIGHT_API_RATE_LIMIT_BACKOFF_SECONDS\" /app/api/hindsight_api/engine/providers/openai_compatible_llm.py'"
```

Patch script details:
- Copies `/app/api/hindsight_api/engine/providers/openai_compatible_llm.py` out of the container.
- Replaces `_strip_code_fences()` with regex-based fence stripping.
- Handles ` ```json`, ` ```JSON`, CRLF, and avoids leaving the language label before JSON.
- Also patches `APIStatusError` retry backoff so HTTP 429 / rate limit / throttling / quota errors sleep `HINDSIGHT_API_RATE_LIMIT_BACKOFF_SECONDS` seconds (default 300) instead of doing quick retry loops. This is especially important before native reflect/consolidation/observations paid-provider windows.
- Important 2026-05-07 pitfall: the patch text must write regexes with single backslashes in the target source, e.g. `r"```(?:[a-zA-Z0-9_-]+)?\s*\n(.*?)```"`. A broken V3 patch wrote `\\s*\\n`, matching literal backslash sequences and causing continued fenced-JSON parse retries. Verify target source grep shows V4 marker and single-backslash regex.
- Does not restart by default. Use `--restart` only if you need active Python process to reload immediately; restarting interrupts active worker, so coordinate with queue state.

Note: patching file inside running container may only affect new processes. Existing Python process keeps already-loaded code. If the queue has high parse-loop risk, restart container after marking active op safe/skipped.

## Safe restart and restore

If you kill an old wait process, start a fresh restore monitor:

```bash
terminal background equivalent:
set -euo pipefail
python3 ~/.hermes/scripts/hindsight_minimax_import.py wait-queue --poll 60 --timeout 43200
python3 ~/.hermes/scripts/hindsight_minimax_import.py normal-local
python3 ~/.hermes/scripts/hindsight_minimax_import.py status
```

Expected import mode:
- `HINDSIGHT_API_LLM_PROVIDER=minimax`
- `HINDSIGHT_API_ENABLE_OBSERVATIONS=false`
- `HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS=0`
- `failed_operations=0`

Expected final mode:
- provider restored to `ollama`
- pending/processing 0

## Important observations from 2026-05-05

- `batch_retain|pending|... payload_null=...` can be normal wrapper state; do not treat as old payload-null bug if `retain` tasks are completing and no STUCK loop.
- MiniMax can produce valid fenced JSON but upstream parser may fail; don't assume every `JSON parse error` means content contamination.
- A single parse error that completes on attempt 2 is less severe, but still wastes calls. Patch parser to avoid repeated waste.
- Always distinguish:
  - natural-language answer to old prompt = split/clean content;
  - fenced JSON = parser patch.
