# Session/native retain operator safety

Use this when running paid Hindsight session/json native retain, especially after reset, embedding migration, provider switch, or cron/manual overlap.

## 2026-05-10 incident distilled

A bge-m3/vector1024 production retain was accidentally started with:
- wrong manifest: all-history/lightweight manifest (`20260509-155655-session-manifest.jsonl`, 1920 records / 419 production / 04-09..05-09)
- wrong extraction config: wrapper reset bank config to `retain_extraction_mode=concise`

The intended run was:
- `week-04-09-04-16-production-20260509-1549.cleaned-user-assistant-only-v2.jsonl`
- 83 production records
- user + Hermes assistant only
- chunk_size=8000
- `retain_extraction_mode=custom`
- bge-m3 / `vector(1024)`
- observations disabled

Root causes:
1. A newer `latest`/all-history manifest was substituted for the previously approved cleaned manifest.
2. `hindsight_minimax_import.py` had hardcoded defaults that re-applied `HINDSIGHT_API_RETAIN_EXTRACTION_MODE=concise` and patched bank config to `concise` during restart/provider switch.
3. The failure was not caused by Hermes cron job 2; Hindsight cron jobs were paused. The active trigger was the manual/background wrapper.

Immediate safe response used:
1. Stop Hindsight Docker to stop paid burn.
2. Kill stale wait/audit monitors.
3. Verify no active retain wrapper remains.
4. Backup current partial DB.
5. Reset and migrate DB back to bge-m3 `vector(1024)`.
6. Patch wrapper defaults and rerun tests.
7. Dry-run the exact intended manifest before `--execute`.
8. Re-run with correct manifest/config and verify container env + bank config after restart.

## Pre-execute checklist

Before any paid `session-manifest-retain-llm --execute`:

1. Exact manifest identity
   - Print absolute path.
   - Compare against the approved path from prior user decision.
   - Do not use `latest.json` or a newly generated manifest unless explicitly approved.

2. Manifest scope
   - Count total records and `action=production` records.
   - Confirm date window.
   - Confirm `content` exists for records to retain, or that the runner intentionally rehydrates content from source JSON.
   - Confirm expected chunk count from actual retained text, not just metadata.

3. Content policy
   - For this user's paid production session retain, default policy is user + Hermes assistant only.
   - Exclude tool output, command output, search traces, thinking/analysis/commentary traces, and credential-like material.
   - Secret/credential-like sessions route to manual review/quarantine, not production retain.

4. Runtime config
   - Container env must show:
     - `HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=BAAI/bge-m3`
     - `HINDSIGHT_API_RETAIN_CHUNK_SIZE=8000` unless user chose otherwise
     - `HINDSIGHT_API_RETAIN_EXTRACTION_MODE=custom`
     - `HINDSIGHT_API_RETAIN_CUSTOM_INSTRUCTIONS` present
     - `HINDSIGHT_API_RETAIN_EXTRACT_CAUSAL_LINKS=false`
     - `HINDSIGHT_API_ENABLE_OBSERVATIONS=false`
   - Bank config must show matching `retain_chunk_size`, `retain_extraction_mode`, custom instructions, and observations disabled.
   - DB schema must be `memory_units.embedding = vector(1024)` for bge-m3.

5. Cron/manual overlap
   - List Hermes cron jobs before blaming cron.
   - Paused jobs are not root cause.
   - Active local background wrappers and Docker workers are more likely during manual runs.

## Post-restart verification commands

Use `sg docker -c` for Docker on this host.

```bash
sg docker -c "docker exec hindsight /bin/sh -lc 'env | grep -E \"HINDSIGHT_API_(LLM_PROVIDER|LLM_MODEL|RETAIN_EXTRACTION_MODE|RETAIN_CHUNK_SIZE|RETAIN_CUSTOM_INSTRUCTIONS|RETAIN_EXTRACT_CAUSAL_LINKS|EMBEDDINGS_LOCAL_MODEL|ENABLE_OBSERVATIONS)\" | sort'"

curl -sS http://127.0.0.1:8888/v1/default/banks/hermes/config | python3 - <<'PY'
import sys,json
cfg=json.load(sys.stdin).get('config',{})
print(json.dumps({k:cfg.get(k) for k in [
  'retain_chunk_size','retain_extraction_mode','retain_custom_instructions',
  'enable_observations','recall_max_tokens','consolidation_llm_batch_size'
]}, ensure_ascii=False, indent=2))
PY

PG=$HOME/.pg0/installation/18.1.0/bin
$PG/psql -h 127.0.0.1 -p 5432 -U hindsight -d hindsight -At -c \
"SELECT format_type(a.atttypid,a.atttypmod) FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid WHERE c.relname='memory_units' AND a.attname='embedding';"
```

## Wrapper patch guidance

Avoid hardcoded paid-retain behavioral defaults in provider-switch wrappers. Defaults should be env-overridable via `HINDSIGHT_NATIVE_RETAIN_*`, and current preferred defaults are:

```python
"HINDSIGHT_API_RETAIN_CHUNK_SIZE": os.environ.get("HINDSIGHT_NATIVE_RETAIN_CHUNK_SIZE", "8000"),
"HINDSIGHT_API_RETAIN_EXTRACTION_MODE": os.environ.get("HINDSIGHT_NATIVE_RETAIN_EXTRACTION_MODE", "custom"),
"HINDSIGHT_API_RETAIN_CUSTOM_INSTRUCTIONS": os.environ.get("HINDSIGHT_NATIVE_RETAIN_CUSTOM_INSTRUCTIONS", "ONLY extract durable user/project facts, decisions, results, preferences, stable environment facts. Skip tool logs, file listings, raw command output, process chatter, greetings. Max 3-5 facts per chunk."),
"HINDSIGHT_API_RETAIN_EXTRACT_CAUSAL_LINKS": os.environ.get("HINDSIGHT_NATIVE_RETAIN_EXTRACT_CAUSAL_LINKS", "false"),
```

Do not patch `retain_extract_causal_links` through bank config unless the Hindsight API exposes it as configurable. In the observed version, bank config API returned HTTP 400 for that key; set it via container env instead.

## Destructive recovery pattern

If wrong paid retain has already begun:
1. Stop Hindsight first to stop burn.
2. Ask/confirm before destructive reset unless user already gave explicit reset authorization.
3. Backup partial DB with counts and sha256.
4. Reset/drop/recreate DB.
5. Run migrations manually if startup migrations are disabled.
6. Explicitly ensure embedding dimension for current local model.
7. Dry-run correct manifest.
8. Start paid run, then verify config and progress.

Never treat a stale wait-audit crossing reset as valid; it may report queue drained with 0 docs/0 units.