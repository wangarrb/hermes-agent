# Hindsight V2 Publish + Cron/Wiki Maintenance Notes (2026-05-07)

Session-specific details from hardening and publishing Offline Hindsight V2 canonical observations.

## What changed

- `hindsight_offline_v2_rebuild.py` was hardened to default to local mode and require explicit publish confirmation:
  - `--mode publish`
  - `--confirm-publish publish-hindsight-v2-canonical`
- Manual publish succeeded with:
  - `published=True`
  - `inserted_documents=32`
  - `inserted_observations=2278`
  - `embedding_provider_used=docker`
  - `embedding_count=2278`
  - backup path under `offline_reflect/v2_publish_backups/`
- Main DB verification used:
  - documents canonical prefix column is `documents.id`, not `documents.document_id`
  - memory units use `memory_units.document_id`

## Verification queries

```bash
PSQL=$HOME/.pg0/installation/18.1.0/bin/psql
$PSQL -h /tmp -p 5432 -U hindsight -d hindsight -At <<'SQL'
SELECT 'canonical_docs|' || COUNT(*)
FROM documents
WHERE bank_id='hermes' AND id LIKE 'hermes-offline-canonical::%';

SELECT 'canonical_obs|' || COUNT(*)
FROM memory_units
WHERE bank_id='hermes' AND document_id LIKE 'hermes-offline-canonical::%';

SELECT 'latest_published_at|' || COALESCE(MAX((metadata->>'published_at')), '')
FROM documents
WHERE bank_id='hermes' AND id LIKE 'hermes-offline-canonical::%';

SELECT 'embedding_count|' || COUNT(*)
FROM memory_units
WHERE bank_id='hermes'
  AND document_id LIKE 'hermes-offline-canonical::%'
  AND embedding IS NOT NULL;
SQL
```

## Cron compatibility pitfall

When publish confirmation became mandatory, `hindsight_offline_cron_runner.py` also had to be updated. Otherwise daily/weekly cron would run V2 rebuild in `--mode publish` but without the confirmation token, producing a local proposal (`published=False`) and overwriting `v2_rebuild/latest.json` even though the previous manual publish had succeeded.

Correct cron-side call:

```bash
python3 ~/.hermes/scripts/hindsight_offline_v2_rebuild.py \
  --mode publish \
  --confirm-publish publish-hindsight-v2-canonical
```

After changing publish safety or cron behavior:

1. Run `python3 -m py_compile ~/.hermes/scripts/hindsight_offline_cron_runner.py`.
2. Check `hermes cron list` for daily/weekly next runs and last status.
3. Verify `~/.hermes/hindsight/offline_reflect/v2_rebuild/latest.json` after any cron/smoke run.
4. Re-check main DB canonical counts, not just the latest JSON.

## Wiki maintenance integration

Wiki maintenance should read high-level Hindsight outputs in this order:

1. `offline_reflect/v2_rebuild/gate/canonical-retain-proposal.md`
2. `offline_reflect/v2_cards/**/*.md`
3. `offline_reflect/weekly/**/*.md`
4. `offline_reflect/daily/**/*.md` only as fallback

The report should include canonical state and conflict summary. It must write only to `wiki/auto-maintenance/`, not the main wiki.

## Status after this session

- Hindsight health: healthy/database connected.
- pending/processing/failed operations: 0/0/0.
- Provider restored to normal-local Ollama.
- Main canonical layer published and embedded.
- Daily cron remains responsible for MiniMax import + daily reflect; weekly cron for weekly reflect; wiki cron for candidate reports.
