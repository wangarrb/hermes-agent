# 2026-05-07 Hindsight full rebuild: weekly backfill + canonical publish FK pitfall

## Context

During a full SQLite retain rebuild and offline Hindsight consolidation run, daily reflect completed but V2 canonical publish initially failed after conflict/eval gate passed.

Key paths:
- Weekly/full pipeline log: `$HOME/.hermes/logs/hindsight-offline-pipeline/20260507-170411-weekly.log`
- Failed combined run log: `$HOME/.hermes/logs/hindsight-offline-pipeline/20260507-163117-both.log`
- V2 rebuild latest: `$HOME/.hermes/hindsight/offline_reflect/v2_rebuild/latest.json`
- Wiki auto-maintenance report: `$HOME/wiki/auto-maintenance/wiki-auto-maintenance-20260507-165519.md`

## What happened

1. Full SQLite retain rebuild completed successfully with MiniMax, concurrency 4, observations disabled.
2. Daily offline reflect for `2026-05-07` generated 5 outputs and posted them back to Hindsight.
3. V2 reduce/conflict/eval gate passed, but direct DB publish failed:

```text
psycopg2.errors.ForeignKeyViolation: insert or update on table "memory_links" violates foreign key constraint "fk_memory_links_to_unit_id_memory_units"
DETAIL: Key (to_unit_id)=(f21b6be6-cb5e-491a-ab73-d777006c50b1) is not present in table "memory_units".
```

Root cause: conflict audit allowed a P2 `dangling_evidence_id` to remain non-blocking. `source_memory_ids` may contain dangling evidence IDs as provenance metadata, but `memory_links.to_unit_id` has a hard FK and must only reference existing `memory_units.id`.

## Fix applied

Patch `hindsight_offline_v2_publish.py` so it:
- builds candidate semantic links from `source_memory_ids`,
- queries existing target IDs from `memory_units`,
- inserts only links whose target exists,
- keeps dangling IDs in metadata/source fields,
- reports `skipped_semantic_links_missing_targets`.

Regression test added in `test_hindsight_workflow_hardening.py`:
- `test_publish_filters_semantic_links_to_existing_memory_units`

Verification:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest $HOME/.hermes/scripts/tests -q
# 37 passed

python3 $HOME/.hermes/scripts/hindsight_offline_v2_rebuild.py \
  --mode publish \
  --confirm-publish publish-hindsight-v2-canonical \
  --json
```

Successful publish produced:
- canonical docs: 3
- canonical observations: 55
- semantic links: 563
- skipped missing semantic target: 1
- bad memory_links: 0

DB verification:

```bash
$HOME/.pg0/installation/18.1.0/bin/psql -h /tmp -p 5432 -U hindsight -d hindsight -Atc "
SELECT 'canonical_docs=' || COUNT(*) FROM documents WHERE id LIKE 'hermes-offline-canonical::%';
SELECT 'canonical_units=' || COUNT(*) FROM memory_units WHERE document_id LIKE 'hermes-offline-canonical::%';
SELECT 'dangling_source_ids=' || COUNT(*) FROM memory_units m, unnest(m.source_memory_ids) sid WHERE m.document_id LIKE 'hermes-offline-canonical::%' AND NOT EXISTS (SELECT 1 FROM memory_units s WHERE s.id=sid);
SELECT 'bad_memory_links=' || COUNT(*) FROM memory_links l WHERE NOT EXISTS (SELECT 1 FROM memory_units m WHERE m.id=l.to_unit_id) OR NOT EXISTS (SELECT 1 FROM memory_units m WHERE m.id=l.from_unit_id);
"
```

Expected after this fix:
- `canonical_docs=3`
- `canonical_units=55`
- `dangling_source_ids` may be non-zero if audit has P2 dangling provenance
- `bad_memory_links=0`

## Weekly backfill decision pattern

Weekly dry-run budget for all-history backfill showed:
- pending units: 85
- pending chars: ~1.88M
- estimated MiniMax calls: 85-170

Because this burns paid quota, do not auto-run unless user explicitly authorizes. When the user says to run it in the background, use high temporary caps to allow the already-reviewed budget through:

```bash
export HINDSIGHT_OFFLINE_LLM_CONCURRENCY=4
stdbuf -oL -eL python3 $HOME/.hermes/scripts/hindsight_offline_cron_runner.py weekly \
  --llm-profile minimax \
  --week-mode current \
  --prefilter safe \
  --poll 60 \
  --timeout 0 \
  --lock-timeout 21600 \
  --weekly-budget-max-pending-units 9999 \
  --weekly-budget-max-pending-chars 999999999
```

Run this as a background process and immediately verify:
- process session id is known,
- latest weekly log path exists,
- budget_decision is `pass`,
- Hindsight health returns after container restart,
- provider is temporarily MiniMax with observations disabled.

Startup note: during wrapper-controlled provider switch, `/health` may return connection refused for tens of seconds. Retry health before declaring failure.

## Wiki maintenance

After publish, `wiki_auto_maintenance.py --days 30` correctly wrote only under `$HOME/wiki/auto-maintenance/` and reported canonical state from `latest.json`. Main wiki content was not changed, except Obsidian may touch `$HOME/wiki/.obsidian/workspace.json`; do not treat this as a content merge.
