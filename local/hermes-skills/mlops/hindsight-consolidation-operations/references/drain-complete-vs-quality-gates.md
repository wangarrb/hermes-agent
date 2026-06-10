# Drain complete vs quality gates / auto-retain checks

Session-derived lesson from 2026-05-12 Hindsight offline processing recovery.

## Distinguish three states

Do not answer "the flow is complete" as a single boolean. Split it into:

1. **Native drain/consolidation complete**
   - `/health` is healthy.
   - `unconsolidated_base = 0` for `world/experience` units with no `consolidation_failed_at`.
   - `failed_base = 0`.
   - `async_operations` has no `pending:consolidation` or `processing:consolidation`.
   - It is OK if `async_operations` still has historical `failed:consolidation` rows, as long as source `failed_base = 0` and no active ops remain.

2. **Quality/conflict gate passed**
   - Run `hindsight_conflict_audit.py` separately.
   - A completed native drain does **not** imply conflict/lineage gate pass.
   - Common blocked state: `blocked_conflict_review_required` with many `dangling_source_document` / `dangling_evidence_id` cases. Treat this as lineage/provenance debt, not proof that content is semantically false.

3. **Runtime auto-retain state**
   - Verify `~/.hermes/hindsight/config.json` directly.
   - Expected normal-local setting in this setup: `auto_retain=false`, `auto_recall=true`.
   - Also report `retain_every_n_turns` and `memory_mode` if present.

## Commands

Health + backlog + async ops:

```bash
curl -sS --max-time 10 http://127.0.0.1:8888/health
$HOME/.hindsight-docker/installation/18.1.0/bin/psql -h /tmp -p 5432 -U hindsight -d hindsight -q -t -A -F $'\t' -c "
select count(*) filter(where fact_type in ('world','experience') and consolidated_at is null and consolidation_failed_at is null) as unconsolidated_base,
       count(*) filter(where fact_type in ('world','experience') and consolidation_failed_at is not null) as failed_base,
       count(*) filter(where fact_type='observation') as observations
from memory_units where bank_id='hermes';
select status, operation_type, count(*)
from async_operations where bank_id='hermes'
group by status, operation_type order by status, operation_type;
"
```

Conflict/lineage audit:

```bash
python3 $HERMES_HOME/scripts/hindsight_conflict_audit.py \
  --bank hermes \
  --api http://127.0.0.1:8888 \
  --block-severity P1 \
  --json > /tmp/hindsight-conflict-audit-current.json
python3 - <<'PY'
import json
r=json.load(open('/tmp/hindsight-conflict-audit-current.json'))
print('decision', r.get('decision'))
print('observation_count', r.get('observation_count'))
print('known_counts', r.get('known_counts'))
print('summary', json.dumps(r.get('summary'), ensure_ascii=False, sort_keys=True))
print('latest_json', r.get('json_path'))
print('latest_md', r.get('markdown_path'))
PY
```

Auto-retain:

```bash
python3 - <<'PY'
import json, pathlib
p=pathlib.Path.home()/'.hermes'/'hindsight'/'config.json'
data=json.load(open(p))
for k in ['auto_retain','auto_recall','retain_every_n_turns','memory_mode']:
    print(k, '=', data.get(k))
PY
```

## Conflict-audit lineage false positives

If the native drain is complete but conflict audit reports many `dangling_source_document` / `dangling_evidence_id` cases, first distinguish true untraceable lineage from audit-coverage gaps:

- Offline consolidation document ids have a content hash suffix, e.g. `hermes-offline-consolidation::daily::DATE::topic::00::HASH`. Regenerating the card can change `HASH` while the stable document identity is still the prefix through slot (`...::00`). Audit code should treat that prefix as an alias, not as missing lineage.
- The API document/memory listing may be insufficient for old offline runs. Also load local offline JSON artifacts under `~/.hermes/hindsight/offline_reflect/` (`daily/`, `weekly/`, v2 card sidecars) and count `document_id`, `source_ids`, `evidence_ids`, and embedded UUIDs as local lineage evidence.
- If an observation has at least one traceable source document/file, stale extra source refs or old UUIDs should be non-blocking (`partial_dangling_source_document` / `stale_evidence_id`, P3) rather than P1/P2. Keep them as cleanup debt, not a quality gate blocker.
- After changing audit semantics, add regression tests and rerun `py_compile`, pytest, and the full conflict audit before declaring the gate cleared.

## Reporting pattern

Report as:

- "Native drain/consolidation: complete/incomplete" with counts.
- "Quality/conflict gate: pass/blocked" with top blocking types and report path.
- "Auto retain: on/off" with config path.

Avoid saying "everything is done" if conflict audit is blocked. Say "processing complete, quality gate not cleared".
