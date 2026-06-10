# Hindsight quality audit and session-incremental governance

Use this reference when auditing Hindsight memory quality or preparing a session/json rebuild/migration.

## Trigger

- User asks whether Hindsight quality is good enough.
- Before expanding native consolidation/observations.
- Before switching Hermes recall target to a new bank.
- After changing session/json ingestion or retain runner logic.

## Read-only first

Do not mutate Hindsight during audit. Do not switch provider. Do not run paid LLM. Start with:

```bash
python3 ~/.hermes/scripts/hindsight_minimax_import.py status
```

Expected safe daily state:

- health healthy
- pending/processing/failed all 0
- provider local/Ollama unless explicitly in a paid window
- `HINDSIGHT_API_ENABLE_OBSERVATIONS=false`
- `HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS=0`

## Current lessons from 2026-05-08 audit

The old `hermes` bank was usable as archive/evidence recall but not clean enough for broad native consolidation.

Observed bad patterns:

- System/source tags dominated the bank: `hermes`, `sqlite`, `incremental`, `daily`, `canonical`.
- Native consolidation uses tags as semantic grouping; broad source tags directly pollute observation scopes.
- Cross-domain contamination existed:
  - `egomotion4d` facts containing patent/OA terms.
  - `egomotion4d` facts containing OpenClaw terms.
  - `openclaw` facts containing patent/OA terms.
  - `openclaw` facts containing Egomotion/VGGT/DAGE/ATE terms.
- Native 36 observations from old bank were mostly single-source rewrites with bad tags; do not expand old-bank native consolidation.
- Offline canonical observations had useful high-level content but topic assignment could be polluted.
- User-preference recall from old Hindsight was weak; keep user preferences in USER/memory/profile or curated clean docs, not broad old-bank recall.

Positive checks from the same audit:

- active queues were empty.
- no exact duplicate text groups.
- no replacement-char/control-char rows.
- memory units with missing document: 0.
- observation source refs mostly existed; only a small number of missing source refs.

## SQL audit probes

Use external PostgreSQL read-only checks when available:

```bash
PSQL=$HOME/.pg0/installation/18.1.0/bin/psql
DB=hindsight
$PSQL -h /tmp -p 5432 -U hindsight -d "$DB" -F $'\t' -A <<'SQL'
SELECT operation_type,status,count(*) AS n,count(*) FILTER (WHERE task_payload IS NULL) AS payload_null
FROM async_operations GROUP BY operation_type,status ORDER BY operation_type,status;

SELECT COALESCE(fact_type,'<null>') AS fact_type, count(*) AS n
FROM memory_units WHERE bank_id='hermes'
GROUP BY fact_type ORDER BY n DESC;

SELECT tag, count(*) AS n
FROM memory_units, LATERAL unnest(tags) AS tag
WHERE bank_id='hermes'
GROUP BY tag ORDER BY n DESC LIMIT 40;

SELECT tag, count(*) AS n
FROM memory_units, LATERAL unnest(tags) AS tag
WHERE bank_id='hermes' AND tag IN ('hermes','sqlite','incremental','daily','canonical','source:tmp')
GROUP BY tag ORDER BY n DESC;

SELECT COALESCE(jsonb_array_length(to_jsonb(source_memory_ids)),0) AS source_count, count(*) AS n
FROM memory_units
WHERE bank_id='hermes' AND fact_type='observation'
GROUP BY source_count ORDER BY source_count;

WITH obs AS (
  SELECT id AS obs_id, unnest(source_memory_ids) AS source_id
  FROM memory_units
  WHERE bank_id='hermes' AND fact_type='observation' AND source_memory_ids IS NOT NULL
)
SELECT count(*) AS source_refs,
       count(*) FILTER (WHERE m.id IS NULL) AS missing_source_refs,
       count(DISTINCT obs_id) FILTER (WHERE m.id IS NULL) AS observations_with_missing_source
FROM obs LEFT JOIN memory_units m ON m.id=obs.source_id;

SELECT count(*) AS unconsolidated_facts
FROM memory_units
WHERE bank_id='hermes' AND fact_type IN ('experience','world')
  AND consolidated_at IS NULL AND consolidation_failed_at IS NULL;
SQL
```

Contamination probes:

```sql
SELECT 'egomotion_tag_patent_terms' AS metric, count(*) FROM memory_units WHERE bank_id='hermes' AND 'egomotion4d'=ANY(tags) AND text ~* '(patent|OA1|office action|专利|审查意见|权利要求)'
UNION ALL SELECT 'egomotion_tag_openclaw_terms', count(*) FROM memory_units WHERE bank_id='hermes' AND 'egomotion4d'=ANY(tags) AND text ~* '(OpenClaw|ClawHub|approval|gateway probe|No session found)'
UNION ALL SELECT 'openclaw_tag_patent_terms', count(*) FROM memory_units WHERE bank_id='hermes' AND 'openclaw'=ANY(tags) AND text ~* '(patent|OA1|office action|专利|审查意见|权利要求)'
UNION ALL SELECT 'openclaw_tag_egomotion_terms', count(*) FROM memory_units WHERE bank_id='hermes' AND 'openclaw'=ANY(tags) AND text ~* '(Egomotion|VGGT|DAGE|ATE|TrackingWorld|trajectory)';
```

## Recall smoke

Run fixed smoke queries and inspect top results manually:

- `Hindsight session json native consolidation discard quarantine observation_scopes`
- `Egomotion4D VGGT DAGE ATE_metric trajectory scale window`
- `专利 OA1 审查意见 权利要求 意见陈述书`
- `OpenClaw ClawHub approval gateway probe No session found`
- `用户偏好 简洁 质疑精神 技术排障 直接执行`
- `CCH gpt-5.5 provider context_length Responses API identity`

Grade recall by:

- relevance of top results
- tag/scope consistency
- whether old source tags dominate
- whether user preference queries are polluted by project facts
- whether observations cite enough source facts

## Session/json manifest governance

The clean session/json pipeline is safer than old SQLite day-topic bundles, but still needs gates.

Manifest script:

```bash
python3 ~/.hermes/scripts/hindsight_session_manifest.py --bank-target hermes_v3 --json
```

Expected manifest safety properties:

- lean output by default, no full conversation content in JSONL.
- records include source mtime/size/file hash and content hashes.
- no system/source tags (`hermes`, `sqlite`, `incremental`, `daily`, `canonical`, `source:tmp`) in proposed tags.
- mixed sessions with multiple project tags or too many semantic tags go to `manual_review`.
- production set can still be large; do not submit full production set blindly.

Incremental semantics:

- Session JSON files are mutable and may grow/change after an earlier manifest.
- `--since-mtime-ns` is only a candidate scan accelerator, not a successful-submit watermark.
- Retain runner uses successful-submit state, default:
  - `~/.hermes/hindsight/session_ingest/submit_state.json`
- Skip unchanged only when same `document_id` has same `content_sha256` in submit-state.
- Changed sessions use same `document_id` + `update_mode=replace` so Hindsight replace/re-retain refreshes facts.
- Dry-run never updates submit-state; execute success does.

Retain runner:

```bash
python3 ~/.hermes/scripts/hindsight_session_retain_runner.py \
  --manifest <manifest.jsonl> \
  --bank hermes_v3 \
  --limit 5 \
  --json
```

Real submit requires explicit confirmation:

```bash
python3 ~/.hermes/scripts/hindsight_session_retain_runner.py \
  --manifest <manifest.jsonl> \
  --bank hermes_v3 \
  --limit 5 \
  --execute \
  --confirm retain-hindsight-session-manifest \
  --json
```

## Recommendation pattern

For current user environment:

1. Keep old `hermes` as archive/evidence bank.
2. Do not run broad native consolidation on old `hermes`.
3. Create clean candidate bank `hermes_v3`.
4. Submit 5-10 production manifest records first.
5. Wait retain queue completion.
6. Audit facts/tags/scopes before expanding.
7. Enable native observations only after clean facts pass audit, starting with 1 job / 50 facts.
8. Use discard-first remediation for any destructive old-bank cleanup.

## Report path convention

Save quality reports under:

```text
~/.hermes/hindsight/reports/YYYY-MM-DD-hindsight-quality-audit.md
```

Reference report from this session:

```text
$HOME/.hermes/hindsight/reports/2026-05-08-hindsight-quality-audit.md
```
