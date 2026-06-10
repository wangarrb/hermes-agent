# Hindsight Recall/Reflect is Not Evidence

> Hindsight's `recall` and `reflect` tools return **derived** facts from the knowledge graph.
> They are NOT a direct query against the actual system state. Always verify claims
> against real data before drawing conclusions.

## The Lesson

In multiple sessions, hindsight recall/reflect returned incomplete or misleading results:

1. **5-16 Weekly Report**: recall returned format hints and category structures but no actual report content. The report existed at the file level (`memory/others/周报_2026-05-16.md`), but hindsight had extracted it into fine-grained facts and lost the original structure.

2. **External Import Completion**: recall returned "145 external docs imported" but couldn't confirm whether `observation` had been generated. Only a direct PostgreSQL query revealed: 0 observations from external docs (because `enable_observations: false` during import-minimax mode).

3. **Consolidation Status**: recall returned "all completed" but didn't catch that the 5-19 daily pipeline had *actually failed* at the session-manifest-retain step. The pipeline log showed `refusing to switch to minimax with existing queue` — a failure mode recall had no record of.

## The Correct Workflow

When asked "is X done?" or "what happened to Y?" about hindsight:

```
Step 1: hindsight recall/reflect → get hypothesis / hint / direction
Step 2: VERIFY against actual system state:
   a. Pipeline logs: /home/wyr/.hermes/logs/hindsight-offline-pipeline/<date>-daily-noagent.log
   b. Docker logs: docker logs hindsight 2>&1 | grep ...
   c. PostgreSQL (direct): psql -h 127.0.0.1 -U hindsight -d hindsight -c "SELECT ..."
   d. Bank stats API: curl -s http://127.0.0.1:8888/v1/default/banks/<bank>/stats
   e. Operations API: curl -s .../operations?exclude_parents=true&limit=...
   f. Submit state files: /home/wyr/.hermes/hindsight/external_import/submit_state*.json
   g. External import manifests: /home/wyr/.hermes/hindsight/external_import/manifests/
```

## Key Tables to Query

```sql
-- Documents in a bank
SELECT COUNT(*) FROM documents WHERE bank_id='hermes';

-- Fact type distribution across memory_units
SELECT fact_type, COUNT(*) FROM memory_units WHERE bank_id='hermes' GROUP BY fact_type;

-- External-origin documents
SELECT id, created_at FROM documents WHERE id LIKE 'external-%' ORDER BY created_at DESC;

-- External memory units by fact_type
SELECT fact_type, COUNT(*) FROM memory_units WHERE document_id LIKE 'external-%' GROUP BY fact_type;

-- Check if observation was generated for external docs
SELECT document_id,
  COUNT(*) as total,
  COUNT(*) FILTER (WHERE fact_type='observation') as obs,
  COUNT(*) FILTER (WHERE fact_type='experience') as exp,
  COUNT(*) FILTER (WHERE fact_type='world') as world
FROM memory_units
WHERE document_id LIKE 'external-%'
GROUP BY document_id;
```

## Pipeline Log Checkpoints

| File | What it tells you |
|------|-------------------|
| `2026MMDD-010100-daily-noagent.log` | Daily offline pipeline: session retain, offline reflect, V2 publish |
| `docker logs hindsight` | Real-time retain/consolidation activity |
| `submit_state.json` | External import documents submitted and their target banks |
| `external_import/evals/*` | External import test/eval results |
| `offline_reflect/offline_reflect_progress.json` | Offline reflect units processed |

## Common Misleading Recall Results

- "pending_consolidation: 0" + recall says "all done" → but daily pipeline may have *skipped* the session-manifest-retain step entirely
- "observations: 19598" + recall says "all imported" → but external docs may have 0 observations (checked via DB only)
- "last_consolidated_at: recent" → doesn't mean the most recent session data was included; the pipeline may have processed stale data