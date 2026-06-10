# Full Pipeline Verification — 2026-05-15 Run

## Context

Second end-to-end `full` pipeline run for this environment. First complete
uninterrupted run after the auto-trigger fix in
`hindsight_wait_native_consolidation.py` (Pitfall #28).

## Run Profile

| Field | Value |
|-------|-------|
| Mode | `full --incremental` |
| Start | 2026-05-15 17:47 CST |
| End | 2026-05-15 19:03 CST |
| Wall time | ~1h16min |
| Exit code | 0, all 13 stages `ok` |
| Log | `~/.hermes/hindsight/pipeline_runs/full-20260515-174702.log` (14K lines) |
| Run report | `~/.hermes/hindsight/pipeline_runs/20260515-190313-full-pipeline-run.json` |

## Key Metrics

### Hindsight Delta

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Documents | 1674 | 1682 | +8 |
| Nodes | 21754 | 21928 | +174 |
| Observations | 14987 | 15122 | +135 |
| Completed ops | 2299 | 2317 | +18 |

### Stage Timing

| Stage | Elapsed | Notes |
|-------|---------|-------|
| preflight | ~1min | All green, 64x8 |
| session manifest | ~1min | 2437 records, 576 production |
| retain (5 new) | ~5min | 571 unchanged, 0 new |
| daily reflect + drain | ~17min | 3 V2 cards; **auto-trigger fired once** |
| V2 publish (daily) | ~10min | bge-m3 CPU embeddings |
| weekly reflect + drain | ~7min | auto-trigger fired again |
| V2 publish (weekly) | ~10min | bge-m3 embeddings |
| conflict audit | ~1min | 56 cases, 0 P1 blocking |
| proposals + review | ~1min | 80 proposals, 0 LLM calls (no execute flag) |

### Auto-Trigger Performance

Both consolidation drains stalled naturally after precision-remote-mode restore.
Auto-trigger detected and self-healed both:

| Drain | pending_conso | trigger | wait_time |
|-------|--------------|---------|-----------|
| After daily retain | 20 | `2f183a68` | ~120s (2 poll cycles) |
| After weekly reflect | 15 | `abce7409` | ~120s (2 poll cycles) |

Neither required manual intervention.

## Expected Normal Baseline

This run represents a **light day** — few new sessions (5). Compare:

| Scenario | Date | Docs Δ | Total time | Consolidation |
|----------|------|--------|-----------|---------------|
| **Light day** | 2026-05-15 | +8 | ~1h16min | 20→0 in ~6min |
| **Heavy day** | 2026-05-14 | +30 | ~3h | 1754→0 in ~2h20m |

Consolidation cost is proportional to unconsolidated backlog, not session count.

## Known Harmless Artifacts

- `alias_map_missing` in V2 publish — `v2_aliases.json` doesn't exist, but
  `alias_normalization` processes fields anyway. Non-blocking.
- 5 historical failed operations — all from 2026-05-12/14, not from this run.
- 80 proposals all `no_go` — expected without `--execute-proposal-review-llm`.

## Verification Command

For follow-up runs:

```bash
# Quick health check
python3 ~/.hermes/scripts/hindsight_consolidation_status.py --skip-psql --json

# Compare per-run report
cat ~/.hermes/hindsight/pipeline_runs/20260515-190313-full-pipeline-run.json

# Check for new failures
curl -s 'http://127.0.0.1:8888/v1/default/banks/hermes/operations?status=failed&exclude_parents=true&limit=3' | \
  python3 -c "import sys,json; [print(f'{o[\"created_at\"][:19]} {o[\"task_type\"]} {o[\"status\"]}') for o in json.load(sys.stdin).get('operations',[])]"
```
