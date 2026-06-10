# Hindsight review-backlog scorer preflight and weekly sidecar workflow

Use this when deciding whether to run a paid LLM scorer over Hindsight review backlog records.

## Positioning

The review-backlog LLM scorer is a **sidecar rescue router**, not a production retain/consolidation step.

It may score whether backlog items are worth later rescue, but it must not:
- write scorer summaries into Hindsight content;
- turn scorer labels into semantic tags or observation scopes;
- mutate production documents/facts/observations;
- trigger native consolidation/observations.

## Default budget and cadence

User correction from 2026-05-09: default should be **weekly 10 scorer packages/calls**, not 50. 50 is only a configurable larger pilot.

Current helper:

```bash
python3 ~/.hermes/scripts/hindsight_review_backlog_llm_scorer.py \
  --input <active-3mo-scorer-sample.jsonl> \
  --output-batches <weekly10-batches.jsonl> \
  --score-output <score-sidecar.jsonl> \
  --summary-json <summary.json>
```

Default behavior:
- `--cadence weekly`
- `--batch-size 5`
- `--max-batches 10` / `--max-llm-calls 10`
- one scorer batch/package = one future LLM call
- overflow records are deferred, not dropped
- no LLM call unless `--execute-score --confirm-score score-review-backlog`

## Required preflight before paid scorer

Run/read a read-only preflight before any paid scorer execution. Minimum gates:

1. Hindsight health is healthy.
2. Runtime provider is safe normal-local when not intentionally in paid mode:
   - `HINDSIGHT_API_LLM_PROVIDER=ollama`
   - `HINDSIGHT_API_ENABLE_OBSERVATIONS=false`
   - `HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS=0`
3. Queue is empty:
   - `pending_operations=0`
   - `processing_operations=0`
   - `failed_operations=0`
4. Active `payload_null` is zero. Historical completed `batch_retain` rows with `task_payload IS NULL` are a warning/diagnostic fact but do not block a read-only scorer if there are no pending/processing rows.
5. Active backlog and scorer sample exist.
6. Active backlog policy is time-window based, normally active 3 months + cold archive.
7. Scorer sample is content-free by default and all rows have `event_date`.
8. Weekly budget is within cap, e.g. `llm_calls_planned <= 10` and no unexpected deferred records.
9. Scorer dry-run confirms `execute=false`, `llm_calls_made=0`, `scores_written=0`.
10. Relevant tests and py_compile pass.

Example preflight outcome from 2026-05-09:
- active backlog: 188 records
- archive: 0 records
- sample40: 40 records, content-free, 40/40 event-dated
- weekly plan: 8 packages / <=10 calls, deferred=0
- dry-run: `llm_calls_made=0`, `scores_written=0`
- recommendation: `GO_SCORER_SIDECAR_ONLY`

## Allowed next step after preflight passes

Only execute the scorer sidecar, e.g.:

```bash
python3 ~/.hermes/scripts/hindsight_review_backlog_llm_scorer.py \
  --input <active-3mo-scorer-sample.jsonl> \
  --output-batches <weekly10-batches.jsonl> \
  --score-output <weekly10-score-sidecar.jsonl> \
  --summary-json <weekly10-score-summary.json> \
  --execute-score \
  --confirm-score score-review-backlog
```

Boundary: even after scorer execution, production mutation remains blocked until the sidecar is reviewed and converted into explicit temp-bank/proposal work.

## Still blocked at this stage

Do not proceed directly from preflight to:
- production replace/re-retain;
- production mutation;
- native consolidation/observations;
- full one-week retain/rerun.

The next safe stage after scorer review is usually: create temp-bank repair candidates, run temp retain, then run fact-quality and recall gates.
