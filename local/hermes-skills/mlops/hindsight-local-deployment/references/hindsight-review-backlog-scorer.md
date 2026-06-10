# Hindsight review backlog scorer workflow

Context: `hindsight_review_backlog.py` builds a recoverable review backlog, `hindsight_review_backlog_sampler.py` selects a content-free sample, and `hindsight_review_backlog_llm_scorer.py` either plans or, with explicit confirmation, scores that sample as a sidecar.

## Key defaults

- Active backlog cleanup is time-window first: keep the last 3 calendar months by default.
- Scorer cost guard is package-based: 1 scorer batch = 1 future LLM call.
- Default cadence is weekly.
- Default cap is 10 packages/calls per week.
- Larger pilots can raise `--max-batches` / `--max-llm-calls` to 50 or another explicit value.
- Overflow rows are deferred, not dropped.

## Safe modes

- Plan only: no LLM, no Hindsight writes.
- Dry-run scoring: if `--score-output` is requested without `--execute-score`, write an empty sidecar and confirm `llm_calls_made=0`.
- Real scoring: requires both `--execute-score` and `--confirm-score score-review-backlog`.
- Real scoring writes sidecar JSONL only; it must not submit to Hindsight or mutate production.

## Sidecar schema

The scorer output is `hindsight-review-backlog-llm-score-v1` and should stay sidecar-only.
Useful fields:
- `scores`: raw per-dimension integer scores, 0-5.
- `scores_normalized`: each score divided by 5.0, 0-1 helper fields for sorting/filtering.
- `score_total_0_20`: raw four-dimension total.
- `score_mean_0_1`: equal-weight mean of normalized dimensions, 0-1 helper for ranking.
- `value_level`
- `information_density`
- `durability`
- `actionability`
- `topic`
- `value_classes`
- `retainability_risk`
- `recommended_route`
- `anomalies`
- `reason_brief`
- `suggested_spans`

Use normalized scores only as helper signals. Route/risk/anomaly/secret gates stay dominant; a low score is not a delete signal.

## Pitfalls

- Do not confuse package count with session count. One package is one future LLM call.
- Do not feed scorer output back into Hindsight tags/content.
- Skip credential-like records before scoring.
- Keep `--max-record-chars` bounded so the scorer prompt stays small.
- Always verify that the score run stayed sidecar-only and that Hindsight submit stayed disabled.
- After real scoring, verify coverage before trusting routes: `score_summary.missing_scores_from_llm` must be 0 or `score_coverage` must meet the explicit gate. A 2026-05-09 MiniMax run with batch_size=5 returned only 12/39 direct scores; the script defaulted 27 missing records to `wait`. Treat defaulted records as unscored, not as model judgment.
- The scorer prompt should include `required_document_ids` and require exactly one score per document_id. The parser should accept both list output and `scores_by_document_id` / dict-shaped scores. If coverage is poor, do not proceed to production repair; rerun a tiny hardened pilot first or switch to batch_size=1 / targeted missing-document retry with a fresh LLM budget.
- On this machine the default profile `.env` may not contain `MINIMAX_API_KEY`; profile-specific keys can live under `~/.hermes/profiles/<profile>/.env`. Load keys without printing them, and never put API keys in command arguments.

## Useful commands

```bash
# Plan weekly scorer batches, no LLM
python3 ~/.hermes/scripts/hindsight_review_backlog_llm_scorer.py \
  --input <review_backlog.jsonl> \
  --output-batches <planned_batches.jsonl> \
  --summary-json <plan_summary.json>

# Dry-run score sidecar, still no LLM
python3 ~/.hermes/scripts/hindsight_review_backlog_llm_scorer.py \
  --input <review_backlog.jsonl> \
  --output-batches <planned_batches.jsonl> \
  --score-output <score_sidecar.dryrun.jsonl> \
  --summary-json <dryrun_summary.json>

# Real scorer execution, explicit confirm required
python3 ~/.hermes/scripts/hindsight_review_backlog_llm_scorer.py \
  --input <review_backlog.jsonl> \
  --output-batches <planned_batches.jsonl> \
  --score-output <score_sidecar.jsonl> \
  --execute-score \
  --confirm-score score-review-backlog \
  --llm-model MiniMax-M2.7 \
  --llm-base-url https://api.minimaxi.com/v1 \
  --llm-api-key-env MINIMAX_API_KEY
```
