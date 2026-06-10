# Hindsight review-backlog scorer MiniMax coverage lesson (2026-05-09)

## Context

A sidecar-only MiniMax scorer run was executed for `sample40 weekly10` review backlog scoring:

- planned batches: 8
- actual MiniMax calls: 8
- input records: 40
- skipped before LLM: 1 credential-like record
- score sidecar rows written: 39
- Hindsight submit / production mutation: disabled throughout

Safety checks after the run showed Hindsight production state unchanged:

- total_documents: 188
- total_nodes: 852
- total_observations: 0
- pending_operations: 0
- failed_operations: 0

## Problem

The run was operationally safe but not trustworthy as an automated routing signal.

MiniMax returned direct scores for only 12 of the 39 prompted records. The script then defaulted the 27 missing records to:

- `recommended_route = wait`
- reason: `LLM response omitted this document_id; defaulted to wait.`

This means route distribution was misleading: many `wait` rows were missing-output fallbacks, not model judgments.

Observed route distribution:

- wait: 32
- cluster_revisit: 6
- raw_only: 1

Effective direct score coverage: 12 / 39 = 30.8%.

## Root cause hypothesis

The prompt/schema did not force one output per input `document_id` strongly enough. The schema example used a one-object list under `scores`, and the model often followed the example shape by returning only one scored record for a batch.

A second robustness gap: parser support was too narrow. It accepted list-shaped `scores`, but not map-shaped outputs like `scores_by_document_id` or dict-shaped `scores`.

## Hardening applied

Patch the scorer script so future runs can detect and reduce this failure mode:

- Add `required_document_ids` and `required_score_count` to the scorer prompt.
- Explicitly require exactly one score object per required `document_id`.
- Add final-check constraints in the prompt.
- Parse list-shaped and map-shaped outputs:
  - `scores: [{...}]`
  - `scores: {document_id: {...}}`
  - `scores_by_document_id: {document_id: {...}}`
  - `score_by_document_id: {document_id: {...}}`
- Add summary coverage metrics:
  - `records_prompted_to_llm`
  - `valid_scores_from_llm`
  - `missing_scores_from_llm`
  - `missing_document_ids`
  - `score_coverage`
  - `coverage_ok`

Regression tests added:

- prompt contains required document IDs and score count
- map-shaped score output is accepted
- missing score coverage is reported correctly

Verification commands:

```bash
$HOME/.hermes/hermes-agent/venv/bin/python -m py_compile $HOME/.hermes/scripts/hindsight_review_backlog_llm_scorer.py
$HOME/.hermes/hermes-agent/venv/bin/python -m pytest $HOME/.hermes/scripts/tests/test_hindsight_review_backlog.py -q
```

Expected test result after patch: `11 passed`.

## Future workflow rule

For review-backlog scorer runs:

1. Treat sidecar-only safety and score coverage as separate gates.
2. Do not use scorer output for production repair if `missing_scores_from_llm > 0` unless the user explicitly accepts partial coverage.
3. Treat defaulted `wait` rows as unscored evidence, not as a low-priority model judgment.
4. If coverage fails:
   - rerun a tiny hardened pilot first, e.g. 2 batches / 10 records;
   - if still bad, switch to `batch_size=1` or implement targeted missing-document retry;
   - ask for a new LLM budget before retrying many missing records.
5. Always re-check Hindsight stats after paid sidecar work to prove no production mutation happened.

## API key pitfall

On this machine, the default profile `.env` may not contain `MINIMAX_API_KEY`; profile-specific keys can live under `~/.hermes/profiles/<profile>/.env`. Load keys into environment without printing them. Never pass API keys directly in command arguments.