# Hindsight session quality hardening diagnostics

Use after a session/json native retain run when quality gates fail, especially `docs_without_units > 0` or recall-smoke shows generic/dominant-topic contamination.

## Purpose

Phase A is read-only. It should not call paid LLMs and should not mutate Hindsight. It converts a failed trial into reusable quality evidence:

- classify zero-unit documents using generic structural features;
- score recall-smoke with top-k metrics instead of eyeballing sample rows;
- generate reviewable manifest-derived benchmark candidates;
- prepare, but not execute, Phase B temp-bank candidate manifests.

## Script

Current helper:

```bash
python3 ~/.hermes/scripts/hindsight_session_quality_hardening.py \
  --bank hermes \
  --manifest <week-production.jsonl> \
  --audit-json <retain-audit-parsed.json> \
  --output-dir <run-root> \
  --stem phase-a-session-quality-hardening
```

Output:

- `<stem>.json`
- `<stem>.md`

The script is intentionally generic. It must not hard-code project/query-specific fixes such as `patent`, `openclaw`, `cch`, or any specific project name.

## Zero-unit classes

Generic classes currently used:

- `true_low_signal`
- `context_bootstrap_or_resume_noise`
- `noisy_transcript`
- `noisy_high_value_transcript`
- `overlong_or_multi_scope`
- `cleaning_lost_context_risk`
- `extraction_too_strict_candidate`
- `unclassified_zero_unit`

Recommended routes:

- `skip_or_raw_only`
- `manual_review_or_raw_only`
- `raw_only`
- `production_windowed`
- `retry_less_aggressive_cleaning`
- `retry_custom_mission`
- `manual_review`

## Recall metrics

The script scores recall-smoke rows with:

- `precision_at_k`
- `off_topic_rate`
- `mrr`
- `first_relevant_rank`
- `dominant_tag_ratio`
- `dominant_doc_prefix_ratio`
- `expected_term_coverage`

This is still a lexical/generic first pass. Treat it as a gate signal and triage aid, not a full semantic judge.

## Manifest-derived benchmark candidates

The script emits `manifest_derived_benchmark_candidates` from observed tags + generic value classes. These are reviewable candidates for a future benchmark JSONL. Do not automatically treat them as production truth.

## Candidate sampling pitfall

The JSON/Markdown report may show a full `high_value_retry_candidate_count` while the materialized `high_value_retry_candidates` list is capped by `--max-samples` for legibility. Do not use the default sample list as the full route/retry set for paid reruns. Before building a full-week v2/all-routes manifest, rerun hardening with `--max-samples` greater than the reported candidate count or otherwise export all candidates, then assert:

```text
routed_parent_count == high_value_retry_candidate_count
```

For route-expanded/windowed manifests, compare runs at the parent-session level, not raw output-document level. Track `source_parent_count`, `parents_with_units`, `parent_zero_count`, and per-route parent coverage.

## Verification

Run:

```bash
python3 -m py_compile ~/.hermes/scripts/hindsight_session_quality_hardening.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  ~/.hermes/scripts/tests/test_hindsight_session_quality_hardening.py \
  ~/.hermes/scripts/tests/test_hindsight_bank_quality_audit.py \
  ~/.hermes/scripts/tests/test_hindsight_session_manifest_and_discard.py \
  ~/.hermes/scripts/tests/test_hindsight_session_manifest_selector.py \
  -q
```

Known good result from 2026-05-08: `34 passed in 0.14s`.

## Phase B boundary

Preparing a Phase B temp-bank candidate manifest is safe if it is only a file write. Executing temp-bank paid retain is not Phase A; it switches provider and spends paid calls, so require explicit confirmation.
