# Hindsight consolidation telemetry contract

Use this when making offline Hindsight operations long-term observable, especially for daily reports and production go/no-go summaries.

## Goal

Daily reports and operator dashboards should read structured Hindsight/offline-pipeline statistics rather than inferring health and cost from one log file or one SQLite state DB.

## Minimum dimensions

Every run/stage counter should include:

- `bank`
- `tenant` when applicable
- `run_id` or `operation_id`
- `stage` (`retain`, `extract`, `consolidate`, `reflect`, `proposal_review`, `conflict_audit`, `publish_gate`, etc.)
- `provider` and `model` for LLM stages
- `started_at`, `finished_at`, `duration_seconds`
- `status` (`pending`, `processing`, `completed`, `failed`, `skipped`, `manual_review_required`)

## Required counters

At minimum expose these per bank/run/stage:

- LLM calls
- input tokens
- output tokens
- total tokens
- retry count
- rate-limit count
- exception count
- created count
- updated count
- deleted/quarantined count
- conflict count
- duplicate/deduped count
- pending human review count
- proposal count
- accepted/rejected/skipped proposal count
- active/pending/completed/failed async operation counts

## Safety rules

- Do not include raw prompts, raw secrets, or credential material in stats endpoints.
- If examples need token-like strings, construct them dynamically in tests instead of storing realistic tokens in markdown.
- Token/cost accounting should be additive and idempotent: repeated daily-report reads must not alter counters.
- A failed stage should include a redacted error class/message and the operation/run id needed for drill-down.

## Daily report usage

Daily reports should aggregate at least three channels:

1. Main Hindsight session/native retain activity.
2. Offline reflect/consolidation pipeline activity.
3. Auxiliary compression or other memory-producing jobs if enabled.

The report should state when a channel is unavailable rather than silently reporting zero. `0 calls` and `unknown calls` are different.

## Suggested JSON shape

```json
{
  "schema_version": "hindsight-ops-stats-v1",
  "generated_at": "2026-01-01T00:00:00Z",
  "bank": "hermes",
  "window": {"start": "...", "end": "..."},
  "totals": {
    "llm_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "created": 0,
    "updated": 0,
    "deleted_or_quarantined": 0,
    "conflicts": 0,
    "pending_human_review": 0,
    "exceptions": 0
  },
  "stages": [
    {
      "stage": "consolidate",
      "status": "completed",
      "llm_calls": 0,
      "total_tokens": 0,
      "created": 0,
      "updated": 0,
      "exceptions": 0
    }
  ]
}
```

## Operator interpretation

- Do not treat missing stats as success.
- Do not use only `state.db` for cost; it can miss offline pipeline and auxiliary jobs.
- If counters disagree, prefer source-of-truth operation/run manifests over derived dashboard totals and flag the mismatch in the daily report.
