# Hindsight memory pipeline modes

Use this as the stable mode taxonomy for Hermes/Hindsight offline memory processing.

## Safety boundary

- Default is plan/dry-run. No production Hindsight mutation unless `--execute --confirm run-hindsight-pipeline` is supplied.
- Underlying session retain still requires its own token: `retain-hindsight-session-manifest`.
- V2 canonical publish still requires: `publish-hindsight-v2-canonical`.
- Production repair-zone merge remains proposal-only until the user gives a separate go/no-go.
- Weekly/full proposal bundles now get a local `proposal_review` packet after proposal generation. Advisory LLM review is available but separately gated by `--execute-proposal-review-llm --confirm-proposal-review review-hindsight-proposals`; final approval remains human-only.
- Do not edit `.env` from this flow.

## Mode 1: `daily`

Purpose: daily incremental session conversation processing.

Default behavior:

1. Build deterministic session/json manifest from `~/.hermes/sessions`.
2. Retain manifest into the target Hindsight bank through `hindsight_minimax_import.py session-manifest-retain-llm`.
3. Enable native observations/consolidation during session retain.
4. Run daily processed-fact reflect/consolidation (`offline_hindsight_reflect_consolidate.py --scope daily --daily-source facts`).
5. Rebuild/gate local V2 cards.

Incremental semantics:

- Default `--history incremental` relies on submit state to skip unchanged documents.
- It may still scan all session JSON files; this is acceptable because the runner performs the durable incremental skip.
- Force full-history reprocessing with `--history all`, which passes `--ignore-submit-state` to retain runner.

Plan command:

```bash
python3 ~/.hermes/scripts/hindsight_memory_pipeline.py daily --plan-json
```

Execution command:

```bash
python3 ~/.hermes/scripts/hindsight_memory_pipeline.py daily --execute --confirm run-hindsight-pipeline
```

## Mode 2: `weekly`

Purpose: higher-dimensional weekly/all-history integration, cleanup, and conflict handling.

Default behavior:

1. Run weekly/all-history reflect from daily outputs:
   - `--weekly-source daily`
   - `--weekly-window all-history`
   - `--weekly-group-by topic`
   - `--backfill-missing-daily`
2. Rebuild/gate V2 cards.
3. Run conflict/lineage audit with P1 block severity.
4. Sweep approved review-repair sidecars into proposal-only canonical bundles.
5. Build local proposal review packets for LLM advisory + human go/no-go review; no production writes.

Repair-zone rule:

- Weekly may search and package the approved repair zone.
- It does not directly copy temp-bank/sidecar facts into production.
- Output remains canonical proposal bundles under `~/.hermes/hindsight/review_repair/proposals/`.
- `hindsight_proposal_review.py` writes human-review packets under `~/.hermes/hindsight/review_repair/reviews/`. Dry-run review blocks production because LLM judgement is pending; advisory LLM judgement requires `--execute-llm --confirm-review review-hindsight-proposals` or the pipeline wrapper flags.
- Production merge/retain requires a separate explicit user approval and rollback/quarantine plan.

Plan command:

```bash
python3 ~/.hermes/scripts/hindsight_memory_pipeline.py weekly --plan-json
```

Execution command:

```bash
python3 ~/.hermes/scripts/hindsight_memory_pipeline.py weekly --execute --confirm run-hindsight-pipeline
```

Execution with advisory LLM proposal review and Hermes/cron notification block:

```bash
python3 ~/.hermes/scripts/hindsight_memory_pipeline.py weekly --execute --confirm run-hindsight-pipeline \
  --execute-proposal-review-llm --confirm-proposal-review review-hindsight-proposals \
  --notify-proposal-review
```

## Mode 3: `full`

Purpose: one-click full flow for catch-up or controlled maintenance windows.

Default behavior:

1. Run `daily` mode first.
2. Run `weekly` mode second.
3. Optionally run long-cycle wiki candidate maintenance with `--include-wiki`.

History modes:

- Default: `--history incremental` — process only changed/new session content via submit-state skip.
- Force all conversations/history: `--history all` — ignores submit state and re-retains all manifest records.

Plan command:

```bash
python3 ~/.hermes/scripts/hindsight_memory_pipeline.py full --plan-json
```

Full-history plan:

```bash
python3 ~/.hermes/scripts/hindsight_memory_pipeline.py full --history all --include-wiki --plan-json
```

Execution command:

```bash
python3 ~/.hermes/scripts/hindsight_memory_pipeline.py full --execute --confirm run-hindsight-pipeline
```

## Current implementation entrypoint

The mode orchestrator is:

```bash
~/.hermes/scripts/hindsight_memory_pipeline.py
```

It builds a JSON plan and, when explicitly confirmed, executes existing specialized scripts rather than reimplementing their internals.

## Current maturity judgment

Ready for local operational use:

- Mode taxonomy is defined.
- Dry-run/plan mode is default.
- Execute path is confirmation-gated.
- Session retain default is incremental.
- Weekly includes repair-zone proposal sweep, proposal review packets, and conflict audit.
- Config/preflight layer exists at `~/.hermes/hindsight/pipeline_config.json` + `hindsight_pipeline_preflight.py`.
- Default consolidation tuning is the balanced 20x3 profile (`batch=20`, `llm_batch=20`, `max_round=60`, external `parallel_batches=3`) and strict preflight can verify the runtime container tuning file.
- Tests cover command planning, confirmation gates, proposal building, and proposal review packet behavior.

Not yet ready as a generic public skill without caveats:

- Paths are configurable through `~/.hermes/hindsight/pipeline_config.json`, but provider profiles and Hindsight runtime patches remain environment-specific.
- Execution path needs at least several scheduled daily/weekly cycles before claiming long-term stability.
- Generic publication still needs a documented report destination and optional cron/gateway notification recipe.

## Verification checklist

After any run, report separately:

1. Runtime health: `/health`, queue status, provider mode.
2. Native drain/consolidation: `unconsolidated_base=0`, `failed_base=0`, no pending/processing consolidation.
3. Quality/conflict gate: conflict audit decision and blocking case counts.
4. Auto-retain: `~/.hermes/hindsight/config.json` has expected `auto_retain=false`.
5. Repair-zone: proposal-only outputs generated; no production merge without separate user go/no-go.
6. Proposal review: review packet exists under `review_repair/reviews/`; if `llm_required=true`, verify either LLM advisory status is `reviewed` or report it as pending/no-go.
7. Tests: `pytest` for pipeline/mode, proposal review, and repair sidecar/proposal scripts.
