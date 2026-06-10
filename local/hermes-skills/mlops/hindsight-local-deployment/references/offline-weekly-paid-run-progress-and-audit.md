# Hindsight offline progress monitor and audit caveats

Use this reference when running long Hindsight offline retain/reflect/consolidation jobs and you need live progress plus a final quality check.

## Live progress monitor

Preferred workflow for long all-history backfills:
- keep the worker running
- monitor from a separate tmux/shell window
- read only logs + DB; do not stop the Hindsight process

Helpful scripts:
- `~/.hermes/scripts/hindsight_progress_bar.py`
- `~/.hermes/scripts/hindsight_progress_live.sh`

Typical attach flow:
```bash
tmux new-session -d -s hindsight-progress -x 120 -y 35 '~/.hermes/scripts/hindsight_progress_live.sh 15'
tmux attach -t hindsight-progress
```

What the monitor should distinguish:
- daily backfill can reach 100% while weekly all-history is still running
- `current_run_started` / `saved_now` / `failed_now` describe the current run only
- `429/throttle hits` should be checked separately from normal queue drain

## Bailian / DashScope concurrency lesson

In this run, Bailian GLM-5 tolerated 8-way concurrency but 16-way weekly all-history triggered quota throttling:
- 16 concurrency produced HTTP 429 / `concurrency allocated quota exceeded`
- the remaining weekly units were successfully completed at 8 concurrency

Practical rule: start at 8 for long paid weekly backfills; use 16 only if the provider quota is known to tolerate it.

## Audit caveats

Read-only API summaries can overstate missing linkage.
- `docs_without_units` from API-only audits can be inflated when `document_id` / source fields are sparsely exposed
- verify with PostgreSQL joins before concluding that units are missing
- for this class of audit, check both document→unit linkage and unit→document linkage explicitly

## Interpretation of `observations=0`

`observations=0` is fine when observations are intentionally disabled:
- retain / daily / weekly documents can still be generated
- the queue can still drain normally
- what you lose is the extra observation/proof/source layer for higher-order abstraction and lineage

So:
- `observations=0` does not mean the run failed
- but it does mean the higher-level observation graph is absent or thin

## Current-run caution from this session

The weekly `history-through-2026-W18` run completed with the queue drained and normal-local restored, but the weekly outputs should still be checked for whether the produced documents actually linked memory units as expected. If weekly documents exist but unit linkage is empty, investigate the post-to-Hindsight path rather than assuming the whole pipeline is healthy.
