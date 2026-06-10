# Hindsight review backlog temp-bank one-week flow

Use this after the review-backlog scorer has passed coverage gates. This is the safe bridge between sidecar scoring and any production mutation.

## Trigger

- Backlog scorer ran sidecar-only.
- Credential-like records were skipped.
- `missing_scores_from_llm == 0` or an explicit coverage gate passed.
- Scorer output contains normalized helper fields (`scores_normalized`, `score_mean_0_1`) for ranking/filtering.

## Decision boundary

Do not interpret "one-week full flow" as direct production replace/re-retain/consolidation/observations. The first full flow should be temp-bank validation:

1. Select only A-route candidates (`A_next_repair_candidate`) unless the user explicitly approves another route.
2. Create a dedicated temp bank, e.g. `hermes_tmp_review_repair_a_<scope>_<date>`.
3. Convert candidates into a retain manifest that references original session/source spans, not scorer summaries.
4. Run retain runner dry-run first.
5. Execute temp-bank retain only after the dry-run count and filters match expectation.
6. Wait for async operations to complete.
7. Run fact-quality gate against the temp bank.
8. Run a recall smoke test on expected topics.
9. Produce a production proposal; do not mutate production until the user gives a separate go.

## Gates

- Secrets: credential-like/API-key/token/password/connection-string records stay manual-review only. Do not send them to LLM, Hindsight, or retain.
- Coverage: every candidate submitted to the LLM scorer must have an actual model score; defaulted/missing scores are unscored, not low-value.
- Ranking: use `score_mean_0_1` for ordering, but never override route/risk/anomaly gates with score alone.
- Production: temp-bank success is necessary but not sufficient for production mutation. Require a separate go/no-go and a rollback/discard plan.
- Recall-zone promotion: useful temp-bank/repair-zone observations may be exported to an approved local sidecar first, then searched during weekly conflict/high-dimensional extraction. Only confirmed/distilled canonical repair proposals may merge into production; do not copy temp-bank units directly.

## Current local ops posture

- `recall_prefetch_method=reflect` is allowed in this deployment; LLM usage is considered quota-covered and tracked by daily reports.
- Daily/weekly paid calls are acceptable when guarded by the existing reports/budget checks.
- Hermes cron scheduling can wait until the user explicitly asks to finish it.
- Native consolidation is expected to use the balanced tuning profile (`consolidation_batch_size=20`, `consolidation_llm_batch_size=20`, external `parallel_batches=3`, official `max_memories_per_round=60`; old local wrappers may show `max_memories_per_job=60`). If live bank config or container env still shows 50x*, treat it as a rollout mismatch to fix at the next safe idle/restart point.

## Commands to prefer

- Retain runner should default to dry-run; execution requires the explicit confirm token used by the script (for example `--execute --confirm retain-hindsight-session-manifest`).
- Fact-quality gate should be read-only and should check: expected docs, DB docs, memory units, observations, docs without units, parent coverage, artifact counts, operations, secret-like/prompt-leak/context-compaction artifacts.
- Approved repair-zone sidecar export is local/read-only with respect to Hindsight:
  - `python3 ~/.hermes/scripts/hindsight_repair_sidecar_export.py --bank <temp_bank> --sidecar <sidecar-cleaned.json> --audit-json <bank-audit-cleaned.json> --output-root ~/.hermes/hindsight/review_repair/approved --stem <safe-stem> --json`
  - Output: `<stem>-observations_index.jsonl`, `<stem>-summary.json`, `<stem>-rejected.jsonl`.
- Layered recall can include approved repair sidecars without replacing normal Hindsight hits:
  - `python3 ~/.hermes/scripts/hindsight_recall_layered.py --mode mixed --limit 3 --raw-limit 20 --repair-sidecar-root ~/.hermes/hindsight/review_repair/approved "<query>"`
  - Expect `layer=approved_repair_sidecar` hits appended after main non-local results when relevant.
- Build production-review proposals from approved sidecar records without writing Hindsight:
  - `python3 ~/.hermes/scripts/hindsight_repair_proposal_build.py --approved-index ~/.hermes/hindsight/review_repair/approved/<stem>-observations_index.jsonl --output-root ~/.hermes/hindsight/review_repair/proposals --stem <stem> --top 80 --json`
  - Outputs: `<stem>-canonical-proposals.json`, `<stem>-canonical-proposals.md`, `<stem>-quality-report.json`.
  - Proposal output remains `proposal_only_no_write`; production merge/retain still requires a separate user go/no-go and rollback/quarantine plan.
- Build the release review packet from proposal JSONs, still without production writes:
  - dry/manual path: `python3 ~/.hermes/scripts/hindsight_proposal_review.py --proposal-json ~/.hermes/hindsight/review_repair/proposals/<stem>-canonical-proposals.json --review-root ~/.hermes/hindsight/review_repair/reviews --json`
  - advisory LLM path: add `--execute-llm --confirm-review review-hindsight-proposals --notify`.
  - LLM `merge_ready` only yields `conditional_go`; final approval remains human-only and any production retain/merge is a separate workflow.

## 2026-05-09 case note

A sample40 review run scored 39 non-secret records with hardened MiniMax scorer coverage 39/39; one credential-like record was skipped. The correct next step was A-route temp-bank validation (13 candidates), not production mutation. This case established normalized 0-1 helper scores plus the rule that route/risk/anomaly gates dominate score-based filtering.
