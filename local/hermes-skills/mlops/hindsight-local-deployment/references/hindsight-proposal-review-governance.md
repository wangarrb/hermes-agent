# Hindsight proposal review governance

Use this when converting approved Hindsight review-repair sidecars into production-review proposals and preparing for release/publication.

## Boundary

- Proposal generation and proposal review are local-file workflows only.
- Production repair-zone merge/retain must stay proposal-only until a separate human go/no-go, rollback plan, and quarantine/temp-bank validation exist.
- Advisory LLM review is not approval. A `merge_ready` LLM result can only become `conditional_go`; final approval remains human-only.
- Do not edit `.env` as part of this flow.

## Current entrypoints

Config/preflight:

```bash
python3 ~/.hermes/scripts/hindsight_pipeline_preflight.py --strict-runtime --json
```

Weekly plan:

```bash
python3 ~/.hermes/scripts/hindsight_memory_pipeline.py weekly --plan-json
```

Build proposal-only bundles:

```bash
python3 ~/.hermes/scripts/hindsight_repair_proposal_build.py \
  --approved-index ~/.hermes/hindsight/review_repair/approved/<stem>-observations_index.jsonl \
  --output-root ~/.hermes/hindsight/review_repair/proposals \
  --stem <stem> --top 80 --json
```

Build local human-review packet without LLM disclosure:

```bash
python3 ~/.hermes/scripts/hindsight_proposal_review.py \
  --proposal-json ~/.hermes/hindsight/review_repair/proposals/<stem>-canonical-proposals.json \
  --review-root ~/.hermes/hindsight/review_repair/reviews \
  --top 80 --notify --json
```

Run advisory LLM review only with explicit confirmation:

```bash
python3 ~/.hermes/scripts/hindsight_proposal_review.py \
  --proposal-json ~/.hermes/hindsight/review_repair/proposals/<stem>-canonical-proposals.json \
  --review-root ~/.hermes/hindsight/review_repair/reviews \
  --top 80 \
  --execute-llm --confirm-review review-hindsight-proposals \
  --notify --json
```

Pipeline wrapper for weekly advisory LLM review:

```bash
python3 ~/.hermes/scripts/hindsight_memory_pipeline.py weekly --execute --confirm run-hindsight-pipeline \
  --execute-proposal-review-llm --confirm-proposal-review review-hindsight-proposals \
  --notify-proposal-review
```

## Required safety behavior

- `hindsight_proposal_review.py` must not call Hindsight retain/merge/delete endpoints.
- Deterministic blocked proposals, especially `secret_like_material`, must not be sent to external LLM. They should receive `llm_judgement.status=skipped_deterministic_block` and remain `no_go`.
- Secret scanners in proposal build and review should cover at least:
  - `sk-...` including dots/dashes/underscores,
  - `api_key/token/secret/password/passwd` assignments,
  - bearer tokens,
  - AWS-style `AKIA...` ids.
- Review packets should state:
  - `production_mutation_allowed=false`,
  - `production_merge_or_retain_executed=false`,
  - human final decision pending,
  - rollback/quarantine plan required before any future production write.

## Verification checklist

Run after edits:

```bash
python3 -m py_compile \
  ~/.hermes/scripts/hindsight_pipeline_common.py \
  ~/.hermes/scripts/hindsight_pipeline_preflight.py \
  ~/.hermes/scripts/hindsight_memory_pipeline.py \
  ~/.hermes/scripts/hindsight_repair_proposal_build.py \
  ~/.hermes/scripts/hindsight_proposal_review.py

python3 -m pytest \
  ~/.hermes/scripts/tests/test_hindsight_memory_pipeline.py \
  ~/.hermes/scripts/tests/test_hindsight_repair_proposal_build.py \
  ~/.hermes/scripts/tests/test_hindsight_proposal_review.py -q

python3 ~/.hermes/scripts/hindsight_pipeline_preflight.py --strict-runtime --json
```

Expected:

- tests pass,
- strict preflight has `blocking=0`, `warnings=0`,
- local and runtime consolidation tuning agree on `parallel_batches=3`,
- weekly dry-run plan contains `proposal_review` and `production_writes_possible=false`.

## Operational note from 2026-05-13

A review found a blocker where deterministic-blocked/secret-like proposals could still be sent to the advisory LLM. The fix is to check deterministic block codes before `llm_fn(build_llm_messages(...))` and skip the LLM call for blocked proposals. Keep this as a regression test: a fake LLM must record zero calls for a proposal containing a secret-like `sk-...` token.
