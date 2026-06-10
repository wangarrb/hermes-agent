# Hindsight Offline v2 Publish Gate

Session: 2026-05-06. Use when deciding whether local-only Offline Hindsight v2 canonical cards are safe to publish/retain into the main Hindsight bank.

## Problem

Local v2 cards can improve source-layer hits while leaving or reducing term recall. Publishing those cards to Hindsight too early can pollute the main bank with broad summaries that outrank more precise daily/weekly/raw evidence.

## Safety Rule

Fail closed. Local cards remain experimental unless a gate passes on both:

- generic/shareable benchmark
- local/private regression benchmark

The gate must perform no LLM calls, no Hindsight API calls, and no Hindsight writes.

## Gate Script

Script:

```bash
~/.hermes/scripts/hindsight_offline_v2_gate.py
```

Inputs:

- baseline eval JSON with `cards_root=null`
- cards eval JSON produced with `--use-local-cards`
- at least two pairs: generic and local
- local cards root, default `~/.hermes/hindsight/offline_reflect/v2_cards`

Core checks per pair:

- same positive `case_count`
- baseline eval has no cards root
- cards eval has cards root
- average layered term recall improves by at least `--min-term-recall-delta` (default `0.001`)
- layered expected layer hits improve by at least `--min-layer-hit-delta` (default `1`)
- average score does not regress
- no case-level layered term recall regressions by default

If all checks pass and `--emit-proposal` is set, the script emits local-only proposal preview files:

- `canonical-retain-proposal.jsonl`
- `canonical-retain-proposal.md`

These are not retained automatically. Actual Hindsight writes still require a separate explicit workflow and user authorization.

## Commands

```bash
python3 ~/.hermes/scripts/hindsight_offline_v2_reduce.py --mode dry-run --scope all
python3 ~/.hermes/scripts/hindsight_offline_v2_reduce.py --mode local --scope all

python3 ~/.hermes/scripts/hindsight_eval.py \
  --benchmark ~/.hermes/hindsight/eval/benchmark_queries.jsonl \
  --output-dir ~/.hermes/hindsight/eval/runs-generic-default
python3 ~/.hermes/scripts/hindsight_eval.py \
  --benchmark ~/.hermes/hindsight/eval/benchmark_queries.jsonl \
  --use-local-cards \
  --output-dir ~/.hermes/hindsight/eval/runs-generic-cards
python3 ~/.hermes/scripts/hindsight_eval.py \
  --benchmark ~/.hermes/hindsight/eval/benchmark_queries.local.jsonl \
  --output-dir ~/.hermes/hindsight/eval/runs-local-default
python3 ~/.hermes/scripts/hindsight_eval.py \
  --benchmark ~/.hermes/hindsight/eval/benchmark_queries.local.jsonl \
  --use-local-cards \
  --output-dir ~/.hermes/hindsight/eval/runs-local-cards

python3 ~/.hermes/scripts/hindsight_offline_v2_gate.py \
  --pair generic ~/.hermes/hindsight/eval/runs-generic-default/latest.json ~/.hermes/hindsight/eval/runs-generic-cards/latest.json \
  --pair local ~/.hermes/hindsight/eval/runs-local-default/latest.json ~/.hermes/hindsight/eval/runs-local-cards/latest.json \
  --emit-proposal
```

## 2026-05-06 Quality Validation Snapshot

Gate result: `eligible_for_local_proposal`.

Change that unblocked the gate:

- reducer writes `observations_index.jsonl` containing all deduped local observations, not just compact card top-N
- local recall uses significant-token/type/numeric scoring and caps experimental local canonical results in top-k
- local cards remain opt-in via `--use-local-cards`

Generic pair passed:

- score_delta `+6.66`
- term_recall_delta `+0.067`
- expected_layer_hits_delta `+1`
- case regressions `0`

Local pair passed:

- score_delta `+11.72`
- term_recall_delta `+0.107`
- expected_layer_hits_delta `+2`
- case regressions `0`

Proposal emitted as local preview only:

- `~/.hermes/hindsight/offline_reflect/v2_publish_gate/canonical-retain-proposal.jsonl`
- `~/.hermes/hindsight/offline_reflect/v2_publish_gate/canonical-retain-proposal.md`

Report paths:

- `~/.hermes/hindsight/offline_reflect/v2_publish_gate/latest.json`
- `~/.hermes/hindsight/offline_reflect/v2_publish_gate/latest.md`

Actual Hindsight retain/post still requires a separate explicit workflow and user authorization; this gate does not write Hindsight.

## Pitfalls

- Do not treat better layer hits alone as sufficient; term recall must also improve.
- Do not run the gate only on the local/private benchmark; generic shareable benchmark is required to avoid overfitting.
- Do not emit or retain canonical docs if the gate says `blocked_keep_local_only`.
- Do not confuse local proposal preview files with actual Hindsight retain.
