# Hindsight offline v2 rebuild notes (2026-05-06)

## What changed
- Offline consolidation now has a local-only v2 reduction layer that builds canonical cards under `~/.hermes/hindsight/offline_reflect/v2_cards/`.
- Layered recall/eval prefers local canonical cards when present.
- Publish gate is fail-closed: only allow local proposal generation when generic + local benchmarks both improve.

## Useful commands
```bash
python3 ~/.hermes/scripts/hindsight_offline_v2_reduce.py --mode dry-run --scope all
python3 ~/.hermes/scripts/hindsight_offline_v2_reduce.py --mode local --scope all
python3 ~/.hermes/scripts/hindsight_eval.py --benchmark ~/.hermes/hindsight/eval/benchmark_queries.local.jsonl
python3 ~/.hermes/scripts/hindsight_eval.py --benchmark ~/.hermes/hindsight/eval/benchmark_queries.local.jsonl --no-local-cards
```

## Guardrails
- Do not write v2 cards straight into the main wiki or Hindsight without the publish gate.
- Keep `observations_index.jsonl` alongside cards; compact cards alone lose numeric detail.
- Use `--include-backups` only for diagnosis; default reduce excludes `_bad-output-backup*`.

## Session-specific result
- Latest rebuild finished with `decision=eligible_for_local_proposal` and `published=true`.
- Main write-back included canonical docs, canonical observations, embeddings, unit entities, and semantic links.
