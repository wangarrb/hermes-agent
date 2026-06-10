# Hindsight Offline V2: sidecar recall + publish runbook

Session learning: when local canonical cards improve high-level recall, do not let them compete for the same top-k slots as source/non-local facts. The safer design is source-preserving augmentation.

## Retrieval merge principle

Preferred context shape:

```text
context = non_local_top_k + local_canonical_sidecar_m
```

Example:

```text
non-local top-5 + local_canonical sidecar 1 => total context 6
```

Why:
- Non-local/source facts preserve grounding, numeric evidence, source document traceability, and expected-layer hits.
- Local canonical cards improve high-level recall, rule/preference recall, and cross-session synthesis.
- A fixed in-top-k cap such as `local_cap = 1` is only a fail-closed guard, not the long-term design. It prevents broad canonical cards from crowding source facts, but still forces a tradeoff that sidecar avoids.

Suggested intent policy:
- Detail / evidence / numeric / ranking query: keep non-local top-k unchanged; append 0-1 local cards only if they add distinct high-level context.
- High-level summary / rule / preference / project-memory query: keep non-local top-k unchanged; append 1-2 local cards.
- Evaluation should report both `non_local_top_k` preservation and `augmented_context` recall, plus token overhead.

## Raw / daily / weekly cleaning model

Do not describe daily/weekly as directly cleaning raw transcripts by default. The safe architecture is layered:

```text
SQLite raw transcript
  -> sqlite import / retain prefilter
  -> Hindsight processed facts
  -> daily consolidation over facts
  -> weekly consolidation over daily outputs
  -> v2 reduce into canonical cards / observations_index
```

Current cleaning responsibilities:
- SQLite import / retain prefilter: handles raw transcript extraction, structured content unwrapping, hard-noise filtering, heartbeat/progress/compression-handoff removal, short low-value turns, and safe session prefilter.
- Daily consolidation: default `--daily-source facts`; deduplicates/merges/abstracts processed facts, not raw.
- Weekly consolidation: default `--weekly-source daily`; merges across daily outputs and history.
- V2 reducer: consumes existing daily/weekly JSON, excludes `_bad-output-backup*` by default, parses `llm_json`, normalizes observations, deduplicates, merges evidence/source docs/tags, and writes `observations_index.jsonl`.

Only use raw daily/weekly modes for explicit diagnosis or special reprocessing, and run dry-run first.

## Meaning of "not published to Hindsight main DB"

Local v2 cards live on disk (for example under an offline_reflect `v2_cards/` directory). They can be consumed by layered recall, but Hindsight API/DB does not know about newly generated cards until publish.

Publishing means inserting/replacing canonical documents and observation memory units in the main Hindsight DB/bank, usually scoped to a prefix like `hermes-offline-canonical::*`.

A local rebuild (`--mode local`) may produce:
- local cards/index files
- eval outputs
- gate report
- proposal preview

But it must not invoke the publisher and should report `published: false`.

## Publish safety policy

Publishing to the main Hindsight DB is a write operation. Require explicit user authorization before running it.

Before publishing, verify:
1. Hindsight health is healthy and database connected.
2. pending/processing/failed operations are all zero or understood.
3. latest local/generic gate is eligible with no case-level regressions.
4. You are not relying on a stale gate if code or inputs changed.
5. DSN / connection string / API keys are never printed; redact as `[REDACTED]`.

Preferred publish path:

```bash
python3 ~/.hermes/scripts/hindsight_offline_v2_rebuild.py --mode publish --json
```

Reason: this reruns reduce + eval + gate, then publishes only if eligible.

Direct publish of existing cards is possible but less safe because it can bypass fresh eval/gate:

```bash
python3 ~/.hermes/scripts/hindsight_offline_v2_publish.py \
  --cards-root ~/.hermes/hindsight/offline_reflect/v2_cards \
  --bank hermes \
  --mode publish \
  --replace \
  --json
```

Use direct publish only when the user explicitly asks to publish exactly the already-built cards and understands the stale-gate risk.

## Rerun decision

- If only publishing current already-built cards: raw/daily/weekly do not need rerun. Safer to rerun v2 rebuild in publish mode so gate is fresh.
- If retrieval merge logic changes (for example switching from in-top-k cap to sidecar augmentation): rerun eval/gate before publish. Raw/daily/weekly still do not need rerun unless source data changed.
- If daily/weekly source JSON changed or bad-output backups were repaired: rerun v2 reduce/eval/gate.
- If retain facts are stale or raw import filtering changed: rerun SQLite retain/import first, then daily/weekly, then v2.
