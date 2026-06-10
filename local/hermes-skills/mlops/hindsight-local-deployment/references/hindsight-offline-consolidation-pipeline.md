# Hindsight Offline Consolidation Pipeline (2026-05-05, revised)

## User correction captured

The offline consolidation design must avoid feeding raw transcripts directly to daily/weekly consolidation by default.

Two-level design:

1. Raw SQLite conversation data enters the system through controlled `retain` / first-pass extraction only.
2. Daily consolidation processes Hindsight processed facts from that day's retained documents.
3. Weekly consolidation performs cross-topic **and cross-history-period** integration over daily consolidation outputs by default, so the scheduled weekly run refreshes the global Hindsight knowledge layer rather than only summarizing the current ISO week.
4. If daily outputs are missing, weekly should first backfill missing retained days from processed facts, then run the global weekly refresh. Direct `--weekly-source facts` is a fallback, not the default.
5. Direct raw SQLite consolidation is a diagnostic / special recompute fallback only; dry-run first because it can explode LLM calls.

## Implemented files

- Wrapper: `~/.hermes/scripts/hindsight_minimax_import.py`
- Worker script: `~/.hermes/scripts/offline_hindsight_reflect_consolidate.py`
- Design doc: `~/.hermes/hindsight/offline_reflect/README.md`

## Default commands

Daily processed-fact consolidation, MiniMax then restore local:

```bash
python3 ~/.hermes/scripts/hindsight_minimax_import.py offline-reflect-minimax -- \
  --scope daily --date yesterday --daily-source facts --mode submit --prefilter safe
```

Weekly/global cross-topic and cross-history consolidation from daily outputs, MiniMax then restore local:

```bash
python3 ~/.hermes/scripts/hindsight_minimax_import.py offline-reflect-minimax -- \
  --scope weekly --weekly-window all-history --weekly-source daily --weekly-group-by all --backfill-missing-daily --mode submit --prefilter safe
```

Current-week-only diagnostic mode, when you explicitly do not want full-history refresh:

```bash
python3 ~/.hermes/scripts/hindsight_minimax_import.py offline-reflect-minimax -- \
  --scope weekly --week previous --weekly-window week --weekly-source daily --weekly-group-by all --mode submit --prefilter safe
```

Weekly direct processed-facts fallback, when daily outputs are unavailable:

```bash
python3 ~/.hermes/scripts/hindsight_minimax_import.py offline-reflect-minimax -- \
  --scope weekly --week previous --weekly-source facts --weekly-group-by all --mode submit --prefilter safe
```

Direct raw re-scan, diagnostic / special recompute only:

```bash
python3 ~/.hermes/scripts/hindsight_minimax_import.py offline-reflect-minimax -- \
  --scope weekly --week previous --weekly-source raw --weekly-group-by all --mode submit --prefilter safe
```

Incremental SQLite retain defaults to safe filtering via wrapper injection:

```bash
python3 ~/.hermes/scripts/hindsight_minimax_import.py sqlite-import-minimax -- \
  --mode submit --group-by day-topic
```

## Design notes

- `weekly` now defaults to `--weekly-window all-history`: it reads all available daily outputs across retained history and produces a global refresh period such as `history-through-2026-W19`.
- `--weekly-window week` is the legacy/current-week-only behavior and should be treated as diagnostic or narrow recompute.
- `sqlite-import-minimax` injects `--prefilter safe` when the caller does not explicitly pass `--prefilter`.
- `offline-reflect-minimax` also injects `--prefilter safe` by default.
- Hindsight built-in observations/consolidation remain disabled by default:
  - `HINDSIGHT_API_ENABLE_OBSERVATIONS=false`
  - `HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS=0`
- The pipeline is an external controlled consolidation flow; it is not Hindsight built-in reflect.
- The wrapper must wait for the Hindsight retain queue to drain before restoring local. Avoid `--no-wait` for formal runs because queued retain may continue under local Ollama after restore.

## Dry-run evidence from implementation

Daily processed-facts dry-run for `2026-05-04`:

```text
source=facts
facts=39
units=2
total_chars=13716
topics: egomotion4d=1, hermes=1
MiniMax calls: 0 (dry-run)
```

Weekly processed-facts cross-topic dry-run for `2026-W18`:

```text
source=facts
facts=814
units=6
total_chars=304241
topic=all for all units
MiniMax calls: 0 (dry-run)
```

This confirmed weekly `--weekly-group-by all` builds cross-topic processed-fact chunks rather than topic-isolated raw chunks. It also reduced the previous raw weekly input from about 1.65M chars to about 304k chars before daily outputs exist; once daily outputs exist, weekly should normally be even smaller.

## Known v1 gaps from 2026-05-06 audit

- Offline consolidation v1 does post retain-friendly markdown back to Hindsight, but its schema has no explicit `observations` / canonical-insight layer. Result: high-level outputs are re-extracted as ordinary `experience/world` facts; `memory_units.fact_type='observation'` stays near-empty because built-in Hindsight observations are intentionally disabled. This is not a queue bug; it is a schema/semantic-layer gap.
- Weekly all-history v1 is a chunked map pass, not a true final global reduce. It splits daily markdown by input size (`max_input_chars`) into many independent `cross-topic__NN` units; each unit may only see 1-5 daily files, so it cannot produce a single coherent global worldview.
- `weekly-group-by all` currently concatenates daily outputs in file/path order, which can mix unrelated topics inside one chunk. Prefer semantic/topic clustering before reduce, then a second-stage global reducer.
- Recall quality is weakened because raw SQLite facts often outrank offline daily/weekly consolidation docs. Add retrieval stratification or a query helper that boosts `hermes-offline-consolidation::{weekly,daily}` for high-level questions and falls back to raw facts for evidence.
- Entity extraction is over-fragmented. Detect alias groups generically (same base name plus suffixes like Frontend/Provider/baseline/config/file extensions, path variants, case variants); do not hard-code a user's local entities. Add alias normalization / entity hygiene before claiming graph-quality parity with original Hindsight.
- Offline outputs may be mixed Chinese/English. The pipeline should expose an explicit output-language option (`auto|zh|en`) and treat Chinese as a first-class language; preserve English only where useful for commands, file paths, variable names, model names, and quoted evidence.
- Add an evaluation harness before changing defaults: coverage, recall precision@k, high-level-answer completeness, source traceability, duplicate/conflict rate, entity alias fragmentation, cost, and queue safety.

## Pitfalls

- If the offline LLM returns `<think>...</think>` plus fenced JSON, do not save the raw chain-of-thought as the consolidation result. Strip `<think>` and parse the final valid JSON object; if the model returns a single knowledge-point object, wrap it into the standard schema.
- Do not post markdown containing a large `## JSON` fenced block back into Hindsight retain. Hindsight retain itself asks the LLM for structured JSON; nested JSON/code fences can trigger MiniMax JSON parse retry loops. Keep full JSON in local `.json/.md` outputs, but post a retain-friendly markdown without the machine-readable JSON block.
- Redact API keys/tokens from offline outputs before saving/posting; record the configuration risk, not the secret value.
- Offline submit to `/memories` must retry transient `ConnectionRefused` / 429 / 5xx. Docker restart windows can make a weekly unit save locally but fail to POST; after such a run, reconcile local `.json/.md` outputs against DB `documents.id` and resubmit missing document IDs under paid-LLM mode.
- If the offline reflect script exits non-zero after some successful POSTs, the wrapper must still wait for the submitted Hindsight retain queue to drain before restoring `normal-local`; otherwise those queued retain jobs may be processed by local Ollama and produce inconsistent quality.
- If the Hindsight queue already has pending/processing tasks, do not start a new MiniMax consolidation run. The wrapper should refuse unless `--allow-existing-queue` is explicit. This prevents overlapping paid MiniMax calls and makes queue accounting understandable.
