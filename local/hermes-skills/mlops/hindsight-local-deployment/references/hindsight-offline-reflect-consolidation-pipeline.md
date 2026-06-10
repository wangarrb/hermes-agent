# Hindsight Offline Reflect / Consolidation Pipeline

Session learning from 2026-05-05: for this user, do not rely on Hindsight built-in reflect as the default long-term knowledge evolution path. Use a controlled offline pipeline that calls MiniMax explicitly, writes auditable artifacts, posts the result back to Hindsight, then restores local Ollama mode.

2026-05-05 correction: daily/weekly consolidation should not directly consume raw transcripts by default. Raw SQLite is too large and can explode LLM calls. Raw data should enter through controlled retain / first-pass extraction; consolidation consumes processed facts or summaries.

## Trigger

Use this when the user asks to design or run daily/weekly reflect, consolidation, long-term knowledge cleanup, or high-quality memory evolution for Hermes/Hindsight.

## Architecture

- Daily consolidation:
  - Default input: Hindsight `memory_units` / processed facts extracted from that day's retained SQLite documents.
  - Filtering: default `--prefilter safe` for any raw fallback path.
  - Grouping: by topic, split long units by fact boundary.
  - LLM: MiniMax-M2.7, direct API call from the offline script.
  - Output: JSON + Markdown under `~/.hermes/hindsight/offline_reflect/daily/<date>/`.
  - Hindsight write: post Markdown as async memory so Hindsight builds searchable facts/nodes.
  - Document ID shape: `hermes-offline-consolidation::daily::<date>::<topic>::<idx>::<hash>`.

- Weekly consolidation:
  - Default input: existing daily consolidation Markdown, not one week of raw conversation.
  - Fallback input: `--weekly-source facts` for one week of processed facts if daily artifacts do not exist yet.
  - Raw input: `--weekly-source raw` only for diagnostic/special recompute; dry-run first because raw weekly input can explode in size.
  - Purpose: merge duplicate daily insights, raise abstraction level, form a weekly knowledge system, list comparisons/risks/open questions.
  - Output: JSON + Markdown under `~/.hermes/hindsight/offline_reflect/weekly/<YYYY-Www>/`.
  - Document ID shape: `hermes-offline-consolidation::weekly::<YYYY-Www>::<topic>::<idx>::<hash>`.

## Commands

Dry-run daily processed-facts consolidation, no MiniMax call:

```bash
python3 ~/.hermes/scripts/offline_hindsight_reflect_consolidate.py \
  --scope daily --date yesterday --daily-source facts --mode dry-run --prefilter safe
```

Submit daily with MiniMax and restore local mode after queue drain:

```bash
python3 ~/.hermes/scripts/hindsight_minimax_import.py offline-reflect-minimax -- \
  --scope daily --date yesterday --daily-source facts --mode submit --prefilter safe
```

Submit weekly consolidation from daily outputs:

```bash
python3 ~/.hermes/scripts/hindsight_minimax_import.py offline-reflect-minimax -- \
  --scope weekly --week previous --weekly-source daily --weekly-group-by all --mode submit --prefilter safe
```

Fallback weekly consolidation directly from processed facts:

```bash
python3 ~/.hermes/scripts/hindsight_minimax_import.py offline-reflect-minimax -- \
  --scope weekly --week previous --weekly-source facts --weekly-group-by all --mode submit --prefilter safe
```

SQLite incremental retain also defaults to safe through the wrapper:

```bash
python3 ~/.hermes/scripts/hindsight_minimax_import.py sqlite-import-minimax -- \
  --mode submit --group-by day-topic
```

The wrapper injects `--prefilter safe` if the caller did not specify a prefilter.

## Safety rules

1. Do not start offline consolidation while Hindsight has pending/processing operations unless the user explicitly accepts `--allow-existing-queue`.
2. Do not use `--no-wait` for formal MiniMax runs: posted docs may be retained after restore by local Ollama instead of MiniMax.
3. Keep Hindsight built-in observations/consolidation off by default:
   - `HINDSIGHT_API_ENABLE_OBSERVATIONS=false`
   - `HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS=0`
4. Only enable Hindsight built-in observations with `--enable-hindsight-observations` when explicitly requested.
5. Always dry-run first and inspect units/chars before paid MiniMax calls.
6. Weekly consolidation should normally run after daily outputs exist; avoid `--scope both` as a formal weekly run because weekly reads already-written daily artifacts.

## Verified dry-run observations from 2026-05-05

- Daily `2026-05-04` processed facts dry-run:
  - source=facts
  - facts=39
  - units=2
  - total_chars=13,716
  - topics: egomotion4d=1, hermes=1
- Weekly `2026-W18` processed facts fallback dry-run:
  - source=facts
  - facts=814
  - units=6
  - total_chars=304,241
  - topic=all for all units
- Previous raw weekly dry-run was about 1.65M chars, confirming raw weekly should not be the normal path.

## Files

- Orchestrator: `~/.hermes/scripts/hindsight_minimax_import.py`
- Offline consolidation script: `~/.hermes/scripts/offline_hindsight_reflect_consolidate.py`
- Local design README: `~/.hermes/hindsight/offline_reflect/README.md`
