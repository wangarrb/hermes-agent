# Hindsight/Hermes daily report cron checklist

Use this when generating the scheduled Hermes daily report from `~/.hermes/scripts/daily_report.py` output plus live Hindsight/Wiki state.

## Why this exists

`daily_report.py` may produce a short/truncated briefing. The user explicitly wants the cron report to preserve that injected content, but also include operational state that the script may omit: local model provider details, Hindsight queue/config, offline consolidation coverage, weekly/global/all-history state, and wiki updates.

## Required enrichment steps

1. Run or preserve `daily_report.py` output first; do not delete its key progress bullets even if they are truncated.
2. Read latest offline pipeline summaries:
   - `~/.hermes/logs/hindsight-offline-pipeline/latest-daily.json`
   - `~/.hermes/logs/hindsight-offline-pipeline/latest-weekly.json`
   - `~/.hermes/logs/hindsight-offline-pipeline/latest-both.json`
   - `~/.hermes/logs/hindsight-offline-pipeline/latest-summary.json`
3. Check live Hindsight status rather than trusting summaries alone:
   - `python3 ~/.hermes/scripts/hindsight_minimax_import.py status`
   - include health, pending/processing/failed, total documents/nodes/observations.
4. Preserve local/background model information in model stats:
   - provider, model, base_url, location for Hindsight daily local model, e.g. `ollama / qwen3.5:9b-local / http://127.0.0.1:11434/v1 / local`.
   - include Hermes session provider/base_url from `~/.hermes/state.db` when available.
5. Report offline consolidation coverage:
   - latest daily dir count and paths under `~/.hermes/hindsight/offline_reflect/daily/<date>/`.
   - if `weekly/` or `global/` directories do not exist, say so explicitly.
   - still check all-history/global V2 artifacts such as `v2_cards/global/global.md`, `v2_cards/manifest.json`, and `v2_rebuild/latest.json`; these can be the actual high-level/global output even when `Weekly 0`.
6. Report V2 rebuild/publish state:
   - input files, observation count, card count, conflict summary, gate decision.
   - publish result: inserted/deleted documents, inserted observations, semantic links, skipped missing targets, backup path.
7. Report wiki update state:
   - `~/wiki/auto-maintenance/latest.md` summary, candidate counts, health findings, and whether main wiki files were modified.
   - note if wiki report is stale relative to latest V2 publish.
8. Report cron/delivery anomalies:
   - `hermes cron list` for job state and delivery errors.
   - Do not call send_message or attempt delivery from the cron response; final response is delivered by the scheduler.

## Pitfalls

- Do not collapse all consolidation to `Daily N Weekly M`; the user expects daily + weekly/global/all-history coverage with counts and recent paths.
- Do not omit Hindsight's local/background model. It is not visible in Hermes session stats but is operationally important.
- Stats API can be insufficient for queue truth in some Hindsight bugs; for suspicious paid-LLM burn, also inspect DB operations and Docker logs as described in the main skill.
- A weekly `BrokenPipeError` may indicate orchestration/stdout failure while Hindsight itself is healthy; report both the failure and live queue status.
- `~/wiki/auto-maintenance/latest.md` can lag behind the newest V2 publish; state the timestamp and prefer `v2_rebuild/latest.json` for canonical publish counts.
