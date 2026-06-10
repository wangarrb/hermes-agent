# Hindsight offline pipeline readiness and publish gate

Use when evaluating whether the Hindsight offline/session pipeline is actually processing useful information, safe for long-term use, or ready to publish as a generic skill.

## Evidence that the pipeline is effective

Treat the pipeline as effective only when multiple independent signals agree:

- Hindsight recall/reflect can retrieve offline consolidation principles and concrete domain knowledge from the main bank.
- Main-bank audit shows queue health: `pending_operations=0`, `pending_consolidation=0`, and no unexplained failed source units.
- Offline outputs contain structured layers such as `executive_summary`, `knowledge_points`, `canonical_observations`, and `source_refs`.
- V2/local layered recall eval improves term recall and expected layer hits without case-level regressions.
- Temp-bank repair runs produce memory units/observations while leaving production untouched.

A useful temp-bank result is not production-ready by itself. It proves candidate value, not database cleanliness.

## Required readiness checks

1. Hindsight runtime and config:
   - `/health` is healthy.
   - `auto_retain=false` after paid/offline runs.
   - Daily auto recall should use `recall_prefetch_method=recall`; `reflect` risks LLM call amplification.
2. Main-bank quality audit:
   - run `hindsight_bank_quality_audit.py --bank hermes --db-fallback always`.
   - inspect contamination counts, broad/system tags, duplicate text groups, docs without units, and missing source refs.
3. Offline/V2 quality:
   - inspect `offline_reflect/v2_cards/manifest.json` and `observations_index.jsonl` counts.
   - compare generic and local eval pairs with and without local cards.
   - rerun publish gate after the latest conflict audit; do not rely on an older blocked/pass result.
4. Conflict/lineage:
   - latest conflict audit must pass at the configured block severity.
   - if it previously blocked, rerun the V2 gate after repair/audit pass.
5. Scheduling:
   - verify the offline pipeline is actually scheduled and enabled. Check Hermes cron, user crontab, and systemd timers/services. Scripts and logs existing are not proof of an active schedule.
6. LLM budget:
   - weekly must pass dry-run budget gates before paid submit.
   - daily should also have max-units/max-chars/max-calls guards before it is treated as unattended-safe.

## Long-term safety rules

- Keep clean production bank, temp repair bank, raw/evidence bank, and local/approved sidecars as separate layers.
- Do not copy temp-bank memory units or sidecar text directly into `hermes`.
- Promotion path: recall/repair hit -> source lineage trace -> confirm against production facts -> distill minimal canonical observation/repair note -> proposal/apply -> mark sidecar as promoted/superseded.
- Use sidecar recall as source-preserving augmentation: append limited sidecar hits after normal Hindsight top-N; do not let sidecar replace source evidence.
- Raw/evidence banks should use chunk/raw indexing and stay out of default auto-recall unless the user asks for evidence/details.

## LLM-call explosion risks

Primary risks:

- `recall_prefetch_method=reflect` in daily Hermes auto-recall.
- daily offline jobs without a weekly-style budget gate.
- native consolidation with large LLM batch size and high parallel scope fanout.
- large temp-bank experiments without route quotas.
- automatic promotion of all sidecar facts into production.

Mitigations:

- Prefer `recall` over `reflect` for daily recall prefetch.
- Keep weekly budget limits, e.g. max pending units / chars, and add similar daily limits.
- Tune consolidation by estimated recall/search fanout, not raw parallelism; stable profiles are around 20x3 or 25x2 rather than blindly increasing batch size.
- Keep review-backlog scorer bounded by package/call budgets.
- Require proposal + gate for production promotion.

## Publishability assessment

Do not publish the current local Hindsight pipeline skill as-is. It is a personal ops skill until it is split into:

- a generic class-level skill: offline memory governance, session clean retain, temp-bank validation, sidecar recall, quality gates, promotion proposals, LLM budget guards, conflict audit;
- a local overlay: `$HOME/...` paths, bank names, providers/models, private benchmark cases, project-specific examples, cron details.

A public generic skill must remove private paths, credentials/provider assumptions, local project names, and one-off run artifacts from the main SKILL.md. Keep those in local references or overlays.

## Quick verdict language

- Effective information processing: yes, if recall/eval/gates show useful structured outputs.
- Production cleanliness: only if main-bank audit and recall smoke pass; temp-bank success is insufficient.
- Unattended stability: not until scheduling and daily budget gates are verified.
- Generic publish readiness: not until local details are parameterized and split from the class-level workflow.
