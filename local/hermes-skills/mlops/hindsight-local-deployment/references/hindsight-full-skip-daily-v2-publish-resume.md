# Full pipeline resume with `--skip-daily` and V2 publish

Use this when a full Hindsight run has already completed session manifest/retain/daily reflect and needs to continue from weekly/V2 gates without redoing daily work.

## Command pattern

```bash
python3 $HERMES_HOME/scripts/hindsight_memory_pipeline.py full \
  --skip-daily \
  --history incremental \
  --include-wiki \
  --strict-runtime \
  --execute --confirm run-hindsight-pipeline \
  --execute-proposal-review-llm \
  --confirm-proposal-review review-hindsight-proposals \
  --notify-proposal-review
```

`--skip-daily` is valid only with `full` mode. It skips:

- session manifest build
- session retain
- daily reflect
- the daily V2 rebuild gate

Then it runs:

1. preflight
2. runtime status
3. weekly reflect, including missing daily backfill if configured
4. V2 rebuild/publish gate
5. conflict audit
6. repair-zone proposal build
7. proposal review
8. optional wiki auto-maintenance

## Operational notes from the 2026-05-13/14 runs

- Weekly all-history backfill can look quiet in the outer log while it is generating daily/weekly markdown files. Check newest files under `$HERMES_HOME/hindsight/offline_reflect/{daily,weekly}/` before assuming stall.
- After weekly submit, wait for native Hindsight operations to drain before continuing; use `hindsight_consolidation_status.py --skip-psql --json`.
- The wrapper may recreate/restart the Hindsight container to restore normal-local/precision remote mode after offline weekly work. Verify `/health` and bank config afterwards.
- V2 publish can appear stuck after gate files are written because a `docker exec` child is computing bge-m3 embeddings inside the container. `docker top hindsight` may show a high-CPU command like `SentenceTransformer(... BAAI/bge-m3 ...)`. Do not kill it if CPU is active and stats/documents are still advancing.
- Local V2 gate files can be written before the publish subprocess returns. Treat completion as the pipeline step returning rc=0, not just file timestamps.
- Do not rely on short-lived `no_agent` cron jobs for long `full --skip-daily` verification. The Hermes cron runner may remove the one-shot job after its scheduling timeout while the intended long wait/run never continues. After scheduling, verify both the cron list and a durable process/log are still alive; if the job disappeared and no verifier process exists, restart the verification as an explicit background watchdog process that waits for Hindsight idle, then runs `full --skip-daily` once. This avoids accidentally rerunning daily/session retain while still exercising weekly/V2/conflict/proposal stages.

## Verification checklist

- `full --skip-daily --plan-json` has exactly this post-status execution sequence: `weekly_reflect`, `v2_rebuild_gate`, `conflict_audit`, `repair_zone_proposals`, `proposal_review` (plus preflight/runtime status before it). It must not include `build_session_manifest`, `retain_session_manifest`, or `daily_reflect`.
- `daily --skip-daily`, `weekly --skip-daily`, and `preflight --skip-daily` fail closed with `--skip-daily is only valid with full mode`.
- `pipeline_runs/<timestamp>-full-pipeline-run.json` exists and has `status=ok`.
- Step list includes `weekly_reflect`, `v2_rebuild_gate`, `conflict_audit`, `repair_zone_proposals`, `proposal_review`, and optional `wiki_auto_maintenance`.
- `v2_rebuild/latest.json` has `mode=publish`, `published=true`, and `errors=[]`.
- `conflict_audit/latest.json` has no blocking cases.
- Proposal review packet states `production_mutation_allowed=false` and `production_merge_or_retain_executed=false`.
- Final Hindsight status has `pending=0` and `active=0`; pre-existing `failed` rows should be reported separately from new failures.
- Installed scripts and packaged skill scripts are byte-identical after any hotfix; otherwise a future installer can regress the active fix.
- Run an independent pre-release reviewer after machine checks. If the reviewer finds low-risk non-blocking cleanup, fix it, rerun focused tests plus final gates, and then treat the release as clean.

## Safety boundary

Proposal review is not production approval. Even `conditional_go` proposals still require a separate exact payload, production snapshot/export, quarantine/temp-bank validation, rollback plan, and explicit human go/no-go before any retain/merge/delete workflow.
