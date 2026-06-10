# Hindsight Cron Schedule — Production Pipeline Jobs

Stable as of 2026-05-13. Three Hermes cron jobs plus the pre-existing daily report.

## Schedule Overview

```
01:00 CST — hindsight-daily-pipeline (每天)
05:00 CST — hindsight-weekly-pipeline (每周日)
08:30 CST — hermes-daily-report (每天，存量)
12:00 CST — wiki-biweekly-maintenance (双周周日，偶数周)
```

## Job Details

### 1. hindsight-daily-pipeline (b631d8d0ca1c)

```bash
hermes cron create '0 1 * * *' \
  --name hindsight-daily-pipeline \
  --skills hindsight-local-deployment,hindsight-consolidation-operations \
  --enabled_toolsets terminal,file \
  --deliver 'weixin:o9cq80zuypUIya6lGtgkneJvZquY@im.wechat'
```

Prompt kernel (self-contained, no user questions):

1. Health check: `curl -s http://127.0.0.1:8888/health` → healthy + connected
2. Dry-run plan: `hindsight_memory_pipeline.py daily --plan-json`
3. Execute: `hindsight_memory_pipeline.py daily --execute --confirm run-hindsight-pipeline`
4. Report: stage durations, doc/obs/mu deltas, anomalies
5. If foreground timeout but Hindsight still working → report "wait for idle, resume from safe stage"
6. Never edit `.env`

Skills loaded: `hindsight-local-deployment`, `hindsight-consolidation-operations`

### 2. hindsight-weekly-pipeline (89327c29066b)

```bash
hermes cron create '0 5 * * 0' \
  --name hindsight-weekly-pipeline \
  --skills hindsight-local-deployment,hindsight-consolidation-operations \
  --enabled_toolsets terminal,file \
  --deliver 'weixin:o9cq80zuypUIya6lGtgkneJvZquY@im.wechat'
```

Prompt kernel:

1. Health check
2. Confirm daily pipeline completed (check recent daily output)
3. Dry-run plan: `hindsight_memory_pipeline.py weekly --plan-json`
4. Execute with proposal review LLM:
   `hindsight_memory_pipeline.py weekly --execute --confirm run-hindsight-pipeline --execute-proposal-review-llm --confirm-proposal-review review-hindsight-proposals`
5. Report: conflict audit (case_count, blocking_cases, by_severity), proposal review packets, V2 publish status, stage durations
6. List any human-review-required proposals separately
7. Never edit `.env`

### 3. wiki-biweekly-maintenance (f381e3f57bf1) — Biweekly Guard

```bash
hermes cron create '0 12 * * 0' \
  --name wiki-biweekly-maintenance \
  --skills llm-wiki,hindsight-local-deployment \
  --enabled_toolsets terminal,file \
  --deliver 'weixin:o9cq80zuypUIya6lGtgkneJvZquY@im.wechat'
```

Prompt kernel:

- **Step 0 — Biweekly guard**: `python3 -c "import datetime; w=datetime.date.today().isocalendar()[1]; print(w, w%2)"`. If `w%2==1` (odd week) → exit with "奇数周不执行". Runs only on even weeks.
- Step 1 — Health check
- Step 2 — Confirm V2 publish is today's: `ls -la ~/.hermes/hindsight/offline_reflect/v2_rebuild/latest.json`; if mtime ≠ today → exit "weekly pipeline 未完成"
- Step 3 — Queue drained: `pending_operations=0`
- Step 4 — Read wiki schema/index/log
- Step 5 — Filter Hindsight V2 cards/weekly output for stable conclusions
- Step 6 — Write candidate report to `~/wiki/auto-maintenance/wiki-auto-maintenance-YYYYMMDD-HHmmss.md`
- Step 7 — Run wiki lint
- Step 8 — Report: candidates, wiki health, source stats
- Step 9 — Never modify main wiki (concepts/ projects/ queries/ etc.)

Skills loaded: `llm-wiki`, `hindsight-local-deployment`

### 4. hermes-daily-report (d913e5e05007) — Pre-existing

Schedule: `30 8 * * *`. Uses script `daily_report.py`. Delivers to WeChat. This job was not modified; it pre-dates the pipeline cron rollout.

## Biweekly Guard Pattern

Since Hermes cron does not natively support "every 2 weeks on Sunday," the wiki job runs every Sunday at noon but exits immediately on odd ISO weeks. The first Python command checks `datetime.date.today().isocalendar()[1] % 2` and skips on odd weeks.

Implication: the wiki job still shows as "scheduled every Sunday" in `hermes cron list`, and it will consume one Hermes invocation on odd weeks (to run the guard). This is an acceptable trade-off until native biweekly cron support exists.

## Job Lifecycle Notes

- All three pipeline jobs have `repeat: forever` and `enabled: true`.
- An old `wiki-auto-maintenance-after-hindsight` job (18d333a5e675, paused, Sunday 05:00) was removed during rollout to avoid overlap with the new weekly pipeline.
- If any job needs pausing, use `hermes cron pause <job_id>` rather than removing it.
- The WeChat delivery target is `weixin:o9cq80zuypUIya6lGtgkneJvZquY@im.wechat` (the user's WeChat UID via iLink bot).

## Comprehensive Guide

A detailed 960-line design and operations guide lives at:
`~/wiki/auto-maintenance/hindsight-offline-guide.md`

It covers Hindsight concepts, architecture, installation, parameter tuning, pitfalls, and the full cron schedule. Point new users or other Hermes sessions to this file for a complete reference.
