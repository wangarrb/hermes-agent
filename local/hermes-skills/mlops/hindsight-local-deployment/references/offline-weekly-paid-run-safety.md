# Hindsight offline weekly paid-run safety notes

Session: 2026-05-10 weekly offline reflect/consolidation run using DeepSeek official provider.

## When to use

Use this reference when asked to run Hindsight offline daily/weekly processing with a paid LLM profile, especially on production banks such as `hermes`.

## Proven safe sequence

1. Start from status in normal-local mode.
   - Health must be healthy and database connected.
   - Queue should be empty or understood: pending=0, processing=0, failed=0 is ideal.
   - Production default should be local Ollama with observations disabled and consolidation slots 0.

2. Run dry-run budget before any paid submit.
   - For weekly runs, inspect pending_units, pending_chars, estimated_llm_calls_min/max, block_reasons, and missing_daily_outputs.
   - Treat budget pass as necessary but not sufficient; also check target week/window/source.

3. Prefer the cron runner over a naked offline script when doing production paid runs.
   - Runner handles status snapshots, provider switch, wait-for-queue, restore normal-local, and summary/log writing.
   - Example shape:
     `python3 $HOME/.hermes/scripts/hindsight_offline_cron_runner.py weekly --llm-profile deepseek-v4-flash --week-mode current --prefilter safe`

4. Keep production native observations/consolidation disabled unless explicitly requested.
   - Offline reflect/consolidation can generate documents/experience nodes while `total_observations` remains 0.
   - `total_observations=0` is normal if observations are intentionally disabled.

5. After submit, wait until the queue drains.
   - Verify pending_operations=0, processing_operations=0, failed_operations=0.
   - Check submit ok/failed counts and final summary status/anomaly_count.
   - For long all-history backfills, show progress with a shell/tmux monitor rather than repeatedly explaining status in chat. Copy `scripts/hindsight_progress_bar.py` and `scripts/hindsight_progress_live.sh` from this skill to `~/.hermes/scripts/`, then use:
     `tmux new-session -d -s hindsight-progress -x 120 -y 35 '~/.hermes/scripts/hindsight_progress_live.sh 15'`
     and optionally open a GUI shell with:
     `gnome-terminal --title='Hindsight Offline Progress' -- bash -lc 'tmux attach -t hindsight-progress'`.
     The monitor must only read logs/DB counts and must not stop the Hindsight worker; Ctrl-C exits the monitor only.
   - When daily backfill reaches 100%, do not declare the whole job complete if weekly/all-history is still running. Switch the monitor/summary to weekly progress (`Weekly period=history-through-...`) and continue until weekly units are 100% and the async queue drains.
   - If a high-concurrency run partly succeeds and fails due to provider 429, do not resubmit everything blindly. Re-run the same command without `--force-repost`; the progress cache will skip successful v2 units and only call the LLM for missing units.

6. Always restore and verify normal-local after paid runs.
   - Expected: provider/base URLs point to local Ollama, observations false, consolidation max slots 0.
   - Do not assume restore succeeded; run the status wrapper and inspect env/state.

## Interpretation guidance

- Production data can be described as long-term usable if the pipeline completed with no parse errors, failed ops, queue residue, or anomalies, but do not claim every memory unit is manually verified or permanently valuable.
- Weekly runs may be `history-through-<week>` rather than exactly the last seven calendar days. Report the actual period/window/source shown in the summary.
- A deprecated/blocked sqlite import path failure is not evidence of production data failure; distinguish wrapper/entrypoint failures from the successful offline weekly path.

## 2026-05-10 concrete outcome pattern

A normal completed weekly run looked like:
- budget pass with 8 pending units and 308710 pending chars
- submit complete ok=8 failed=0
- documents 83 -> 91, nodes 700 -> 716
- observations stayed 0 by design
- queue drained, failed_operations=0
- summary status=ok, anomaly_count=0
- normal-local restored

## 2026-05-10 historical weekly backfill pattern

When catching up visible historical weeks through an alternate paid/provider route, a healthy pattern was:
- profile/base_url label: `opencode-go`
- model: `deepseek-v4-flash`
- concurrency: 32 at the per-week runner level
- weeks attempted: `2026-W11` through `2026-W19`
- weeks submitted ok: W11, W12, W13, W15, W16, W19
- weeks skipped: W14, W17, W18 because `pending_units=0`
- final queue state: pending=0, processing=0, failed=0
- final status summary: completed operations only, increased documents/nodes, normal-local restored

Report skipped zero-pending weeks cautiously. They are successful no-op skips only if the source layer is known complete. If raw Hermes `state.db` has messages for those dates but `pending_units=0`, treat it as a source-coverage gap until proven otherwise: check local `offline_reflect/daily/<date>/` markdown files, Hindsight `documents`/`memory_units` coverage, and whether the script is querying legacy `hermes-sqlite::day-topic::YYYY-MM-DD` ids while production now uses native `hermes-session::YYYYMMDD_*` document ids. In that mismatch case, `pending_units=0` means the weekly runner had no daily markdowns and no matched processed facts to backfill; it is not evidence that the week contained no useful raw conversation.

If the user asks whether this is “正式发布”, distinguish pipeline completion from publication gate state: a run can finish `status=ok` while v2 rebuild/publish remains `blocked_keep_local_only` and `published=false`. In that case say it is local candidate/gate output, not a formal main-bank publication.

If the offline script is older than 2026-05-10, patch `offline_hindsight_reflect_consolidate.py` before relying on weekly-source=daily with native session retain. The required fix is: `query_facts_for_days()` must match both legacy `hermes-sqlite::day-topic::YYYY-MM-DD__...` and native `hermes-session::YYYYMMDD...` / `hermes-session::session_YYYYMMDD...`; `parse_doc_day_topic()` should parse native session dates; and `retained_sqlite_days()` should include native session document dates for all-history windows. After patching, verify with a known native-covered day, e.g. `query_facts_for_days('hermes',['2026-04-16'])` returns facts, then run weekly dry-run budgets. This may reveal previously hidden pending daily rebuilds for W15/W16, not just W17/W18 gaps.

Important: never print API keys or secret values from provider env/logs. Refer to key env names only, e.g. `DEEPSEEK_API_KEY`, not their values.
