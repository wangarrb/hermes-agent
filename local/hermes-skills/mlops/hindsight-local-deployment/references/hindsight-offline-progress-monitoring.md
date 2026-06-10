# Hindsight offline progress monitoring

Session-derived workflow for long paid Hindsight backfills and weekly consolidation runs.

## Use when
- A long daily/weekly backfill is still running and the user wants a live progress view.
- You need to confirm whether the run is still active, how far the daily backfill got, and whether weekly all-history is still processing.
- You want to keep the monitor separate from the worker so Ctrl-C only stops the monitor.

## Known-good monitor setup
- Shell monitor script: `$HOME/.hermes/scripts/hindsight_progress_live.sh`
- Status probe script: `$HOME/.hermes/scripts/hindsight_progress_bar.py`
- Tmux session name: `hindsight-progress`

Start the monitor:
```bash
tmux new-session -d -s hindsight-progress -x 120 -y 35 '$HOME/.hermes/scripts/hindsight_progress_live.sh 15'
```

Attach from another terminal if needed:
```bash
tmux attach -t hindsight-progress
```

Optional GUI attach:
```bash
gnome-terminal --title='Hindsight Offline Progress' -- bash -lc 'tmux attach -t hindsight-progress'
```

## What the monitor should report
- Current state: `running` vs `finished_or_not_found`
- Active concurrency: `start`, `min`, `429_backoff`
- Daily backfill completeness: done days / total missing days
- Current weekly all-history stage: `period`, `batch`, `rem_after`, `started`, `failed`
- 429/throttle hit count from the log
- Production DB summary: retain/batch_retain/consolidation counts plus docs/units

## Session-specific lessons
- A long run can finish the daily backfill first and then continue into weekly all-history consolidation; the monitor should switch wording accordingly.
- The 2026-05-11 Bailian/DashScope GLM-5 run hit provider throttling at concurrency 16 with `HTTP 429` / `concurrency allocated quota exceeded. please try again later.`
- Fallback to concurrency 8 with `min-concurrency 4` continued the weekly run without additional 429s in the observed restart.
- Do not infer progress from the cumulative number of saved weekly markdowns alone; distinguish current-run `started/saved_now/failed_now/in_flight_now` from previously processed v2 units.
- When tailing the log, use the newest `*bailian-history-submit-c*.log` file if multiple concurrency runs exist.
- Monitoring should only read log/DB/file state; it must never stop or restart the Hindsight worker.

## Verification snippets
```bash
python3 $HOME/.hermes/scripts/hindsight_progress_bar.py
tmux capture-pane -t hindsight-progress -p | tail -30
pgrep -af 'hindsight_minimax_import.py offline-reflect-llm|offline_hindsight_reflect_consolidate.py'
```
