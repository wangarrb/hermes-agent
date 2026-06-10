# Bailian offline reflect concurrency and live progress monitor

Session: 2026-05-11 Hindsight all-history offline reflect/consolidation using `bailian` profile (DashScope GLM-5 via OpenAI-compatible endpoint).

## What worked

- Retain queue had drained before follow-on reflect/consolidation: no claimable child `retain` pending/processing rows and no failed operations.
- Offline reflect/consolidation was run via:
  ```bash
  HINDSIGHT_PSQL=$HOME/.hindsight-docker/installation/18.1.0/bin/psql \
  HINDSIGHT_OFFLINE_LLM_CONCURRENCY=16 \
  python3 ~/.hermes/scripts/hindsight_minimax_import.py offline-reflect-llm \
    --llm-profile bailian --allow-existing-queue -- \
    --scope both --mode submit \
    --weekly-source daily --weekly-window all-history \
    --backfill-missing-daily --llm-label bailian \
    --concurrency 16 --min-concurrency 8
  ```
- The old concurrency-4 process was stopped with SIGTERM before starting the concurrency-16 process. Exit code 143 from the old wrapper was expected and not a Hindsight failure.
- No real 429 lines were observed during the checked segment. Do not infer a 16-way provider limit from small daily batches: many daily periods only have 1-7 units, so they cannot fully exercise concurrency 16. The weekly/history reduce is the real stress test.

## Live progress UX

User accepted a separate shell window rather than modifying Hermes TUI/statusbar.

Recommended pattern:
1. Copy support scripts from this skill into `~/.hermes/scripts/`:
   - `scripts/hindsight_progress_bar.py`
   - `scripts/hindsight_progress_live.sh`
2. Make the shell script executable:
   ```bash
   chmod +x ~/.hermes/scripts/hindsight_progress_live.sh
   ```
3. Start a tmux monitor:
   ```bash
   tmux new-session -d -s hindsight-progress -x 120 -y 35 '~/.hermes/scripts/hindsight_progress_live.sh 15'
   ```
4. If a GUI is available (`DISPLAY` set, `gnome-terminal` installed), open a terminal attached to it:
   ```bash
   gnome-terminal --title='Hindsight Offline Progress' -- bash -lc 'tmux attach -t hindsight-progress'
   ```
5. If no GUI window appears, the user can attach manually:
   ```bash
   tmux attach -t hindsight-progress
   ```

The monitor should report, at minimum:
- current status (`running` / `finished_or_not_found`)
- adaptive concurrency start/min/backoff
- daily backfill day progress
- known unit outputs
- current day/topic unit progress
- 429 count
- DB operation summary
- log path

Ctrl-C in the monitor exits the monitor only. It must not stop the Hindsight offline worker.

## Pitfalls

- `gnome-terminal` may exit with TTY/job-control warnings when launched from a non-interactive agent shell; verify by checking `tmux list-clients -t hindsight-progress` and `tmux capture-pane` rather than assuming the GUI failed.
- Do not create recurring Hermes cron jobs for this temporary progress view unless the user explicitly wants scheduled delivery. A local cron delivery may not reliably show in the current CLI session; a tmux/terminal window is clearer.
- Never print API keys or `.env` values in progress output.
