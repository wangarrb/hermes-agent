# Monitor window interpretation notes

Session-derived notes for Hindsight consolidation drain monitoring.

## Key behaviors observed

- The live monitor samples every 30 seconds and writes JSONL snapshots.
- In the current script, `window=N` in the rate display means the last N samples, not N minutes.
- With 30s sampling, `window=10` covers about 4.5 minutes and `window=20` covers about 9.5 minutes.
- `ETA=unknown` is expected when `unconsolidated_base` does not decrease over the sampled span.
- `observations` can continue to rise while `unconsolidated_base` stays flat during an active `processing:consolidation=1` job.
- A flat `unconsolidated_base` during a short window is only meaningful if `processing:consolidation=0` and `last_obs_age` is growing. If `processing:consolidation=1` and `last_obs_age` is fresh, keep watching.
- The most useful live monitor line is one that includes all of: `processing`, `completed`, `last_obs_age`, `uncon_delta`, and `obs_delta`.

## Practical check sequence when the tmux pane looks stale

1. Confirm the monitor process exists:
   - `pgrep -af hindsight_observations_monitor.py`
2. Confirm the JSONL file is advancing:
   - `stat $HERMES_HOME/logs/hindsight-observations/20260511-monitor-live.jsonl`
   - `tail -3 $HERMES_HOME/logs/hindsight-observations/20260511-monitor-live.jsonl`
3. Confirm tmux is showing the same process output:
   - `tmux capture-pane -t hindsight-obs-monitor -p -S -30`
4. Only after the above, consider a tmux redraw (`Ctrl-b r`) or reattach.

## Diagnostic rule of thumb

- If `observations` and `last_observation` move but `unconsolidated_base` is flat for a short window, keep watching.
- If `completed:consolidation` increases across jobs but `unconsolidated_base` never drops, inspect whether the consolidator is still marking source memories with `consolidated_at` on commit.
