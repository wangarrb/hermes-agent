# Hindsight observations drain false-positive lesson (2026-05-12)

Session-specific lesson from an interrupted offline drain:

- A live tmux monitor window is not proof that the pipeline is still advancing.
- In this session, the monitor kept refreshing while the actual drain worker was absent.
- The correct recovery check is the combination of:
  1. a live `hindsight_observations_drain.py` process,
  2. fresh JSONL entries in the drain/monitor log,
  3. a real decline in `unconsolidated_base` across samples.
- If those three do not hold, treat the run as stalled even if the UI looks active.
- After restarting the drain worker, verify that the new log file gets fresh rows before telling the user that the backlog is moving again.

This reference complements `references/precision-remote-observations-drain.md` and is intended to be read together with it during interrupted-run recovery.