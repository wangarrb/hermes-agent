# External content import (OpenClaw, chat-memo)

## Overview

Build/retain defaults now target `hermes`. Keep `--bank-target hermes` and `--bank hermes` explicit in production commands; use a separate `external_*_10pct_*` bank only for isolated smoke/eval runs.

These scripts are **manual-only** — not integrated into the daily/weekly pipeline. They have their own submit-state tracking (`external_import/submit_state.json`) separate from Hermes session retain.

## Supported sources

| Source | Script flag | Input |
|--------|-------------|-------|
| OpenClaw lcm.db | `--source openclaw-lcm --db /path/to/lcm.db` | SQLite (112 conversations, 43K messages typical) |
| Chat-memo txt exports | `--source chat-memo --chat-memo-dir /path/to/dir` | `.txt` files with Title/URL/Platform/Created headers |

## Workflow

### 1. Build manifest (dry-run, no Hindsight writes)

```bash
# OpenClaw
python3 $HERMES_HOME/scripts/hindsight_external_manifest.py \
  --source openclaw-lcm \
  --db /home/wyr/.openclaw/lcm.db \
  --bank-target hermes \
  --include-content \
  --json

# Chat-memo
python3 $HERMES_HOME/scripts/hindsight_external_manifest.py \
  --source chat-memo \
  --chat-memo-dir /path/to/chat-memo-dir \
  --bank-target hermes \
  --include-content \
  --json
```

Output: `external_import/manifests/<timestamp>-external-manifest.jsonl` (records + metadata + tags)

### 2. Submit manifest retain + observation drain

```bash
python3 $HERMES_HOME/scripts/hindsight_external_retain_runner.py \
  --manifest /path/to/manifest.jsonl \
  --bank hermes \
  --action production \
  --execute \
  --confirm retain-hindsight-external-manifest \
  --enable-observations \
  --wait-consolidation \
  --consolidation-timeout-s 86400 \
  --json
```

`--bank` now defaults to `hermes`, but keep it explicit in production commands for readability.

Key flags:
- `--action production` — only submit production-tagged records; manual_review/skip are filtered out
- `--execute` — without this it's dry-run (safe default)
- `--confirm retain-hindsight-external-manifest` — required to actually submit
- `--enable-observations` — patch bank to enable native observations for consolidation
- `--wait-consolidation` — default on; after retain ops complete, POST `/consolidate` and wait until `pending_consolidation==0` and no child pending/processing operations remain
- `--no-wait-consolidation` — emergency/debug only; leaves observation drain to another watchdog
- `--no-wait` — skip waiting for async retain completion (submit_state won't be updated and consolidation wait is skipped)
- `--limit N` — limit the number of records (useful for smoke tests)

## Pitfalls

1. **Do not stop at retain success.** Current `hindsight_external_retain_runner.py` has post-retain consolidation integrated by default. Keep `--wait-consolidation` enabled for production imports; otherwise verify with `hindsight_wait_native_consolidation.py --bank hermes` before claiming observations are complete.

2. **`--enable-observations` is necessary but not sufficient on `hermes`.** The runner patches bank config before retain, but the normal/daily Hindsight mode may restore `enable_observations=false` after paid/offline windows. If config is false while `pending_consolidation > 0`, observations will not continue unless a drain gate re-enables/triggers consolidation. Always verify `/config` and `/stats` after external retain.

3. **10pct finalize is bank-specific.** `hindsight_external_10pct_finalize.py` waits the sample banks (`external_chatmemo_10pct_...`, `external_openclaw_10pct_...`) only. A later full import into `hermes` must still run/verify the hermes post-retain drain.

4. **Connection refused during wait.** The runner submits retain_batch asynchronously, then polls for completion via `wait_for_operation_ids`. If Hindsight is briefly unreachable during wait (Connection refused), the runner crashes with exit 1 BUT the retain was already submitted. Verify with Hindsight stats (`total_documents` increase) and operation status — don't blindly re-run.

5. **Argument confusion.** The runner uses `--action production --execute --confirm retain-hindsight-external-manifest`, NOT `--retain / --filter-action / --confirm-retain`. Check `--help` before running.

6. **manifest/latest.json is a metadata dict, not jsonl.** The retainer expects the `.jsonl` path directly — pass `--manifest /path/to/<timestamp>-external-manifest.jsonl`.

7. **Manual_review records go to hermes bank only after review.** By default only `--action production` records are submitted. manual_review records require separate evaluation before retain.

## Output destinations

- Manifests: `$HERMES_HOME/hindsight/external_import/manifests/<timestamp>-external-manifest.jsonl`
- Submit state: `$HERMES_HOME/hindsight/external_import/submit_state.json`
- Bank: `hermes` (same bank as session retain)
- Per-source sub-directories: `manifests/<run-name>/<source>/` for 10pct-sampled runs