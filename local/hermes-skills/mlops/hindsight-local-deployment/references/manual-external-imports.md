# Manual external conversation imports

Use this reference when importing non-Hermes conversations (ChatGPT/Gemini/豆包 chat-memo exports, OpenClaw `lcm.db`) into local Hindsight.

## Policy

External imports are manual-only:

- Do not add them to `hindsight_memory_pipeline.py`, daily/full/weekly cron, or daily reports.
- Use separate banks, not the production `hermes` bank:
  - `external_chatmemo_smoke`, `external_chatmemo`
  - `external_openclaw_smoke`, `external_openclaw`
- Default to smoke banks and `enable_observations=false` until parsing and recall quality are verified.
- Default manifest output omits content and stores source pointers/hashes; retain runner rehydrates from source.
- Manual-review/secret-like records are not retained by default.

## Scripts

Active local scripts:

```bash
~/.hermes/scripts/hindsight_external_manifest.py
~/.hermes/scripts/hindsight_external_retain_runner.py
```

These are separate from daily/session retain scripts and intentionally report:

```text
manual_only=true
daily_pipeline_integrated=false
```

## Chat-memo workflow

Generate a smoke manifest:

```bash
python3 ~/.hermes/scripts/hindsight_external_manifest.py \
  --source chat-memo \
  --path "/path/to/chat-memo_dir" \
  --bank-target external_chatmemo_smoke \
  --min-file-age-seconds 0 \
  --json
```

Dry-run a small retain:

```bash
python3 ~/.hermes/scripts/hindsight_external_retain_runner.py \
  --manifest <external-manifest.jsonl> \
  --bank external_chatmemo_smoke \
  --limit 5 \
  --json
```

Execute only after checking the dry-run:

```bash
python3 ~/.hermes/scripts/hindsight_external_retain_runner.py \
  --manifest <external-manifest.jsonl> \
  --bank external_chatmemo_smoke \
  --limit 5 \
  --batch-size 2 \
  --wait-timeout-s 1800 \
  --execute \
  --confirm retain-hindsight-external-manifest \
  --json
```

## OpenClaw workflow

Use `~/.openclaw/lcm.db` as the primary source. Do not default to agent JSONL/checkpoint/trajectory files.

OpenClaw rules:

- read `messages` table only;
- keep only `role in ('user', 'assistant')`;
- do not read `message_parts`, `summaries`, tool logs, reasoning, compaction, checkpoints, trajectory, deleted/reset files;
- allow `agent:main:main`, `agent:main:tui-*`, and `agent:main:dingtalk:direct:*`;
- exclude cron/subagent/acp/empty session keys;
- skip `title='历史:*'` aggregate conversations by default;
- content denylist includes untrusted system/sender metadata, `[cron:]`, HEARTBEAT, process/status chatter, LCM compaction, and low-signal greetings.

Generate a smoke manifest:

```bash
python3 ~/.hermes/scripts/hindsight_external_manifest.py \
  --source openclaw-lcm \
  --db ~/.openclaw/lcm.db \
  --bank-target external_openclaw_smoke \
  --min-file-age-seconds 0 \
  --json
```

Default segmentation:

```text
max_segment_turns=60
max_segment_chars=80000
gap_split_hours=6
```

## Validation checklist

After a smoke import:

1. Check bank config and stats:
   - `enable_observations=false` unless explicitly testing observations.
   - `total_documents` equals submitted item count.
   - `pending_operations=0`, `failed_operations=0`.
   - `total_observations=0` for raw/no-observation smoke.
2. Recall from the external bank with source-specific queries.
3. Verify isolation:
   - `hermes` bank should return zero documents for `q=external-chatmemo` / `q=external-openclaw`.
   - external bank should return the submitted documents.
4. Inspect `manual_review` counts before larger imports; secret-like records should stay blocked.
5. Watch for duplicated exports. The chat-memo importer deduplicates identical conversation/document IDs and keeps the newest source file by mtime/size.

## Pitfalls

- `--limit` currently selects the first production records in manifest order; it is good for smoke but not topic-diverse sampling. For quality evaluation, prefer future `--sample-diverse` behavior or manually select document IDs.
- Disabling observations does not mean Hindsight stores no memory units; retain still creates facts/source units. Expect `total_nodes > 0` and possibly `pending_consolidation > 0` in the external bank.
- Keyword tagging may be broad on old chat exports; keep broad or secret-like items in `manual_review` and avoid importing them into the main bank.
- Never compensate for external import by running full daily/session retain; external imports are manual-only by design.
