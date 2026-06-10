## Reference Implementation: 2026-05-19 OpenClaw Markdown Artifact Pipeline

### What was added

Extended the markdown artifact discovery to also scan OpenClaw session JSONL files, because many produced .md files (like weekly reports) appear in session JSONL tool events but NOT in lcm.db.

### Key code added

**New functions in hindsight_external_manifest.py:**

```
iter_openclaw_session_files(roots, limit_sessions)
is_openclaw_session_cron(session_path, event)
discover_markdown_paths_from_openclaw_sessions(
    roots, include_cron, min_file_age_seconds, limit_sessions, allowed_roots
)
```

**Modified functions:**
- `discover_conversation_markdown_artifact_paths()` — added `include_openclaw_sessions=True` parameter
- `records_from_markdown_artifacts()` — passes through `include_openclaw_sessions` and `openclaw_session_roots`
- CLI `main()` — added `--markdown-include-openclaw-sessions`, `--no-markdown-include-openclaw-sessions`, `--openclaw-session-roots` flags

### OpenClaw session JSONL structure

Each `.jsonl` file contains one event per line. Events have:
- `type`: "message", "session.started", "model.completed", "trace.artifacts", etc.
- For `type:message` events: contains `message.role` + `message.content`
- The toolResult role has `toolName: write` and content like `"Successfully wrote 4257 bytes to /path/file.md"`
- Cron sessions identified by `sessionKey` containing `agent:main:cron:` — filtered when `include_cron=False`

### Discovery algorithm

1. For each JSONL file, attempt to read lines
2. Skip cron sessions (check first event's sessionKey)
3. For remaining lines, extract all text values via `iter_text_values()` (recursive dict/list walk)
4. Check each text value: does it have a positive MD production hint? (wrote/written/created/saved/已写入/写入/保存到)
5. If yes, extract .md path candidates
6. Validate each path against allowed_roots, control file list, and disk existence

### CLI usage examples

```bash
# Default: discovers from OpenClaw lcm.db + session JSONL
python3 hindsight_external_manifest.py \
  --source markdown-artifact --bank-target hermes --min-file-age-seconds 0

# Explicit OpenClaw session roots
python3 hindsight_external_manifest.py \
  --source markdown-artifact --openclaw-session-roots ~/.openclaw/agents/main/sessions

# Disable session JSONL discovery (lcm.db only)
python3 hindsight_external_manifest.py \
  --source markdown-artifact --no-markdown-include-openclaw-sessions
```

### Production backfill result (2026-05-19)

```
total_records_in_manifest: 776
production_submitted: 769
manual_review_skipped: 7 (suspected secrets)
retain_operations: completed (all batches)
consolidation: ready (0 pending, 0 failed)

Document count: 2053 -> 2822 (+769)
Observation count: 19602 -> 20152

All 13 discovered md files were real files on disk.
Control files (AGENTS.md, HEARTBEAT.md, etc.) were filtered out.
5-16 weekly report split into 47 records (1 outline + 46 items).
```

### Recall verification

Three queries verified consolidated observations were searchable:

| Query | Hits | Notes |
|---|---|---|
| 外厂主目标测距测速误差对比融合版本整体最优 | 34 | Hit 5-16 weekly report items |
| 单目多模态大模型优化 FPN 深度精度 | 36 | Hit specific accuracy numbers |
| S3-J5国科微单目3D模型20260511 | 42 | Hit exact version numbers |

### Consolidation performance with 769-document backfill

- Started with ~600 pending_consolidation after all retain ops completed
- Processing rate: ~24 items per 2 minutes (batch_size=64, llm_batch_size=8, parallel_batches=8)
- Middle phase (~450 remaining) briefly stalled (LLM provider rate limit), auto-recovered
- Total consolidation wall time: ~45 minutes
- Final stall at <30 items resolved by auto-trigger (runner's `trigger_on_stall`)
- No manual intervention needed; no failed consolidation