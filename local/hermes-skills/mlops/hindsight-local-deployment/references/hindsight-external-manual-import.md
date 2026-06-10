# Hindsight External Manual Import

Use this reference when designing or operating imports of third-party conversation data (OpenClaw, exported ChatGPT/Gemini/Doubao chat memos, etc.) into local Hindsight.

## User policy

External conversation data is **manual-only**:

- Do not add external imports to `hindsight_memory_pipeline.py` daily/weekly/full plans.
- Do not create cron jobs or scheduled jobs for external imports.
- Do not include external import runs/banks in daily reports or normal Hindsight usage statistics.
- Do not import third-party data into the main Hermes bank by default.
- Use separate banks with `external_*` names and keep normal Hermes auto-recall pointed at the main bank unless explicitly requested.

Recommended banks:

```text
external_openclaw_smoke
external_openclaw
external_chatmemo_smoke
external_chatmemo
```

## OpenClaw source of truth

For OpenClaw conversation import, use only:

```text
~/.openclaw/lcm.db
```

The reliable tables/columns observed in this install:

- `conversations`: `conversation_id`, `session_id`, `session_key`, `title`, `created_at`, `updated_at`, `active`
- `messages`: `message_id`, `conversation_id`, `seq`, `role`, `content`, `created_at`

Avoid these by default because they contain tool traces, compression, checkpoints, or runtime artifacts:

- `~/.openclaw/agents/*/sessions/*.jsonl`
- `*.checkpoint.*`
- `*.trajectory.jsonl`, `*.trajectory-path.json`
- `*.deleted.*`, `*.reset.*`
- `dingtalk-state/*`
- `summaries` table
- `message_parts` table (`reasoning`, `tool`, `compaction`, `file` parts)

## OpenClaw session filters

Default allowlist:

```text
agent:main:main
agent:main:tui-*
agent:main:dingtalk:direct:170941191029642680
```

Default denylist:

```text
agent:*:cron:*
agent:*:subagent:*
agent:*:acp:*
empty session_key
title == "历史: 当前会话"
```

The DingTalk direct session is intentionally allowed for this user. Cron/subagent/ACP are execution/process streams, not direct user conversation.

## Message filters

Read from `messages` only, and keep only:

```sql
role in ('user', 'assistant')
```

Then apply deterministic content filters. Drop messages/lines dominated by:

```text
System (untrusted):
Sender (untrusted metadata):
[cron:
Read HEARTBEAT.md
HEARTBEAT_OK
LCM compaction
Command still running
Process still running
Process exited
(no new output)
The user received a system notification
脚本正在运行 / 进程还在运行 / 继续等待 / 纯备份完成状态
```

Drop a segment if, after filtering:

- it has fewer than 1 user and 1 assistant message;
- effective content is too short or pure greeting/status;
- it is mostly process/tool/system noise;
- it matches credential/secret patterns (`api_key`, `secret`, `token`, `password`, `sk-...`, bearer tokens, etc.). Secret-like content should go to manual review, not external LLM review.

## Timestamps and segmentation

OpenClaw `lcm.db` has enough timestamp information:

- conversation-level: `conversations.created_at`, `conversations.updated_at`
- message-level: `messages.created_at`

Set Hindsight `event_date` to the first valid user message in the segment. Store segment metadata:

```text
conversation_created_at
conversation_updated_at
segment_started_at
segment_ended_at
message_id_start/message_id_end
seq_start/seq_end
```

Default segmentation for OpenClaw long conversations:

```text
max_segment_turns = 60
max_segment_chars = 80000
gap_split_hours = 6
```

Use stable document IDs such as:

```text
external-openclaw::<conversation_id>::seg-001
external-openclaw::<conversation_id>::seg-002
```

## Observations policy

Do not enable Hindsight observations for external imports until deterministic extraction rules are stable and smoke-tested.

Recommended workflow:

1. Generate manifest only (read-only).
2. Import 5-10 clean samples into `external_*_smoke`.
3. Run recall smoke checks and inspect generated documents/observations.
4. Only after rules are stable, allow explicit manual `--enable-observations` for `external_*` banks.

## Incremental import

Use a separate submit state under:

```text
~/.hermes/hindsight/external_import/*-submit-state.json
```

Incremental authority should be `content_sha256`, not only file mtime or `updated_at`. Source timestamps are useful for candidate prefiltering, but final skip/replace decision should compare stable content hashes.

Never automatically delete Hindsight documents when a source file disappears. Mark stale or require explicit manual delete confirmation.

## Chat memo exports

For directories like:

```text
/home/wyr/桌面/temp/chat-memo_34_20260409201656
```

Parse files with headers:

```text
Title:
URL:
Platform:
Created:
Messages:
User: [timestamp]
AI: [timestamp]
```

Map `AI` to `assistant`; use URL conversation ID when available for stable document IDs. If URL is missing, fall back to file stem + source hash.

## Non-goals

External manual import should not:

- mutate the main Hermes bank by default;
- alter provider/runtime mode outside a bounded manual command;
- change cron schedules;
- become part of daily/full/weekly pipeline statistics;
- participate in normal Hermes auto-recall unless the user explicitly asks for cross-bank recall.
