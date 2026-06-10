# Manual External Conversation Import

Use this when importing non-Hermes conversations into Hindsight, e.g. exported ChatGPT/Gemini/豆包 `chat-memo` text files or OpenClaw `lcm.db` conversations.

## User Requirements Captured

- External data is manual-only: never add it to daily/weekly/full Hindsight cron pipelines.
- External data must not be included in daily usage/data statistics.
- External data goes to separate banks, not the main `hermes` bank, until explicitly reviewed.
- OpenClaw import must identify true dialogue by strict deterministic rules and avoid tool/log/system/checkpoint pollution.
- DingTalk direct OpenClaw conversations are allowed.
- Observations should remain disabled by default; enable only after smoke tests show the rules are clean.
- Long OpenClaw conversations may be split into segments.
- `external_*` bank naming is acceptable.

## Implemented Manual Scripts

Local scripts:

```bash
~/.hermes/scripts/hindsight_external_manifest.py
~/.hermes/scripts/hindsight_external_retain_runner.py
```

Tests:

```bash
~/.hermes/scripts/tests/test_hindsight_external_manifest.py
~/.hermes/scripts/tests/test_hindsight_external_retain_runner.py
```

Verification used:

```bash
python3 -m py_compile \
  ~/.hermes/scripts/hindsight_external_manifest.py \
  ~/.hermes/scripts/hindsight_external_retain_runner.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  ~/.hermes/scripts/tests/test_hindsight_external_manifest.py \
  ~/.hermes/scripts/tests/test_hindsight_external_retain_runner.py \
  ~/.hermes/scripts/tests/test_hindsight_session_retain_runner.py -q
```

Expected: all pass.

## Banks

Smoke/default banks:

```text
external_chatmemo_smoke
external_openclaw_smoke
```

Reviewed/manual production-style external banks:

```text
external_chatmemo
external_openclaw
```

Keep these outside regular Hermes memory recall/statistics unless the user explicitly asks for cross-bank queries.

## Chat Memo Adapter

Source pattern:

```text
Title: ...
URL: ...
Platform: ChatGPT/Gemini/豆包
Created: ...
Messages: ...

User: [timestamp]
...

AI: [timestamp]
...
```

Rules:

- `AI` maps to assistant.
- Document IDs prefer URL conversation IDs:
  - `external-chatmemo::chatgpt::<chatgpt_c_id>`
  - `external-chatmemo::doubao::<doubao_chat_id>`
- No URL: use stable file stem/hash fallback.
- Default manifest omits content and stores source path/hash for rehydration.

Manual manifest command:

```bash
python3 ~/.hermes/scripts/hindsight_external_manifest.py \
  --source chat-memo \
  --path "/home/wyr/桌面/temp/chat-memo_34_20260409201656" \
  --bank-target external_chatmemo_smoke \
  --min-file-age-seconds 0 \
  --json
```

## OpenClaw LCM Adapter

Primary source:

```text
~/.openclaw/lcm.db
```

Use only `messages` table with:

```sql
role IN ('user', 'assistant')
```

Do not read/import by default:

- `message_parts`
- `summaries`
- `~/.openclaw/agents/*/sessions/*.jsonl`
- `*.checkpoint.*`
- `*.trajectory*`
- `*.deleted.*`
- `*.reset.*`
- tool outputs / reasoning / compaction / files

Allowed `session_key` by default:

```text
agent:main:main
agent:main:tui-*
agent:main:dingtalk:direct:*
```

Excluded by default:

```text
agent:*:cron:*
agent:*:subagent:*
agent:*:acp:*
empty session_key
title starting with 历史:
```

Content-level deterministic drops:

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
The user received a system notification
(no new output)
short pure operational-status replies such as 脚本正在运行 / 进程还在运行 / 继续等待
low-signal greetings
```

OpenClaw timestamps are available:

- `conversations.created_at`
- `conversations.updated_at`
- `messages.created_at`

Use segment first valid user message time as `event_date`; store segment start/end and conversation timestamps in metadata.

Default segmentation:

```text
max_segment_turns=60
max_segment_chars=80000
gap_split_hours=6
```

Manual manifest command:

```bash
python3 ~/.hermes/scripts/hindsight_external_manifest.py \
  --source openclaw-lcm \
  --db ~/.openclaw/lcm.db \
  --bank-target external_openclaw_smoke \
  --min-file-age-seconds 0 \
  --json
```

## Retain Runner

Dry-run first:

```bash
python3 ~/.hermes/scripts/hindsight_external_retain_runner.py \
  --manifest <manifest.jsonl> \
  --bank external_openclaw_smoke \
  --limit 3 \
  --json
```

Real smoke import requires explicit confirmation:

```bash
python3 ~/.hermes/scripts/hindsight_external_retain_runner.py \
  --manifest <manifest.jsonl> \
  --bank external_openclaw_smoke \
  --limit 10 \
  --execute \
  --confirm retain-hindsight-external-manifest \
  --json
```

The runner maintains a separate incremental state by default:

```text
~/.hermes/hindsight/external_import/submit_state.json
```

Dry-run never writes submit state.

## Observations Policy

Default manifest has empty `observation_scopes` and does not enable observations. After smoke review, the user may manually opt into observation scopes or enable bank observations for a controlled run. Do not enable observations automatically from any daily/offline pipeline.

## Validation Snapshot from 2026-05-18

Chat memo dry-run:

```text
files_seen=35
records=36
production=31
manual_review=5
estimated_retain_chunks=185
```

OpenClaw dry-run:

```text
conversations_seen=112
included_conversations=10
records=15
production=9
manual_review=6
excluded: session_key_excluded=78, no_valid_segments=22, history_aggregate=2
dropped messages: openclaw_system_noise=367, empty=116, low_signal=1
estimated_retain_chunks=54
```

## Pitfalls

1. Do not add these scripts to `hindsight_memory_pipeline.py` or cron prompts.
2. Do not include `external_*` banks in daily reports/statistics unless the user explicitly requests external import accounting.
3. Do not use OpenClaw agent JSONL/checkpoint/trajectory files as the default source; they are noisy and contain execution artifacts.
4. Do not treat role=`user` as inherently clean in OpenClaw; system notifications can be stored as user content and must pass content-level deny rules.
5. Do not enable observations before rule smoke tests; external bank separation limits blast radius but does not prevent bad observations inside that bank.
