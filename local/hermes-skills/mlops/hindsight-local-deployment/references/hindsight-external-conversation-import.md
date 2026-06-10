# Hindsight external conversation import design

Use this when extending the offline Hindsight pipeline beyond native Hermes session JSON, e.g. ChatGPT/Gemini/Doubao text exports, OpenClaw LCM conversations, or other third-party chat archives.

## Core architecture

Do not hardwire third-party formats into `hindsight_session_manifest.py`. Keep Hermes session import stable and add a separate external import layer:

- `hindsight_external_manifest.py` — read-only adapter-based manifest builder.
- optional `hindsight_external_retain_runner.py`, or reuse/extend `hindsight_session_retain_runner.py` after record rehydration supports external sources.
- state/output root: `$HERMES_HOME/hindsight/external_import/`.

Flow:

1. Source adapter reads third-party data.
2. Normalize to canonical conversation schema.
3. Clean user/assistant turns; drop tools/system/checkpoints/logs/secrets.
4. Emit Hindsight manifest JSONL.
5. Retain runner submits only changed production records using submit-state.
6. Reuse native consolidation, V2 rebuild, conflict audit, and proposal review gates.

## Canonical conversation schema

Normalize every source to:

```json
{
  "source_kind": "chat_memo_txt | openclaw_lcm | openclaw_agent_jsonl",
  "source_name": "chat-memo-34 | openclaw-main",
  "external_conversation_id": "...",
  "title": "...",
  "url": "...",
  "platform": "ChatGPT | Gemini | 豆包 | OpenClaw",
  "created_at": "...",
  "updated_at": "...",
  "messages": [
    {"role": "user | assistant", "timestamp": "...", "content": "..."}
  ],
  "source_path": "...",
  "source_hash": "...",
  "adapter_version": "external-import-v1"
}
```

Render Hindsight content as plain dialogue with title/platform/url/created/source headers, then `User: [timestamp]` and `Assistant: [timestamp]` blocks.

## Hindsight manifest shape

Use distinct IDs and source tags. Example:

```json
{
  "document_id": "external-chat::chatgpt::69af968b-90a0-8320-9568-a02489f0ec42",
  "bank_target": "hermes",
  "action": "production",
  "reason": "semantic_tags_detected",
  "context": "external_conversation",
  "event_date": "2026-03-10 11:57:39",
  "tags": ["source:external-chat", "platform:chatgpt", "domain:autodrive"],
  "observation_scopes": [["domain:autodrive"], ["source:external-chat"]],
  "metadata": {
    "source_kind": "chat_memo_txt",
    "source_label": "chat-memo-34",
    "source_path": "...",
    "url": "...",
    "platform": "ChatGPT",
    "external_conversation_id": "...",
    "content_sha256": "...",
    "source_mtime_ns": "...",
    "adapter_version": "external-import-v1",
    "cleaning_version": "external-clean-v1"
  },
  "update_mode": "replace"
}
```

Keep `event_date` from the original conversation time, not import/runtime time.

## Adapter notes

### `chat_memo_txt`

Observed format in `/home/wyr/桌面/temp/chat-memo_34_20260409201656/*.txt`:

```text
Title: ...
URL: ...
Platform: ChatGPT|Gemini|豆包
Created: YYYY-MM-DD HH:MM:SS
Messages: N

User: [timestamp]
...

AI: [timestamp]
...
```

Parse headers, split `User:`/`AI:` turns, and map `AI` to `assistant`. Prefer URL conversation IDs for document identity; fallback to file stem/hash.

Document IDs:

```text
external-chat::<platform_slug>::<url_id_or_file_stem>
```

### `openclaw_lcm`

Prefer OpenClaw structured DB over session files. Observed primary DB:

```text
~/.openclaw/lcm.db
```

Useful tables include `conversations`, `messages`, `message_parts`, and `summaries`. Start with:

```sql
SELECT
  c.conversation_id,
  c.session_id,
  c.session_key,
  c.title,
  c.created_at,
  c.updated_at,
  m.message_id,
  m.seq,
  m.role,
  m.content,
  m.created_at
FROM conversations c
JOIN messages m ON m.conversation_id = c.conversation_id
ORDER BY c.conversation_id, m.seq;
```

Keep only user/assistant by default. Drop system/tool/developer/function/synthetic/ignored parts, shell output, patch blobs, gateway/status/log/heartbeat/bootstrap/system prompt material, and large raw file payloads.

Document ID:

```text
external-chat::openclaw-lcm::<conversation_id>
```

### `openclaw_agent_jsonl`

Treat as fallback only, not primary source. Input root: `~/.openclaw/agents/*/sessions/`. Exclude by default:

```text
*.checkpoint.*
*.trajectory.jsonl
*.trajectory-path.json
*.deleted.*
*.reset.*
*/dingtalk-state/*
```

If the same session/conversation is present in `lcm.db`, skip JSONL to avoid duplicates.

## Incremental import

Use a dedicated submit-state, not just mtime:

```text
$HERMES_HOME/hindsight/external_import/submit_state.json
```

Schema sketch:

```json
{
  "schema_version": "external-submit-state-v1",
  "documents": {
    "hermes::external-chat::chatgpt::xxx": {
      "document_id": "external-chat::chatgpt::xxx",
      "bank": "hermes",
      "source_kind": "chat_memo_txt",
      "source_path": "...",
      "content_sha256": "...",
      "full_content_sha256": "...",
      "source_file_sha256": "...",
      "source_mtime_ns": 123,
      "source_size_bytes": 456,
      "external_updated_at": "...",
      "message_count": 56,
      "adapter_version": "external-import-v1",
      "cleaning_version": "external-clean-v1",
      "last_submitted_at": "..."
    }
  }
}
```

Incremental rules:

1. Missing document_id -> submit.
2. Same document_id but different `content_sha256` -> resubmit with `update_mode=replace`.
3. Same `content_sha256` -> skip unchanged.
4. Source disappeared -> mark stale only; do not delete Hindsight documents automatically.
5. Active OpenClaw conversations or recently modified files -> skip until older than `min_file_age_seconds` (default 900s).

Use mtime/watermarks only to reduce candidate scans; content hash remains authoritative.

## Safety classification

Actions:

- `production`: semantic tags present, no secret-like material, not tool/system/log/checkpoint noise.
- `manual_review`: no tags, multi-scope, speculative/overbroad, config/path-sensitive, possible secret, or tool-heavy content.
- `skip`: empty, low-signal, checkpoint, trajectory, cron/system/bootstrap, raw tool/log output, heartbeat/status.

Secret-like content must not be sent to external LLM proposal review; route to deterministic manual review.

## Pipeline integration

After stabilization, insert external import between session retain and daily reflect:

```text
build_session_manifest
retain_session_manifest
build_external_manifest
retain_external_manifest
daily_reflect
wait_native_consolidation
v2_rebuild
conflict_audit
proposal_review
```

For first rollout, run separately and smoke-test in a temp bank such as `hermes_external_smoke` before writing to the production `hermes` bank.

## Suggested commands

Dry-run manifest:

```bash
python3 ~/.hermes/scripts/hindsight_external_manifest.py \
  --source chat-memo \
  --path "/home/wyr/桌面/temp/chat-memo_34_20260409201656" \
  --source openclaw-lcm \
  --db ~/.openclaw/lcm.db \
  --bank-target hermes \
  --output-dir ~/.hermes/hindsight/external_import/manifests \
  --json
```

Smoke retain:

```bash
python3 ~/.hermes/scripts/hindsight_external_retain_runner.py \
  --manifest <manifest.jsonl> \
  --bank hermes_external_smoke \
  --limit 5 \
  --execute \
  --confirm retain-hindsight-external-manifest
```

Production incremental retain:

```bash
python3 ~/.hermes/scripts/hindsight_external_retain_runner.py \
  --manifest <manifest.jsonl> \
  --bank hermes \
  --submit-state ~/.hermes/hindsight/external_import/submit_state.json \
  --execute \
  --confirm retain-hindsight-external-manifest
```

## Recommended config

```json
{
  "schema_version": "external-import-config-v1",
  "bank": "hermes",
  "submit_state": "~/.hermes/hindsight/external_import/submit_state.json",
  "output_dir": "~/.hermes/hindsight/external_import/manifests",
  "min_file_age_seconds": 900,
  "max_document_chars": 120000,
  "retain_chunk_size": 8000,
  "sources": [
    {
      "name": "chat-memo-34",
      "type": "chat_memo_txt",
      "path": "/home/wyr/桌面/temp/chat-memo_34_20260409201656",
      "enabled": true,
      "tags": ["source:chat-memo"]
    },
    {
      "name": "openclaw-lcm",
      "type": "openclaw_lcm",
      "path": "~/.openclaw/lcm.db",
      "enabled": true,
      "tags": ["source:openclaw"]
    },
    {
      "name": "openclaw-agent-jsonl",
      "type": "openclaw_agent_jsonl",
      "path": "~/.openclaw/agents",
      "enabled": false,
      "tags": ["source:openclaw"],
      "exclude_globs": ["*.checkpoint.*", "*.trajectory*", "*.deleted.*", "*.reset.*", "*/dingtalk-state/*"]
    }
  ]
}
```

## Common pitfalls

1. Importing OpenClaw JSONL/checkpoint/trajectory files directly; this pollutes Hindsight with tool traces and internal state.
2. Using filesystem mtime as the only incremental gate; use content hashes and submit-state as authority.
3. Losing original event dates and making historical facts look newly created.
4. Deleting production Hindsight documents when source files disappear; mark stale and require explicit confirmation for deletion.
5. Splitting knowledge by long-term separate source banks; prefer one production bank with source/platform tags, after smoke validation.
