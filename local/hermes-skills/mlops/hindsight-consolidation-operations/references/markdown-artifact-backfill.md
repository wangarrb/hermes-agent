# Markdown Artifact Backfill for Hindsight

This reference captures the session-specific lessons from extending Hindsight external imports to third-party markdown artifacts.

## What changed

- Markdown artifacts are treated as a separate `source_kind`: `markdown_artifact_md`.
- Default discovery should not blindly scan broad markdown roots.
- Discovery must prefer conversation-produced evidence and only import files that exist on disk.
- OpenClaw LCM discovery alone is incomplete for some artifacts; OpenClaw session JSONL files can contain the decisive `write_file` / `Successfully wrote ... .md` evidence.

## Discovery order

1. Hermes sessions (optional, default off for markdown artifacts because Hermes native pipeline already handles them).
2. OpenClaw lcm.db messages with explicit `.md` write evidence.
3. OpenClaw session JSONL files under `~/.openclaw/agents/main/sessions/` for write-file/tool-output evidence.
4. Explicit `--markdown-path` inputs if the caller already knows the path.

## Path resolution rules

- Resolve only paths that end with `.md`.
- If the path cannot be resolved to an existing file, skip it.
- Restrict OpenClaw artifacts to `~/.openclaw/workspace/` by default.
- Exclude root control files such as `AGENTS.md`, `MEMORY.md`, `SOUL.md`, `USER.md`, `HEARTBEAT.md`, `IDENTITY.md`, `CLAUDE.md`, `DREAMS.md`, `TOOLS.md`.

## Markdown parsing

Use structure-aware parsing:

- document outline record
- section records
- item records

Preserve metadata fields such as:

- `title`
- `report_date`
- `artifact_type`
- `section_path`
- `item_index`
- `block_type`

Do not merge different numbered items or separate sections.

## Retain policy

Use a dedicated retain instruction for `external_markdown_artifact` records:

- preserve markdown hierarchy
- keep concrete metrics, versions, paths, dates, and acceptance/failure details
- avoid collapsing structured document content into generic conversation facts

## Validation checklist

- Dry-run should show discovered artifact paths, missing/skipped candidates, and source breakdown.
- Spot-check a target report such as `周报_2026-05-16.md` to confirm it yields one outline record plus item records.
- Verify that the retained records are classified as `source_kind=markdown_artifact_md` and `context=external_markdown_artifact`.
- Confirm production backfill only submits `production` records and leaves `manual_review` records out.
