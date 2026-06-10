---
name: hindsight-external-import-design
description: Class-level workflow for designing and rolling up third-party content import into Hindsight. Covers source classification, discovery, normalization, retain policy, validation, and production backfill. Reusable across any external source type (chat exports, OpenClaw sessions, markdown artifacts, meeting notes, etc.).
version: 1.0.0
author: Agent
license: MIT
metadata:
  hermes:
    tags: [hindsight, external-import, memory, consolidation, markdown-artifact]
    related_skills: [hindsight-local-deployment, hindsight-consolidation-operations, systematic-debugging]
---

# Hindsight External Import Design

## Overview

This is the class-level workflow for importing third-party content into Hindsight (i.e., content that did NOT originate through the native Hermes session pipeline). It covers the full lifecycle: source discovery → normalization into manifest records → retain policy separation → validation statistics → production backfill → recall smoke verification.

Core principle: **external content is NOT "just another conversation"**. Different source types need different parsing, different retain instructions, different dedup logic, and different validation.

## When to Use

Use this skill when:

- You need to import chat exports from third-party platforms (ChatGPT, Gemini, Doubao, etc.)
- You need to import OpenClaw conversation records (`lcm.db` or session JSONL)
- You need to import produced markdown artifacts (weekly reports, plans, notes) discovered from agent conversations
- You are adding a NEW external source type and need to decide where it fits in the pipeline
- The question is about content quality — dedup, skip rules, secret detection, tag inference

## Architecture: The 4-Layer Model

Every external import pipeline should be structured as 4 separate layers:

### Layer 1: Discovery
Find content on disk — files, databases, session logs. No content parsing or semantic logic.

Functions:
- `discover_<source>_paths(root)` → list of paths
- `iter_<source>_files(roots, min_age, limit)` → yield paths
- Include diagnostics: files_seen, skipped_candidates, missing_or_skipped

### Layer 2: Normalization (Adapter)
Convert raw source content → unified manifest record schema.

Each source type gets its own adapter function:
- `records_from_<source>(path, ...)` → list of manifest records

Schema:
- `document_id`: deterministic from content (prevents dupes)
- `source_kind`: one of the standard kinds (see below)
- `content`: cleaned, structured text
- `metadata`: all structural info (title, date, section_path, etc.)
- `tags`: inferred semantic tags
- `action`/`reason`: skip/manual_review/production
- `observation_scopes`: for downstream consolidation

### Layer 3: Retain Policy
Different source kinds need different retain instructions injected into Hindsight's `retain_custom_instructions`.

Two main policies:
1. **external_conversation**: extract durable facts (decisions, preferences, results); skip tool logs and chatter
2. **external_markdown_artifact**: preserve document structure; keep section/item boundaries; do NOT merge different numbered items or sections

### Layer 4: Validation
Four-part check:

1. **Discovery stats**: paths_seen, paths_found, skipped (missing / too_recent / control_file)
2. **Content stats**: records, by_action, by_reason, by_source, by_record_kind
3. **Retain stats**: submitted_items, unchanged_skipped, failed
4. **Recall smoke**: can consolidated observations be retrieved by title? by date? by specific metric?

## Source Types

Define a small, fixed set of source_kind values. Add new types only when the content structure genuinely differs.

### Standard source_kind values

| source_kind | Source | Parsing |
|---|---|---|
| `chat_memo_txt` | ChatGPT/Gemini/Doubao exported .txt | Header-based (Title/URL/Platform/Created), message-based |
| `openclaw_lcm` | OpenClaw `lcm.db` conversations | SQL query + strict filtering (drop system/untrusted/tool messages) |
| `openclaw_session_jsonl` | OpenClaw session JSONL files | JSONL line-by-line, extract write events for markdown artifacts |
| `markdown_artifact_md` | Any .md produced in conversation | heading/list/paragraph structural parsing |

chat_memo_txt — for direct chat exports from platforms like ChatGPT, Gemini, Doubao
openclaw_lcm — for OpenClaw conversation records stored in its LCM SQLite database
openclaw_session_jsonl — for OpenClaw session JSONL files (used to discover markdown that was written by tools; NOT stored in lcm.db)
markdown_artifact_md — for any .md file discovered/produced by conversation that exists on disk

### What does NOT need a new type

- Content from a new source but same structure → just add a new discovery function, reuse existing adapter
- New file format but same semantics → convert to one of the existing types before creating manifest records

## Discovery Rules

### What to discover
- Files that are known to be conversation-produced (explicit write events, "Successfully wrote" in tool output)
- Files that exist on disk AND are under allowed workspace roots
- All OpenClaw lcm.db conversations except cron/history-aggregate/excluded-key sessions

### What to skip
- Root control files: AGENTS.md, MEMORY.md, SOUL.md, USER.md, HEARTBEAT.md, IDENTITY.md, CLAUDE.md, DREAMS.md, TOOLS.md
- Files outside allowed roots
- Files too recent (below min_file_age_seconds)
- Files in non-project directories (SKIP_MD_DIRS: .git, node_modules, etc.)
- Files that cannot be resolved to a real path on disk ("if找不到就放弃")

### Dedup rules
- Same file from multiple discovery sources: dedup at path level (dict[str, Path])
- Same conversation from multiple raw exports: prefer newest source file (compare mtime_ns)
- Same content in subsequent pipeline runs: content_sha256 matching in submit_state → skip unchanged

## Retain Instructions for Different Source Kinds

### external_conversation (default)
```
Extract durable user/project facts, decisions, results, preferences, stable environment facts.
Skip tool logs, file listings, raw command output, process chatter, greetings.
For external_conversation records, keep only high-signal durable facts (usually 3-5 per chunk).
```

### external_markdown_artifact
```
For external_markdown_artifact records, preserve the Markdown structure provided in the content header:
report_date/title/artifact_type/section_path/item_index.
Do not merge different numbered items or sections.
Keep concrete metrics, project names, versions, chips, platforms, paths, dates, and acceptance/failure details.
```

## Validation Checklist

Before declaring any external import pipeline production-ready:

- [ ] All source_kind values have at least one integration test
- [ ] Discovery skips control files and out-of-bounds paths
- [ ] Content records have deterministic document_id (no collision across runs)
- [ ] Dry-run matches expected counts
- [ ] Retain instruction is source_kind-specific (not a generic catch-all)
- [ ] Submit state survives across runs (incremental skip works)
- [ ] Recall smoke test: search for a specific metric/value from the imported content
- [ ] No Hermes native sessions are double-imported (Hermes is already handled by native pipeline)

## Typical Execution Order

1. Understand source format (read samples of raw data)
2. Define source_kind and document_id namespace
3. Write discovery function (iter_*, discover_*)
4. Write adapter (records_from_*) with structural tagging
5. Add to the source enum in CLI / main orchestration
6. Write unit tests for the new source
7. Run dry-run, inspect manifest JSONL
8. Run production backfill with small batch first
9. Wait for retention + consolidation to drain
10. Run recall smoke test

## Reference Implementation

See `references/openclaw-markdown-artifact-2026-05-19.md` for the session-specific implementation notes:
- OpenClaw session JSONL discovery addition
- Markdown artifact retain policy integration
- Production backfill result: 769 documents imported, 20152 observations, 0 failed

## Common Pitfalls

1. Using the same retain instruction for all external types. **Don't.** Markdown artifacts need structural preservation; chat exports need fact extraction.
2. Scanning workspace roots blindly. **Don't.** Only import files that are explicitly "written" in conversations. Use file_age + conversation evidence to gate.
3. Importing Hermes native sessions as external. **Don't.** Hermes has its own native pipeline. Set `include_hermes=False` by default.
4. Merging different numbered items during retain. **Don't.** Each work item in a weekly report should produce separate observations.
5. Forgetting to exclude control files. **Always filter:** AGENTS.md, MEMORY.md, SOUL.md, USER.md, HEARTBEAT.md, IDENTITY.md, CLAUDE.md, DREAMS.md, TOOLS.md.
6. Hardcoding paths. **Always use** `allowed_roots` parameter; restrict OpenClaw LCM discovery to `.openclaw/workspace/` by default.
7. Not checking submit_state on subsequent runs. Unchanged records waste queue capacity.