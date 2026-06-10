# Hindsight tag/scope cleanup vs session JSON v3 clean route

Use this reference when the user asks what “tag/scope cleanup” or “session JSON v3 clean route” means after a SQLite/day-topic rebuild or quality audit.

## Core distinction

- **Tag/scope cleanup**: remediate the existing production/archive bank in place or document-by-document. This is a post-hoc repair path.
- **Session JSON v3 clean route**: rebuild a clean candidate bank from Hermes session JSON with manifest-first controls. This is a source-level clean-ingestion path.

A successful SQLite `--full` rebuild proves the service and retain queue are healthy; it does **not** prove the bank is a clean canonical knowledge base.

## Why SQLite day-topic imports can be structurally polluted

Observed after the 2026-05-08 full rebuild:

- 118 documents, 6298 facts/nodes, 0 failed ops, provider restored to normal-local.
- Broad/source tags (`hermes`, `sqlite`, `incremental`) were present on all 6298 facts.
- Topic contamination remained measurable, e.g. `egomotion4d`-tagged documents/facts containing patent/OpenClaw material.

Reason: day-topic bundling can mix multiple discussion scopes from the same day/topic bucket; Hindsight native consolidation treats tags/scopes as semantic grouping signals. If upstream tags are broad or wrong, native observations inherit the wrong scope.

## What tag/scope cleanup means

Read-only first:

1. Audit broad/system tag counts.
2. Audit cross-domain contamination, e.g. egomotion tag + patent/OpenClaw terms, openclaw tag + Egomotion/VGGT/DAGE/ATE terms.
3. List affected documents, facts, derived observations, and source refs.
4. Produce a discard/quarantine snapshot plan before any mutation.

Possible repair actions, from lower to higher risk:

- Treat broad/source tags (`hermes`, `sqlite`, `incremental`, `daily`, `canonical`) as metadata rather than semantic tags.
- Patch document tags where the document is clearly single-scope.
- Set explicit `observation_scopes`, e.g. `[["project:egomotion4d"], ["domain:autodrive"]]`.
- Clear/re-run derived observations after tag repair.
- Delete/re-retain a polluted document only after backup + discard-first snapshot + explicit confirmation.

Do **not** do broad in-place cleanup blindly. Current Hindsight lacks convenient exact fact/observation edit/upsert APIs, so many corrections are document-level and destructive enough to require confirmation.

## What session JSON v3 clean route means

Use JSON sessions as primary historical evidence and create a clean candidate bank such as `hermes_v3` / `hermes_sessions_v3`.

Key design:

- One session or deterministic part-split = one Hindsight document.
- Stable `document_id`, e.g. `hermes-session::<session_id>` or `::part-000`.
- Raw evidence remains in JSON/SQLite outside Hindsight for provenance/re-retain.
- Manifest-first: inspect coverage, cost, tags, scopes, hashes, and action before any submit.
- Tags are semantic only: `domain:*`, `project:*`, `topic:*`.
- Source/system labels (`hermes`, `sqlite`, `incremental`, `daily`, `canonical`) go to metadata, not tags.
- Explicit `observation_scopes` are proposed before enabling native observations.
- Mixed, context-bootstrap, memory-recall, credential-bearing, and low-signal sessions are routed to `manual_review` or `skip` rather than production.

Current implementation pieces:

- `~/.hermes/scripts/hindsight_session_manifest.py`
- `~/.hermes/scripts/hindsight_session_retain_runner.py`
- `~/.hermes/scripts/hindsight_bank_quality_audit.py`
- `~/.hermes/scripts/hindsight_minimax_import.py session-manifest-retain-minimax`

Latest known v3 manifest from 2026-05-08:

- records: 1800
- production: 389
- manual_review: 1336
- skip: 75
- secret/credential material routed to manual review

Latest MiniMax v4 smoke bank result:

- 15 documents
- 295 facts
- 0 observations
- broad/system tag count: 0
- contamination probes: 0 for the checked egomotion/openclaw/patent mismatches

This smoke result is a promising ingestion-quality signal, not enough by itself to switch Hermes auto-recall.

## Pros / cons

### Tag/scope cleanup

Pros:

- Faster for small, obvious pollution cases.
- Keeps existing bank and recall coverage.
- Useful when only a few documents need correction.

Cons:

- Mutates the current bank; requires backup and explicit confirmation.
- Coarse repair granularity because exact fact/observation edits are not well exposed.
- Does not undo day-topic mixing when the original document is inherently multi-scope.
- Risky before native consolidation because wrong tags can become wrong high-level observations.

### Session JSON v3 clean route

Pros:

- Fixes the source of pollution rather than patching symptoms.
- Preserves natural session provenance and stable document ids.
- Supports blue/green migration: build `hermes_v3`, benchmark, then switch recall target later.
- Better suited for Hindsight native observations/consolidation.

Cons:

- More up-front work and more audits.
- Manual-review set can be large.
- Initial coverage is lower than the old archive bank until backfill progresses.
- Paid retain/consolidation must be batched with cost controls and long/no-timeout runners.

## Recommended answer pattern

When explaining this to the user:

1. Say the SQLite rebuild is operationally successful but not a clean canonical memory.
2. Define tag/scope in terms of Hindsight consolidation grouping, not just labels.
3. Distinguish post-hoc repair (tag/scope cleanup) from clean re-ingestion (session JSON v3).
4. Emphasize current best path: keep old `hermes` as archive/evidence; use discard-first for local repairs; build `hermes_v3` as the long-term clean production candidate.
5. Do not imply destructive cleanup or full production migration has already been done unless live audit confirms it.
