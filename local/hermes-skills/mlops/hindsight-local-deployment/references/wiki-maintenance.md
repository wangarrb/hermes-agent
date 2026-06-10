---
name: hindsight-wiki-maintenance
description: Maintain a markdown/Obsidian wiki from Hindsight high-level memory outputs without directly editing the curated wiki; generates reviewable candidate pages/reports from canonical observations, weekly/daily summaries, and source notes.
tags: wiki, hindsight, knowledge-base, obsidian, maintenance, canonical-observations
---

# Hindsight Wiki Maintenance

Use this skill when maintaining a markdown/Obsidian wiki from Hindsight or another memory system's high-level outputs.

The goal is not to dump memory facts into a wiki. The goal is to turn stable, high-level, well-supported knowledge into reviewable wiki candidates while preserving provenance and avoiding accidental edits to curated pages.

## When to Use

- A wiki/knowledge base should be refreshed from Hindsight offline outputs.
- A scheduled maintenance job should run after daily/weekly memory consolidation.
- The user asks whether high-level memory outputs should feed wiki pages.
- The user wants candidate wiki updates, not automatic writes to the main wiki.
- The memory system has canonical observations, weekly summaries, daily summaries, conflict audits, lineage traces, or repair proposals.

## Core Principle

Use high-level Hindsight outputs as candidate evidence, not as final wiki truth.

Preferred source order:
1. Published or publish-ready canonical observations / retain proposals.
2. V2/local canonical cards or observation index.
3. Weekly cross-topic summaries.
4. Daily summaries only as fallback or provenance.
5. Raw transcripts only for targeted lineage/debugging, not broad wiki ingestion.

## Safety Rule

Automated maintenance must write to an isolated review area first.

Do not directly modify the main wiki directories such as `concepts/`, `projects/`, `queries/`, `entities/`, or `comparisons/` unless the user explicitly approves that merge.

Recommended layout:

```text
wiki/
  auto-maintenance/
    latest.md
    raw/
    candidates/
    reports/
```

If the existing wiki has a different review folder, use that.

## Standard Workflow

1. Orient to the wiki:
   - Read `SCHEMA.md`.
   - Read `index.md`.
   - Read recent `log.md` entries.
   - Search existing pages before proposing new ones.

2. Wait for memory pipeline completion:
   - Ensure daily/weekly consolidation is complete.
   - Ensure queue/pending work is drained.
   - If a lock file or status endpoint exists, honor it.

3. Read high-level Hindsight state:
   - Latest rebuild/gate/publish report if present.
   - Conflict audit summary.
   - Canonical proposal or canonical cards.
   - Weekly summaries.
   - Daily summaries only when needed.

4. Filter candidates:
   - Keep stable research/work/project knowledge, decisions, risks, constraints, and reusable lessons.
   - Drop transient progress logs, one-off command output, raw tool noise, credentials, personal/private details, and routine status chatter.
   - Prefer items with provenance: evidence IDs, source documents, dates, conflict audit status, or lineage references.

5. Organize candidates by topic:
   - Evolution / timeline.
   - Key findings.
   - Current state.
   - Open questions / risks.
   - Proposed target page.
   - Provenance.

6. Produce isolated outputs:
   - Human-readable candidate report.
   - Optional candidate page drafts under review folder.
   - Machine-readable JSON block or sidecar for later diff/merge automation.

7. Do not merge automatically:
   - The user reviews candidate pages.
   - Only after explicit approval, merge into main wiki and update `index.md`/`log.md`.

## Candidate Report Template

```markdown
# Wiki Maintenance Candidates - YYYY-MM-DD

## Summary
- mode: isolated candidates only
- memory source: Hindsight canonical / weekly / daily
- canonical state: published|local|missing
- conflict gate: passed|blocked|unknown
- candidates: N
- main wiki modified: no

## Topic: <topic>

### Evolution
- YYYY-MM-DD: concise milestone with provenance.

### Key Findings
- Stable conclusion. Evidence: <source/evidence id/path>.

### Current State
- Latest known state and scope.

### Open Questions / Risks
- Unresolved conflict, weak evidence, or stale item.

### Suggested Wiki Action
- update existing page: `concepts/foo.md`, or
- create candidate page: `auto-maintenance/candidates/foo.md`, or
- no action / keep as memory only.

### Provenance
- Hindsight source: <canonical proposal/card/weekly path>
- Evidence IDs / source docs: <ids or paths if available>
```

## Hindsight-Specific Quality Gates

Before recommending a merge candidate, check:

- Conflict audit has no blocking cases for this topic.
- Candidate has enough provenance to trace back to source observations/facts.
- The claim is stable: not just a transient experiment status unless it changes the accepted conclusion.
- The claim has scope: dataset/project/version/date/conditions where applicable.
- If numeric metrics are included, keep units and comparison baseline.
- If an older wiki page conflicts, mark it as a proposed correction rather than silently overwriting it.

## Good Candidate Signals

- Repeated across multiple observations or weekly summaries.
- Encodes a durable decision, architecture rule, workflow, failure mode, or evaluation result.
- Has explicit applicability and limitations.
- Has evidence IDs or source document references.
- Changes how future work should be done.

## Reject / Defer Signals

- One-off progress update with no durable lesson.
- Raw command output or tool logs.
- Secrets, API keys, tokens, private personal info.
- Contradicted by a blocking conflict case.
- Missing provenance for a strong claim.
- Too project-specific for a general concept page; keep it in a project candidate instead.

## Generic Cron Pattern

A scheduled wiki job should run after the offline memory pipeline:

```text
Daily memory pipeline -> Weekly/global consolidation -> Canonical/gate/publish or local proposal -> Wiki maintenance candidate report
```

Cron prompt should be self-contained:

```text
Run wiki auto-maintenance after Hindsight offline pipeline. Wait for the offline pipeline lock/queue to drain. Generate isolated candidate reports under the wiki auto-maintenance directory. Do not edit main wiki pages. Include canonical state, conflict summary, candidate topics, proposed merge targets, and output paths.
```

## Implementation Notes

A reusable script should avoid hard-coded user/project terms.

Recommended configurable inputs:
- `WIKI_PATH`
- `HINDSIGHT_OFFLINE_DIR`
- `WIKI_MAINTENANCE_KEYWORDS`
- lookback days
- source note directories
- output subdirectory
- queue/status command

Domain-specific terms should come from the wiki schema/taxonomy or environment variables, not the generic skill.

When reporting Hindsight state, prefer main DB / publish report verification over a single mutable `latest.json` file. Cron or smoke tests may overwrite `latest.json`; include canonical DB counts or publish metadata when available.

## Verification

After a maintenance run, verify:

- Main wiki files were not modified unless explicitly approved. Ignore incidental Obsidian UI state churn such as `.obsidian/workspace.json` when judging content safety; report it separately if noticed.
- Candidate report path exists.
- The report includes canonical Hindsight state and conflict summary.
- Candidate count and source file count are reported.
- Queue/pipeline status was checked.
- No secrets are printed in the report.

## Merge Procedure After User Approval

1. Read candidate page/report.
2. Read target wiki page(s).
3. Apply minimal edits preserving existing structure.
4. Add/update provenance markers.
5. Update frontmatter `updated` date.
6. Update `index.md` and `log.md`.
7. Run lightweight wiki lint.
8. Report exact files changed.
