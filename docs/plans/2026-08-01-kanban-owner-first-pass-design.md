# Kanban Owner First-Pass Reliability Design

## Goal

Raise planner/designer first-pass acceptance and same-card completion without
adding reviewer gates, branches, worktrees, or polling tasks.

## Design

1. **Current evidence before stale history.** For goal-mode worker context,
   render the latest durable corrective handback before attempt summaries,
   show full attempts only from the current generation, and omit cross-task
   role history. Older generations remain durable in SQLite and `show`, but
   appear in the injected prompt only as a count.
2. **Safe Hermes wake boundary.** Treat the exact idle status line
   `⚕ ❯ msg=interrupt ...` as idle. The visually similar active line without
   `❯` is busy and must never receive a verdict wake or completion check.
3. **Evidence-complete preflight.** Add one concise planner/designer rule:
   before formal review, map every submitted success gate to a real command or
   artifact and run the primary production-path falsifier in the same modality
   as final acceptance. Missing rows mean continue implementation, not submit.

## Boundaries

- Do not change Kanban task lifecycle, review-count rules, or goal completion
  semantics.
- Do not delete history; only reduce stale material in active goal prompts.
- Do not add a new task schema or mechanical evidence gate.
- Keep ordinary non-goal worker context unchanged.

## Verification

- A reworked goal sees its newest corrective comment before history, current-
  generation attempts in full, and only a collapsed count for older attempts.
- A normal task retains existing prior-attempt and recent-role-history output.
- Hermes active `⚕ msg=interrupt ...` is rejected; exact
  `⚕ ❯ msg=interrupt ...` is accepted.
- Planner/designer role prompt contains one compact evidence-matrix rule and
  no duplicated workflow.
