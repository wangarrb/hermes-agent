# Kanban Independent Review Timeout Completion

**Status:** approved for implementation

**Date:** 2026-07-30

## Decision

An implementer-owned independent review is an internal quality check, not the
final acceptance gate. If the independent reviewer returns an explicit
`FAIL`/`REWORK`, completion remains blocked. If the review process reaches a
real timeout without a verdict, the implementer records the timeout and may
complete immediately; planner or designer performs the later acceptance.

## Minimal Contract

`hermes kanban complete` accepts this completion metadata for an implementer
review timeout:

```json
{
  "independent_review_outcome": "TIMEOUT",
  "independent_review_note": "deleg_x timed out after 600s"
}
```

The note is required so the completed task retains an auditable reviewer id,
timeout duration, or equivalent evidence. The completion record is normalized
to a stored `TIMEOUT` review outcome. This path does not require a PASS review
artifact or Git delivery lifecycle binding.

Normal independent-review behavior is unchanged:

- `PASS` uses the existing bound `independent_review.v1` artifact.
- `FAIL`/`REWORK` cannot use the timeout path.
- Missing review evidence without an explicit timeout remains blocked.

## Agent Guidance

The implementer prompt documents the exact timeout-completion metadata. It
must not convert a known negative verdict into `TIMEOUT`. Planner/designer
remain responsible for downstream acceptance and can return the completed
delivery for rework.

## Tests

- An implementer completion with `TIMEOUT` plus a non-empty note bypasses the
  PASS artifact gate and stores normalized timeout evidence.
- `TIMEOUT` without a note is rejected.
- Existing missing-artifact and non-PASS verdict tests remain unchanged.
- Non-implementer completion behavior remains unchanged.
