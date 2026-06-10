# V4 recall smoke and conflict-driven repair loop

Session-specific lessons from the 2026-05-08 V4 smoke / audit pass.

## What changed

- A bank can look clean on counts and still be semantically contaminated.
- Query-level recall smoke is required; tag/broad/system counts alone are not enough.
- Mixed high-level tags such as `domain:hindsight`, `domain:autodrive`, `topic:memory-management`, `project:egomotion4d`, and `project:openclaw` can still cause cross-topic recall bleed.
- Secret/credential-like sessions should be routed to manual review, and audit outputs must redact credential-like strings before writing JSON/Markdown/stdout.
- Paid/native consolidation can need a generous health timeout because local embedding/model reloads are slow; background+notify is safer than foreground timeouts.

## V4 lessons to carry forward

- Keep a temporary smoke bank for validation; do not treat it as production truth.
- After any paid/native run, do a tag audit plus recall smoke.
- Add topic/tag co-occurrence drift checks to weekly gates.
- Use conflict-driven repair:
  1. detect high-level anomaly
  2. trace lineage downward
  3. isolate the faulty layer
  4. write a repair proposal only
  5. keep destructive actions behind explicit confirmation

## Repair loop inputs

- wrong recall answer
- wrong observation
- conflict in high-level summaries
- user-reported memory error
- recall-smoke contamination
- useful knowledge discovered in the recall/repair zone during weekly conflict review or high-dimensional information extraction

## Weekly recall-zone promotion loop

When weekly conflict handling or high-level/high-dimensional extraction searches the recall/repair zone and finds useful knowledge, treat it as promotion evidence, not as production truth.

Flow:

1. Search the recall/repair zone sidecars, temp-bank reports, and raw/evidence bank with source-preserving recall.
2. For each useful hit, trace lineage back to source document/span or temp memory ids.
3. Confirm it against source context and existing production facts; classify as new fact, refinement, duplicate, conflict, or obsolete.
4. Rewrite it into a minimal canonical observation / repair note with clear scope, applicability, evidence ids, and no temp-bank/debug metadata.
5. Merge only through a production proposal/apply stage. Do not auto-copy temp-bank memory units or sidecar text into `hermes`.
6. After merge, rerun recall smoke and conflict checks; if the sidecar item is now represented in production, mark the sidecar/proposal as promoted or superseded to avoid duplicate truth sources.

This loop lets useful recall-zone knowledge graduate into the main bank, while keeping unverified or noisy temp results outside production.

## Practical guardrails

- Use a temporary build/smoke bank if it reduces risk.
- Prefer long health timeouts for paid/background runs.
- Do not rely on counts alone to validate semantic cleanliness.
- Keep credential-like material out of audit logs and reports.
