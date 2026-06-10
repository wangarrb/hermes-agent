# Hindsight LLM cleanup / repair loop

Session learning: 2026-05-10 user wanted periodic audit plus a Hermes-large-model pass that can clean conflicts/errors in Hindsight. The safe design is LLM-assisted planning, not direct autonomous mutation of production memory.

## Trigger

Use this reference when the user asks to:
- periodically audit Hindsight memory quality;
- clean conflicts, stale claims, cross-topic contamination, missing lineage, or low-value memories;
- use a large Hermes/paid model to process many Hindsight conflicts at once;
- repair production banks such as `hermes` after offline retain/reflect/consolidation.

## Current capability boundary

Existing scripts are mostly read-only / proposal-first:
- `hindsight_bank_quality_audit.py`: read-only bank quality audit; status/stats/tag pollution/cross-domain contamination/lineage/recall smoke.
- `hindsight_conflict_audit.py`: read-only conflict audit; detects contamination, missing lineage, dangling evidence, numeric divergence, polarity conflicts; accepts manual claims.
- `hindsight_repair_proposal.py`: deterministic repair proposal from conflict cases; no LLM calls and no Hindsight writes.
- `hindsight_review_backlog_llm_scorer.py`: can call an OpenAI-compatible LLM, but only writes sidecar scores; `production_mutation_allowed=false`.
- `hindsight_discard_manager.py`: discard/quarantine snapshot manager; snapshots document/facts/derived observations before any separate mutation.
- `hindsight_offline_v2_rebuild.py`: reduce -> conflict audit -> eval gate -> optional publish; publish requires explicit confirm token.

Do not claim that a fully automatic production-cleaning loop already exists unless an apply tool has been implemented and verified. As of this learning, the robust design is proposal-only first, then gated apply after confirmation.

## Recommended architecture

1. Read-only audit input
   - Run bank quality audit and conflict audit.
   - Optionally sample review backlog and run LLM scorer.
   - Gather source document ids, evidence ids, raw-span snippets, and daily/weekly output summaries.

2. Hermes LLM cleanup pass
   - Send only redacted/source-backed evidence to the model.
   - Ask for a strict JSON repair plan, not direct memory content rewrites.
   - The model may classify, merge, split scope, identify stale facts, and propose quarantine/supersede/re-retain candidates.

3. Programmatic validator
   - Fail closed if evidence/source ids are missing.
   - Secret/credential-like material must route to manual review.
   - `delete` is never executable directly; downgrade to `delete_candidate`.
   - `confidence != high` cannot auto-apply.
   - Cross-topic moves require explicit source evidence.
   - Supersede requires old claim, new claim, and applicability/scope.
   - Any production mutation requires a discard snapshot first.

4. Proposal files only
   - Write local artifacts under e.g. `$HOME/.hermes/hindsight/offline_reflect/llm_cleaner/<timestamp>/`:
     - `repair_plan.json`
     - `repair_plan.md`
     - optional `validator_report.json`
   - No Hindsight writes in the LLM pass.

5. Apply only after confirmation
   - Snapshot impacted document/facts/observations with `hindsight_discard_manager.py`.
   - Apply low-risk/non-destructive actions first.
   - Destructive operations (delete/overwrite/clear observation/re-retain production/publish canonical cards) require explicit user confirmation and built-in confirm tokens.
   - Re-run audit + recall smoke after apply.

## Repair plan schema

Use a schema like:

```json
{
  "schema_version": "hindsight-llm-cleanup-plan-v1",
  "generated_at": "ISO-8601",
  "source_reports": [],
  "repairs": [
    {
      "case_id": "conflict::...",
      "target_id": "document/fact/observation id",
      "action": "keep|quarantine|supersede|split_scope|re_retain|delete_candidate|manual_review",
      "confidence": "high|medium|low",
      "evidence_ids": [],
      "source_document_ids": [],
      "corrected_claim": "source-backed corrected claim or empty",
      "supersedes": [],
      "scope": "when/config/date/topic this claim applies to",
      "reason": "short source-backed reason; no chain of thought",
      "risk": "low|medium|high"
    }
  ]
}
```

## What can be automated vs not

Safe to auto-generate:
- clean canonical-card candidates;
- quarantine candidates;
- supersede proposals;
- re-retain candidate manifests;
- regression/audit rules;
- local markdown/json reports.

Do not auto-execute without confirmation:
- deleting production documents;
- overwriting production facts;
- clearing derived observations;
- re-retaining production documents;
- publishing observations/canonical cards to the main bank.

## User preference captured

The user wants Hermes large-model help for periodic cleanup and conflict/error handling, but the implementation should remain evidence-first, gated, and recoverable. Favor a one-shot LLM cleanup window that produces a structured repair plan, with programmatic validation and discard snapshots before any production mutation.
