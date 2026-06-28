#!/usr/bin/env python3
"""Fail-closed gate for Offline Hindsight v2 local canonical cards.

Purpose:
- Compare baseline eval reports against eval reports with --use-local-cards.
- Decide whether local v2 cards are eligible for a future Hindsight retain proposal.
- Optionally emit local-only proposal preview files when all gates pass.

Safety:
- No LLM calls.
- No Hindsight API calls.
- No Hindsight writes.
- Writes only local report/proposal files under --output-dir.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import hindsight_conflict_core as conflict_core
except Exception:  # pragma: no cover - gate still reports missing dependency via conflict audit check
    conflict_core = None

HOME = Path.home()
HERMES_HOME = HOME / ".hermes"
DEFAULT_CARDS_ROOT = HERMES_HOME / "hindsight" / "offline_reflect" / "v2_cards"
DEFAULT_OUTPUT_DIR = HERMES_HOME / "hindsight" / "offline_reflect" / "v2_publish_gate"


def slugify(text: str, *, prefix: str = "item") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:80]
    return slug or prefix


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"failed to load JSON: {path}: {e}")


def summary(report: dict[str, Any]) -> dict[str, Any]:
    s = report.get("summary")
    if not isinstance(s, dict):
        raise SystemExit("eval report missing summary")
    return s


def result_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(report.get("results") or []):
        rid = str(row.get("id") or f"case_{i:04d}")
        out[rid] = row
    return out


def fnum(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def inum(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def evaluate_pair(
    label: str,
    baseline_path: Path,
    cards_path: Path,
    *,
    min_term_recall_delta: float,
    min_layer_hit_delta: int,
    min_score_delta: float,
    max_case_term_regressions: int,
    case_regression_tolerance: float,
) -> dict[str, Any]:
    baseline = load_json(baseline_path)
    cards = load_json(cards_path)
    bs = summary(baseline)
    cs = summary(cards)
    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: str, severity: str = "blocker") -> None:
        checks.append({"name": name, "passed": bool(passed), "severity": severity, "detail": detail})

    b_cases = inum(bs.get("case_count"))
    c_cases = inum(cs.get("case_count"))
    add_check("same_case_count", b_cases == c_cases and b_cases > 0, f"baseline={b_cases}, cards={c_cases}")

    b_cards_root = baseline.get("cards_root")
    c_cards_root = cards.get("cards_root")
    add_check("baseline_has_no_cards", b_cards_root in {None, "", "None"}, f"baseline cards_root={b_cards_root!r}")
    add_check("cards_eval_has_cards_root", bool(c_cards_root), f"cards cards_root={c_cards_root!r}")

    b_term = fnum(bs.get("layered_avg_term_recall"))
    c_term = fnum(cs.get("layered_avg_term_recall"))
    term_delta = round(c_term - b_term, 6)
    add_check(
        "avg_term_recall_improved",
        term_delta >= min_term_recall_delta,
        f"baseline={b_term}, cards={c_term}, delta={term_delta}, required>={min_term_recall_delta}",
    )

    b_hits = inum(bs.get("layered_expected_layer_hits"))
    c_hits = inum(cs.get("layered_expected_layer_hits"))
    hit_delta = c_hits - b_hits
    add_check(
        "expected_layer_hits_improved",
        hit_delta >= min_layer_hit_delta,
        f"baseline={b_hits}, cards={c_hits}, delta={hit_delta}, required>={min_layer_hit_delta}",
    )

    b_score = fnum(bs.get("layered_avg_score"))
    c_score = fnum(cs.get("layered_avg_score"))
    score_delta = round(c_score - b_score, 6)
    add_check(
        "avg_score_not_regressed",
        score_delta >= min_score_delta,
        f"baseline={b_score}, cards={c_score}, delta={score_delta}, required>={min_score_delta}",
    )

    b_results = result_by_id(baseline)
    c_results = result_by_id(cards)
    regressions = []
    for rid, brow in b_results.items():
        crow = c_results.get(rid)
        if not crow:
            regressions.append({"id": rid, "reason": "missing_in_cards_eval"})
            continue
        b_case_term = fnum(((brow.get("layered") or {}).get("term_recall")))
        c_case_term = fnum(((crow.get("layered") or {}).get("term_recall")))
        delta = round(c_case_term - b_case_term, 6)
        if delta < -case_regression_tolerance:
            regressions.append({"id": rid, "baseline": b_case_term, "cards": c_case_term, "delta": delta})
    add_check(
        "case_term_recall_regressions_within_limit",
        len(regressions) <= max_case_term_regressions,
        f"regressions={len(regressions)}, allowed<={max_case_term_regressions}",
    )

    blockers = [c for c in checks if c["severity"] == "blocker" and not c["passed"]]
    return {
        "label": label,
        "baseline_eval": str(baseline_path),
        "cards_eval": str(cards_path),
        "cards_root": c_cards_root,
        "metrics": {
            "baseline_layered_avg_score": b_score,
            "cards_layered_avg_score": c_score,
            "score_delta": score_delta,
            "baseline_layered_avg_term_recall": b_term,
            "cards_layered_avg_term_recall": c_term,
            "term_recall_delta": term_delta,
            "baseline_layered_expected_layer_hits": b_hits,
            "cards_layered_expected_layer_hits": c_hits,
            "expected_layer_hits_delta": hit_delta,
            "case_term_recall_regressions": len(regressions),
        },
        "checks": checks,
        "regressions": regressions[:50],
        "passed": not blockers,
    }


def load_cards(cards_root: Path) -> list[dict[str, Any]]:
    manifest = cards_root / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"cards manifest not found: {manifest}")
    cards = []
    for p in sorted(list((cards_root / "topics").glob("*.json")) + list((cards_root / "global").glob("*.json"))):
        cards.append(load_json(p))
    if not cards:
        raise SystemExit(f"no cards found under {cards_root}")

    # Match publish semantics: proposal previews must show the complete detailed
    # observations_index when present, not only compact card top-N observations.
    index_path = cards_root / "observations_index.jsonl"
    if index_path.exists():
        index_obs: list[dict[str, Any]] = []
        for line in index_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("insight"):
                index_obs.append(obj)
        if index_obs:
            topic_cards = {str(c.get("topic") or ""): c for c in cards if c.get("scope") == "topic"}
            global_card = next((c for c in cards if c.get("scope") == "global"), None) or cards[-1]
            for c in cards:
                c["compact_card_observation_count"] = len(c.get("canonical_observations") or [])
                c["canonical_observations"] = []
            for obs in index_obs:
                topic = str(obs.get("topic") or "global")
                target = topic_cards.get(topic, global_card)
                target.setdefault("canonical_observations", []).append(obs)
            cards = [c for c in cards if c.get("canonical_observations")]
    return cards


def card_to_candidate_document(card: dict[str, Any]) -> dict[str, Any]:
    scope = str(card.get("scope") or "topic")
    topic = str(card.get("topic") or "global")
    card_id = str(card.get("card_id") or "")
    suffix = re.sub(r"[^a-f0-9]", "", card_id.lower())[-12:] or slugify(topic)
    document_id = f"hermes-offline-canonical::{scope}::{slugify(topic, prefix='topic')}::{suffix}"
    lines = []
    lines.append(f"# Offline Hindsight Canonical Card: {topic}")
    lines.append("")
    lines.append(f"card_id: {card_id}")
    lines.append(f"schema_version: {card.get('schema_version')}")
    lines.append(f"scope: {scope}")
    lines.append(f"generated_at: {card.get('generated_at')}")
    lines.append("")
    lines.append("## Executive Summary")
    for item in card.get("executive_summary") or []:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Canonical Observations")
    for obs in card.get("canonical_observations") or []:
        lines.append(f"- id: {obs.get('id')}")
        lines.append(f"  confidence: {obs.get('confidence')}")
        lines.append(f"  type: {obs.get('type')}")
        lines.append(f"  insight: {obs.get('insight')}")
        if obs.get("applicability"):
            lines.append(f"  applicability: {obs.get('applicability')}")
        if obs.get("evidence_ids"):
            lines.append(f"  evidence_ids: {', '.join(map(str, obs.get('evidence_ids')[:12]))}")
        if obs.get("source_documents"):
            lines.append(f"  sources: {', '.join(map(str, obs.get('source_documents')[:5]))}")
    lines.append("")
    lines.append("## Evidence Index")
    for e in card.get("evidence_index") or []:
        lines.append(f"- {e}")
    return {
        "document_id": document_id,
        "context": "offline_hindsight_canonical_v2_local_proposal",
        "content": "\n".join(lines).strip() + "\n",
        "metadata": {
            "source_card_id": card_id,
            "scope": scope,
            "topic": topic,
            "observation_count": len(card.get("canonical_observations") or []),
            "compact_card_observation_count": card.get("compact_card_observation_count"),
            "safety": "proposal_preview_only_not_retained",
        },
    }


def write_proposal(cards_root: Path, output_dir: Path) -> dict[str, Any]:
    cards = load_cards(cards_root)
    docs = [card_to_candidate_document(c) for c in cards]
    proposal_jsonl = output_dir / "canonical-retain-proposal.jsonl"
    proposal_md = output_dir / "canonical-retain-proposal.md"
    proposal_jsonl.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs) + "\n", encoding="utf-8")
    lines = ["# Offline Hindsight v2 Canonical Retain Proposal", "", "Status: local preview only; not retained to Hindsight.", ""]
    for doc in docs:
        lines.append(f"## {doc['document_id']}")
        lines.append(f"topic: {doc['metadata'].get('topic')}, observations: {doc['metadata'].get('observation_count')}")
        lines.append("")
    proposal_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return {"document_count": len(docs), "jsonl_path": str(proposal_jsonl), "markdown_path": str(proposal_md)}


def evaluate_conflict_audit(path: Path | None, *, block_severity: str = "P1", required: bool = False) -> dict[str, Any]:
    """Convert conflict/raw-lineage audit JSON into a gate check.

    The check is read-only and fail-closed when required=True. It blocks when any
    case severity is >= block_severity.
    """
    if path is None:
        return {
            "name": "conflict_audit_passed",
            "passed": not required,
            "severity": "blocker",
            "detail": "conflict audit not provided" + ("; required" if required else "; optional skipped"),
            "blocking_case_count": None,
            "blocking_examples": [],
        }
    p = Path(path).expanduser()
    if not p.exists():
        return {
            "name": "conflict_audit_passed",
            "passed": False,
            "severity": "blocker",
            "detail": f"conflict audit missing: {p}",
            "blocking_case_count": None,
            "blocking_examples": [],
        }
    try:
        audit = load_json(p)
    except Exception as e:
        return {
            "name": "conflict_audit_passed",
            "passed": False,
            "severity": "blocker",
            "detail": f"failed to load conflict audit: {p}: {e}",
            "blocking_case_count": None,
            "blocking_examples": [],
        }
    cases = audit.get("cases") or []
    blocking = []
    if conflict_core is not None:
        for case in cases:
            if conflict_core.severity_at_least(str(case.get("severity") or "OK"), block_severity):
                blocking.append(case)
    else:
        order = {"OK": 0, "P3": 1, "P2": 2, "P1": 3, "P0": 4}
        threshold = order.get(block_severity, 3)
        for case in cases:
            if order.get(str(case.get("severity") or "OK").upper(), 0) >= threshold:
                blocking.append(case)
    summary = audit.get("summary") or {}
    recomputed_blocking_count = len(blocking)
    summary_blocking_count = None
    if summary.get("blocking_cases") is not None:
        try:
            summary_blocking_count = int(summary.get("blocking_cases") or 0)
        except Exception:
            summary_blocking_count = None
    summary_mismatch = summary_blocking_count is not None and summary_blocking_count != recomputed_blocking_count
    # Fail closed: the case list is the source of truth. If the summary disagrees,
    # use the stricter count and mark the audit inconsistent.
    blocking_count = max(recomputed_blocking_count, summary_blocking_count or 0)
    examples = [
        {"case_id": c.get("case_id"), "type": c.get("type"), "severity": c.get("severity"), "target": c.get("target"), "title": c.get("title")}
        for c in (blocking or cases)[:10]
    ]
    decision_blocked = str(audit.get("decision") or "pass") == "blocked_conflict_review_required"
    passed = blocking_count == 0 and not decision_blocked and not summary_mismatch
    detail_extra = "; summary_mismatch=true" if summary_mismatch else ""
    return {
        "name": "conflict_audit_passed",
        "passed": passed,
        "severity": "blocker",
        "detail": f"audit={p}, decision={audit.get('decision')}, cases={len(cases)}, blocking_cases={blocking_count}, recomputed_blocking_cases={recomputed_blocking_count}, threshold>={block_severity}{detail_extra}",
        "audit_path": str(p),
        "blocking_case_count": blocking_count,
        "recomputed_blocking_case_count": recomputed_blocking_count,
        "summary_blocking_case_count": summary_blocking_count,
        "summary_mismatch": summary_mismatch,
        "blocking_examples": examples if not passed else [],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = []
    lines.append("# Offline Hindsight v2 Publish Gate")
    lines.append("")
    lines.append(f"generated_at: {report['generated_at']}")
    lines.append(f"decision: {report['decision']}")
    lines.append(f"cards_root: {report.get('cards_root')}")
    lines.append("")
    lines.append("## Safety")
    for item in report.get("safety") or []:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Global Checks")
    for c in report.get("global_checks") or []:
        mark = "PASS" if c.get("passed") else "FAIL"
        lines.append(f"- {mark} {c.get('name')}: {c.get('detail')}")
    lines.append("")
    lines.append("## Eval Pairs")
    for pair in report.get("pairs") or []:
        lines.append(f"### {pair['label']}")
        lines.append(f"passed: {pair['passed']}")
        m = pair.get("metrics") or {}
        lines.append(
            "metrics: "
            f"score_delta={m.get('score_delta')}, "
            f"term_recall_delta={m.get('term_recall_delta')}, "
            f"expected_layer_hits_delta={m.get('expected_layer_hits_delta')}, "
            f"case_term_regressions={m.get('case_term_recall_regressions')}"
        )
        for c in pair.get("checks") or []:
            mark = "PASS" if c.get("passed") else "FAIL"
            lines.append(f"- {mark} {c.get('name')}: {c.get('detail')}")
        lines.append("")
    proposal = report.get("proposal")
    if proposal:
        lines.append("## Proposal Preview")
        lines.append(f"document_count: {proposal.get('document_count')}")
        lines.append(f"jsonl_path: {proposal.get('jsonl_path')}")
        lines.append(f"markdown_path: {proposal.get('markdown_path')}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Fail-closed gate for Offline Hindsight v2 local canonical cards")
    ap.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("LABEL", "BASELINE_EVAL_JSON", "CARDS_EVAL_JSON"),
        help="Eval pair to compare. Repeat for generic and local benchmarks.",
    )
    ap.add_argument("--cards-root", default=str(DEFAULT_CARDS_ROOT))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--min-pairs", type=int, default=2, help="Fail unless at least this many eval pairs are provided")
    ap.add_argument("--min-term-recall-delta", type=float, default=0.001, help="Strict positive average term-recall improvement required")
    ap.add_argument("--min-layer-hit-delta", type=int, default=1, help="Expected layer hits must improve by at least this many")
    ap.add_argument("--min-score-delta", type=float, default=0.0, help="Average score must not regress")
    ap.add_argument("--max-case-term-regressions", type=int, default=0, help="Allowed per-case layered term recall regressions")
    ap.add_argument("--case-regression-tolerance", type=float, default=0.0)
    ap.add_argument("--emit-proposal", action="store_true", help="If gate passes, emit local-only canonical retain proposal preview files")
    ap.add_argument("--conflict-audit-json", help="Conflict/raw-lineage audit JSON. Blocks publish when cases >= --conflict-block-severity exist.")
    ap.add_argument("--require-conflict-audit", action="store_true", help="Fail gate if --conflict-audit-json is absent or missing")
    ap.add_argument("--conflict-block-severity", choices=["P0", "P1", "P2", "P3"], default="P1")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    pairs_arg = args.pair or []
    cards_root = Path(args.cards_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    global_checks: list[dict[str, Any]] = []

    def add_global_check(name: str, passed: bool, detail: str) -> None:
        global_checks.append({"name": name, "passed": bool(passed), "severity": "blocker", "detail": detail})

    add_global_check("min_eval_pairs", len(pairs_arg) >= args.min_pairs, f"provided={len(pairs_arg)}, required>={args.min_pairs}")
    add_global_check("cards_manifest_exists", (cards_root / "manifest.json").exists(), f"manifest={cards_root / 'manifest.json'}")
    global_checks.append(
        evaluate_conflict_audit(
            Path(args.conflict_audit_json).expanduser() if args.conflict_audit_json else None,
            block_severity=args.conflict_block_severity,
            required=args.require_conflict_audit,
        )
    )

    pair_reports = []
    for label, baseline, cards in pairs_arg:
        pair_reports.append(
            evaluate_pair(
                label,
                Path(baseline).expanduser(),
                Path(cards).expanduser(),
                min_term_recall_delta=args.min_term_recall_delta,
                min_layer_hit_delta=args.min_layer_hit_delta,
                min_score_delta=args.min_score_delta,
                max_case_term_regressions=args.max_case_term_regressions,
                case_regression_tolerance=args.case_regression_tolerance,
            )
        )

    blockers = [c for c in global_checks if not c["passed"]]
    blockers.extend(pair for pair in pair_reports if not pair.get("passed"))
    decision = "eligible_for_local_proposal" if not blockers else "blocked_keep_local_only"

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision": decision,
        "cards_root": str(cards_root),
        "thresholds": {
            "min_pairs": args.min_pairs,
            "min_term_recall_delta": args.min_term_recall_delta,
            "min_layer_hit_delta": args.min_layer_hit_delta,
            "min_score_delta": args.min_score_delta,
            "max_case_term_regressions": args.max_case_term_regressions,
            "case_regression_tolerance": args.case_regression_tolerance,
        },
        "safety": [
            "No LLM calls.",
            "No Hindsight API calls.",
            "No Hindsight writes.",
            "Conflict/raw-lineage audit is read-only and can block proposal when high-severity cases are open.",
            "Proposal files, if emitted, are local preview only and require separate explicit retain workflow.",
        ],
        "global_checks": global_checks,
        "pairs": pair_reports,
        "proposal": None,
    }

    if args.emit_proposal and decision == "eligible_for_local_proposal":
        report["proposal"] = write_proposal(cards_root, output_dir)
    elif args.emit_proposal:
        report["proposal"] = {"status": "not_emitted_gate_blocked"}

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"publish-gate-{ts}.json"
    md_path = output_dir / f"publish-gate-{ts}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    (output_dir / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "latest.md").write_text(render_markdown(report), encoding="utf-8")

    if args.json:
        print(json.dumps({"json_path": str(json_path), "markdown_path": str(md_path), **report}, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(report))
        print(f"saved_json: {json_path}")
        print(f"saved_markdown: {md_path}")


if __name__ == "__main__":
    main()
