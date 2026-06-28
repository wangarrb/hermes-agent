#!/usr/bin/env python3
"""Build production-review proposals from approved Hindsight repair sidecars.

This script is intentionally local-file only. It does not call Hindsight APIs and
never mutates production memory. Output is a user-reviewable proposal bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "hindsight-repair-canonical-proposals-v1"

SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-.]{12,}\b"),
    re.compile(r"(?i)\b(api[_ -]?key|token|secret|password|passwd)\b\s*[:=]\s*[^\s,;]{6,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}\b"),
    re.compile(r"(?i)\bAKIA[0-9A-Z]{16}\b"),
]
ARTIFACT_PATTERNS = [
    re.compile(r"CONTEXT COMPACTION", re.I),
    re.compile(r"REFERENCE ONLY", re.I),
    re.compile(r"__HERMES_|HERMES_CWD", re.I),
    re.compile(r"hermes_tmp_review_repair", re.I),
    re.compile(r"traceback \(most recent call last\)", re.I),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text


def stable_id(text: str, prefix: str = "proposal") -> str:
    return f"{prefix}::{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def has_secret(text: str) -> bool:
    return any(p.search(text or "") for p in SECRET_PATTERNS)


def artifact_reasons(text: str) -> list[str]:
    reasons = []
    for p in ARTIFACT_PATTERNS:
        if p.search(text or ""):
            reasons.append(p.pattern)
    return reasons


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                rows.append({"_parse_error": str(e), "_line": lineno, "raw": line[:500]})
                continue
            obj["_line"] = lineno
            rows.append(obj)
    return rows


def merge_unique(*lists: Any) -> list[str]:
    seen = set()
    out: list[str] = []
    for xs in lists:
        if xs is None:
            continue
        if isinstance(xs, str) or not hasattr(xs, "__iter__"):
            xs = [xs]
        for x in xs:
            s = str(x)
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def priority_score(item: dict[str, Any]) -> float:
    text = item.get("canonical_text", "")
    lower = text.lower()
    score = 0.0
    score += min(20, 4 * item.get("evidence_count", 0))
    score += min(15, 5 * item.get("source_document_count", 0))
    if any(w in lower for w in ["must", "never", "默认", "必须", "禁止", "不应", "不要"]):
        score += 12
    if any(w in lower for w in ["p0", "priority", "critical", "风险", "gate", "质量", "rollback"]):
        score += 10
    if any(w in lower for w in ["hindsight", "consolidation", "retain", "recall", "offline"]):
        score += 5
    if any(w in lower for w in ["egomotion4d", "can", "gtsam", "vggt", "dage", "phase"]):
        score += 4
    # Prefer compact, durable facts over long transcript-like paragraphs.
    n = len(text)
    if 80 <= n <= 500:
        score += 5
    elif n > 900:
        score -= 8
    return round(score, 2)


def build_proposals(approved_index: str | Path, *, top: int | None = None) -> dict[str, Any]:
    path = Path(approved_index).expanduser()
    rows = load_jsonl(path)
    rejected: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    raw_accepted = 0

    for row in rows:
        if row.get("_parse_error"):
            rejected.append({"line": row.get("_line"), "reason": "json_parse_error", "detail": row.get("_parse_error")})
            continue
        text = normalize_text(row.get("insight") or row.get("text") or "")
        if not text:
            rejected.append({"line": row.get("_line"), "id": row.get("id"), "reason": "empty_text"})
            continue
        if row.get("status") not in (None, "approved"):
            rejected.append({"line": row.get("_line"), "id": row.get("id"), "reason": "not_approved"})
            continue
        if has_secret(text):
            rejected.append({"line": row.get("_line"), "id": row.get("id"), "reason": "secret_like_material"})
            continue
        reasons = artifact_reasons(text)
        if reasons:
            rejected.append({"line": row.get("_line"), "id": row.get("id"), "reason": "artifact_or_prompt_leak", "patterns": reasons})
            continue

        raw_accepted += 1
        key = normalize_text(text).lower()
        g = groups.setdefault(
            key,
            {
                "proposal_id": stable_id(key),
                "canonical_text": text,
                "topic": row.get("topic") or "unknown",
                "type": row.get("type") or "technical_lesson",
                "tags": [],
                "sidecar_ids": [],
                "source_documents": [],
                "source_fact_ids": [],
                "evidence_ids": [],
                "candidate_ids": [],
                "source_lines": [],
                "layer": "approved_repair_sidecar",
                "merge_gate": "user_approval_required",
                "production_action": "proposal_only_no_write",
            },
        )
        g["sidecar_ids"] = merge_unique(g.get("sidecar_ids"), row.get("id"))
        g["source_documents"] = merge_unique(g.get("source_documents"), row.get("source_documents"), row.get("provenance", {}).get("document_id"), row.get("provenance", {}).get("parent_document_id"))
        g["source_fact_ids"] = merge_unique(g.get("source_fact_ids"), row.get("source_fact_ids"))
        g["evidence_ids"] = merge_unique(g.get("evidence_ids"), row.get("evidence_ids"), row.get("source_fact_ids"))
        g["tags"] = merge_unique(g.get("tags"), row.get("tags"))
        cand = (row.get("provenance") or {}).get("candidate_id")
        g["candidate_ids"] = merge_unique(g.get("candidate_ids"), cand)
        g["source_lines"] = merge_unique(g.get("source_lines"), row.get("_line"))

    proposals = list(groups.values())
    for p in proposals:
        p["evidence_count"] = len(p.get("evidence_ids") or p.get("source_fact_ids") or [])
        p["source_document_count"] = len(p.get("source_documents") or [])
        p["sidecar_record_count"] = len(p.get("sidecar_ids") or [])
        p["priority_score"] = priority_score(p)
        p["quality_flags"] = []
        if p["evidence_count"] < 1:
            p["quality_flags"].append("no_evidence_ids")
        if p["source_document_count"] < 1:
            p["quality_flags"].append("no_source_documents")
        if len(p["canonical_text"]) > 900:
            p["quality_flags"].append("long_text_manual_distill_recommended")

    proposals.sort(key=lambda x: (x.get("priority_score", 0), x.get("evidence_count", 0), x.get("source_document_count", 0)), reverse=True)
    if top is not None and top > 0:
        proposals_out = proposals[:top]
    else:
        proposals_out = proposals

    topic_counts = Counter(p.get("topic") or "unknown" for p in proposals)
    type_counts = Counter(p.get("type") or "unknown" for p in proposals)
    quality = {
        "rows_seen": len(rows),
        "raw_accepted_rows": raw_accepted,
        "accepted_unique": len(proposals),
        "emitted_proposals": len(proposals_out),
        "deduped_rows": raw_accepted - len(proposals),
        "rejected": len(rejected),
        "reject_reasons": dict(Counter(r.get("reason", "unknown") for r in rejected)),
        "topic_counts": dict(topic_counts),
        "type_counts": dict(type_counts),
        "source": str(path),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "source": str(path),
        "quality": quality,
        "proposals": proposals_out,
        "rejected": rejected,
    }


def render_markdown(bundle: dict[str, Any]) -> str:
    q = bundle["quality"]
    lines = [
        "# Hindsight repair canonical proposals",
        "",
        f"generated_at: {bundle['generated_at']}",
        f"source: {bundle['source']}",
        "",
        "## Quality summary",
        "",
        f"- rows_seen: {q['rows_seen']}",
        f"- raw_accepted_rows: {q['raw_accepted_rows']}",
        f"- accepted_unique: {q['accepted_unique']}",
        f"- emitted_proposals: {q['emitted_proposals']}",
        f"- deduped_rows: {q['deduped_rows']}",
        f"- rejected: {q['rejected']} {q.get('reject_reasons', {})}",
        f"- topic_counts: {q.get('topic_counts', {})}",
        "",
        "## Production boundary",
        "",
        "This file is proposal-only. Do not mutate the production Hindsight bank until the user explicitly approves a merge/retain plan.",
        "",
        "## Proposals",
        "",
    ]
    for i, p in enumerate(bundle["proposals"], 1):
        lines.extend([
            f"### {i}. {p['proposal_id']} score={p['priority_score']}",
            "",
            f"- topic/type: {p.get('topic')} / {p.get('type')}",
            f"- evidence_count: {p.get('evidence_count')} ; source_document_count: {p.get('source_document_count')} ; sidecar_record_count: {p.get('sidecar_record_count')}",
            f"- tags: {p.get('tags')}",
            f"- source_documents: {p.get('source_documents')}",
            f"- quality_flags: {p.get('quality_flags')}",
            f"- merge_gate: {p.get('merge_gate')}",
            "",
            p.get("canonical_text", ""),
            "",
        ])
    return "\n".join(lines)


def write_outputs(bundle: dict[str, Any], output_root: Path, stem: str) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    proposal_json = output_root / f"{stem}-canonical-proposals.json"
    proposal_md = output_root / f"{stem}-canonical-proposals.md"
    quality_json = output_root / f"{stem}-quality-report.json"
    proposal_json.write_text(json.dumps({k: v for k, v in bundle.items() if k != "rejected"}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    proposal_md.write_text(render_markdown(bundle), encoding="utf-8")
    quality_json.write_text(json.dumps({"schema_version": bundle["schema_version"], "generated_at": bundle["generated_at"], "quality": bundle["quality"], "rejected": bundle["rejected"]}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"proposal_json": str(proposal_json), "proposal_md": str(proposal_md), "quality_json": str(quality_json)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build local production-review proposal bundle from approved Hindsight repair sidecar")
    ap.add_argument("--approved-index", required=True, type=Path)
    ap.add_argument("--output-root", type=Path, default=Path.home() / ".hermes" / "hindsight" / "review_repair" / "proposals")
    ap.add_argument("--stem", default=None)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    bundle = build_proposals(args.approved_index, top=args.top)
    stem = args.stem or Path(args.approved_index).name.replace("-observations_index.jsonl", "")
    paths = write_outputs(bundle, args.output_root.expanduser(), stem)
    result = {"paths": paths, "quality": bundle["quality"]}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"wrote {paths['proposal_json']}")
        print(f"wrote {paths['proposal_md']}")
        print(f"wrote {paths['quality_json']}")
        print(json.dumps(bundle["quality"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
