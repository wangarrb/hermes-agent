#!/usr/bin/env python3
"""Post-retain fact-quality gate for Hindsight temp-bank pilots.

Read-only. Counts parent coverage, zero-unit documents, and common fact text
artifacts introduced during extraction. Designed for temp-bank retry runs where
Hindsight-visible metadata is intentionally empty and provenance is kept in a
sidecar JSON.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hindsight_bank_quality_audit import load_db_collections, redact_secrets  # noqa: E402

REASONING_RE = re.compile(r"<\s*/?\s*(?:think|thinking)\b|reasoning_content|codex_reasoning_items|chain[-_ ]of[-_ ]thought|\breasoning\b", re.I)
CONTEXT_COMPACTION_RE = re.compile(r"\[CONTEXT COMPACTION\b|END OF CONTEXT SUMMARY|Active Task", re.I)
GENERIC_USER_RE = re.compile(r"Involving:\s*用户\b")
TEMP_BANK_RE = re.compile(r"hermes_tmp|weektrial|first[-_ ]useful[-_ ]window|clean[-_ ]v2", re.I)
PROMPT_LEAK_RE = re.compile(r"Metadata:|Narrator:|custom mission|retain-ready|Retain durable facts|schema_version|Hindsight user_message", re.I)
HEADER_FRAGMENT_RE = re.compile(r"\b(JSON|paths?|provided|chunk|assistant|metadata|document_id|source_json_path)\b", re.I)
SECRET_LIKE_RE = re.compile(r"(?i)(api[_-]?key|secret|password|token|sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16})")
WHEN_FIELD_RE = re.compile(r"When:\s*([^|\n]+)")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_sidecar(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"docs": []}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def suspicious_when_values(text: str) -> list[str]:
    bad: list[str] = []
    for match in WHEN_FIELD_RE.finditer(text):
        value = match.group(1).strip().strip(".;, ")
        compact = re.sub(r"\s+", " ", value)
        if not compact:
            continue
        # Valid-ish values usually look like ISO dates/datetimes or N/A. The
        # local extractor's bad values are truncated prefixes or header fragments.
        if compact.upper() in {"N/A", "NA", "UNKNOWN", "NONE"}:
            continue
        if re.fullmatch(r"20\d{2}(-\d{1,2}){0,2}(T[0-9:.+-Z]+)?", compact):
            continue
        if re.fullmatch(r"20\d{2}年(\d{1,2}月(\d{1,2}日)?)?", compact):
            continue
        if compact in {"20", "202", "2026", "2026-", "2026-0"}:
            bad.append(compact)
            continue
        if HEADER_FRAGMENT_RE.search(compact):
            bad.append(compact[:120])
            continue
        if re.search(r"20\d{2}[^0-9\-年T ]+[A-Za-z_]+", compact):
            bad.append(compact[:120])
    return bad


def artifact_flags(text: str, *, bank: str) -> dict[str, Any]:
    flags: dict[str, Any] = {}
    checks = {
        "reasoning_like": REASONING_RE.search(text),
        "context_compaction": CONTEXT_COMPACTION_RE.search(text),
        "generic_involving_user": GENERIC_USER_RE.search(text),
        "temp_bank_or_window_leak": TEMP_BANK_RE.search(text) or (bank and bank in text),
        "prompt_or_metadata_leak": PROMPT_LEAK_RE.search(text),
        "secret_like": SECRET_LIKE_RE.search(text),
    }
    for key, value in checks.items():
        flags[key] = bool(value)
    bad_when = suspicious_when_values(text)
    flags["malformed_when"] = bool(bad_when)
    if bad_when:
        flags["malformed_when_values"] = bad_when[:5]
    return flags


def doc_id(item: dict[str, Any]) -> str:
    return str(item.get("document_id") or item.get("id") or "")


def parent_from_doc_id(document_id: str) -> str:
    marker = "::clean-v2-first-useful-window"
    if marker in document_id:
        return document_id.split(marker, 1)[0]
    marker = "::first-useful-window"
    if marker in document_id:
        return document_id.split(marker, 1)[0]
    return document_id


def run_gate(*, bank: str, sidecar_path: str | Path | None = None) -> dict[str, Any]:
    sidecar = load_sidecar(sidecar_path)
    expected_docs = [str(d.get("document_id")) for d in sidecar.get("docs") or [] if d.get("document_id")]
    doc_to_parent = {str(d.get("document_id")): str(d.get("parent_document_id") or parent_from_doc_id(str(d.get("document_id")))) for d in sidecar.get("docs") or [] if d.get("document_id")}
    parent_to_docs: dict[str, list[str]] = defaultdict(list)
    for did, parent in doc_to_parent.items():
        parent_to_docs[parent].append(did)

    data = load_db_collections(bank)
    docs = data.get("documents") or []
    memories = data.get("memories") or []
    operations = data.get("operations") or []

    units_by_doc = Counter(doc_id(m) for m in memories if doc_id(m))
    docs_in_db = {doc_id(d) for d in docs if doc_id(d)}
    expected_doc_set = set(expected_docs) or docs_in_db
    docs_without_units = sorted(did for did in expected_doc_set if units_by_doc.get(did, 0) == 0)
    docs_missing_in_db = sorted(did for did in expected_doc_set if did not in docs_in_db)

    sidecar_doc_by_id = {str(d.get("document_id")): d for d in sidecar.get("docs") or [] if d.get("document_id")}
    parent_attrs: dict[str, dict[str, Any]] = {}
    for did, d in sidecar_doc_by_id.items():
        parent = doc_to_parent.get(did) or parent_from_doc_id(did)
        cand = d.get("candidate") or {}
        parent_attrs[parent] = {
            "recommended_route": cand.get("recommended_route") or "<unknown>",
            "zero_unit_class": cand.get("zero_unit_class") or "<unknown>",
        }

    doc_attrs: dict[str, dict[str, Any]] = {}
    for did, d in sidecar_doc_by_id.items():
        cand = d.get("candidate") or {}
        variant = d.get("variant") or cand.get("variant")
        if not variant:
            if "::whole-session" in did:
                variant = "whole-session"
            elif "::first-useful-window" in did:
                variant = "first-useful-window"
            else:
                variant = "<unknown>"
        doc_attrs[did] = {
            "parent_document_id": doc_to_parent.get(did) or parent_from_doc_id(did),
            "variant": str(variant),
            "recommended_route": cand.get("recommended_route") or "<unknown>",
            "zero_unit_class": cand.get("zero_unit_class") or "<unknown>",
        }

    if doc_to_parent:
        parent_unit_counts: dict[str, int] = {}
        for parent, child_docs in parent_to_docs.items():
            parent_unit_counts[parent] = sum(units_by_doc.get(did, 0) for did in child_docs)
        parents_with_units = sorted(parent for parent, n in parent_unit_counts.items() if n > 0)
        parent_zero = sorted(parent for parent, n in parent_unit_counts.items() if n == 0)
    else:
        parent_unit_counts = {parent_from_doc_id(did): count for did, count in units_by_doc.items()}
        parents_with_units = sorted(parent for parent, n in parent_unit_counts.items() if n > 0)
        parent_zero = []

    def grouped_parent_stats(attr: str) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for parent in parent_unit_counts:
            grouped[str((parent_attrs.get(parent) or {}).get(attr) or "<unknown>")].append(parent)
        out: dict[str, dict[str, Any]] = {}
        for key, parents in sorted(grouped.items()):
            with_units = [p for p in parents if parent_unit_counts.get(p, 0) > 0]
            out[key] = {
                "parents": len(parents),
                "parents_with_units": len(with_units),
                "parent_coverage_ratio": round(len(with_units) / len(parents), 4) if parents else 0,
                "memory_units": sum(parent_unit_counts.get(p, 0) for p in parents),
                "parent_zero_count": len(parents) - len(with_units),
            }
        return out

    def grouped_doc_stats(attr: str) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for did in expected_doc_set:
            grouped[str((doc_attrs.get(did) or {}).get(attr) or "<unknown>")].append(did)
        out: dict[str, dict[str, Any]] = {}
        for key, doc_ids in sorted(grouped.items()):
            with_units = [did for did in doc_ids if units_by_doc.get(did, 0) > 0]
            parents = {str((doc_attrs.get(did) or {}).get("parent_document_id") or parent_from_doc_id(did)) for did in doc_ids}
            parents_with_units = {str((doc_attrs.get(did) or {}).get("parent_document_id") or parent_from_doc_id(did)) for did in with_units}
            out[key] = {
                "documents": len(doc_ids),
                "docs_with_units": len(with_units),
                "doc_coverage_ratio": round(len(with_units) / len(doc_ids), 4) if doc_ids else 0,
                "parents": len(parents),
                "parents_with_units": len(parents_with_units),
                "parent_coverage_ratio": round(len(parents_with_units) / len(parents), 4) if parents else 0,
                "memory_units": sum(units_by_doc.get(did, 0) for did in doc_ids),
            }
        return out

    artifact_counts: Counter[str] = Counter()
    artifact_samples: list[dict[str, Any]] = []
    for m in memories:
        text = str(m.get("text") or "")
        flags = artifact_flags(text, bank=bank)
        active = [k for k, v in flags.items() if v is True]
        for key in active:
            artifact_counts[key] += 1
        if active and len(artifact_samples) < 30:
            artifact_samples.append({
                "id": m.get("id"),
                "document_id": m.get("document_id"),
                "flags": {k: flags[k] for k in active},
                "malformed_when_values": flags.get("malformed_when_values"),
                "text": text[:360],
            })

    operations_by_status = Counter(str(op.get("status") or "<unknown>") for op in operations)
    operations_by_type_status = Counter((str(op.get("operation_type") or "<unknown>"), str(op.get("status") or "<unknown>")) for op in operations)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bank": bank,
        "sidecar": str(sidecar_path) if sidecar_path else None,
        "counts": {
            "expected_documents": len(expected_doc_set),
            "db_documents": len(docs),
            "memory_units": len(memories),
            "observations": sum(1 for m in memories if str(m.get("fact_type") or "") == "observation"),
            "docs_without_units": len(docs_without_units),
            "docs_missing_in_db": len(docs_missing_in_db),
            "parents": len(parent_to_docs) if parent_to_docs else None,
            "parents_with_units": len(parents_with_units) if parent_to_docs else None,
            "parent_zero_count": len(parent_zero) if parent_to_docs else None,
            "parent_coverage_ratio": round(len(parents_with_units) / len(parent_to_docs), 4) if parent_to_docs else None,
        },
        "per_doc_units": dict(sorted((did, units_by_doc.get(did, 0)) for did in expected_doc_set)),
        "parent_unit_counts": dict(sorted(parent_unit_counts.items())),
        "parent_stats_by_recommended_route": grouped_parent_stats("recommended_route") if parent_attrs else {},
        "parent_stats_by_zero_unit_class": grouped_parent_stats("zero_unit_class") if parent_attrs else {},
        "variant_stats": grouped_doc_stats("variant") if doc_attrs else {},
        "parent_zero": parent_zero,
        "docs_without_units": docs_without_units,
        "docs_missing_in_db": docs_missing_in_db,
        "artifact_counts": dict(artifact_counts.most_common()),
        "artifact_unit_count": sum(1 for m in memories if any(v is True for v in artifact_flags(str(m.get("text") or ""), bank=bank).values())),
        "artifact_samples": artifact_samples,
        "operations": {
            "by_status": dict(operations_by_status.most_common()),
            "by_type_status": [[typ, status, n] for (typ, status), n in sorted(operations_by_type_status.items())],
            "payload_null_completed_batch_retain": sum(1 for op in operations if str(op.get("operation_type")) == "batch_retain" and str(op.get("status")) == "completed" and op.get("payload_null")),
        },
    }
    return redact_secrets(result)


def write_md(result: dict[str, Any]) -> str:
    lines = ["# Hindsight fact quality gate", ""]
    for key in ["generated_at", "bank", "sidecar"]:
        lines.append(f"- {key}: `{result.get(key)}`")
    lines.append("")
    lines.append("## Counts")
    lines.append("```json")
    lines.append(json.dumps(result.get("counts", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Artifact counts")
    lines.append("```json")
    lines.append(json.dumps(result.get("artifact_counts", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Parent stats by original route")
    lines.append("```json")
    lines.append(json.dumps(result.get("parent_stats_by_recommended_route", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Parent stats by original zero-unit class")
    lines.append("```json")
    lines.append(json.dumps(result.get("parent_stats_by_zero_unit_class", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Variant stats")
    lines.append("```json")
    lines.append(json.dumps(result.get("variant_stats", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Operations")
    lines.append("```json")
    lines.append(json.dumps(result.get("operations", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    parent_zero = result.get("parent_zero") or []
    if parent_zero:
        lines.append("## Parent zero-unit docs")
        for parent in parent_zero[:50]:
            lines.append(f"- `{parent}`")
        lines.append("")
    samples = result.get("artifact_samples") or []
    if samples:
        lines.append("## Artifact samples (redacted)")
        for sample in samples[:10]:
            lines.append(f"- `{sample.get('document_id')}` flags={sample.get('flags')} — {sample.get('text')}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only fact-quality gate for Hindsight temp banks")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--sidecar", type=Path)
    ap.add_argument("--output-json", required=True, type=Path)
    ap.add_argument("--output-md", required=True, type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    result = run_gate(bank=args.bank, sidecar_path=args.sidecar)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.write_text(write_md(result), encoding="utf-8")
    summary = {
        "bank": args.bank,
        "json": str(args.output_json),
        "md": str(args.output_md),
        "counts": result.get("counts"),
        "artifact_counts": result.get("artifact_counts"),
        "operations": result.get("operations"),
    }
    print(json.dumps(summary if args.json else summary, ensure_ascii=False, indent=2 if args.json else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
