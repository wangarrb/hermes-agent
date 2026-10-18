#!/usr/bin/env python3
"""Export approved repair-zone observations to a local sidecar.

Read-only with respect to Hindsight. It reads a temp/repair bank, keeps only clean
observation units with source lineage, and writes local sidecar files that layered
recall can append without polluting the production bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hindsight_bank_quality_audit import load_db_collections, redact_secrets  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path.home() / ".hermes" / "hindsight" / "review_repair" / "approved"
SCHEMA_VERSION = "hindsight-repair-sidecar-v1"

ARTIFACT_PATTERNS: dict[str, re.Pattern[str]] = {
    "reasoning_like": re.compile(r"<\s*/?\s*(?:think|thinking)\b|reasoning_content|chain[-_ ]of[-_ ]thought|\breasoning\b", re.I),
    "context_compaction": re.compile(r"\[CONTEXT COMPACTION\b|END OF CONTEXT SUMMARY|Active Task", re.I),
    "prompt_or_metadata_leak": re.compile(r"Metadata:|Narrator:|custom mission|retain-ready|Retain durable facts|schema_version|Hindsight user_message", re.I),
    "temp_bank_or_window_leak": re.compile(r"hermes_tmp|weektrial|first[-_ ]useful[-_ ]window|clean[-_ ]v2", re.I),
    "secret_like": re.compile(r"(?i)(\bsk-[A-Za-z0-9_.-]{12,}\b|(?:api[_ -]?key|secret|token|password)\s*[:=]\s*[^\s,;|]+)"),
}

TYPE_FROM_VALUE_TAG = {
    "value:user_preference": "user_preference",
    "value:durable_decision": "project_decision",
    "value:project_state": "project_state",
    "value:experiment_result": "experiment_result",
    "value:tool_lesson": "tooling_lesson",
    "value:error_root_cause": "technical_lesson",
    "value:environment_fact": "tooling_lesson",
    "value:open_question": "open_question",
}


def load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(redact_secrets(rec), ensure_ascii=False, sort_keys=True) + "\n")


def memory_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or "")


def memory_text(item: dict[str, Any]) -> str:
    return str(item.get("text") or item.get("content") or "").strip()


def memory_type(item: dict[str, Any]) -> str:
    return str(item.get("fact_type") or item.get("type") or "")


def tags_of(item: dict[str, Any]) -> list[str]:
    tags = item.get("tags") or []
    return [str(t) for t in tags if t]


def source_ids_of(item: dict[str, Any]) -> list[str]:
    ids = item.get("source_memory_ids") or item.get("source_fact_ids") or []
    if isinstance(ids, str):
        ids = [ids]
    return [str(x) for x in ids if x]


def stable_short_id(*parts: str) -> str:
    raw = "\n".join(str(p) for p in parts if p)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def artifact_flags(text: str, *, bank: str = "") -> dict[str, bool]:
    flags = {name: bool(pattern.search(text or "")) for name, pattern in ARTIFACT_PATTERNS.items()}
    if bank and bank in (text or ""):
        flags["temp_bank_or_window_leak"] = True
    return flags


def infer_topic(tags: list[str], sidecar_doc: dict[str, Any] | None = None) -> str:
    candidate = (sidecar_doc or {}).get("candidate") or {}
    topic_group = str(candidate.get("topic_group") or "").strip()
    if topic_group:
        if "hindsight" in topic_group or "hermes" in topic_group:
            return "hindsight"
        if "egomotion" in topic_group or "autodrive" in topic_group:
            return "egomotion4d"
    for prefix in ["project:", "domain:", "topic:"]:
        for tag in tags:
            if tag.startswith(prefix):
                return tag.split(":", 1)[1]
    return "repair"


def infer_type(tags: list[str]) -> str:
    for tag in tags:
        if tag in TYPE_FROM_VALUE_TAG:
            return TYPE_FROM_VALUE_TAG[tag]
    if any(t.startswith("value:") for t in tags):
        return str(next(t for t in tags if t.startswith("value:"))).split(":", 1)[1]
    return "technical_lesson"


def sidecar_doc_index(sidecar: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for doc in sidecar.get("docs") or []:
        if not isinstance(doc, dict):
            continue
        for key in ["document_id", "parent_document_id"]:
            val = doc.get(key)
            if val:
                out[str(val)] = doc
    return out


def reject(record_id: str, reason: str, item: dict[str, Any], *, flags: dict[str, bool] | None = None) -> dict[str, Any]:
    return {
        "id": record_id,
        "reason": reason,
        "document_id": item.get("document_id"),
        "tags": tags_of(item),
        "flags": {k: v for k, v in (flags or {}).items() if v},
        "text_preview": memory_text(item)[:300],
    }


def build_sidecar_records(*, memories: list[dict[str, Any]], sidecar: dict[str, Any], bank: str) -> dict[str, Any]:
    by_id = {memory_id(m): m for m in memories if memory_id(m)}
    doc_index = sidecar_doc_index(sidecar)
    approved: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for item in memories:
        if memory_type(item) != "observation":
            continue
        rid = memory_id(item)
        text = memory_text(item)
        if not rid or not text:
            rejected.append(reject(rid or "<missing-id>", "empty_observation", item))
            continue
        flags = artifact_flags(text, bank=bank)
        active_flags = [name for name, active in flags.items() if active]
        if active_flags:
            rejected.append(reject(rid, "artifact_flags:" + ",".join(sorted(active_flags)), item, flags=flags))
            continue
        source_ids = source_ids_of(item)
        if not source_ids:
            rejected.append(reject(rid, "missing_source_memory_ids", item))
            continue
        missing = [sid for sid in source_ids if sid not in by_id]
        if missing:
            rejected.append(reject(rid, "missing_source_records:" + ",".join(missing[:5]), item))
            continue
        source_docs = sorted({str((by_id[sid].get("document_id") or "")) for sid in source_ids if by_id[sid].get("document_id")})
        if not source_docs:
            rejected.append(reject(rid, "missing_source_documents", item))
            continue
        source_doc = source_docs[0]
        sc_doc = doc_index.get(source_doc)
        tags = tags_of(item)
        safe_id = stable_short_id(bank, rid, text)
        rec = {
            "schema_version": SCHEMA_VERSION,
            "id": f"repair-sidecar::{safe_id}",
            "layer": "approved_repair_sidecar",
            "status": "approved",
            "topic": infer_topic(tags, sc_doc),
            "type": infer_type(tags),
            "insight": text,
            "text": text,
            "tags": tags,
            "evidence_ids": source_ids,
            "source_fact_ids": source_ids,
            "source_documents": source_docs,
            "source_observation_id": rid,
            "provenance": {
                "source_bank_hash": stable_short_id(bank),
                "document_id": source_doc,
                "candidate_id": ((sc_doc or {}).get("candidate") or {}).get("candidate_id"),
                "parent_document_id": (sc_doc or {}).get("parent_document_id"),
                "variant": (sc_doc or {}).get("variant"),
            },
        }
        approved.append(rec)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bank": bank,
        "counts": {
            "approved": len(approved),
            "rejected": len(rejected),
            "observations_seen": sum(1 for m in memories if memory_type(m) == "observation"),
        },
        "rejection_reasons": dict(Counter(r["reason"].split(":", 1)[0] for r in rejected)),
        "approved": approved,
        "rejected": rejected,
    }


def load_memories_from_audit_or_bank(*, bank: str, audit_json: str | Path | None = None) -> list[dict[str, Any]]:
    if audit_json:
        data = load_json(audit_json)
        audit = data.get("audit") or {}
        # Full audit JSON intentionally does not include all memory rows. Fall back
        # to DB unless a future caller provides them explicitly.
        if isinstance(audit.get("memories"), list):
            return audit["memories"]
    return load_db_collections(bank).get("memories") or []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export clean temp-bank repair observations to a local approved sidecar; no Hindsight writes.")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--sidecar", required=True, type=Path)
    ap.add_argument("--audit-json", type=Path, help="Optional audit JSON; DB fallback is used for full memory rows")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--stem", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    memories = load_memories_from_audit_or_bank(bank=args.bank, audit_json=args.audit_json)
    sidecar = load_json(args.sidecar)
    result = build_sidecar_records(memories=memories, sidecar=sidecar, bank=args.bank)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = args.stem or f"repair-sidecar-{args.bank}-{stamp}"
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    approved_path = root / f"{stem}-observations_index.jsonl"
    rejected_path = root / f"{stem}-rejected.jsonl"
    summary_path = root / f"{stem}-summary.json"
    latest_path = root / "latest.json"

    write_jsonl(approved_path, result["approved"])
    write_jsonl(rejected_path, result["rejected"])
    summary = {k: v for k, v in result.items() if k not in {"approved", "rejected"}}
    summary.update({
        "approved_path": str(approved_path),
        "rejected_path": str(rejected_path),
        "summary_path": str(summary_path),
    })
    summary_path.write_text(json.dumps(redact_secrets(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest_path.write_text(json.dumps(redact_secrets(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(redact_secrets(summary), ensure_ascii=False, indent=2 if args.json else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
