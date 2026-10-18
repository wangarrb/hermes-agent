#!/usr/bin/env python3
"""Build first-useful-window retry manifests for Hindsight zero-unit sessions.

This is a local/temp-bank helper. It intentionally keeps Hindsight-visible
metadata empty by default; provenance goes to a sidecar JSON joined by
document_id so metadata does not pollute fact extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import hindsight_session_retain_runner as retain_runner  # noqa: E402

ROLE_RE = re.compile(r"(?m)^(User|Assistant):")
CONTEXT_COMPACTION_RE = re.compile(r"\[CONTEXT COMPACTION\b|Active Task|END OF CONTEXT SUMMARY", re.I)
THINK_OR_REASONING_RE = re.compile(r"<\s*/?\s*(?:think|thinking)\b|reasoning_content|codex_reasoning_items|chain[-_ ]of[-_ ]thought", re.I)
SHORT_LOW_SIGNAL_RE = re.compile(r"^(User:\s*)?(继续|ok|好|hi|hello|在吗|刚聊到哪[儿了]*|刚才聊了什么)\s*[。.!！?？]*\s*$", re.I)
SECRET_LIKE_RE = re.compile(r"(?i)(api[_-]?key|secret|password|token|sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16})")

DEFAULT_RETAIN_CHUNK_SIZE = 8000
DEFAULT_MAX_WINDOW_CHARS = 4000


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_route_quota(values: list[str] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"invalid --route-quota {item!r}; expected route=N")
        route, raw_n = item.split("=", 1)
        out[route.strip()] = int(raw_n)
    return out


def parent_id_from_window_doc_id(doc_id: str) -> str:
    # Previous window manifests use parent::clean-v2-window...::variant.
    marker = "::clean-v2-window"
    if marker in doc_id:
        return doc_id.split(marker, 1)[0]
    marker = "::first-useful-window"
    if marker in doc_id:
        return doc_id.split(marker, 1)[0]
    return doc_id


def load_excluded_parent_ids(paths: list[str] | None) -> set[str]:
    out: set[str] = set()
    for path in paths or []:
        for rec in load_jsonl(path):
            doc_id = str(rec.get("document_id") or "")
            meta = rec.get("metadata") or {}
            parent = meta.get("parent_document_id") or parent_id_from_window_doc_id(doc_id)
            if parent:
                out.add(str(parent))
    return out


def select_candidates(hardening_json: str | Path, *, limit: int, route_quota: dict[str, int] | None = None, exclude_parents: set[str] | None = None) -> list[dict[str, Any]]:
    data = json.loads(Path(hardening_json).read_text(encoding="utf-8"))
    candidates = list(((data.get("zero_unit_report") or {}).get("high_value_retry_candidates") or []))
    exclude_parents = exclude_parents or set()
    candidates = [c for c in candidates if str(c.get("document_id") or "") not in exclude_parents]
    if route_quota:
        # Return the whole candidate pool for quota-controlled routes. build_records
        # enforces accepted quotas after secret/reasoning skips, so skipped records
        # are backfilled by later candidates instead of silently shrinking sample size.
        allowed = set(route_quota)
        return [c for c in candidates if str(c.get("recommended_route") or "") in allowed]
    return candidates[:limit]


def split_role_blocks(text: str) -> list[dict[str, Any]]:
    matches = list(ROLE_RE.finditer(text))
    if not matches:
        clean = text.strip()
        return [{"role": "Text", "start": 0, "end": len(text), "text": clean}] if clean else []
    blocks: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block_text = text[start:end].strip()
        if block_text:
            blocks.append({"role": m.group(1), "start": start, "end": end, "text": block_text})
    return blocks


def useful_start_index(blocks: list[dict[str, Any]]) -> int:
    for i, block in enumerate(blocks):
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        if CONTEXT_COMPACTION_RE.search(text):
            continue
        # If the session starts with a short continuation, keep it as context by
        # starting at zero once a useful block appears immediately after it.
        if SHORT_LOW_SIGNAL_RE.match(text):
            continue
        return 0 if i <= 2 else i
    return 0


def first_useful_window(text: str, *, max_chars: int = DEFAULT_MAX_WINDOW_CHARS) -> tuple[str, dict[str, Any]]:
    blocks = split_role_blocks(text)
    if not blocks:
        return "", {"block_count": 0, "blocks": [0, -1], "truncated": False}
    start_idx = useful_start_index(blocks)
    selected: list[str] = []
    end_idx = start_idx - 1
    truncated = False
    for idx in range(start_idx, len(blocks)):
        candidate = str(blocks[idx]["text"])
        proposed = ("\n\n".join(selected + [candidate])).strip()
        if selected and len(proposed) > max_chars:
            break
        if not selected and len(candidate) > max_chars:
            cut = candidate[:max_chars]
            # Prefer cutting at a line boundary but keep enough content.
            line_cut = cut.rfind("\n")
            if line_cut >= int(max_chars * 0.65):
                cut = cut[:line_cut]
            selected.append(cut.strip())
            end_idx = idx
            truncated = True
            break
        selected.append(candidate)
        end_idx = idx
        if len("\n\n".join(selected)) >= max_chars:
            break
    window = "\n\n".join(selected).strip()
    return window, {
        "block_count": len(blocks),
        "blocks": [start_idx, end_idx],
        "selected_block_count": max(0, end_idx - start_idx + 1),
        "truncated": truncated,
    }


def build_records(
    *,
    source_manifest: str | Path,
    hardening_json: str | Path,
    limit: int,
    route_quota: dict[str, int] | None,
    exclude_parents: set[str],
    max_window_chars: int,
    retain_chunk_size: int,
    document_suffix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_records = {str(r.get("document_id")): r for r in load_jsonl(source_manifest)}
    candidates = select_candidates(hardening_json, limit=limit, route_quota=route_quota, exclude_parents=exclude_parents)
    out_records: list[dict[str, Any]] = []
    sidecar_docs: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    by_route: Counter[str] = Counter()
    by_class: Counter[str] = Counter()
    total_chars = 0
    accepted_by_route: Counter[str] = Counter()
    for cand in candidates:
        if len(out_records) >= limit:
            break
        route = str(cand.get("recommended_route") or "<unknown>")
        if route_quota and accepted_by_route[route] >= route_quota.get(route, 0):
            continue
        parent_id = str(cand.get("document_id") or "")
        src = source_records.get(parent_id)
        if not src:
            skipped["missing_source_manifest_record"] += 1
            continue
        rec = retain_runner.rehydrate_record(src)
        content = str(rec.get("content") or "")
        if not content.strip():
            skipped["empty_rehydrated_content"] += 1
            continue
        if SECRET_LIKE_RE.search(content):
            skipped["secret_like_content"] += 1
            continue
        window, win_meta = first_useful_window(content, max_chars=max_window_chars)
        if not window.strip():
            skipped["empty_window"] += 1
            continue
        if THINK_OR_REASONING_RE.search(window):
            skipped["reasoning_like_window"] += 1
            continue
        doc_id = f"{parent_id}::{document_suffix}::first-useful-window"
        tags = list(rec.get("tags") or src.get("tags") or cand.get("tags") or [])
        obs = list(rec.get("observation_scopes") or src.get("observation_scopes") or [])
        out_records.append({
            "schema_version": "hindsight-first-useful-window-v1",
            "document_id": doc_id,
            "action": "production",
            "reason": "first_useful_window_expansion_sample",
            "context": "hermes_session_window",
            "event_date": rec.get("event_date") or src.get("event_date") or (src.get("metadata") or {}).get("started_at") or cand.get("event_date"),
            "content": window,
            "content_chars": len(window),
            "estimated_retain_chunks": max(1, math.ceil(len(window) / max(1, retain_chunk_size))),
            "tags": tags,
            "observation_scopes": obs,
            "metadata": {},
            "update_mode": "replace",
        })
        sidecar_docs.append({
            "document_id": doc_id,
            "parent_document_id": parent_id,
            "parent_tags": tags,
            "source_manifest_record": parent_id,
            "source_json_path": (src.get("metadata") or {}).get("json_path"),
            "source_event_date": rec.get("event_date") or src.get("event_date") or (src.get("metadata") or {}).get("started_at") or cand.get("event_date"),
            "source_content_chars": len(content),
            "window_content_chars": len(window),
            "window_sha256": sha256_text(window),
            "window": win_meta,
            "candidate": {
                "recommended_route": cand.get("recommended_route"),
                "zero_unit_class": cand.get("zero_unit_class"),
                "semantic_score": cand.get("semantic_score"),
                "noise_ratio": cand.get("noise_ratio"),
                "chars": cand.get("chars"),
                "primary_value_classes": cand.get("primary_value_classes"),
                "model": cand.get("model"),
            },
        })
        by_route[route] += 1
        by_class[str(cand.get("zero_unit_class") or "<unknown>")] += 1
        accepted_by_route[route] += 1
        total_chars += len(window)
    sidecar = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_manifest),
        "hardening_json": str(hardening_json),
        "limit": limit,
        "route_quota": route_quota or {},
        "exclude_parent_count": len(exclude_parents),
        "max_window_chars": max_window_chars,
        "retain_chunk_size": retain_chunk_size,
        "document_suffix": document_suffix,
        "record_count": len(out_records),
        "parent_count": len({d["parent_document_id"] for d in sidecar_docs}),
        "total_content_chars": total_chars,
        "estimated_retain_chunks": sum(int(r.get("estimated_retain_chunks") or 0) for r in out_records),
        "by_recommended_route": dict(by_route.most_common()),
        "by_zero_unit_class": dict(by_class.most_common()),
        "skipped": dict(skipped.most_common()),
        "docs": sidecar_docs,
    }
    return out_records, sidecar


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_md(sidecar: dict[str, Any]) -> str:
    lines = ["# Hindsight first-useful-window expansion manifest", ""]
    for key in ["generated_at", "source_manifest", "hardening_json", "record_count", "parent_count", "total_content_chars", "estimated_retain_chunks", "max_window_chars", "exclude_parent_count"]:
        lines.append(f"- {key}: `{sidecar.get(key)}`")
    lines.append("")
    lines.append("## By recommended route")
    lines.append("```json")
    lines.append(json.dumps(sidecar.get("by_recommended_route", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## By zero-unit class")
    lines.append("```json")
    lines.append(json.dumps(sidecar.get("by_zero_unit_class", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Hygiene")
    lines.append("- Hindsight-visible manifest metadata is intentionally `{}` for every record.")
    lines.append("- Provenance is stored in sidecar JSON by `document_id`, not in the LLM-visible metadata section.")
    lines.append("- Input is natural `User:` / `Assistant:` conversation slices; no custom mission preamble is added.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate first-useful-window zero-unit retry manifest")
    ap.add_argument("--source-manifest", required=True, type=Path)
    ap.add_argument("--hardening-json", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--sidecar", required=True, type=Path)
    ap.add_argument("--summary-md", type=Path)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--route-quota", action="append", help="Route quota like production_windowed=12; may repeat")
    ap.add_argument("--exclude-manifest", action="append", help="Existing window manifest whose parent ids should be excluded")
    ap.add_argument("--max-window-chars", type=int, default=DEFAULT_MAX_WINDOW_CHARS)
    ap.add_argument("--retain-chunk-size", type=int, default=DEFAULT_RETAIN_CHUNK_SIZE)
    ap.add_argument("--document-suffix", default="clean-v2-first-useful-window-expand-v1")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    route_quota = parse_route_quota(args.route_quota)
    exclude_parents = load_excluded_parent_ids(args.exclude_manifest)
    records, sidecar = build_records(
        source_manifest=args.source_manifest,
        hardening_json=args.hardening_json,
        limit=args.limit,
        route_quota=route_quota,
        exclude_parents=exclude_parents,
        max_window_chars=args.max_window_chars,
        retain_chunk_size=args.retain_chunk_size,
        document_suffix=args.document_suffix,
    )
    write_jsonl(args.output, records)
    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    args.sidecar.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.summary_md:
        args.summary_md.write_text(write_summary_md(sidecar), encoding="utf-8")
    result = {
        "manifest": str(args.output),
        "sidecar": str(args.sidecar),
        "summary_md": str(args.summary_md) if args.summary_md else None,
        "record_count": len(records),
        "parent_count": sidecar.get("parent_count"),
        "estimated_retain_chunks": sidecar.get("estimated_retain_chunks"),
        "by_recommended_route": sidecar.get("by_recommended_route"),
        "by_zero_unit_class": sidecar.get("by_zero_unit_class"),
        "skipped": sidecar.get("skipped"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
