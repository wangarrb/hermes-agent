#!/usr/bin/env python3
"""Lightweight selector for Hindsight session retain manifests.

Reads a lean JSONL manifest produced by hindsight_session_manifest.py and writes a
small curated production-only manifest for smoke/e2e retain runs. It is
non-mutating: no Hindsight calls, no content rehydration, no provider switch.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_OUTPUT_DIR = Path.home() / ".hermes" / "hindsight" / "session_ingest" / "manifests"
PART_SUFFIX_RE = re.compile(r"::part-\d+$")

TAG_WEIGHTS = {
    "domain:hindsight": 60,
    "topic:memory-management": 45,
    "topic:native-consolidation": 45,
    "topic:recall-cache": 35,
    "project:egomotion4d": 55,
    "domain:autodrive": 45,
    "project:vggt-long": 45,
    "project:openclaw": 35,
    "domain:paper": 25,
    "domain:patent": 25,
}


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
    return records


def session_root(document_id: str) -> str:
    return PART_SUFFIX_RE.sub("", document_id or "")


def record_score(record: dict[str, Any]) -> tuple[int, int, str]:
    tags = record.get("tags") or []
    tag_score = sum(TAG_WEIGHTS.get(str(tag), 10 if str(tag).startswith(("domain:", "project:", "topic:")) else 0) for tag in tags)
    content_chars = int(record.get("content_chars") or 0)
    # Cap content contribution so very long broad sessions do not dominate.
    length_score = min(content_chars, 20_000) // 100
    # Lower estimated chunks slightly; small smoke batches should avoid extreme requests.
    chunk_penalty = max(0, int(record.get("estimated_retain_chunks") or 1) - 1) * 5
    score = tag_score + length_score - chunk_penalty
    return (score, content_chars, str(record.get("document_id") or ""))


def select_records(records: Iterable[dict[str, Any]], *, limit: int, one_per_session_root: bool = True) -> list[dict[str, Any]]:
    production = [r for r in records if r.get("action") == "production"]
    ranked = sorted(production, key=record_score, reverse=True)
    selected: list[dict[str, Any]] = []
    seen_roots: set[str] = set()
    for record in ranked:
        root = session_root(str(record.get("document_id") or ""))
        if one_per_session_root and root in seen_roots:
            continue
        selected.append(record)
        seen_roots.add(root)
        if len(selected) >= limit:
            break
    return selected


def summarize(records: list[dict[str, Any]], *, source_manifest: str | None = None) -> dict[str, Any]:
    by_tag: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    by_candidate_filter_version: Counter[str] = Counter()
    total_chars = 0
    total_chunks = 0
    for record in records:
        total_chars += int(record.get("content_chars") or 0)
        total_chunks += int(record.get("estimated_retain_chunks") or 0)
        reason = record.get("reason")
        if reason:
            by_reason[str(reason)] += 1
        for tag in record.get("tags") or []:
            by_tag[str(tag)] += 1
        version = (record.get("metadata") or {}).get("candidate_filter_version")
        if version:
            by_candidate_filter_version[str(version)] += 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": source_manifest,
        "selected_records": len(records),
        "total_content_chars": total_chars,
        "estimated_retain_chunks": total_chunks,
        "by_tag": dict(by_tag.most_common()),
        "by_reason": dict(by_reason.most_common()),
        "by_candidate_filter_version": dict(by_candidate_filter_version.most_common()),
        "document_ids": [r.get("document_id") for r in records],
    }


def write_curated_manifest(records: list[dict[str, Any]], output_dir: str | Path = DEFAULT_OUTPUT_DIR, *, stem: str | None = None, source_manifest: str | None = None) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if stem is None:
        stem = datetime.now().strftime("%Y%m%d-%H%M%S-curated-session-manifest")
    manifest_path = output_dir / f"{stem}.jsonl"
    summary_path = output_dir / f"{stem}-summary.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = summarize(records, source_manifest=source_manifest)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), "summary": str(summary_path)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest from hindsight_session_manifest.py")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--allow-multiple-parts", action="store_true", help="Do not dedupe split parts by session root")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    records = load_manifest(args.manifest)
    selected = select_records(records, limit=args.limit, one_per_session_root=not args.allow_multiple_parts)
    paths = write_curated_manifest(selected, args.output_dir, stem=args.stem, source_manifest=str(args.manifest))
    result = {
        "input_records": len(records),
        "selected_records": len(selected),
        "paths": paths,
        "summary": summarize(selected, source_manifest=str(args.manifest)),
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"selected_records={len(selected)}")
        print(f"manifest={paths['manifest']}")
        print(f"summary={paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
