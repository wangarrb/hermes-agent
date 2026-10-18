#!/usr/bin/env python3
"""Select a representative, content-free sample from a Hindsight review backlog.

Read-only: no Hindsight API, no LLM call, no production mutation.  The output is a
sidecar for a later value-scorer dry run.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_BUCKET_QUOTAS = {
    "manual_review": 5,
    "zero_with_retry_evidence": 12,
    "zero_without_retry_evidence": 12,
    "quality_review": 8,
    "monitor_has_units": 5,
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def bucket_for(row: dict[str, Any]) -> str:
    route = ((row.get("review") or {}).get("recommended_route")) or ""
    status = ((row.get("current_retain_outcome") or {}).get("status")) or ""
    if route == "manual_review":
        return "manual_review"
    if route == "cluster_revisit" and row.get("retry_evidence"):
        return "zero_with_retry_evidence"
    if route == "cluster_revisit":
        return "zero_without_retry_evidence"
    if route == "quality_review":
        return "quality_review"
    if route == "monitor" and status == "has_units":
        return "monitor_has_units"
    return "other"


def priority_key(row: dict[str, Any]) -> tuple:
    outcome = row.get("current_retain_outcome") or {}
    hardening = row.get("hardening_overlay") or {}
    anomalies = row.get("deterministic_anomalies") or []
    return (
        -int(bool(row.get("retry_evidence"))),
        -int(outcome.get("memory_unit_count") or 0),
        -int(hardening.get("semantic_score") or 0),
        -len(row.get("value_class_guess") or []),
        -len(anomalies),
        -int(row.get("content_chars") or 0),
        str(row.get("document_id") or ""),
    )


def round_robin_topic(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_topic: dict[str, deque] = defaultdict(deque)
    for row in sorted(rows, key=priority_key):
        by_topic[str(row.get("topic_key") or "<unknown>")].append(row)
    topics = deque(sorted(by_topic.keys(), key=lambda k: (-len(by_topic[k]), k)))
    out: list[dict[str, Any]] = []
    while topics and len(out) < limit:
        topic = topics.popleft()
        q = by_topic[topic]
        if q:
            out.append(q.popleft())
        if q:
            topics.append(topic)
    return out


def parse_quotas(raw: str | None) -> dict[str, int]:
    quotas = dict(DEFAULT_BUCKET_QUOTAS)
    if not raw:
        return quotas
    for part in raw.split(","):
        if not part.strip():
            continue
        key, _, value = part.partition("=")
        if key and value:
            quotas[key.strip()] = int(value.strip())
    return quotas


def select_sample(rows: list[dict[str, Any]], *, size: int, quotas: dict[str, int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[bucket_for(row)].append(row)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    bucket_counts: Counter[str] = Counter()

    for bucket, quota in quotas.items():
        if len(selected) >= size:
            break
        chosen = round_robin_topic(buckets.get(bucket, []), max(0, min(quota, size - len(selected))))
        for row in chosen:
            doc_id = str(row.get("document_id"))
            if doc_id in selected_ids:
                continue
            selected_ids.add(doc_id)
            selected.append(row)
            bucket_counts[bucket] += 1

    if len(selected) < size:
        leftovers = [r for r in rows if str(r.get("document_id")) not in selected_ids]
        for row in round_robin_topic(leftovers, size - len(selected)):
            doc_id = str(row.get("document_id"))
            if doc_id in selected_ids:
                continue
            selected_ids.add(doc_id)
            selected.append(row)
            bucket_counts[bucket_for(row)] += 1

    generated_at = datetime.now(timezone.utc).isoformat()
    output_rows = []
    for idx, row in enumerate(selected, 1):
        bucket = bucket_for(row)
        out = dict(row)
        out["sample"] = {
            "schema_version": "hindsight-review-backlog-scorer-sample-v1",
            "generated_at": generated_at,
            "index": idx,
            "bucket": bucket,
            "reason": "representative_backlog_value_scorer_dry_run",
            "llm_call_allowed": False,
            "hindsight_submit_allowed": False,
        }
        output_rows.append(out)

    summary = {
        "schema_version": "hindsight-review-backlog-scorer-sample-v1",
        "generated_at": generated_at,
        "input_records": len(rows),
        "sample_size": len(output_rows),
        "requested_size": size,
        "bucket_quotas": quotas,
        "by_bucket": dict(bucket_counts.most_common()),
        "by_review_route": dict(Counter((r.get("review") or {}).get("recommended_route") for r in output_rows).most_common()),
        "by_retain_status": dict(Counter((r.get("current_retain_outcome") or {}).get("status") for r in output_rows).most_common()),
        "by_topic_key": dict(Counter(r.get("topic_key") for r in output_rows).most_common()),
        "with_retry_evidence": sum(1 for r in output_rows if r.get("retry_evidence")),
        "with_event_date": sum(1 for r in output_rows if r.get("event_date")),
        "contains_content_fields": sum(1 for r in output_rows if "content" in r or "content_preview" in r),
    }
    return output_rows, summary


def write_summary_md(path: str | Path, summary: dict[str, Any]) -> None:
    lines = ["# Hindsight review backlog scorer sample", ""]
    for key in ["generated_at", "input_records", "sample_size", "requested_size", "with_retry_evidence", "with_event_date", "contains_content_fields"]:
        lines.append(f"- {key}: `{summary.get(key)}`")
    for title, key in [
        ("By bucket", "by_bucket"),
        ("By review route", "by_review_route"),
        ("By retain status", "by_retain_status"),
        ("By topic key", "by_topic_key"),
    ]:
        lines.extend(["", f"## {title}", "```json", json.dumps(summary.get(key) or {}, ensure_ascii=False, indent=2, sort_keys=True), "```"])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backlog", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary-json")
    ap.add_argument("--summary-md")
    ap.add_argument("--size", type=int, default=40)
    ap.add_argument("--bucket-quotas", help="Comma list like manual_review=5,zero_with_retry_evidence=12")
    ap.add_argument("--json", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.backlog)
    sample, summary = select_sample(rows, size=args.size, quotas=parse_quotas(args.bucket_quotas))
    summary["backlog"] = args.backlog
    write_jsonl(args.output, sample)
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary_md:
        write_summary_md(args.summary_md, summary)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"wrote {len(sample)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
