#!/usr/bin/env python3
"""Compact / archive a Hindsight review backlog.

Primary policy is time-based: keep a hot recoverable backlog for the recent
retention window (default 3 months) and move older items to cold archive JSONL.
An optional count cap remains available as a safety valve, but it is not the
default.  The script is read-only with respect to Hindsight.
"""
from __future__ import annotations

import argparse
import calendar
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

DEFAULT_QUOTAS = {
    "manual_review": 5,
    "zero_with_retry_evidence": 24,
    "zero_without_retry_evidence": 36,
    "quality_review": 25,
    "monitor_has_units": 30,
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
    return (
        -int(bool(row.get("retry_evidence"))),
        -int(outcome.get("memory_unit_count") or 0),
        -int(hardening.get("semantic_score") or 0),
        -len(row.get("value_class_guess") or []),
        -len(row.get("deterministic_anomalies") or []),
        -int(row.get("content_chars") or 0),
        str(row.get("document_id") or ""),
    )


def topic_key(row: dict[str, Any]) -> str:
    return str(row.get("topic_key") or "<unknown>")


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        try:
            dt = datetime.fromisoformat(text[:10])
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def row_event_dt(row: dict[str, Any]) -> datetime | None:
    for value in [
        row.get("event_date"),
        ((row.get("source") or {}).get("event_date")),
        ((row.get("source") or {}).get("started_at")),
        ((row.get("source") or {}).get("session_start")),
    ]:
        dt = parse_dt(value)
        if dt:
            return dt
    return None


def subtract_months(dt: datetime, months: int) -> datetime:
    if months <= 0:
        return dt
    month_index = dt.month - months
    year = dt.year + (month_index - 1) // 12
    month = (month_index - 1) % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def cutoff_datetime(*, reference_date: str | datetime | None, retention_months: int, retention_days: int | None) -> datetime | None:
    if retention_days is not None and retention_days <= 0:
        return None
    if retention_days is None and retention_months <= 0:
        return None
    if isinstance(reference_date, datetime):
        ref = reference_date
    elif reference_date:
        ref = parse_dt(reference_date)
        if ref is None:
            raise ValueError(f"invalid reference_date: {reference_date}")
    else:
        ref = datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    ref = ref.astimezone(timezone.utc)
    if retention_days is not None:
        return ref - timedelta(days=retention_days)
    return subtract_months(ref, retention_months)


def round_robin_select(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    by_topic: dict[str, deque] = defaultdict(deque)
    for row in sorted(rows, key=priority_key):
        by_topic[topic_key(row)].append(row)
    topic_queue = deque(sorted(by_topic.keys(), key=lambda k: (-len(by_topic[k]), k)))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    while topic_queue and len(out) < limit:
        topic = topic_queue.popleft()
        q = by_topic[topic]
        while q and str(q[0].get("document_id")) in seen:
            q.popleft()
        if q:
            row = q.popleft()
            doc_id = str(row.get("document_id"))
            if doc_id not in seen:
                out.append(row)
                seen.add(doc_id)
        if q:
            topic_queue.append(topic)
    return out


def parse_quotas(raw: str | None) -> dict[str, int]:
    quotas = dict(DEFAULT_QUOTAS)
    if not raw:
        return quotas
    for part in raw.split(","):
        if not part.strip():
            continue
        key, _, value = part.partition("=")
        if key and value:
            quotas[key.strip()] = int(value.strip())
    return quotas


def parse_buckets(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def with_cleanup(row: dict[str, Any], *, state: str, reason: str, generated_at: str, cutoff: datetime | None) -> dict[str, Any]:
    out = dict(row)
    out["cleanup"] = {
        "schema_version": "hindsight-review-backlog-cleanup-v2",
        "state": state,
        "reason": reason,
        "generated_at": generated_at,
        "cutoff_event_date": cutoff.isoformat() if cutoff else None,
    }
    return out


def compact_backlog(
    rows: list[dict[str, Any]],
    quotas: dict[str, int],
    max_records: int = 0,
    *,
    retention_months: int = 3,
    retention_days: int | None = None,
    reference_date: str | datetime | None = None,
    pin_buckets: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return (active, archive, summary).

    `retention_months=3` is the default hot-window policy.  `max_records=0`
    disables the old quantity cap; set it only as an additional safety valve.
    """
    generated_at = datetime.now(timezone.utc).isoformat()
    pin_buckets = pin_buckets or set()
    cutoff = cutoff_datetime(reference_date=reference_date, retention_months=retention_months, retention_days=retention_days)

    # Deduplicate by document_id, keeping highest-priority row.
    best_by_doc: dict[str, dict[str, Any]] = {}
    for row in rows:
        doc_id = str(row.get("document_id") or "")
        if not doc_id:
            continue
        if doc_id not in best_by_doc or priority_key(row) < priority_key(best_by_doc[doc_id]):
            best_by_doc[doc_id] = row
    rows = list(best_by_doc.values())

    hot_candidates: list[dict[str, Any]] = []
    age_archive: list[dict[str, Any]] = []
    missing_date_count = 0
    pinned_count = 0
    older_than_cutoff_count = 0

    for row in rows:
        bucket = bucket_for(row)
        event_dt = row_event_dt(row)
        if event_dt is None:
            missing_date_count += 1
            hot_candidates.append(with_cleanup(row, state="active", reason="missing_event_date_kept_active", generated_at=generated_at, cutoff=cutoff))
            continue
        if cutoff and event_dt < cutoff and bucket not in pin_buckets:
            older_than_cutoff_count += 1
            age_archive.append(with_cleanup(row, state="archived", reason="older_than_retention_window", generated_at=generated_at, cutoff=cutoff))
            continue
        if cutoff and event_dt < cutoff and bucket in pin_buckets:
            pinned_count += 1
            hot_candidates.append(with_cleanup(row, state="active", reason="pinned_bucket_over_age", generated_at=generated_at, cutoff=cutoff))
        else:
            hot_candidates.append(with_cleanup(row, state="active", reason="within_retention_window" if cutoff else "no_time_cutoff", generated_at=generated_at, cutoff=cutoff))

    selected: list[dict[str, Any]] = []
    overflow_archive: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()

    if max_records and max_records > 0 and len(hot_candidates) > max_records:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in hot_candidates:
            buckets[bucket_for(row)].append(row)
        selected_ids: set[str] = set()
        for bucket, quota in quotas.items():
            if len(selected) >= max_records:
                break
            chosen = round_robin_select(buckets.get(bucket, []), min(quota, max_records - len(selected)))
            for row in chosen:
                doc_id = str(row.get("document_id"))
                if doc_id in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(doc_id)
                selected_counts[bucket] += 1
        if len(selected) < max_records:
            leftovers = [row for row in hot_candidates if str(row.get("document_id")) not in selected_ids]
            for row in round_robin_select(leftovers, max_records - len(selected)):
                doc_id = str(row.get("document_id"))
                if doc_id in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(doc_id)
                selected_counts[bucket_for(row)] += 1
        overflow_archive = [
            with_cleanup(row, state="archived", reason="quantity_overflow_safety_cap", generated_at=generated_at, cutoff=cutoff)
            for row in hot_candidates
            if str(row.get("document_id")) not in selected_ids
        ]
    else:
        selected = hot_candidates
        selected_counts = Counter(bucket_for(r) for r in selected)

    archive = age_archive + overflow_archive
    archive_reason_counts = Counter((r.get("cleanup") or {}).get("reason") for r in archive)
    active_reason_counts = Counter((r.get("cleanup") or {}).get("reason") for r in selected)
    summary = {
        "schema_version": "hindsight-review-backlog-cleanup-v2",
        "generated_at": generated_at,
        "input_records": len(rows),
        "active_records": len(selected),
        "archived_records": len(archive),
        "retention_months": retention_months,
        "retention_days": retention_days,
        "cutoff_event_date": cutoff.isoformat() if cutoff else None,
        "max_records": max_records,
        "count_cap_enabled": bool(max_records and max_records > 0),
        "pin_buckets": sorted(pin_buckets),
        "quotas": quotas,
        "missing_event_date_kept_active": missing_date_count,
        "older_than_cutoff_count": older_than_cutoff_count,
        "pinned_over_age_count": pinned_count,
        "by_active_bucket": dict(selected_counts.most_common()),
        "by_active_status": dict(Counter((r.get("current_retain_outcome") or {}).get("status") for r in selected).most_common()),
        "by_active_route": dict(Counter((r.get("review") or {}).get("recommended_route") for r in selected).most_common()),
        "by_active_reason": dict(active_reason_counts.most_common()),
        "by_archive_bucket": dict(Counter(bucket_for(r) for r in archive).most_common()),
        "by_archive_status": dict(Counter((r.get("current_retain_outcome") or {}).get("status") for r in archive).most_common()),
        "by_archive_route": dict(Counter((r.get("review") or {}).get("recommended_route") for r in archive).most_common()),
        "by_archive_reason": dict(archive_reason_counts.most_common()),
        "active_topic_keys": dict(Counter(topic_key(r) for r in selected).most_common(20)),
        "archive_topic_keys": dict(Counter(topic_key(r) for r in archive).most_common(20)),
    }
    return selected, archive, summary


def write_summary_md(path: str | Path, summary: dict[str, Any]) -> None:
    lines = ["# Hindsight review backlog cleanup", ""]
    for key in [
        "generated_at",
        "input_records",
        "active_records",
        "archived_records",
        "retention_months",
        "retention_days",
        "cutoff_event_date",
        "max_records",
        "count_cap_enabled",
        "missing_event_date_kept_active",
        "older_than_cutoff_count",
    ]:
        lines.append(f"- {key}: `{summary.get(key)}`")
    for title, key in [
        ("By active bucket", "by_active_bucket"),
        ("By active status", "by_active_status"),
        ("By active route", "by_active_route"),
        ("By active reason", "by_active_reason"),
        ("By archive bucket", "by_archive_bucket"),
        ("By archive status", "by_archive_status"),
        ("By archive route", "by_archive_route"),
        ("By archive reason", "by_archive_reason"),
        ("Active topic keys", "active_topic_keys"),
        ("Archive topic keys", "archive_topic_keys"),
    ]:
        lines.extend(["", f"## {title}", "```json", json.dumps(summary.get(key) or {}, ensure_ascii=False, indent=2, sort_keys=True), "```"])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-active", required=True)
    ap.add_argument("--output-archive", required=True)
    ap.add_argument("--summary-json")
    ap.add_argument("--summary-md")
    ap.add_argument("--retention-months", type=int, default=3, help="Hot backlog retention window in calendar months; default 3")
    ap.add_argument("--retention-days", type=int, default=None, help="Alternative hot backlog window in days; overrides --retention-months")
    ap.add_argument("--reference-date", help="Reference date for deterministic cleanup tests/runs; default now")
    ap.add_argument("--max-records", type=int, default=0, help="Optional count cap safety valve. 0 disables quantity cap (default).")
    ap.add_argument("--bucket-quotas", help="Only used with --max-records > 0. Comma list like manual_review=5,zero_with_retry_evidence=24")
    ap.add_argument("--pin-buckets", default="", help="Comma list of buckets to keep active even if older than retention window; default none")
    ap.add_argument("--json", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.input)
    active, archive, summary = compact_backlog(
        rows,
        parse_quotas(args.bucket_quotas),
        max_records=args.max_records,
        retention_months=args.retention_months,
        retention_days=args.retention_days,
        reference_date=args.reference_date,
        pin_buckets=parse_buckets(args.pin_buckets),
    )
    summary["input"] = args.input
    summary["output_active"] = args.output_active
    summary["output_archive"] = args.output_archive
    write_jsonl(args.output_active, active)
    write_jsonl(args.output_archive, archive)
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary_md:
        write_summary_md(args.summary_md, summary)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"wrote {len(active)} active and {len(archive)} archived rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
