#!/usr/bin/env python3
"""Audit Hindsight offline pipeline coverage.

Reports which layer has been processed:
- SQLite raw conversation coverage in Hermes state.db
- Hindsight retained SQLite documents/facts
- Offline daily consolidation documents
- Offline weekly consolidation documents
- Current incremental backlog since sqlite_import_progress cutoff

This is read-only: no Hindsight writes and no LLM calls.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hindsight_native_client import DEFAULT_API, DEFAULT_BANK, HindsightNativeClient

HOME = Path.home()
HERMES_HOME = HOME / ".hermes"
DEFAULT_DB = HERMES_HOME / "state.db"
DEFAULT_PROGRESS = HERMES_HOME / "hindsight" / "sqlite_import_progress.json"
DEFAULT_OFFLINE_PROGRESS = HERMES_HOME / "hindsight" / "offline_reflect" / "offline_reflect_progress.json"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": repr(e), "path": str(path)}


def sqlite_scalar_rows(db: Path, sql: str) -> list[tuple[Any, ...]]:
    con = sqlite3.connect(str(db))
    try:
        cur = con.execute(sql)
        return cur.fetchall()
    finally:
        con.close()



def date_counts_from_sqlite(db: Path) -> dict[str, Any]:
    sessions = sqlite_scalar_rows(
        db,
        "SELECT count(*), datetime(min(started_at),'unixepoch','localtime'), datetime(max(started_at),'unixepoch','localtime') FROM sessions",
    )[0]
    messages = sqlite_scalar_rows(
        db,
        "SELECT count(*), datetime(min(timestamp),'unixepoch','localtime'), datetime(max(timestamp),'unixepoch','localtime') FROM messages",
    )[0]
    session_days = sqlite_scalar_rows(
        db,
        "SELECT date(started_at,'unixepoch','localtime') d, count(*) FROM sessions GROUP BY d ORDER BY d",
    )
    message_days = sqlite_scalar_rows(
        db,
        "SELECT date(timestamp,'unixepoch','localtime') d, count(*) FROM messages GROUP BY d ORDER BY d",
    )
    return {
        "sessions": {"count": sessions[0], "min": sessions[1], "max": sessions[2]},
        "messages": {"count": messages[0], "min": messages[1], "max": messages[2]},
        "session_days": dict(session_days),
        "message_days": dict(message_days),
    }


def classify_doc(document_id: str) -> str:
    if document_id.startswith("hermes-offline-canonical::"):
        return "canonical"
    if document_id.startswith("hermes-offline-consolidation::weekly::"):
        return "offline_weekly"
    if document_id.startswith("hermes-offline-consolidation::daily::"):
        return "offline_daily"
    if document_id.startswith("hermes-sqlite::"):
        return "sqlite_import"
    return (document_id.split("::", 1)[0] or "other") if "::" in document_id else "other"


def hindsight_layer_counts(*, api: str = DEFAULT_API, bank: str = DEFAULT_BANK, client: HindsightNativeClient | None = None) -> dict[str, Any]:
    client = client or HindsightNativeClient(api=api, bank=bank)
    errors: list[str] = []
    try:
        docs = client.list_all_documents(max_items=100000)
    except Exception as e:
        docs = []
        errors.append(f"documents API scan failed: {repr(e)[:500]}")
    try:
        memories = list(client.iter_memories(types=["world", "experience", "observation"], max_items=200000))
    except Exception as e:
        memories = []
        errors.append(f"memories API scan failed: {repr(e)[:500]}")
    try:
        operations = list(client.iter_operations(max_items=10000))
    except Exception as e:
        operations = []
        errors.append(f"operations API scan failed: {repr(e)[:500]}")

    documents_by_kind: Counter[str] = Counter()
    sqlite_days: Counter[str] = Counter()
    daily_days: Counter[str] = Counter()
    weekly_periods: Counter[str] = Counter()
    for doc in docs:
        doc_id = str(doc.get("id") or "")
        documents_by_kind[classify_doc(doc_id)] += 1
        if doc_id.startswith("hermes-sqlite::day-topic::"):
            m = re.search(r"(\d{4}-\d{2}-\d{2})", doc_id)
            if m:
                sqlite_days[m.group(1)] += 1
        m_daily = re.search(r"::daily::(\d{4}-\d{2}-\d{2})", doc_id)
        if m_daily:
            daily_days[m_daily.group(1)] += 1
        m_week = re.search(r"::weekly::([^:]+)::", doc_id)
        if m_week:
            weekly_periods[m_week.group(1)] += 1

    facts_by_bank = {bank: len(memories)}
    op_counter: Counter[tuple[str, str]] = Counter()
    for op in operations:
        op_counter[(str(op.get("task_type") or op.get("operation_type") or "unknown"), str(op.get("status") or "unknown"))] += 1

    return {
        "source": "official_api",
        "errors": errors,
        "documents_by_bank": {bank: len(docs)},
        "facts_by_bank": facts_by_bank,
        "documents_by_kind": [[kind, str(n), "", ""] for kind, n in sorted(documents_by_kind.items())],
        "sqlite_import_days": dict(sorted(sqlite_days.items())),
        "daily_consolidated_days": dict(sorted(daily_days.items())),
        "weekly_consolidated_periods": dict(sorted(weekly_periods.items())),
        "operations": [[typ, status, str(n)] for (typ, status), n in sorted(op_counter.items())],
    }


def incremental_backlog(db: Path, progress: dict[str, Any]) -> dict[str, Any]:
    cutoff = float(progress.get("last_imported_timestamp") or 0.0)
    if cutoff <= 0:
        return {"cutoff": cutoff, "sessions_with_new_messages": None, "max_message_at": None, "note": "no cutoff"}
    rows = sqlite_scalar_rows(
        db,
        "SELECT count(DISTINCT s.id), datetime(max(m.timestamp),'unixepoch','localtime') "
        "FROM sessions s JOIN messages m ON m.session_id=s.id WHERE m.timestamp > ?",
    ) if False else []
    con = sqlite3.connect(str(db))
    try:
        cur = con.execute(
            "SELECT count(DISTINCT s.id), datetime(max(m.timestamp),'unixepoch','localtime') "
            "FROM sessions s JOIN messages m ON m.session_id=s.id WHERE m.timestamp > ?",
            (cutoff,),
        )
        n, max_at = cur.fetchone()
    finally:
        con.close()
    return {"cutoff": cutoff, "cutoff_iso": progress.get("last_imported_iso"), "sessions_with_new_messages": n, "max_message_at": max_at}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--progress", default=str(DEFAULT_PROGRESS))
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--bank", default=DEFAULT_BANK)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    db = Path(args.db).expanduser()
    progress_path = Path(args.progress).expanduser()
    progress = load_json(progress_path, {})
    offline_progress = load_json(DEFAULT_OFFLINE_PROGRESS, {})
    sqlite_cov = date_counts_from_sqlite(db) if db.exists() else {"error": f"missing db {db}"}
    layers = hindsight_layer_counts(api=args.api, bank=args.bank)
    sqlite_days = set(layers.get("sqlite_import_days", {}).keys())
    daily_days = set(layers.get("daily_consolidated_days", {}).keys())
    today = datetime.now().date().isoformat()
    missing_daily = sqlite_days - daily_days
    open_current_missing = sorted(d for d in missing_daily if d >= today)
    closed_missing = sorted(d for d in missing_daily if d < today)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "hindsight_bank": args.bank,
        "hindsight_api": args.api,
        "sqlite": sqlite_cov,
        "sqlite_import_progress": {
            "path": str(progress_path),
            "processed_count": len(progress.get("processed", []) or []),
            "last_imported_iso": progress.get("last_imported_iso"),
            "last_run": progress.get("last_run"),
            "total_sessions_imported": progress.get("total_sessions_imported"),
            "total_bundles_imported": progress.get("total_bundles_imported"),
        },
        "incremental_backlog": incremental_backlog(db, progress) if db.exists() else {},
        "hindsight_layers": layers,
        "offline_reflect_progress": {
            "path": str(DEFAULT_OFFLINE_PROGRESS),
            "processed_document_ids": len(offline_progress.get("processed_document_ids", []) or []),
            "last_run": offline_progress.get("last_run"),
        },
        "coverage_summary": {
            "sqlite_retained_days": len(sqlite_days),
            "daily_consolidated_days": len(daily_days),
            "sqlite_days_without_daily_consolidation": sorted(missing_daily),
            "closed_sqlite_days_without_daily_consolidation": closed_missing,
            "open_current_sqlite_days_without_daily_consolidation": open_current_missing,
            "daily_days_without_sqlite_docs": sorted(daily_days - sqlite_days),
            "weekly_consolidated_periods": sorted(layers.get("weekly_consolidated_periods", {}).keys()),
        },
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print("Hindsight Offline Coverage Audit")
    print(json.dumps(report["sqlite_import_progress"], ensure_ascii=False, indent=2))
    print("incremental_backlog:", json.dumps(report["incremental_backlog"], ensure_ascii=False))
    print("documents_by_bank:", report["hindsight_layers"].get("documents_by_bank"))
    if report["hindsight_layers"].get("errors"):
        print("api_errors:", json.dumps(report["hindsight_layers"].get("errors"), ensure_ascii=False))
    print("facts_by_bank:", report["hindsight_layers"].get("facts_by_bank"))
    print("documents_by_kind:")
    for row in report["hindsight_layers"].get("documents_by_kind", []):
        print("  ", "\t".join(row))
    print("coverage_summary:")
    cs = report["coverage_summary"]
    print(f"  sqlite_retained_days={cs['sqlite_retained_days']}")
    print(f"  daily_consolidated_days={cs['daily_consolidated_days']}")
    print(f"  weekly_consolidated_periods={cs['weekly_consolidated_periods']}")
    missing = cs["sqlite_days_without_daily_consolidation"]
    closed_missing = cs.get("closed_sqlite_days_without_daily_consolidation", [])
    open_missing = cs.get("open_current_sqlite_days_without_daily_consolidation", [])
    print(f"  sqlite_days_without_daily_consolidation={len(missing)}")
    print(f"  closed_sqlite_days_without_daily_consolidation={len(closed_missing)}")
    if closed_missing:
        print("  closed_missing_daily_days=", ", ".join(closed_missing[:80]))
    if open_missing:
        print("  open_current_missing_daily_days=", ", ".join(open_missing[:80]))


if __name__ == "__main__":
    main()
