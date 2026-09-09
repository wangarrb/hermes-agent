#!/usr/bin/env python3
"""Backfill gap-window kanban sessions: state.db -> native session JSON snapshots.

The 2026-08-27 ~ 2026-09-04 ingest blackout left ~53 kanban-profile sessions
(~15K messages) never ingested: they predate the write_json_snapshots:true
config flip, and the offline pipeline's manifest scanner only reads
sessions/session_*.json files. This one-off migration materializes each gap
session from its profile state.db into the exact native snapshot format the
scanner already understands:

    {"session_id", "model", "base_url", "platform", "session_start",
     "last_updated", "system_prompt", "tools", "message_count",
     "messages": [{"role", "content", "timestamp", "_db_persisted"}, ...]}

Per-message timestamps are included so the manifest's event_date logic (fixed
2026-09-04 to prefer max message timestamp) dates extracted facts correctly
instead of at the backfill/import date.

Usage:
    python3 hindsight_backfill_gap_sessions.py --dry-run          # stats only
    python3 hindsight_backfill_gap_sessions.py --write            # materialize JSON files
Safety:
  - Never overwrites an existing session_*.json (skip + report).
  - Skips sessions whose id already has a live snapshot file.
  - Read-only DB access (immutable=1) — safe while Hermes is running.
  - Only user/assistant messages are exported; tool messages are dropped by
    the manifest cleaner anyway and would just bloat files.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

CST = timezone(timedelta(hours=8))
GAP_START = datetime(2026, 8, 27, 0, 0, tzinfo=CST)
GAP_END = datetime(2026, 9, 5, 0, 0, tzinfo=CST)
MIN_MESSAGES = 5  # skip only tiny noise sessions

PROFILES = ["designer", "coordinator", "planner", "implementer", "critic", "lj", "reviewer"]


def epoch_range() -> tuple[int, int]:
    return int(GAP_START.timestamp()), int(GAP_END.timestamp())


def gap_sessions(db_path: Path) -> list[dict]:
    start, end = epoch_range()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True, timeout=5)
    rows = conn.execute(
        """
        SELECT m.session_id,
               count(*) AS n,
               min(CAST(m.timestamp AS INTEGER)) AS t0,
               max(CAST(m.timestamp AS INTEGER)) AS t1,
               max(s.title) AS title,
               max(s.cwd) AS cwd,
               max(s.model) AS model
        FROM messages m JOIN sessions s ON s.id = m.session_id
        WHERE CAST(m.timestamp AS INTEGER) BETWEEN ? AND ?
          AND m.role IN ('user', 'assistant')
          AND m.content IS NOT NULL AND length(m.content) > 0
        GROUP BY m.session_id
        HAVING n >= ?
        ORDER BY n DESC
        """,
        (start, end, MIN_MESSAGES),
    ).fetchall()
    conn.close()
    out = []
    for sid, n, t0, t1, title, cwd, model in rows:
        out.append(
            {
                "session_id": sid,
                "ua_msgs": n,
                "t0": t0,
                "t1": t1,
                "title": title or "",
                "cwd": cwd or "",
                "model": model or "",
            }
        )
    return out


def export_messages(db_path: Path, session_id: str) -> list[dict]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True, timeout=5)
    rows = conn.execute(
        """
        SELECT role, content, CAST(timestamp AS INTEGER)
        FROM messages
        WHERE session_id = ? AND role IN ('user','assistant')
          AND content IS NOT NULL AND length(content) > 0
        ORDER BY CAST(timestamp AS INTEGER), id
        """,
        (session_id,),
    ).fetchall()
    conn.close()
    msgs = []
    for role, content, ts in rows:
        msgs.append(
            {
                "role": role,
                "content": content,
                "timestamp": str(ts),  # epoch-seconds string, same as live DB rows
                "_db_persisted": True,
            }
        )
    return msgs


def build_snapshot(session: dict, messages: list[dict]) -> dict:
    t0 = datetime.fromtimestamp(session["t0"], CST).isoformat()
    t1 = datetime.fromtimestamp(session["t1"], CST).isoformat()
    return {
        "session_id": session["session_id"],
        "model": session["model"],
        "base_url": "",
        "platform": "backfill",
        "session_start": t0,
        "last_updated": t1,
        "system_prompt": "",
        "tools": [],
        "message_count": len(messages),
        "messages": messages,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report stats without writing files")
    ap.add_argument("--write", action="store_true", help="materialize session JSON files")
    args = ap.parse_args()
    if not (args.dry_run or args.write):
        args.dry_run = True

    hermes = Path("/home/wyr/.hermes")
    grand_sessions = 0
    grand_msgs = 0
    grand_chars = 0
    plan = []

    for profile in PROFILES:
        db_path = hermes / "profiles" / profile / "state.db"
        if not db_path.exists():
            continue
        sessions = gap_sessions(db_path)
        if not sessions:
            continue
        sessions_dir = hermes / "profiles" / profile / "sessions"
        for session in sessions:
            out_path = sessions_dir / f"session_{session['session_id']}.json"
            exists = out_path.exists()
            msgs = export_messages(db_path, session["session_id"])
            total_chars = sum(len(str(m["content"])) for m in msgs)
            entry = {
                "profile": profile,
                "session_id": session["session_id"],
                "ua_msgs": len(msgs),
                "content_chars": total_chars,
                "window": f"{datetime.fromtimestamp(session['t0'], CST):%m-%d %H:%M}~{datetime.fromtimestamp(session['t1'], CST):%m-%d %H:%M}",
                "out_path": str(out_path),
                "skip_exists": exists and args.write,
            }
            plan.append(entry)
            grand_sessions += 1
            grand_msgs += len(msgs)
            grand_chars += total_chars

            if args.write and not exists:
                snapshot = build_snapshot(session, msgs)
                sessions_dir.mkdir(parents=True, exist_ok=True)
                tmp = out_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
                # Match live snapshot file permissions (0600)
                tmp.chmod(0o600)
                tmp.replace(out_path)

    # Report
    print(f"{'profile':12s} {'session':26s} {'msgs':>5s} {'chars':>8s}  window / out")
    for e in plan:
        flag = " [EXISTS skip]" if e["skip_exists"] else ""
        print(
            f"{e['profile']:12s} {e['session_id']:26s} {e['ua_msgs']:5d} {e['content_chars']:8d}  "
            f"{e['window']}{flag}"
        )
    print(
        f"\nTOTAL: {grand_sessions} sessions, {grand_msgs} msgs, {grand_chars} chars "
        f"(~{grand_chars // 8000} retain chunks @8K)"
    )
    if args.write:
        written = sum(1 for e in plan if not e["skip_exists"])
        print(f"written: {written} files")
    else:
        print("dry-run only — rerun with --write to materialize")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
