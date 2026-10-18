#!/usr/bin/env python3
"""Guard/marker for N-day Sunday wiki maintenance.

The Hermes cron scheduler cannot express "every 2 weeks on Sunday after another job"
cleanly with a single cron expression when a strict after-Hindsight order is required.
Schedule the job on Sunday, then use this guard to decide whether it is due.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

DEFAULT_MARKER = Path.home() / ".hermes" / "hindsight" / "wiki_maintenance_progress.json"


def parse_date(s: str) -> date:
    return date.fromisoformat(s)


def load_marker(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_due(today: date, *, anchor: date, cycle_days: int, marker: dict) -> tuple[bool, str]:
    if today.weekday() != 6:  # Monday=0, Sunday=6
        return False, f"not Sunday: {today.isoformat()}"
    if today < anchor:
        return False, f"before anchor Sunday: today={today.isoformat()} anchor={anchor.isoformat()}"
    last = marker.get("last_success_date")
    if last:
        try:
            last_date = parse_date(str(last))
        except ValueError:
            return True, f"invalid marker last_success_date={last!r}; run to repair marker"
        days = (today - last_date).days
        if days >= cycle_days:
            return True, f"due: {days} days since last_success_date={last_date.isoformat()}"
        return False, f"not due: {days}/{cycle_days} days since last_success_date={last_date.isoformat()}"
    return True, f"due: no previous success marker and today>=anchor {anchor.isoformat()}"


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("check")
    check.add_argument("--anchor", default="2026-05-10", help="first eligible Sunday, YYYY-MM-DD")
    check.add_argument("--cycle-days", type=int, default=14)
    check.add_argument("--marker", default=str(DEFAULT_MARKER))
    check.add_argument("--today", help="override date for tests, YYYY-MM-DD")

    mark = sub.add_parser("mark")
    mark.add_argument("--marker", default=str(DEFAULT_MARKER))
    mark.add_argument("--date", help="success date, YYYY-MM-DD; default today")
    mark.add_argument("--log-path", default="")

    args = parser.parse_args()
    marker_path = Path(args.marker).expanduser()

    if args.cmd == "check":
        today = parse_date(args.today) if args.today else date.today()
        marker = load_marker(marker_path)
        due, reason = is_due(today, anchor=parse_date(args.anchor), cycle_days=args.cycle_days, marker=marker)
        print(json.dumps({"due": due, "reason": reason, "today": today.isoformat(), "marker": str(marker_path)}, ensure_ascii=False))
        raise SystemExit(0 if due else 2)

    if args.cmd == "mark":
        day = parse_date(args.date) if args.date else date.today()
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        data = load_marker(marker_path)
        data.update(
            {
                "last_success_date": day.isoformat(),
                "last_success_at": datetime.now(timezone.utc).isoformat(),
                "cycle_days": data.get("cycle_days", 14),
                "log_path": args.log_path,
            }
        )
        marker_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"marked": True, "date": day.isoformat(), "marker": str(marker_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
