#!/usr/bin/env python3
"""Compact Hindsight offline reflect progress report for Hermes/tmux monitoring.
No secrets are printed.

Reads the latest Hindsight offline reflect log plus production DB counts and prints
both daily backfill and weekly all-history progress. Safe/read-only.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

HOME = Path.home()
LOG_DIR = HOME / ".hermes" / "logs" / "hindsight-offline-reflect"
OUT_DIR = HOME / ".hermes" / "hindsight" / "offline_reflect"
DAILY_DIR = OUT_DIR / "daily"
WEEKLY_DIR = OUT_DIR / "weekly"
PG_INSTANCE = HOME / ".hindsight-docker" / "instances" / "hindsight" / "instance.json"
PSQL = HOME / ".hindsight-docker" / "installation" / "18.1.0" / "bin" / "psql"


def latest_log() -> Path | None:
    patterns = ["*bailian-history-submit-c*.log", "*bailian-history-submit*.log"]
    files: list[Path] = []
    for pat in patterns:
        files.extend(LOG_DIR.glob(pat))
    files = sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def read_text(path: Path | None) -> str:
    if not path:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def is_running() -> bool:
    try:
        r = subprocess.run(
            ["pgrep", "-af", r"hindsight_minimax_import.py offline-reflect-llm|offline_hindsight_reflect_consolidate.py"],
            text=True,
            capture_output=True,
            timeout=5,
        )
    except Exception:
        return False
    lines = []
    for line in r.stdout.splitlines():
        if "hindsight_progress_bar.py" in line or "hindsight_progress_live" in line:
            continue
        if "pgrep -af" in line:
            continue
        lines.append(line)
    return bool(lines)


def bar(done: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "[?]"
    done = max(0, min(done, total))
    filled = int(width * done / total)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {done}/{total}"


def parse_daily(text: str):
    missing: list[str] = []
    m = re.search(r"backfilling missing/incomplete daily outputs before weekly: (.*)", text)
    if m:
        missing = [x.strip() for x in m.group(1).split(",") if x.strip()]

    expected: dict[str, int] = {}
    for day, n in re.findall(r"Daily period=(\d{4}-\d{2}-\d{2}) source=facts facts=\d+\nUnits: (\d+) ", text):
        expected[day] = int(n)

    completed = {day: len(list((DAILY_DIR / day).glob("*.md"))) for day in missing}
    done_days = [d for d in missing if expected.get(d) is not None and completed.get(d, 0) >= expected[d] and expected[d] > 0]
    partial_days = [d for d in missing if d not in done_days and completed.get(d, 0) > 0]
    known_total_units = sum(expected.get(d, 0) for d in missing)
    known_done_units = sum(min(completed.get(d, 0), expected.get(d, 0)) for d in missing)
    return {
        "missing": missing,
        "expected": expected,
        "completed": completed,
        "done_days": done_days,
        "partial_days": partial_days,
        "known_total_units": known_total_units,
        "known_done_units": known_done_units,
    }


def parse_weekly(text: str):
    matches = list(re.finditer(
        r"Weekly period=([^\s]+) window=([^\s]+) source=([^\s]+) group_by=([^\s]+) days=(\d+) daily_topics=(\d+) units=(\d+)",
        text,
    ))
    if not matches:
        return None
    m = matches[-1]
    block = text[m.start():]
    period, window, source, group_by, days, topics, units = m.groups()
    total = int(units)

    started = set(int(x) for x in re.findall(r"\[(\d+)/\d+\] bailian weekly", block))
    saved_paths = re.findall(r"saved: (.*?/weekly/" + re.escape(period) + r"/[^\s]+\.md)", block)
    saved_now = len(set(saved_paths))
    failed = len(re.findall(r"LLM call failed: \[\d+/\d+\] weekly", block))
    batch_m = list(re.finditer(r"Batch start: concurrency=(\d+) batch=(\d+) remaining_after_batch=(\d+)", block))
    last_batch = None
    if batch_m:
        c, b, rem = batch_m[-1].groups()
        last_batch = {"concurrency": int(c), "batch": int(b), "remaining_after_batch": int(rem)}

    disk_saved = len(list((WEEKLY_DIR / period).glob("*.md")))
    saved_total = max(saved_now, disk_saved)
    in_flight_now = max(0, len(started) - saved_now - failed)
    return {
        "period": period,
        "window": window,
        "source": source,
        "group_by": group_by,
        "days": int(days),
        "topics": int(topics),
        "total": total,
        "started": len(started),
        "saved": saved_total,
        "saved_now": saved_now,
        "failed": failed,
        "in_flight": in_flight_now,
        "last_batch": last_batch,
    }


def parse_concurrency(text: str):
    concs = re.findall(r"Adaptive LLM concurrency: start=(\d+) min=(\d+) 429_backoff=(\d+)s", text)
    if not concs:
        return None
    s, mi, b = concs[-1]
    return {"start": int(s), "min": int(mi), "backoff_s": int(b)}


def count_429(text: str) -> int:
    return len(re.findall(r"HTTP 429|Too Many Requests|concurrency allocated quota exceeded", text, flags=re.I))


def db_summary() -> str:
    if not (PG_INSTANCE.exists() and PSQL.exists()):
        return "DB: unavailable"
    try:
        cfg = json.loads(PG_INSTANCE.read_text())
        env = os.environ.copy()
        env["PGPASSWORD"] = cfg.get("password", "")
        q = """
        SELECT operation_type,status,count(*) FROM async_operations GROUP BY 1,2 ORDER BY 1,2;
        SELECT 'docs_units', (SELECT count(*) FROM documents WHERE bank_id='hermes'), (SELECT count(*) FROM memory_units WHERE bank_id='hermes');
        """
        r = subprocess.run(
            [str(PSQL), "-h", "127.0.0.1", "-p", str(cfg.get("port", 5432)), "-U", cfg.get("username", "hindsight"), "-d", cfg.get("database", "hindsight"), "-tA", "-c", q],
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
        )
        if r.returncode != 0:
            return "DB: query_failed"
        counts = []
        docs_units = None
        for line in r.stdout.splitlines():
            parts = line.split("|")
            if len(parts) == 3 and parts[0] != "docs_units":
                counts.append(f"{parts[0]}:{parts[1]}={parts[2]}")
            elif len(parts) == 3 and parts[0] == "docs_units":
                docs_units = f"docs={parts[1]} units={parts[2]}"
        return "DB: " + ", ".join(counts + ([docs_units] if docs_units else []))
    except Exception as e:
        return f"DB: error={type(e).__name__}"


def main() -> None:
    log = latest_log()
    text = read_text(log)
    running = is_running()
    daily = parse_daily(text)
    weekly = parse_weekly(text)
    concurrency = parse_concurrency(text)
    n429 = count_429(text)

    print(f"HINDSIGHT_PROGRESS {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"状态: {'running' if running else 'finished_or_not_found'}")
    if concurrency:
        print(f"并发: start={concurrency['start']} min={concurrency['min']} 429_backoff={concurrency['backoff_s']}s")
    if daily["missing"]:
        print(f"Daily backfill days: {bar(len(daily['done_days']), len(daily['missing']))} partial={len(daily['partial_days'])}")
        print(f"Daily unit outputs: {bar(daily['known_done_units'], daily['known_total_units']) if daily['known_total_units'] else '[unknown]'}")
    if weekly:
        print(
            f"Weekly all-history outputs: {bar(weekly['saved'], weekly['total'])} "
            f"current_run_started={weekly['started']} saved_now={weekly.get('saved_now', 0)} "
            f"failed_now={weekly['failed']} in_flight_now={weekly['in_flight']}"
        )
        if weekly.get("last_batch"):
            b = weekly["last_batch"]
            print(f"当前: weekly {weekly['period']} batch={b['batch']} rem_after={b['remaining_after_batch']} topics={weekly['topics']} days={weekly['days']}")
    print(f"429/throttle hits: {n429}")
    print(db_summary())
    if log:
        print(f"log: {log}")


if __name__ == "__main__":
    main()
