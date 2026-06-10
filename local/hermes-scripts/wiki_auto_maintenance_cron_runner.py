#!/usr/bin/env python3
"""Run wiki auto-maintenance on a biweekly Sunday cadence.

Scheduled by Hermes cron every Sunday 05:00. This wrapper enforces the 14-day
cycle and writes only isolated candidate reports under ~/wiki/auto-maintenance/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

HOME = Path.home()
HERMES_HOME = HOME / ".hermes"
GUARD = HERMES_HOME / "scripts" / "wiki_maintenance_cycle_guard.py"
MAINT = HERMES_HOME / "scripts" / "wiki_auto_maintenance.py"
MARKER = HERMES_HOME / "hindsight" / "wiki_maintenance_progress.json"
LOG_DIR = HERMES_HOME / "logs" / "wiki-auto-maintenance"
SUMMARY_DIR = LOG_DIR / "summaries"
LATEST_SUMMARY = LOG_DIR / "latest-summary.json"

ANOMALY_RE = re.compile(
    r"(Traceback|ERROR:|\bERROR\b|Exception|Connection refused|ConnectionError|TimeoutError|timed out|"
    r"timeout within|did not .* within|broken_links|missing wrapper|missing maintenance|\bEXIT\s+[1-9][0-9]*\b|"
    r"\"status\"\s*:\s*\"failed\"|status=failed)",
    re.I,
)
BENIGN_RE = re.compile(r"(\"status\"\s*:\s*\"ok\"|\"ok\"\s*:\s*true)", re.I)


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run(cmd: list[str], log_fh) -> int:
    line = "RUN " + " ".join(cmd) + "\n"
    print(line, end="", flush=True)
    log_fh.write(line); log_fh.flush()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for out in proc.stdout:
        print(out, end="", flush=True)
        log_fh.write(out); log_fh.flush()
    code = proc.wait()
    msg = f"EXIT {code} {' '.join(cmd)}\n"
    print(msg, end="", flush=True)
    log_fh.write(msg); log_fh.flush()
    return code


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def scan_log_anomalies(log_path: Path, *, limit: int = 30) -> list[dict[str, object]]:
    if not log_path.exists():
        return [{"line": 0, "text": f"missing log file: {log_path}"}]
    findings: list[dict[str, object]] = []
    for line_no, line in enumerate(log_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        stripped = line.strip()
        if not stripped or BENIGN_RE.search(stripped):
            continue
        if ANOMALY_RE.search(stripped):
            findings.append({"line": line_no, "text": stripped[-800:]})
    return findings[-limit:]


def finalize_summary(summary: dict[str, object], *, log_path: Path) -> Path:
    anomalies = scan_log_anomalies(log_path)
    summary["log_path"] = str(log_path)
    summary["anomaly_count"] = len(anomalies)
    summary["anomalies"] = anomalies
    summary_path = SUMMARY_DIR / f"{stamp()}.json"
    summary["summary_path"] = str(summary_path)
    write_json_atomic(summary_path, summary)
    write_json_atomic(LATEST_SUMMARY, summary)
    (LOG_DIR / "latest.log.path").write_text(str(log_path) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="run even if guard says not due")
    ap.add_argument("--anchor", default="2026-05-10")
    ap.add_argument("--cycle-days", type=int, default=14)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--wait-timeout", type=int, default=21600)
    ap.add_argument("--wait-poll", type=int, default=60)
    ap.add_argument("--lock-timeout", type=int, default=21600)
    args = ap.parse_args()

    if not GUARD.exists():
        raise SystemExit(f"missing guard: {GUARD}")
    if not MAINT.exists():
        raise SystemExit(f"missing maintenance script: {MAINT}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{stamp()}.log"
    summary = {"started_at": datetime.now().isoformat(timespec="seconds"), "log_path": str(log_path), "status": "unknown"}
    with open(log_path, "a", encoding="utf-8") as log_fh:
        check_cmd = [sys.executable, str(GUARD), "check", "--anchor", args.anchor, "--cycle-days", str(args.cycle_days), "--marker", str(MARKER)]
        code = run(check_cmd, log_fh)
        if code != 0 and not args.force:
            summary.update({"status": "skipped", "reason": "not due", "finished_at": datetime.now().isoformat(timespec="seconds")})
            summary_path = finalize_summary(summary, log_path=log_path)
            msg = "SUMMARY " + json.dumps(summary, ensure_ascii=False) + "\n"
            print(msg, end=""); print(f"summary_path={summary_path}")
            log_fh.write(msg); log_fh.write(f"summary_path={summary_path}\n")
            return
        maint_cmd = [
            sys.executable, str(MAINT),
            "--days", str(args.days),
            "--wait-hindsight",
            "--wait-timeout", str(args.wait_timeout),
            "--wait-poll", str(args.wait_poll),
            "--lock-timeout", str(args.lock_timeout),
        ]
        code = run(maint_cmd, log_fh)
        if code != 0:
            summary.update({"status": "failed", "exit_code": code, "finished_at": datetime.now().isoformat(timespec="seconds")})
            summary_path = finalize_summary(summary, log_path=log_path)
            msg = "SUMMARY " + json.dumps(summary, ensure_ascii=False) + "\n"
            print(msg, end=""); print(f"summary_path={summary_path}")
            log_fh.write(msg); log_fh.write(f"summary_path={summary_path}\n")
            raise SystemExit(code)
        # Mark only after successful report generation.
        mark_cmd = [sys.executable, str(GUARD), "mark", "--marker", str(MARKER), "--date", date.today().isoformat(), "--log-path", str(log_path)]
        code = run(mark_cmd, log_fh)
        if code != 0:
            summary.update({"status": "failed", "exit_code": code, "stage": "mark", "finished_at": datetime.now().isoformat(timespec="seconds")})
            summary_path = finalize_summary(summary, log_path=log_path)
            msg = "SUMMARY " + json.dumps(summary, ensure_ascii=False) + "\n"
            print(msg, end=""); print(f"summary_path={summary_path}")
            log_fh.write(msg); log_fh.write(f"summary_path={summary_path}\n")
            raise SystemExit(code)
        summary.update({"status": "ok", "finished_at": datetime.now().isoformat(timespec="seconds")})
        summary_path = finalize_summary(summary, log_path=log_path)
        msg = "SUMMARY " + json.dumps(summary, ensure_ascii=False) + "\n"
        print(msg, end=""); print(f"summary_path={summary_path}")
        log_fh.write(msg); log_fh.write(f"summary_path={summary_path}\n")


if __name__ == "__main__":
    main()
