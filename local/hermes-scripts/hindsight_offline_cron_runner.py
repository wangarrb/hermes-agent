#!/usr/bin/env python3
"""Cron-safe Hindsight offline pipeline runner.

Daily: SQLite incremental retain -> processed-facts daily consolidation.
Weekly: global refresh from all historical daily outputs (cross-topic + cross-period).

The paid LLM is profile-based (minimax/glm/deepseek/custom) and resolved by
hindsight_minimax_import.py; default can be changed with
HINDSIGHT_OFFLINE_LLM_PROFILE in env or ~/.hermes/.env.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

HOME = Path.home()
HERMES_HOME = HOME / ".hermes"
SCRIPT = HERMES_HOME / "scripts" / "hindsight_minimax_import.py"
OFFLINE_REFLECT_SCRIPT = HERMES_HOME / "scripts" / "offline_hindsight_reflect_consolidate.py"
V2_REBUILD_SCRIPT = HERMES_HOME / "scripts" / "hindsight_offline_v2_rebuild.py"
V2_PUBLISH_CONFIRM = "publish-hindsight-v2-canonical"
LOG_DIR = HERMES_HOME / "logs" / "hindsight-offline-pipeline"
SUMMARY_DIR = LOG_DIR / "summaries"
LATEST_SUMMARY = LOG_DIR / "latest-summary.json"
LOCK_PATH = HERMES_HOME / "hindsight" / "offline_pipeline.lock"

ANOMALY_RE = re.compile(
    r"(Traceback|ERROR:|\bERROR\b|Exception|Connection refused|ConnectionRefused|ConnectionError|"
    r"JSON parse error|STUCK|\b429\b|TimeoutError|timed out|timeout within|did not .* within|"
    r"post failed|failed with code|\bEXIT\s+[1-9][0-9]*\b|\"status\"\s*:\s*\"failed\"|status=failed)",
    re.I,
)
BENIGN_RE = re.compile(
    r"(failed_operations['\" ]*[:=]['\" ]*0|\"status\"\s*:\s*\"ok\"|last_status.*ok)",
    re.I,
)


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def resolve_daily_date(mode: str) -> str:
    now = datetime.now()
    if mode == "today":
        return now.date().isoformat()
    if mode == "yesterday":
        return (now.date() - timedelta(days=1)).isoformat()
    if mode == "auto":
        # Scheduled daily runs at 02:00 should process yesterday; manual evening runs process today.
        return (now.date() - timedelta(days=1) if now.hour < 6 else now.date()).isoformat()
    # Explicit YYYY-MM-DD.
    return mode


def resolve_week(mode: str) -> str:
    now = datetime.now().date()
    if mode == "current":
        iso = now.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if mode == "previous":
        iso = (now - timedelta(days=7)).isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    return mode


def open_log(task: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return LOG_DIR / f"{stamp}-{task}.log"


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def scan_log_anomalies(log_path: Path, *, limit: int = 30) -> list[dict[str, object]]:
    """Extract high-signal anomaly lines from a cron log.

    This intentionally keeps raw context small so the daily report can show a
    useful tail without flooding WeChat. Benign zero-failure status lines are
    skipped; real failures, non-zero exits, provider errors, and retry-loop
    symptoms are retained.
    """
    if not log_path.exists():
        return [{"line": 0, "text": f"missing log file: {log_path}"}]
    findings: list[dict[str, object]] = []
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    found_queue_drained = any("queue drained" in line.lower() for line in lines)
    for line_no, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or BENIGN_RE.search(stripped):
            continue
        if "queue_poll_error" in stripped and "Connection refused" in stripped and found_queue_drained:
            continue
        if ANOMALY_RE.search(stripped):
            findings.append({"line": line_no, "text": stripped[-800:]})
    return findings[-limit:]


def finalize_summary(summary: dict[str, object], *, log_path: Path, task: str) -> Path:
    anomalies = scan_log_anomalies(log_path)
    summary["log_path"] = str(log_path)
    summary["anomaly_count"] = len(anomalies)
    summary["anomalies"] = anomalies
    summary_path = SUMMARY_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{task}.json"
    summary["summary_path"] = str(summary_path)
    write_json_atomic(summary_path, summary)
    write_json_atomic(LOG_DIR / f"latest-{task}.json", summary)
    write_json_atomic(LATEST_SUMMARY, summary)
    (LOG_DIR / f"latest-{task}.log.path").write_text(str(log_path) + "\n", encoding="utf-8")
    return summary_path


def run_capture_json(cmd: list[str], log_fh) -> dict[str, object]:
    printable = " ".join(shlex.quote(x) for x in cmd)
    header = f"\n[{iso_now()}] RUN_JSON {printable}\n"
    print(header, end="", flush=True)
    log_fh.write(header); log_fh.flush()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
        log_fh.write(proc.stdout)
    if proc.stderr:
        print(proc.stderr, end="", flush=True)
        log_fh.write(proc.stderr)
    footer = f"[{iso_now()}] EXIT {proc.returncode} {printable}\n"
    print(footer, end="", flush=True)
    log_fh.write(footer); log_fh.flush()
    text = (proc.stdout or "").strip()
    obj: dict[str, object] | None = None
    for idx in [text.find("{"), text.rfind("\n{")]:
        if idx is None or idx < 0:
            continue
        try:
            candidate = text[idx + (1 if text[idx:idx+2] == "\n{" else 0):]
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                obj = parsed
        except Exception:
            pass
    if obj is None:
        raise SystemExit(f"JSON command did not produce parseable JSON: {printable}")
    if proc.returncode != 0:
        raise SystemExit(f"JSON command failed with code {proc.returncode}: {printable}; decision={obj.get('budget_decision')}")
    return obj

def run(cmd: list[str], log_fh, *, env: dict[str, str] | None = None) -> None:
    printable = " ".join(shlex.quote(x) for x in cmd)
    header = f"\n[{iso_now()}] RUN {printable}\n"
    print(header, end="", flush=True)
    log_fh.write(header)
    log_fh.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_fh.write(line)
        log_fh.flush()
    code = proc.wait()
    footer = f"[{iso_now()}] EXIT {code} {printable}\n"
    print(footer, end="", flush=True)
    log_fh.write(footer)
    log_fh.flush()
    if code != 0:
        raise SystemExit(code)


def common_wrapper_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    if args.llm_profile:
        out += ["--llm-profile", args.llm_profile]
    out += ["--poll", str(args.poll), "--timeout", str(args.timeout)]
    return out


def offline_reflect_llm_args(args: argparse.Namespace) -> list[str]:
    """Return read-only LLM settings used for both budget keys and submit.

    The paid wrapper injects these settings before it calls the offline reflect
    script. The budget gate must compute cache keys with the exact same knobs;
    otherwise a profile change can make the budget report reuse stale keys while
    submit sees every unit as pending.
    """
    injected = getattr(args, "_offline_reflect_llm_args", None)
    if injected is not None:
        return list(injected)
    scripts_dir = str(HERMES_HOME / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import hindsight_minimax_import as llm_manager  # type: ignore

    profile = llm_manager.get_llm_profile(getattr(args, "llm_profile", None))
    out = [
        "--llm-model",
        str(profile["model"]),
        "--llm-label",
        str(profile.get("label") or "llm"),
        "--llm-base-url",
        str(profile["base_url"]),
    ]
    if not profile.get("response_format", True):
        out.append("--no-response-format")
    return out


def should_run_daily(args: argparse.Namespace) -> bool:
    return args.task in {"daily", "both"} and not bool(getattr(args, "dry_run_budget_only", False))


def should_run_weekly(args: argparse.Namespace) -> bool:
    return args.task in {"weekly", "both"}


def status(log_fh) -> None:
    run([sys.executable, str(SCRIPT), "status"], log_fh)


def weekly_budget_cmd(args: argparse.Namespace, week: str) -> list[str]:
    return [
        sys.executable,
        str(OFFLINE_REFLECT_SCRIPT),
        "--scope",
        "weekly",
        "--week",
        week,
        "--weekly-window",
        "all-history",
        "--weekly-source",
        "daily",
        "--weekly-group-by",
        "topic",
        "--backfill-missing-daily",
        "--mode",
        "dry-run",
        "--prefilter",
        args.prefilter,
        "--budget-json",
        "--budget-max-pending-units",
        str(args.weekly_budget_max_pending_units),
        "--budget-max-pending-chars",
        str(args.weekly_budget_max_pending_chars),
        *offline_reflect_llm_args(args),
    ]


def weekly_submit_cmd(args: argparse.Namespace, week: str) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "offline-reflect-llm",
        *common_wrapper_args(args),
        "--",
        "--scope",
        "weekly",
        "--week",
        week,
        "--weekly-window",
        "all-history",
        "--weekly-source",
        "daily",
        "--weekly-group-by",
        "topic",
        "--backfill-missing-daily",
        "--mode",
        "submit",
        "--prefilter",
        args.prefilter,
        *offline_reflect_llm_args(args),
    ]


def refresh_v2_cards(log_fh) -> None:
    """Run the default V2 rebuild after daily/weekly consolidation.

    V2 rebuild is deterministic and does not retain raw SQLite or call an LLM.
    Since the rebuild script now fails closed and requires an explicit publish
    confirmation string, scheduled maintenance must pass that string here;
    otherwise the cron job would stop at local proposal generation even after
    all gates pass.
    """
    if not V2_REBUILD_SCRIPT.exists():
        msg = f"[{iso_now()}] V2 rebuild skipped; missing script: {V2_REBUILD_SCRIPT}\n"
        print(msg, end="", flush=True)
        log_fh.write(msg); log_fh.flush()
        return
    run([
        sys.executable,
        str(V2_REBUILD_SCRIPT),
        "--mode",
        "publish",
        "--confirm-publish",
        V2_PUBLISH_CONFIRM,
    ], log_fh)


def daily(args: argparse.Namespace, log_fh) -> str:
    day = resolve_daily_date(args.date_mode)
    print(f"[{iso_now()}] DAILY target_date={day} profile={args.llm_profile or 'default'}", flush=True)
    log_fh.write(f"[{iso_now()}] DAILY target_date={day} profile={args.llm_profile or 'default'}\n")
    log_fh.flush()

    status(log_fh)
    run(
        [
            sys.executable,
            str(SCRIPT),
            "sqlite-import-llm",
            *common_wrapper_args(args),
            "--",
            "--mode",
            "submit",
            "--group-by",
            "day-topic",
            "--prefilter",
            args.prefilter,
        ],
        log_fh,
    )
    run(
        [
            sys.executable,
            str(SCRIPT),
            "offline-reflect-llm",
            *common_wrapper_args(args),
            "--",
            "--scope",
            "daily",
            "--date",
            day,
            "--daily-source",
            "facts",
            "--mode",
            "submit",
            "--prefilter",
            args.prefilter,
        ],
        log_fh,
    )
    refresh_v2_cards(log_fh)
    status(log_fh)
    return day


def weekly(args: argparse.Namespace, log_fh) -> str:
    week = resolve_week(args.week_mode)
    print(f"[{iso_now()}] WEEKLY target_week={week} profile={args.llm_profile or 'default'}", flush=True)
    log_fh.write(f"[{iso_now()}] WEEKLY target_week={week} profile={args.llm_profile or 'default'}\n")
    log_fh.flush()

    status(log_fh)
    # If another process posted retain work just before this task, wait rather than racing the worker.
    run([sys.executable, str(SCRIPT), "wait-queue", "--poll", str(args.poll), "--timeout", str(args.timeout)], log_fh)
    budget_report = run_capture_json(weekly_budget_cmd(args, week), log_fh)
    setattr(args, "_weekly_budget_report", budget_report)
    print(f"[{iso_now()}] WEEKLY budget_decision={budget_report.get('budget_decision')} pending_units={budget_report.get('pending_units')} pending_chars={budget_report.get('pending_chars')}", flush=True)
    if getattr(args, "dry_run_budget_only", False):
        log_fh.write(f"[{iso_now()}] WEEKLY dry-run budget only; skip paid LLM submit and V2 publish\n")
        log_fh.flush()
        return week
    run(weekly_submit_cmd(args, week), log_fh)
    refresh_v2_cards(log_fh)
    status(log_fh)
    return week


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hindsight offline daily/weekly pipeline with a lock and logs")
    parser.add_argument("task", choices=["daily", "weekly", "both"])
    parser.add_argument("--llm-profile", help="minimax/glm/deepseek/custom; default from HINDSIGHT_OFFLINE_LLM_PROFILE or minimax")
    parser.add_argument("--date-mode", default="auto", help="auto/today/yesterday/YYYY-MM-DD; auto=yesterday before 06:00 else today")
    parser.add_argument("--week-mode", default="current", help="current/previous/YYYY-Www; Sunday schedule uses current ISO week")
    parser.add_argument("--prefilter", default="safe", choices=["safe", "balanced", "strict"])
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=0, help="0 means no timeout")
    parser.add_argument("--weekly-budget-max-pending-units", type=int, default=12, help="Fail closed before paid LLM if regular weekly would process more pending units")
    parser.add_argument("--weekly-budget-max-pending-chars", type=int, default=500000, help="Fail closed before paid LLM if regular weekly pending chars exceeds this")
    parser.add_argument("--dry-run-budget-only", action="store_true", help="For weekly/both: run read-only weekly budget check and skip paid LLM submit/V2 publish")
    parser.add_argument("--lock-timeout", type=int, default=21600, help="seconds to wait for pipeline lock")
    args = parser.parse_args()

    if not SCRIPT.exists():
        raise SystemExit(f"missing wrapper script: {SCRIPT}")

    task_for_log = args.task
    log_path = open_log(task_for_log)
    print(f"log_path={log_path}", flush=True)

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w", encoding="utf-8") as lock_fh, open(log_path, "a", encoding="utf-8") as log_fh:
        start = time.time()
        while True:
            try:
                fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - start > args.lock_timeout:
                    raise SystemExit(f"could not acquire lock within {args.lock_timeout}s: {LOCK_PATH}")
                print(f"[{iso_now()}] waiting for lock {LOCK_PATH}", flush=True)
                time.sleep(30)

        summary: dict[str, object] = {
            "started_at": iso_now(),
            "task": args.task,
            "log_path": str(log_path),
            "llm_profile": args.llm_profile or "default",
            "prefilter": args.prefilter,
            "poll": args.poll,
            "timeout": args.timeout,
            "dry_run_budget_only": bool(args.dry_run_budget_only),
        }
        try:
            if should_run_daily(args):
                summary["daily_date"] = daily(args, log_fh)
            if should_run_weekly(args):
                summary["weekly_week"] = weekly(args, log_fh)
                if getattr(args, "_weekly_budget_report", None):
                    summary["weekly_llm_budget"] = getattr(args, "_weekly_budget_report")
            summary["finished_at"] = iso_now()
            summary["status"] = "ok"
            summary_path = finalize_summary(summary, log_path=log_path, task=args.task)
            msg = "\nSUMMARY " + json.dumps(summary, ensure_ascii=False) + "\n"
            print(msg, end="", flush=True)
            print(f"summary_path={summary_path}", flush=True)
            log_fh.write(msg)
            log_fh.write(f"summary_path={summary_path}\n")
        except BaseException as e:
            summary["finished_at"] = iso_now()
            summary["status"] = "failed"
            summary["error"] = repr(e)
            summary_path = finalize_summary(summary, log_path=log_path, task=args.task)
            msg = "\nSUMMARY " + json.dumps(summary, ensure_ascii=False) + "\n"
            print(msg, end="", flush=True)
            print(f"summary_path={summary_path}", flush=True)
            log_fh.write(msg)
            log_fh.write(f"summary_path={summary_path}\n")
            raise


if __name__ == "__main__":
    main()
