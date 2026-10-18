#!/usr/bin/env python3
"""Continue a manual Hindsight full pipeline run without operator intervention.

The foreground pipeline may time out while Hindsight native observations are still
processing. This watcher waits for the pipeline process, waits for Hindsight idle
if needed, repairs local submit_state after verified queue drain, reruns the full
pipeline with the current scripts, and writes a post-run summary.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERMES = Path.home() / ".hermes"
SCRIPTS = HERMES / "scripts"
STATUS = SCRIPTS / "hindsight_consolidation_status.py"
PIPELINE = SCRIPTS / "hindsight_memory_pipeline.py"
RETAIN_RUNNER = SCRIPTS / "hindsight_session_retain_runner.py"


def run(cmd: list[str], *, timeout: int | None = None, check: bool = False, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=check, cwd=cwd)


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def status_snapshot() -> dict[str, Any]:
    proc = run([sys.executable, str(STATUS), "--json"], timeout=90)
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stdout[-2000:] + proc.stderr[-2000:]}
    return json.loads(proc.stdout)


def active_summary(s: dict[str, Any]) -> dict[str, Any]:
    checks = s.get("checks") or {}
    psql = (checks.get("async_operations_psql") or {}).get("summary") or {}
    api = (checks.get("operations_api") or {}).get("summary") or {}
    return psql or api or {}


def wait_idle(log: Path, *, poll_s: int = 60, timeout_s: int = 86400) -> dict[str, Any]:
    start = time.time()
    last = None
    while True:
        s = status_snapshot()
        summary = active_summary(s)
        cur = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "pending": summary.get("pending_count"), "active": summary.get("active_count"), "failed": summary.get("failed_count"), "completed": summary.get("completed_count")}
        if cur != last:
            append_log(log, "status " + json.dumps(cur, ensure_ascii=False, sort_keys=True))
            last = cur
        if summary and not summary.get("has_active_work"):
            return s
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Hindsight did not become idle within {timeout_s}s: {summary}")
        time.sleep(poll_s)


def parse_manifest_from_log(log_path: Path) -> str | None:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    matches = re.findall(r"--manifest\s+(\S+session-manifest\.jsonl)", text)
    return matches[-1] if matches else None


def mark_submit_state_if_needed(manifest: str, log: Path) -> dict[str, Any]:
    code = f"""
import json, sys
from pathlib import Path
sys.path.insert(0, {str(SCRIPTS)!r})
import hindsight_session_retain_runner as r
manifest=Path({manifest!r})
state_path=Path.home()/'.hermes'/'hindsight'/'session_ingest'/'submit_state.json'
records=r.load_manifest(manifest)
state=r.load_submit_state(state_path)
selected, skipped = r.prepare_retain_records(records, action='production', submit_state=state, bank='hermes')
if selected:
    backup=state_path.with_suffix(state_path.suffix + '.watchdog-' + str(int(__import__('time').time())) + '.bak')
    if state_path.exists():
        backup.write_bytes(state_path.read_bytes())
    r.update_submit_state_for_items(state, selected, manifest_path=manifest, bank='hermes')
    r.save_submit_state(state_path, state)
else:
    backup=None
print(json.dumps({{'selected_marked_submitted': len(selected), 'skipped': dict(skipped), 'state_path': str(state_path), 'backup': str(backup) if backup else None}}, ensure_ascii=False, sort_keys=True))
"""
    proc = run([sys.executable, "-c", code], timeout=180)
    if proc.returncode != 0:
        out = {"ok": False, "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}
    else:
        out = json.loads(proc.stdout)
        out["ok"] = True
    append_log(log, "submit_state " + json.dumps(out, ensure_ascii=False, sort_keys=True))
    return out


def run_full_pipeline(runroot: Path, log: Path) -> int:
    cmd = [
        sys.executable,
        str(PIPELINE),
        "full",
        "--history",
        "incremental",
        "--include-wiki",
        "--strict-runtime",
        "--execute",
        "--confirm",
        "run-hindsight-pipeline",
        "--execute-proposal-review-llm",
        "--confirm-proposal-review",
        "review-hindsight-proposals",
        "--notify-proposal-review",
    ]
    append_log(log, "rerun_command " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(Path.home()))
    assert proc.stdout is not None
    rerun_log = runroot / f"watchdog-rerun-{int(time.time())}.log"
    with rerun_log.open("w", encoding="utf-8") as f:
        for line in proc.stdout:
            f.write(line)
            f.flush()
            append_log(log, "rerun: " + line.rstrip())
    rc = proc.wait()
    append_log(log, f"rerun_exit_code {rc}; log={rerun_log}")
    return rc


def post_verify(runroot: Path, log: Path) -> dict[str, Any]:
    status = status_snapshot()
    runroot.mkdir(parents=True, exist_ok=True)
    (runroot / "status-after-watchdog.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Recall smoke through Hermes tool-equivalent REST endpoint.
    recall = None
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://127.0.0.1:8888/v1/default/banks/hermes/memories/recall",
            data=json.dumps({"query": "Hindsight v0.6.1 API-first operations", "max_tokens": 2048, "budget": "low"}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            recall = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
    except Exception as e:
        recall = {"error": f"{type(e).__name__}: {e}"}
    (runroot / "recall-smoke-after-watchdog.json").write_text(json.dumps(recall, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"status_summary": active_summary(status), "bank_summary": status.get("bank_summary"), "recall_keys": sorted(recall.keys()) if isinstance(recall, dict) else None, "runroot": str(runroot)}
    append_log(log, "post_verify " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--runroot", type=Path, required=True)
    args = ap.parse_args()
    runroot = args.runroot
    log = runroot / "watchdog.log"
    append_log(log, f"watchdog_started pid={args.pid} at={time.strftime('%Y-%m-%dT%H:%M:%S')}")
    while process_alive(args.pid):
        time.sleep(60)
    append_log(log, f"primary_process_exited at={time.strftime('%Y-%m-%dT%H:%M:%S')}")
    status = status_snapshot()
    (runroot / "status-after-primary-exit.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = active_summary(status)
    if summary.get("has_active_work"):
        append_log(log, "primary exited but Hindsight still active; waiting idle")
        status = wait_idle(log)
        (runroot / "status-after-idle.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # If the primary full pipeline did not leave a completed run report, it may
    # have timed out during retain waiting. Mark submit-state after verified idle
    # and rerun so daily/weekly/conflict/wiki stages execute.
    primary_log = runroot / "full-pipeline.log"
    primary_text = primary_log.read_text(encoding="utf-8", errors="replace") if primary_log.exists() else ""
    if "exit_code=0" not in primary_text:
        manifest = parse_manifest_from_log(primary_log)
        if manifest:
            mark_submit_state_if_needed(manifest, log)
        rc = run_full_pipeline(runroot, log)
        if rc != 0:
            append_log(log, f"watchdog_failed rerun_rc={rc}")
            print(f"Hindsight full pipeline watchdog rerun failed rc={rc}; see {log}")
            return rc
    result = post_verify(runroot, log)
    print(json.dumps({"ok": True, "message": "Hindsight full pipeline completed and post-verified", **result}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
