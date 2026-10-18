#!/usr/bin/env python3
"""Wait for the current Hindsight consolidation op to finish, then restart API so staged DB pool cap takes effect.

This avoids interrupting an active consolidation job. It uses docker logs instead of psql because
psql may fail with `too many clients already` during heavy consolidation.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

DRAIN_SCRIPT = "/home/wyr/.hermes/scripts/hindsight_observations_drain.py"
OLD_WATCHER = "/home/wyr/.hermes/scripts/hindsight_drain_completion_watch.py"
LOG_PATH = Path("/home/wyr/.hermes/logs/hindsight-observations/pool-cap-idle-restart.log")
TMUX_DRAIN_SESSION = "hindsight-obs-drain"
MAX_WAIT_SECONDS = 4 * 3600


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], *, check: bool = True, timeout: int = 120) -> str:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"cmd failed {cmd}: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout


def docker_logs_since(since: str = "2h") -> str:
    return run(["docker", "logs", "--since", since, "hindsight"], check=False, timeout=120)


def latest_consolidation_op() -> str | None:
    out = docker_logs_since("2h")
    ops = re.findall(r"op=([0-9a-f-]{36}) type=consolidation", out)
    return ops[-1] if ops else None


def op_completed(op_id: str) -> bool:
    out = docker_logs_since("2h")
    return f"Marked async operation as completed: {op_id}" in out or f"Marked async operation as failed: {op_id}" in out


def stop_processes_containing(needle: str) -> None:
    try:
        out = run(["pgrep", "-af", needle], check=False)
    except Exception as exc:
        log(f"WARN pgrep failed for {needle}: {exc}")
        return
    me = os.getpid()
    for line in out.splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1] if len(parts) > 1 else ""
        if pid == me or "pool_cap_idle_restart.py" in cmd:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            log(f"sent SIGTERM pid={pid} cmd={cmd[:160]}")
        except ProcessLookupError:
            pass
        except Exception as exc:
            log(f"WARN failed to terminate pid={pid}: {exc}")


def wait_health(timeout_s: int = 180) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            with urlopen("http://127.0.0.1:8888/health", timeout=5) as r:
                body = r.read().decode("utf-8", errors="replace")
            if "healthy" in body and "connected" in body:
                log(f"health ok: {body}")
                return
            last = body
        except Exception as exc:
            last = repr(exc)
        time.sleep(5)
    raise RuntimeError(f"health did not recover: {last}")


def start_drain_tmux() -> None:
    run(["tmux", "kill-session", "-t", TMUX_DRAIN_SESSION], check=False)
    run(["tmux", "new-session", "-d", "-s", TMUX_DRAIN_SESSION, "python3", DRAIN_SCRIPT])
    log(f"started drain tmux session {TMUX_DRAIN_SESSION}")


def main() -> int:
    op_id = latest_consolidation_op()
    log(f"current_op={op_id or 'none'}; staged pool cap restart watcher started")

    # Stop the external drain driver so it cannot queue a new consolidation job before restart.
    stop_processes_containing(DRAIN_SCRIPT)
    stop_processes_containing(OLD_WATCHER)

    if op_id:
        start = time.time()
        while not op_completed(op_id):
            if time.time() - start > MAX_WAIT_SECONDS:
                log(f"TIMEOUT waiting for op completion: {op_id}")
                return 3
            log(f"waiting current consolidation op to finish before restart: {op_id}")
            time.sleep(60)
        log(f"current consolidation op finished: {op_id}")
    else:
        log("no active op found in logs; restarting immediately")

    log("restarting hindsight container to apply DB pool cap")
    run(["docker", "restart", "-t", "30", "hindsight"], timeout=180)
    wait_health()
    start_drain_tmux()
    log("DONE pool-cap restart completed; drain resumed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
