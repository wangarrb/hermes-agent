#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from urllib import request as urlrequest

PSQL = "/home/wyr/.hindsight-docker/installation/18.1.0/bin/psql"
LOG = Path("/home/wyr/.hermes/logs/hindsight-observations/consolidator-patch-idle-restart.log")
LOG.parent.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], *, timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"cmd failed rc={proc.returncode}: {' '.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}")
    return proc


def psql(sql: str) -> str:
    cmd = [PSQL, "-h", "/tmp", "-p", "5432", "-U", "hindsight", "-d", "hindsight", "-q", "-t", "-A", "-F", "\t", "-c", sql]
    return run(cmd, timeout=30).stdout.strip()


def consolidation_counts() -> dict[str, int]:
    out = psql("""
select status, count(*)
from async_operations
where bank_id='hermes' and operation_type='consolidation'
group by status;
""")
    counts: dict[str, int] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        status, n = line.split("\t")
        counts[status] = int(n)
    return counts


def health_ok(timeout_s: int = 90) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urlrequest.urlopen("http://127.0.0.1:8888/health", timeout=5) as r:
                body = r.read().decode("utf-8", "replace")
            if "healthy" in body:
                log(f"health ok: {body}")
                return True
        except Exception as e:
            log(f"waiting health: {e}")
        time.sleep(3)
    return False


def main() -> int:
    log("START patch idle restart watcher; patched consolidator.py already copied, waiting for current processing op to finish")
    # Stop the external drain trigger so it cannot queue another task while we are trying to restart.
    run(["tmux", "kill-session", "-t", "hindsight-obs-drain"], check=False)
    log("stopped tmux session hindsight-obs-drain (API worker task, if any, keeps running)")

    last_report = 0.0
    while True:
        try:
            counts = consolidation_counts()
        except Exception as e:
            log(f"psql failed; retrying: {e}")
            time.sleep(10)
            continue
        processing = counts.get("processing", 0)
        pending = counts.get("pending", 0)
        if time.time() - last_report > 60 or processing == 0:
            log(f"counts={json.dumps(counts, sort_keys=True)}")
            last_report = time.time()
        if processing == 0:
            break
        time.sleep(5)

    log("no processing consolidation op; restarting container to load patched consolidator parallel_batches fix")
    run(["docker", "restart", "-t", "30", "hindsight"], timeout=90)
    if not health_ok():
        log("ERROR health did not recover after restart")
        return 2

    # Restart the drain loop under tmux. It will pick up pending/backlog with patched code.
    run(["tmux", "kill-session", "-t", "hindsight-obs-drain"], check=False)
    run([
        "tmux", "new-session", "-d", "-s", "hindsight-obs-drain", "-x", "170", "-y", "45",
        "python3", "/home/wyr/.hermes/scripts/hindsight_observations_drain.py",
    ])
    log("restarted tmux session hindsight-obs-drain")
    log("DONE patch restart applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
