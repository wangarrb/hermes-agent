#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from urllib import request as urlrequest

PSQL = "/home/wyr/.hindsight-docker/installation/18.1.0/bin/psql"
LOG = Path("/home/wyr/.hermes/logs/hindsight-observations/balanced-tuning-idle-restart.log")
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
    return run([PSQL, "-h", "/tmp", "-p", "5432", "-U", "hindsight", "-d", "hindsight", "-q", "-t", "-A", "-F", "\t", "-c", sql], timeout=30).stdout.strip()


def counts() -> dict[str, int]:
    out = psql("select status,count(*) from async_operations where bank_id='hermes' and operation_type='consolidation' group by status;")
    d: dict[str, int] = {}
    for line in out.splitlines():
        if line.strip():
            k, v = line.split("\t")
            d[k] = int(v)
    return d


def backlog() -> str:
    return psql("""
select count(*) filter(where fact_type='observation'),
       count(*) filter(where fact_type in ('world','experience') and consolidated_at is null and consolidation_failed_at is null),
       count(*) filter(where fact_type in ('world','experience') and consolidation_failed_at is not null)
from memory_units where bank_id='hermes';
""")


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
    log("START balanced tuning idle restart watcher: target batch=25, llm_batch=25, parallel_batches=2, max_job=50")
    run(["tmux", "kill-session", "-t", "hindsight-obs-drain"], check=False)
    log("stopped hindsight-obs-drain until restart, avoiding immediate retrigger on old runtime")
    last = 0.0
    while True:
        c = counts()
        if time.time() - last > 60 or c.get("processing", 0) == 0:
            log(f"counts={json.dumps(c, sort_keys=True)} backlog={backlog()}")
            last = time.time()
        if c.get("processing", 0) == 0:
            break
        time.sleep(10)
    log("no processing consolidation op; restarting container for balanced tuning")
    run(["docker", "restart", "-t", "30", "hindsight"], timeout=90)
    if not health_ok():
        log("ERROR health failed after restart")
        return 2
    run(["tmux", "kill-session", "-t", "hindsight-obs-drain"], check=False)
    run(["tmux", "new-session", "-d", "-s", "hindsight-obs-drain", "-x", "170", "-y", "45", "python3", "/home/wyr/.hermes/scripts/hindsight_observations_drain.py"])
    log("restarted hindsight-obs-drain")
    log("DONE balanced tuning applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
