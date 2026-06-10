#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urlrequest

PSQL = "/home/wyr/.hindsight-docker/installation/18.1.0/bin/psql"
BANK = "hermes"
CONTAINER = "hindsight"
DRAIN = "/home/wyr/.hermes/scripts/hindsight_observations_drain.py"
LOG_DIR = Path.home() / ".hermes" / "logs" / "hindsight-observations"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG = LOG_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-tuning-restart-watcher.log"
MAX_WAIT_SECONDS = 3 * 3600


def log(msg: str, **fields: object) -> None:
    row = {"ts": datetime.now(timezone.utc).isoformat(), "msg": msg, **fields}
    line = json.dumps(row, ensure_ascii=False)
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def psql(sql: str) -> str:
    last = ""
    for attempt in range(12):
        proc = subprocess.run(
            [PSQL, "-h", "/tmp", "-p", "5432", "-U", "hindsight", "-d", "hindsight", "-q", "-t", "-A", "-F", "\t", "-c", sql],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
        last = proc.stderr.strip() or proc.stdout.strip()
        if "too many clients already" in last or "connection to server" in last:
            time.sleep(min(5 + attempt * 5, 60))
            continue
        raise RuntimeError(last)
    raise RuntimeError(last)


def active_counts() -> dict[str, int]:
    raw = psql(
        """
        select status||':'||operation_type, count(*)
        from async_operations
        where bank_id='hermes'
          and status in ('pending','processing')
        group by 1
        order by 1;
        """
    )
    counts: dict[str, int] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        k, v = line.split("\t")
        counts[k] = int(v)
    return counts


def snapshot() -> dict[str, object]:
    raw = psql(
        """
        select
          count(*) filter(where fact_type='observation') as observations,
          count(*) filter(where fact_type in ('experience','world') and consolidated_at is null and consolidation_failed_at is null) as unconsolidated_base,
          count(*) filter(where fact_type in ('experience','world') and consolidation_failed_at is not null) as failed_base
        from memory_units where bank_id='hermes';
        """
    )
    parts = (raw or "0\t0\t0").split("\t")
    return {"observations": int(parts[0] or 0), "unconsolidated_base": int(parts[1] or 0), "failed_base": int(parts[2] or 0)}


def wait_health() -> None:
    for i in range(60):
        try:
            with urlrequest.urlopen("http://127.0.0.1:8888/health", timeout=5) as r:
                data = json.loads(r.read().decode("utf-8"))
            if data.get("status") == "healthy":
                log("health_ok", health=data)
                return
        except Exception as exc:
            if i % 5 == 0:
                log("health_wait", error=str(exc))
        time.sleep(2)
    raise RuntimeError("Hindsight health check did not become healthy")


def main() -> int:
    log("watcher_start", log_path=str(LOG))
    start = time.time()
    while True:
        try:
            counts = active_counts()
            snap = snapshot()
            log("poll", active=counts, snapshot=snap)
            if not counts:
                break
        except Exception as exc:
            log("poll_failed", error=str(exc))
        if time.time() - start > MAX_WAIT_SECONDS:
            log("timeout", seconds=MAX_WAIT_SECONDS)
            return 2
        time.sleep(30)

    log("idle_reached_restart_container")
    subprocess.run(["sg", "docker", "-c", f"docker restart -t 30 {CONTAINER}"], check=True)
    wait_health()
    log("restart_done_exec_drain", drain=DRAIN)
    os.execvp("python3", ["python3", DRAIN])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
