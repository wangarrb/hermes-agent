#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PSQL = "/home/wyr/.hindsight-docker/installation/18.1.0/bin/psql"
SCRIPTS = Path("/home/wyr/.hermes/scripts")
LOG_DIR = Path("/home/wyr/.hermes/logs/hindsight-observations")
LOG_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_LOG = LOG_DIR / "20260511-drain-bailian-after-idle.log"


def processing_count() -> int:
    cmd = [
        PSQL, "-h", "/tmp", "-p", "5432", "-U", "hindsight", "-d", "hindsight",
        "-q", "-t", "-A",
        "-c",
        "select count(*) from async_operations where bank_id='hermes' and status='processing';",
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return int(out or 0)


def switch_to_bailian() -> None:
    code = r'''
import os, sys
sys.path.insert(0, "/home/wyr/.hermes/scripts")
import hindsight_minimax_import as h
h.switch_mode("normal-local", allow_existing_queue=True, health_timeout_s=1200)
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def start_drain() -> None:
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPTS / "hindsight_observations_drain.py")],
        cwd=str(SCRIPTS),
        stdout=open(DRIVE_LOG, "a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"started drain pid={proc.pid} log={DRIVE_LOG}")


def main() -> int:
    print("waiting for current processing consolidation to finish before switching to bailian/glm...")
    while True:
        try:
            n = processing_count()
        except Exception as e:
            print(f"poll error: {e!r}; retry in 10s")
            time.sleep(10)
            continue
        print(f"processing={n}")
        if n == 0:
            break
        time.sleep(15)
    print("idle reached; switching Hindsight to precision remote bailian/glm mode with observations enabled...")
    switch_to_bailian()
    print("switch complete; restarting observations drain...")
    start_drain()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
