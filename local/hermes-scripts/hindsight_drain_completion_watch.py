#!/usr/bin/env python3
"""
Watch Hindsight observation/consolidation drain through the existing monitor JSONL.

This watcher deliberately avoids opening new PostgreSQL connections during heavy drain,
because the Hindsight API pool can temporarily occupy ~100 connections and external
psql/monitor probes may hit `too many clients already`.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time
from datetime import datetime

LOG_PATH = pathlib.Path("/home/wyr/.hermes/logs/hindsight-observations/20260511-monitor-live.jsonl")
POLL_SECONDS = 60
STALE_SECONDS = 10 * 60
REPORT_EVERY_SECONDS = 10 * 60
MAX_SECONDS = 48 * 3600


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def latest_snapshot() -> tuple[dict | None, float | None]:
    if not LOG_PATH.exists():
        return None, None
    mtime = LOG_PATH.stat().st_mtime
    try:
        lines = LOG_PATH.read_text(errors="replace").splitlines()
    except Exception as exc:
        print(f"{now()} WARN read monitor log failed: {exc}", flush=True)
        return None, mtime
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line), mtime
        except Exception:
            continue
    return None, mtime


def drain_process_alive() -> bool:
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", r"hindsight_observations_drain\.py"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except subprocess.CalledProcessError:
        return False


def fmt(snapshot: dict, mtime: float | None) -> str:
    ops = snapshot.get("ops") or {}
    stale = "?" if mtime is None else f"{time.time() - mtime:.0f}s"
    return (
        f"ts={snapshot.get('ts')} "
        f"unconsolidated_base={snapshot.get('unconsolidated_base')} "
        f"observations={snapshot.get('observations')} "
        f"failed_base={snapshot.get('failed_base')} "
        f"processing={ops.get('processing:consolidation', 0)} "
        f"completed={ops.get('completed:consolidation', 0)} "
        f"last_obs={snapshot.get('last_observation')} "
        f"log_age={stale}"
    )


def main() -> int:
    print(f"{now()} START Hindsight drain completion watcher", flush=True)
    start = time.time()
    last_report = 0.0
    last_key = None
    stale_reported = False

    while True:
        elapsed = time.time() - start
        if elapsed > MAX_SECONDS:
            print(f"{now()} TIMEOUT after {MAX_SECONDS}s", flush=True)
            return 4

        snapshot, mtime = latest_snapshot()
        alive = drain_process_alive()
        mtime_age = None if mtime is None else time.time() - mtime

        if snapshot is None:
            if time.time() - last_report >= REPORT_EVERY_SECONDS:
                print(f"{now()} WAIT no valid monitor snapshot yet; drain_alive={alive}", flush=True)
                last_report = time.time()
            time.sleep(POLL_SECONDS)
            continue

        uncon = snapshot.get("unconsolidated_base")
        failed = snapshot.get("failed_base") or 0
        obs = snapshot.get("observations")
        ops = snapshot.get("ops") or {}
        key = (uncon, failed, obs, ops.get("processing:consolidation"), ops.get("completed:consolidation"))
        should_report = key != last_key or (time.time() - last_report >= REPORT_EVERY_SECONDS)
        if should_report:
            print(f"{now()} STATUS {fmt(snapshot, mtime)} drain_alive={alive}", flush=True)
            last_key = key
            last_report = time.time()
            stale_reported = False

        if failed:
            print(f"{now()} ALERT failed_base={failed}; stop watching for manual inspection", flush=True)
            return 3
        if uncon == 0:
            print(f"{now()} DONE backlog cleared: {fmt(snapshot, mtime)}", flush=True)
            return 0
        if not alive:
            print(f"{now()} ALERT drain process is not alive but backlog is not zero: {fmt(snapshot, mtime)}", flush=True)
            return 2
        if mtime_age is not None and mtime_age > STALE_SECONDS and not stale_reported:
            print(
                f"{now()} WAIT monitor log stale for {mtime_age:.0f}s; likely DB pool pressure; continuing without DB probes",
                flush=True,
            )
            stale_reported = True

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
