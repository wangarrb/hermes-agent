#!/usr/bin/env python3
"""Monitor Hindsight hermes bank async queue until drained.

每轮读取 /stats，并写入 ~/.hermes/hindsight/hermes_queue_monitor.jsonl。
达到 pending=0 且无 processing 时退出。
"""

import json
import time
from datetime import datetime
from pathlib import Path

import requests

API = "http://localhost:8888"
BANK = "hermes"
INTERVAL_SECONDS = 300
LOG_PATH = Path.home() / ".hermes" / "hindsight" / "hermes_queue_monitor.jsonl"
MAX_STALE_ROUNDS = 24  # 2 hours with no pending/completed/doc progress


def read_stats(bank):
    r = requests.get(f"{API}/v1/default/banks/{bank}/stats", timeout=30)
    r.raise_for_status()
    return json.loads(r.text.replace("\x00", ""))


def simplify(d):
    ops = d.get("operations_by_status") or {}
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "total_documents": d.get("total_documents"),
        "total_nodes": d.get("total_nodes"),
        "pending_operations": d.get("pending_operations"),
        "ops_pending": ops.get("pending", 0),
        "ops_processing": ops.get("processing", 0),
        "ops_completed": ops.get("completed", 0),
        "ops_failed": ops.get("failed", 0),
        "last_consolidated_at": d.get("last_consolidated_at"),
    }


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    last_key = None
    stale = 0
    print(f"Monitoring Hindsight bank={BANK}; log={LOG_PATH}", flush=True)
    while True:
        try:
            rec = simplify(read_stats(BANK))
        except Exception as e:
            rec = {"time": datetime.now().isoformat(timespec="seconds"), "error": repr(e)}

        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps(rec, ensure_ascii=False), flush=True)

        if not rec.get("error"):
            pending = int(rec.get("pending_operations") or 0)
            processing = int(rec.get("ops_processing") or 0)
            if pending == 0 and processing == 0:
                print("DONE: hermes queue drained", flush=True)
                return

            key = (rec.get("total_documents"), rec.get("total_nodes"), rec.get("pending_operations"), rec.get("ops_completed"))
            if key == last_key:
                stale += 1
            else:
                stale = 0
            last_key = key
            if stale >= MAX_STALE_ROUNDS:
                print(f"STALE: no progress for {MAX_STALE_ROUNDS * INTERVAL_SECONDS}s", flush=True)
                return

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
