#!/usr/bin/env python3
"""Tight-poll watchdog: wait for active Hindsight consolidation to finish, then restart."""

import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

API = "http://127.0.0.1:8888"
TENANT = "default"
BANK = "hermes"
POLL_S = 3
MAX_WAIT_S = 7200  # 2-hour safety timeout

def ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def op_count(status: str) -> int:
    url = f"{API}/v1/{TENANT}/banks/{BANK}/operations"
    params = {"status": status, "limit": 1, "exclude_parents": "true"}
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{qs}", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        return int(data.get("total", 0))
    except Exception as e:
        print(f"[{ts()}] operations API error: {e}", flush=True)
        return -1

def patch_loaded() -> bool:
    """Check if the container file has the new code."""
    r = subprocess.run(
        ["sg", "docker", "-c",
         "docker exec hindsight python3 -c \"from pathlib import Path; t=Path('/app/api/hindsight_api/engine/consolidation/consolidator.py').read_text(); print('OK' if 'AdaptiveLLMConcurrencyLimiter' in t else 'OLD')\""],
        capture_output=True, text=True, timeout=30
    )
    return "OK" in r.stdout

def health_ok() -> bool:
    try:
        with urllib.request.urlopen(f"{API}/health", timeout=10) as resp:
            data = json.loads(resp.read().decode())
        return data.get("status") == "healthy"
    except Exception:
        return False

def restart_container() -> bool:
    print(f"[{ts()}] RESTARTING hindsight container...", flush=True)
    r = subprocess.run(["/usr/bin/sg", "docker", "-c", "docker restart hindsight"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"[{ts()}] restart failed: {r.stderr.strip()}", flush=True)
        return False
    print(f"[{ts()}] restart command issued, waiting for healthy...", flush=True)
    for _ in range(60):
        time.sleep(2)
        if health_ok():
            print(f"[{ts()}] health OK", flush=True)
            # Give the worker a moment to import modules
            time.sleep(3)
            return True
        print(f"[{ts()}] waiting for health...", flush=True)
    print(f"[{ts()}] health timeout after restart", flush=True)
    return False

def main():
    print(f"[{ts()}] Hindsight consolidation restart watchdog started", flush=True)
    print(f"[{ts()}]   polling every {POLL_S}s, max wait {MAX_WAIT_S}s", flush=True)
    print(f"[{ts()}]   will restart when processing==0 (current round finishes)", flush=True)

    started = time.time()
    last_processing = None
    idle_count = 0

    while time.time() - started < MAX_WAIT_S:
        processing = op_count("processing")
        pending = op_count("pending")

        if processing < 0:
            time.sleep(POLL_S)
            continue

        if last_processing != processing or idle_count == 0:
            print(f"[{ts()}] processing={processing} pending={pending}", flush=True)
            last_processing = processing

        if processing == 0:
            idle_count += 1
            # Need 2 consecutive idle reads to avoid racing with a brief gap
            if idle_count >= 1:
                print(f"[{ts()}] processing==0 confirmed, restarting now", flush=True)
                if restart_container():
                    if patch_loaded():
                        print(f"[{ts()}] VERIFIED: AdaptiveLLMConcurrencyLimiter is loaded", flush=True)
                    else:
                        print(f"[{ts()}] WARNING: new code not detected after restart", flush=True)
                    print(f"[{ts()}] DONE. Container restarted with 429-adaptive parallel consolidator.", flush=True)
                return 0
        else:
            idle_count = 0

        time.sleep(POLL_S)

    print(f"[{ts()}] TIMEOUT after {MAX_WAIT_S}s, giving up", flush=True)
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
