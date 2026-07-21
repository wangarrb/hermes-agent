#!/usr/bin/env python3
"""Argument-free wrapper for Hermes cron: runs the full daily mental-model cycle.

This script is the SINGLE entry point for Hermes cron scheduling.
It internally calls hindsight_daily_noagent.py --mental-model-daily with
the correct REAL_HOME so that cron profile redirection doesn't interfere.

Usage (from Hermes cron):
    script: daily_mental_model_wrapper.py

The wrapper:
    1. Sets HOME to the real user home (not Hermes profile home)
    2. Runs Stage A + Stage B when stale, then target-isolated smoke for every
       accepted logical model even when no evidence changed
    3. Exits with the adjudication exit code

Exit codes:
    0 = operational success (PASS, REJECT, or ESCALATE research outcome)
    1 = infra error

The child keeps 2=REJECT and 3=ESCALATE_D_REVIEW for direct callers. Cron
must not classify those fail-closed research outcomes as scheduler failures.
"""

import os
import sys
import subprocess

REAL_HOME = "/home/wyr"
SCRIPT = os.path.join(REAL_HOME, ".hermes", "scripts", "hindsight_daily_noagent.py")

def run() -> int:
    if not os.path.exists(SCRIPT):
        print(f"ERROR: script not found at {SCRIPT}", file=sys.stderr)
        return 1

    env = {
        **os.environ,
        "HOME": REAL_HOME,
        "HERMES_PROFILE": "coordinator",
        "PYTHONUNBUFFERED": "1",
    }
    result = subprocess.run(
        [sys.executable, SCRIPT, "--mental-model-daily"],
        env=env,
        capture_output=False,  # pass through to stderr/stdout
        timeout=1800,
    )
    if result.returncode in (0, 2, 3):
        print(f"MENTAL_MODEL_DAILY_OUTCOME={result.returncode}")
        return 0
    return result.returncode


def main():
    sys.exit(run())

if __name__ == "__main__":
    main()
