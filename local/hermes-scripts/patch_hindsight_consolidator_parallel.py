#!/usr/bin/env python3
"""Apply the local parallel Hindsight consolidator patch to the running container.

This patch keeps recall/search, LLM calls, and DB writes independently bounded:
- HINDSIGHT_API_CONSOLIDATION_RECALL_MAX_CONCURRENT limits recall fanout.
- HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT limits provider LLM calls.
- HINDSIGHT_API_CONSOLIDATION_PARALLEL_BATCHES limits local batch tasks.
- observation writes and source consolidated_at commits are serialized.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PATCH_CANDIDATES = [
    Path.home() / ".hermes" / "patches" / "hindsight-consolidator-parallel" / "consolidator.py",
    Path(__file__).with_name("hindsight_consolidator_parallel_patched.py"),
]
TARGET = "/app/api/hindsight_api/engine/consolidation/consolidator.py"
CONTAINER = "hindsight"


def patched_source() -> Path:
    for candidate in PATCH_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(p) for p in PATCH_CANDIDATES)
    raise FileNotFoundError(f"patched consolidator missing; searched: {searched}")


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True)


def main() -> int:
    try:
        patched = patched_source()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    # Compile host-side first.
    proc = run([sys.executable, "-m", "py_compile", str(patched)])
    if proc.returncode != 0:
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
        return proc.returncode

    # Container must exist/running enough for docker cp.
    proc = run(["docker", "inspect", CONTAINER])
    if proc.returncode != 0:
        print(proc.stderr.strip() or proc.stdout.strip(), file=sys.stderr)
        return proc.returncode

    backup = Path.home() / ".hermes" / "patches" / "hindsight-consolidator-parallel" / "last-container-consolidator.py.bak"
    proc = run(["docker", "cp", f"{CONTAINER}:{TARGET}", str(backup)])
    if proc.returncode != 0:
        print("WARNING: could not save in-container backup", proc.stderr.strip(), file=sys.stderr)

    proc = run(["docker", "cp", str(patched), f"{CONTAINER}:{TARGET}"])
    if proc.returncode != 0:
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
        return proc.returncode

    proc = run(["docker", "exec", CONTAINER, "python", "-m", "py_compile", TARGET])
    if proc.returncode != 0:
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
        return proc.returncode

    print(f"applied parallel consolidator patch to {CONTAINER}:{TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
