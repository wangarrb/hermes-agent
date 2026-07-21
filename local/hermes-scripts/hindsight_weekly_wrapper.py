#!/usr/bin/env python3
"""Run the deterministic full Hindsight maintenance pipeline without an agent."""

from __future__ import annotations

import subprocess
import sys


PIPELINE = "/home/wyr/.hermes/scripts/hindsight_daily_noagent.py"


def build_command(python: str = sys.executable) -> list[str]:
    return [python, PIPELINE, "--mode", "full", "--include-wiki"]


def main() -> int:
    return subprocess.run(build_command(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
