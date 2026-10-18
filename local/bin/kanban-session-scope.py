#!/usr/bin/env python3
"""Print the latest saved session ID for one exact workspace."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plugins.kanban.session_scope import latest_hermes_session  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="backend", required=True)
    hermes = subparsers.add_parser("hermes")
    hermes.add_argument("--db", required=True)
    hermes.add_argument("--workspace", required=True)
    args = parser.parse_args()

    session_id = latest_hermes_session(args.db, args.workspace)
    if not session_id:
        return 1
    print(session_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
