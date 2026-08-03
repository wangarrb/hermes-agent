"""Fake watcher for testing SIGUSR1 hot-reload and lock adoption.

This fixture is used by test_kanban_watcher_runtime.py to simulate a real
watcher process that holds a lock, has an active claim, and responds to
SIGUSR1 by self-executing.  It imports the real watcher_runtime module
and uses the same lock/identity/handoff primitives.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KANBAN = REPO / "plugins" / "kanban"
sys.path.insert(0, str(KANBAN))

from watcher_runtime import (  # type: ignore[import-not-found]
    WatcherIdentity,
    WatcherLock,
)


def build_identity_from_args(args: argparse.Namespace) -> WatcherIdentity:
    return WatcherIdentity.from_values(
        args.board, args.profile, args.session, args.pane,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--pane", required=True)
    parser.add_argument("--handoff-dir", default="")
    parser.add_argument("--claim-file", default="")
    args = parser.parse_args(argv)

    identity = build_identity_from_args(args)

    try:
        lock = WatcherLock.acquire(identity)
    except SystemExit:
        print("LOCK_REJECTED", flush=True)
        return 1

    print(f"LOCK_ACQUIRED pid={os.getpid()}", flush=True)

    # Write claim fixture if requested
    if args.claim_file:
        claim_path = Path(args.claim_file)
        claim_path.parent.mkdir(parents=True, exist_ok=True)
        claim_path.write_text(json.dumps({
            "task_id": "t_fake",
            "run_id": 1,
            "generation": 1,
            "claim_lock": f"host:{os.getpid()}:fake-interactive",
            "worker_pid": os.getpid(),
        }))

    # Set up SIGUSR1 handler (just sets a flag)
    reload_requested = False

    def _handle_usr1(signum, frame):
        nonlocal reload_requested
        reload_requested = True

    signal.signal(signal.SIGUSR1, _handle_usr1)

    # Main loop — heartbeat and check for reload
    heartbeat_count = 0
    while True:
        heartbeat_count += 1
        print(f"HEARTBEAT {heartbeat_count}", flush=True)

        if reload_requested:
            print("RELOAD_REQUESTED", flush=True)
            # In a real watcher we'd do preflight + execve here.
            # For testing, just write an ACK and continue.
            if args.handoff_dir:
                ack_path = Path(args.handoff_dir) / f"ack.{identity.digest}.json"
                ack_path.write_text(json.dumps({
                    "nonce": "test-nonce",
                    "pid": os.getpid(),
                    "proc_start_time": 0,
                    "status": "ACK",
                }))
            reload_requested = False

        time.sleep(0.2)

        if heartbeat_count > 100:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
