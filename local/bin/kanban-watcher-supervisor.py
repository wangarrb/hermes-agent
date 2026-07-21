#!/usr/bin/env python3
"""Kanban watcher supervisor — single process, monitors all watcher children.

Scans for ``--watch-child`` kanban listener processes (all listener types),
records their cmdline, and restarts them if they exit.  Designed to be
launched once per zellij session (e.g. by ``start-kanban.sh``) and run as
a daemon thread.

Usage:
    kanban-watcher-supervisor.py --session kanban-egomotion4d [--poll-s 30]

Detection: finds processes whose cmdline contains ``--watch-child`` plus
any known listener script (hermes-kanban-role-context-listener.py /
deepseek_kanban_interactive.py / codex_kanban_interactive.py /
reasonix_kanban_interactive.py / claude_kanban_interactive.py).
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

WATCHER_PATTERNS = [
    "hermes-kanban-role-context-listener.py",
    "deepseek_kanban_interactive.py",
    "codex_kanban_interactive.py",
    "reasonix_kanban_interactive.py",
    "claude_kanban_interactive.py",
]
WATCH_CHILD_FLAG = "--watch-child"


def _read_cmdline(pid: int) -> list[str] | None:
    """Read /proc/<pid>/cmdline and return as a list of args."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
        if not raw:
            return None
        return raw.rstrip(b"\x00").split(b"\x00")
    except (OSError, ProcessLookupError):
        return None


def _is_watcher(cmdline: list[str]) -> bool:
    """Check if a cmdline is a kanban watcher child process."""
    joined = " ".join(cmdline)
    if WATCH_CHILD_FLAG not in joined:
        return False
    return any(p in joined for p in WATCHER_PATTERNS)


def _is_conda_run_wrapper(cmdline: list[str]) -> bool:
    """True only for a real ``conda run`` layer, not Miniconda's Python."""
    for index, arg in enumerate(cmdline[:4]):
        if Path(arg).name == "conda":
            return index + 1 < len(cmdline) and cmdline[index + 1] == "run"
    return False


def _process_alive(pid: int) -> bool:
    """Check if a process is still alive (not zombie, not dead)."""
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            stat = f.read().split()
        # state is the 3rd field (index 2)
        state = stat[2]
        return state not in ("Z",)  # zombie = dead
    except (OSError, IndexError, ProcessLookupError):
        return False


_WATCHER_KEY_FLAGS = (
    "--board",
    "--profile",
    "--zellij-session",
    "--zellij-pane-id",
)
_SUPERVISOR_FALLBACK_ENV = "HERMES_KANBAN_SUPERVISOR_FALLBACK"


def _watcher_key(cmdline: list[str]) -> tuple[str, ...] | None:
    """Return the logical identity of a watcher, or ``None`` if incomplete."""
    values: list[str] = []
    for flag in _WATCHER_KEY_FLAGS:
        try:
            index = cmdline.index(flag)
            value = cmdline[index + 1]
        except (ValueError, IndexError):
            return None
        values.append(value)
    return tuple(values)


def _has_live_replacement(
    current: dict[int, list[str]],
    dead_pid: int,
    dead_cmdline: list[str],
) -> bool:
    """Whether another live process already owns the dead watcher's identity."""
    key = _watcher_key(dead_cmdline)
    if key is None:
        return False
    return any(
        pid != dead_pid and _watcher_key(cmdline) == key
        for pid, cmdline in current.items()
    )


def _read_process_identity(pid: int) -> tuple[int, int] | None:
    """Return ``(ppid, start_time)`` from procfs for PID-reuse-safe checks."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return int(fields[3]), int(fields[21])
    except (OSError, IndexError, ValueError):
        return None


def _duplicate_cleanup_candidates(
    current: dict[int, list[str]],
    *,
    parent_by_pid: dict[int, int],
    supervisor_pid: int,
    supervisor_spawned_pids: set[int] | None = None,
) -> list[int]:
    """Select only supervisor-owned duplicate watchers for cleanup."""
    groups: dict[tuple[str, ...], list[int]] = {}
    for pid, cmdline in current.items():
        key = _watcher_key(cmdline)
        if key is not None:
            groups.setdefault(key, []).append(pid)

    candidates: list[int] = []
    marked = supervisor_spawned_pids or set()
    for pids in groups.values():
        if len(pids) < 2:
            continue
        owned = sorted(
            pid
            for pid in pids
            if parent_by_pid.get(pid) == supervisor_pid or pid in marked
        )
        foreign = [pid for pid in pids if pid not in owned]
        if foreign:
            candidates.extend(owned)
        elif len(owned) > 1:
            candidates.extend(owned[1:])
    return sorted(candidates)


def _terminate_owned_duplicate(
    pid: int,
    *,
    expected_key: tuple[str, ...],
    supervisor_pid: int,
    timeout_s: float = 2.0,
) -> bool:
    """Gracefully stop an exact supervisor child, escalating only if unchanged."""
    cmdline = _read_cmdline(pid)
    identity = _read_process_identity(pid)
    if cmdline is None or identity is None:
        return False
    decoded = [arg.decode("utf-8", errors="replace") for arg in cmdline]
    if (
        identity[0] != supervisor_pid
        or not _is_watcher(decoded)
        or _watcher_key(decoded) != expected_key
    ):
        return False

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _process_alive(pid):
            return True
        time.sleep(0.05)

    if _read_process_identity(pid) != identity:
        return True
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return True


def discover_watchers() -> dict[int, list[str]]:
    """Find all running watcher child processes. Returns {pid: cmdline_list}."""
    result: dict[int, list[str]] = {}
    proc_dir = Path("/proc")
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        cmdline = _read_cmdline(pid)
        if cmdline is None:
            continue
        cmdline_str = [a.decode("utf-8", errors="replace") for a in cmdline]
        if _is_watcher(cmdline_str):
            result[pid] = cmdline_str
    return result


def _exec_str(cmd: list[str]) -> str:
    """Decode cmdline bytes to str list."""
    return [c.decode("utf-8", errors="replace") if isinstance(c, bytes) else c for c in cmd]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kanban watcher supervisor")
    parser.add_argument("--session", default="", help="Zellij session name (for logging only)")
    parser.add_argument("--poll-s", type=float, default=30.0, help="Poll interval in seconds")
    parser.add_argument("--max-restarts", type=int, default=5, help="Max restarts per process before giving up")
    parser.add_argument("--restart-delay-s", type=float, default=5.0, help="Delay before restarting a crashed watcher")
    parser.add_argument("--log-dir", default="/home/wyr/.hermes/hermes-agent/kanban_logs/egomotion4d", help="Log directory")
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "watcher-supervisor.log"

    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    session_label = args.session or "unknown"
    log(f"supervisor started: session={session_label} poll={args.poll_s}s")

    # Track: {pid: {"cmdline": [...], "restarts": N, "env": {...}}}
    tracked: dict[int, dict] = {}
    restart_counts: dict[str, int] = {}  # key=profile, value=restart count

    while True:
        try:
            # Discover current watchers
            current = discover_watchers()

            # Find dead ones
            dead_pids = [pid for pid in tracked if pid not in current and not _process_alive(pid)]
            for pid in dead_pids:
                info = tracked.pop(pid)
                cmdline = info["cmdline"]
                # Extract profile for logging
                profile = "?"
                for i, arg in enumerate(cmdline):
                    if arg == "--profile" and i + 1 < len(cmdline):
                        profile = cmdline[i + 1]
                        break
                log(f"watcher died: pid={pid} profile={profile} rc=unknown; will restart")
                # Reclaim zombie if we are the parent; otherwise the launcher will reap it
                try:
                    os.waitpid(pid, os.WNOHANG)
                except (OSError, ChildProcessError):
                    pass  # not our child — launcher will reap

                if _has_live_replacement(current, pid, cmdline):
                    log(
                        f"  skipping restart: profile={profile} already has "
                        "a live watcher for the same board/session/pane"
                    )
                    continue

                # Avoid restarting conda wrapper processes — only restart the inner python
                restart_cmd = list(cmdline)
                # If this is a conda wrapper, skip restart — the launcher will handle it
                if _is_conda_run_wrapper(restart_cmd):
                    log(f"  skipping restart: pid={pid} profile={profile} is a conda wrapper (launcher manages it)")
                    continue

                # Restart
                key = profile
                restart_counts[key] = restart_counts.get(key, 0) + 1
                if restart_counts[key] > args.max_restarts:
                    log(f"  giving up on profile={profile}: exceeded max_restarts={args.max_restarts}")
                    continue

                time.sleep(args.restart_delay_s)
                try:
                    env = dict(info.get("env") or os.environ)
                    env[_SUPERVISOR_FALLBACK_ENV] = "1"
                    new_proc = subprocess.Popen(
                        restart_cmd,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=env,
                        start_new_session=True,
                    )
                    tracked[new_proc.pid] = {"cmdline": cmdline, "restarts": restart_counts[key], "env": env}
                    log(f"  restarted: pid={new_proc.pid} profile={profile} (attempt {restart_counts[key]}/{args.max_restarts})")
                except OSError as exc:
                    log(f"  restart failed: profile={profile} error={exc}")

            # Track new ones we haven't seen before
            for pid, cmdline in current.items():
                if pid not in tracked:
                    # Reconstruct env from /proc/<pid>/environ
                    env: dict[str, str] = {}
                    try:
                        with open(f"/proc/{pid}/environ", "rb") as f:
                            for line in f.read().split(b"\x00"):
                                if b"=" in line:
                                    k, v = line.split(b"=", 1)
                                    env[k.decode("utf-8", errors="replace")] = v.decode("utf-8", errors="replace")
                    except (OSError, ProcessLookupError):
                        env = dict(os.environ)

                    profile = "?"
                    for i, arg in enumerate(cmdline):
                        if arg == "--profile" and i + 1 < len(cmdline):
                            profile = cmdline[i + 1]
                            break
                    tracked[pid] = {"cmdline": cmdline, "restarts": 0, "env": env}
                    log(f"tracking: pid={pid} profile={profile}")

            parent_by_pid = {
                pid: identity[0]
                for pid in current
                if (identity := _read_process_identity(pid)) is not None
            }
            supervisor_spawned_pids = {
                pid
                for pid, info in tracked.items()
                if info.get("env", {}).get(_SUPERVISOR_FALLBACK_ENV) == "1"
            }
            for pid in _duplicate_cleanup_candidates(
                current,
                parent_by_pid=parent_by_pid,
                supervisor_pid=os.getpid(),
                supervisor_spawned_pids=supervisor_spawned_pids,
            ):
                cmdline = current.get(pid)
                key = _watcher_key(cmdline) if cmdline is not None else None
                if key is None:
                    continue
                if _terminate_owned_duplicate(
                    pid,
                    expected_key=key,
                    supervisor_pid=os.getpid(),
                ):
                    tracked.pop(pid, None)
                    log(
                        f"stopped duplicate supervisor watcher: pid={pid} "
                        f"board={key[0]} profile={key[1]} pane={key[3]}"
                    )

            time.sleep(args.poll_s)
        except Exception as exc:
            log(f"FATAL: unhandled exception in main loop: {exc}")
            import traceback as _tb
            log(f"FATAL: traceback:\n{''.join(_tb.format_exception(type(exc), exc, exc.__traceback__))}")
            log(f"FATAL: supervisor will restart watchers before exiting")
            # Try one last restart of all dead tracked watchers before dying
            for pid, info in list(tracked.items()):
                if not _process_alive(pid):
                    log(f"FATAL: orphaned watcher pid={pid} profile={info.get('profile','?')}")
            raise  # re-raise so stderr captures the crash


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("supervisor interrupted", flush=True)
        raise SystemExit(0)
