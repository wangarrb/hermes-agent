"""Kanban watcher runtime: identity, local flock, and reload handoff primitives.

This module owns watcher identity, local ``flock``-based singleton enforcement,
and the atomic state files used during safe hot-reload.  It deliberately contains
no task body, prompt, token, or credential data — only process identity, lock
metadata, and reload nonces.

Identity
--------
A watcher is uniquely identified by ``(board, profile, zellij_session, zellij_pane_id)``.
Two watchers with the same identity cannot both run claim loops; the second exits
with a log message naming the owner PID/start-time.

Lock
----
Uses ``fcntl.LOCK_EX | LOCK_NB`` on a per-identity lock file under
``$XDG_RUNTIME_DIR/hermes-kanban/watchers/`` (fallback: a mode-0700 UID-specific
directory under ``tempfile.gettempdir()``).  A metadata JSON file (mode 0600)
records PID, proc start-time, identity digest, and code revision for diagnostics.
Correctness depends on the kernel lock, not the metadata file — a stale metadata
file without a held kernel lock does not block acquisition.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


# ── WatcherIdentity ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WatcherIdentity:
    """Immutable identity for a watcher claim loop."""
    board: str
    profile: str
    session: str
    pane: str

    @classmethod
    def from_values(
        cls, board: str, profile: str, session: str, pane: str,
    ) -> "WatcherIdentity":
        if not board:
            raise ValueError("board must be non-empty")
        if not profile:
            raise ValueError("profile must be non-empty")
        if not session:
            raise ValueError("session must be non-empty")
        if not pane:
            raise ValueError("pane must be non-empty")
        return cls(board=board, profile=profile, session=session, pane=pane)

    @property
    def digest(self) -> str:
        """Stable hex digest for lock file naming."""
        raw = f"{self.board}\0{self.profile}\0{self.session}\0{self.pane}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def to_dict(self) -> dict[str, str]:
        return {
            "board": self.board,
            "profile": self.profile,
            "session": self.session,
            "pane": self.pane,
            "digest": self.digest,
        }


# ── runtime_root ─────────────────────────────────────────────────────────────

def runtime_root(env_root: Path | None = None, uid: int | None = None) -> Path:
    """Return the mode-0700 runtime root directory for Kanban watcher state.

    Prefers ``$XDG_RUNTIME_DIR``; falls back to a UID-specific directory under
    ``tempfile.gettempdir()``.  Never hardcodes ``~/.hermes``.
    """
    if uid is None:
        uid = os.getuid()

    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg and Path(xdg).exists():
        base = Path(xdg)
    else:
        base = Path(tempfile.gettempdir()) / f"hermes-kanban-{uid}"

    root = base / "hermes-kanban" / "watchers"
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except PermissionError:
        pass
    return root


def _proc_start_time(pid: int) -> int:
    """Read procfs start-time (field 22, 1-indexed field 21) for PID-reuse safety."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return int(fields[21])
    except (OSError, IndexError, ValueError):
        return 0


def _code_revision() -> str:
    """Best-effort code revision identifier (git SHA or 'unknown')."""
    return os.environ.get("HERMES_KANBAN_CODE_REVISION", "unknown")


# ── WatcherLock ──────────────────────────────────────────────────────────────

class WatcherLock:
    """Non-blocking exclusive flock for a single watcher identity.

    Use as a context manager::

        with WatcherLock.acquire(ident) as lock:
            ...

    If the lock cannot be acquired (another process holds it), raises
    ``SystemExit(1)`` after logging the owner.  The lock FD is kept alive
    for the entire context duration and released on exit.
    """

    def __init__(self, identity: WatcherIdentity, fd: int, lock_path: Path, meta_path: Path):
        self._identity = identity
        self._fd = fd
        self._lock_path = lock_path
        self._meta_path = meta_path
        self._held = True

    @property
    def is_held(self) -> bool:
        return self._held

    @property
    def fd(self) -> int:
        return self._fd

    @property
    def identity(self) -> WatcherIdentity:
        return self._identity

    @property
    def lock_path(self) -> Path:
        return self._lock_path

    @classmethod
    def acquire(cls, identity: WatcherIdentity) -> "WatcherLock":
        """Try to acquire the lock.  Exit non-zero if contended."""
        root = runtime_root()
        lock_dir = root / identity.digest
        lock_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(lock_dir, 0o700)
        except PermissionError:
            pass

        lock_path = lock_dir / "watcher.lock"
        meta_path = lock_dir / "metadata.json"

        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Another process holds the kernel lock — read its metadata
            os.close(fd)
            owner_info = {}
            try:
                owner_info = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
            owner_pid = owner_info.get("pid", "?")
            owner_start = owner_info.get("proc_start_time", "?")
            print(
                f"[watcher-lock] identity {identity.digest} already held by "
                f"PID {owner_pid} (start_time={owner_start}); exiting.",
                file=sys.stderr, flush=True,
            )
            raise SystemExit(1)

        # We hold the lock — write metadata
        pid = os.getpid()
        start_time = _proc_start_time(pid)
        metadata = {
            "pid": pid,
            "proc_start_time": start_time,
            "identity": identity.digest,
            "code_revision": _code_revision(),
        }
        # Atomic write: write to temp then rename
        tmp_meta = meta_path.with_suffix(".tmp")
        tmp_meta.write_text(json.dumps(metadata, ensure_ascii=False))
        os.chmod(str(tmp_meta), 0o600)
        tmp_meta.rename(meta_path)

        return cls(identity, fd, lock_path, meta_path)

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass

    def __enter__(self) -> "WatcherLock":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()
