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

    def make_inheritable(self) -> int:
        """Mark the lock FD inheritable for ``os.execve``.  Returns the FD."""
        os.set_inheritable(self._fd, True)
        return self._fd

    def make_non_inheritable(self) -> None:
        """Revert FD to close-on-exec after a failed or completed reload."""
        os.set_inheritable(self._fd, False)

    @classmethod
    def adopt_inherited(
        cls, identity: WatcherIdentity, fd: int,
    ) -> "WatcherLock":
        """Adopt an inherited lock FD after ``os.execve``.

        Strictly verifies, before adoption (any mismatch raises
        ``RuntimeError`` and the caller must fail closed — never fall back to
        :meth:`acquire` while the inherited FD may still hold the kernel lock):

        1. The FD is a valid open file descriptor.
        2. ``fstat`` dev/ino of the FD matches the expected ``watcher.lock``.
        3. ``/proc/self/fd/<fd>`` resolves to the same lock path.
        4. The kernel lock is still held on the FD.
        5. Lock metadata matches the identity digest, this PID and this
           process's proc start-time.
        """
        if fd < 0:
            raise RuntimeError(f"invalid inherited lock FD {fd}")

        # Expected lock layout — same as acquire()
        root = runtime_root()
        lock_dir = root / identity.digest
        lock_path = lock_dir / "watcher.lock"
        meta_path = lock_dir / "metadata.json"

        # 1) FD is a valid open file
        try:
            fd_stat = os.fstat(fd)
        except OSError as exc:
            raise RuntimeError(f"inherited lock FD {fd} is not open: {exc}") from exc

        # 2) FD points at the expected lock file (dev/ino identity)
        try:
            expected_stat = lock_path.stat()
        except OSError as exc:
            raise RuntimeError(f"expected lock path missing: {lock_path}: {exc}") from exc
        if (fd_stat.st_dev, fd_stat.st_ino) != (
            expected_stat.st_dev, expected_stat.st_ino,
        ):
            raise RuntimeError(
                f"inherited FD {fd} does not point at {lock_path} "
                f"(dev/ino mismatch: fd={fd_stat.st_dev}:{fd_stat.st_ino} "
                f"expected={expected_stat.st_dev}:{expected_stat.st_ino})"
            )

        # 3) /proc/self/fd/<fd> resolves to the same path
        try:
            resolved = Path(f"/proc/self/fd/{fd}").resolve()
        except OSError as exc:
            raise RuntimeError(f"cannot resolve /proc/self/fd/{fd}: {exc}") from exc
        if resolved != lock_path.resolve():
            raise RuntimeError(
                f"/proc/self/fd/{fd} -> {resolved}, expected {lock_path.resolve()}"
            )

        # 4) The kernel lock is still held on this FD
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            raise RuntimeError(f"inherited lock FD {fd} is not held: {exc}") from exc

        # 5) Metadata matches identity / PID / proc start-time
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read lock metadata {meta_path}: {exc}") from exc
        if meta.get("identity") != identity.digest:
            raise RuntimeError(
                f"lock metadata identity mismatch: {meta.get('identity')} != {identity.digest}"
            )
        if meta.get("pid") != os.getpid():
            raise RuntimeError(
                f"lock metadata pid {meta.get('pid')} != current pid {os.getpid()}"
            )
        if meta.get("proc_start_time") != _proc_start_time(os.getpid()):
            raise RuntimeError(
                "lock metadata proc_start_time mismatch with current process"
            )

        # Re-set to non-inheritable
        os.set_inheritable(fd, False)

        return cls(identity, fd, lock_path, meta_path)


# ── Atomic file helpers ─────────────────────────────────────────────────────

def _atomic_write_json(path: Path, data: dict[str, Any], mode: int = 0o600) -> None:
    """Write JSON atomically with mode ``mode``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True))
    os.chmod(str(tmp), mode)
    tmp.rename(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ── ReloadRequest ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReloadRequest:
    """Request for a watcher to self-exec reload at the next safe boundary."""
    nonce: str
    identity_digest: str
    owner_pid: int
    owner_start_time: int
    request_time: int
    code_revision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "nonce": self.nonce,
            "identity": self.identity_digest,
            "owner_pid": self.owner_pid,
            "owner_start_time": self.owner_start_time,
            "request_time": self.request_time,
            "code_revision": self.code_revision,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReloadRequest":
        return cls(
            nonce=str(d["nonce"]),
            identity_digest=str(d["identity"]),
            owner_pid=int(d["owner_pid"]),
            owner_start_time=int(d["owner_start_time"]),
            request_time=int(d["request_time"]),
            code_revision=str(d.get("code_revision", "")),
        )


def reload_request_path(root: Path, identity_digest: str) -> Path:
    return root / identity_digest / "reload_request.json"


def write_reload_request(root: Path, request: ReloadRequest) -> Path:
    path = reload_request_path(root, request.identity_digest)
    _atomic_write_json(path, request.to_dict())
    return path


def read_reload_request(root: Path, identity_digest: str) -> ReloadRequest | None:
    data = _read_json(reload_request_path(root, identity_digest))
    if data is None:
        return None
    return ReloadRequest.from_dict(data)


def delete_reload_request(root: Path, identity_digest: str) -> None:
    try:
        reload_request_path(root, identity_digest).unlink()
    except FileNotFoundError:
        pass


# ── ReloadHandoff ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReloadHandoff:
    """Snapshot of active claim state for preservation across self-exec."""
    nonce: str
    identity_digest: str
    task_id: str
    run_id: int
    generation: int
    claim_lock: str
    worker_pid: int
    original_pid: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "nonce": self.nonce,
            "identity": self.identity_digest,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "generation": self.generation,
            "claim_lock": self.claim_lock,
            "worker_pid": self.worker_pid,
            "original_pid": self.original_pid,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReloadHandoff":
        return cls(
            nonce=str(d["nonce"]),
            identity_digest=str(d["identity"]),
            task_id=str(d["task_id"]),
            run_id=int(d["run_id"]),
            generation=int(d["generation"]),
            claim_lock=str(d["claim_lock"]),
            worker_pid=int(d["worker_pid"]),
            original_pid=int(d["original_pid"]),
        )


def reload_handoff_path(root: Path, identity_digest: str) -> Path:
    return root / identity_digest / "reload_handoff.json"


def write_reload_handoff(root: Path, handoff: ReloadHandoff) -> Path:
    path = reload_handoff_path(root, handoff.identity_digest)
    _atomic_write_json(path, handoff.to_dict())
    return path


def read_reload_handoff(root: Path, identity_digest: str) -> ReloadHandoff | None:
    data = _read_json(reload_handoff_path(root, identity_digest))
    if data is None:
        return None
    return ReloadHandoff.from_dict(data)


def delete_reload_handoff(root: Path, identity_digest: str) -> None:
    try:
        reload_handoff_path(root, identity_digest).unlink()
    except FileNotFoundError:
        pass


# ── ReloadACK ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReloadACK:
    """Acknowledgement that a watcher reloaded (or failed to reload).

    ``ok=True`` means the watcher successfully adopted the inherited lock and
    reached the post-reload health requirement (idle: one healthy DB tick;
    active: DB claim equality + one heartbeat).  ``ok=False`` carries a
    machine-readable ``error`` reason (e.g. ``preflight``, ``execve_failed``,
    ``owner_mismatch``) so the CLI can stop rolling immediately instead of
    timing out.  ``identity_digest`` pins the ACK to the exact watcher
    identity that was asked to reload.
    """
    nonce: str
    pid: int
    proc_start_time: int
    code_revision: str
    ok: bool = True
    error: str = ""
    identity_digest: str = ""
    task_id: str = ""
    run_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "nonce": self.nonce,
            "pid": self.pid,
            "proc_start_time": self.proc_start_time,
            "code_revision": self.code_revision,
            "ok": self.ok,
            "error": self.error,
            "identity": self.identity_digest,
            "task_id": self.task_id,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReloadACK":
        return cls(
            nonce=str(d["nonce"]),
            pid=int(d["pid"]),
            proc_start_time=int(d["proc_start_time"]),
            code_revision=str(d.get("code_revision", "")),
            ok=bool(d.get("ok", True)),
            error=str(d.get("error", "")),
            identity_digest=str(d.get("identity", "")),
            task_id=str(d.get("task_id", "")),
            run_id=int(d.get("run_id", 0)),
        )


def reload_ack_path(root: Path, identity_digest: str) -> Path:
    return root / identity_digest / "reload_ack.json"


def write_reload_ack(root: Path, identity_digest: str, ack: ReloadACK) -> Path:
    path = reload_ack_path(root, identity_digest)
    _atomic_write_json(path, ack.to_dict())
    return path


def read_reload_ack(root: Path, identity_digest: str) -> ReloadACK | None:
    data = _read_json(reload_ack_path(root, identity_digest))
    if data is None:
        return None
    return ReloadACK.from_dict(data)


def delete_reload_ack(root: Path, identity_digest: str) -> None:
    try:
        reload_ack_path(root, identity_digest).unlink()
    except FileNotFoundError:
        pass


def cleanup_reload_state(root: Path, identity_digest: str) -> None:
    """Remove all reload state files for an identity after successful ACK."""
    delete_reload_request(root, identity_digest)
    delete_reload_handoff(root, identity_digest)
    delete_reload_ack(root, identity_digest)
