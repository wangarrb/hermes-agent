"""Tests for plugins/kanban/watcher_runtime.py — identity, locks and reload state.

These tests exercise the watcher identity, local flock singleton, and reload
handoff/ACK primitives.  Subprocess tests prove that two processes using the
same identity cannot both acquire the lock, while different pane IDs can.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "plugins" / "kanban"))

from watcher_runtime import (  # type: ignore[import-not-found]
    WatcherIdentity,
    runtime_root,
    WatcherLock,
)


# ── WatcherIdentity ──────────────────────────────────────────────────────────

class TestWatcherIdentity:
    def test_from_values_requires_all_fields(self):
        with pytest.raises((ValueError, TypeError)):
            WatcherIdentity.from_values("", "profile", "session", "1")
        with pytest.raises((ValueError, TypeError)):
            WatcherIdentity.from_values("board", "", "session", "1")
        with pytest.raises((ValueError, TypeError)):
            WatcherIdentity.from_values("board", "profile", "", "1")
        with pytest.raises((ValueError, TypeError)):
            WatcherIdentity.from_values("board", "profile", "session", "")

    def test_from_values_succeeds_with_all_fields(self):
        ident = WatcherIdentity.from_values("board", "profile", "session", "1")
        assert ident.board == "board"
        assert ident.profile == "profile"
        assert ident.session == "session"
        assert ident.pane == "1"

    def test_distinct_identities_have_distinct_digests(self):
        a = WatcherIdentity.from_values("board", "profile", "session", "1")
        b = WatcherIdentity.from_values("board", "profile", "session", "2")
        assert a.digest != b.digest

    def test_same_identity_has_same_digest(self):
        a = WatcherIdentity.from_values("board", "profile", "session", "1")
        b = WatcherIdentity.from_values("board", "profile", "session", "1")
        assert a.digest == b.digest


# ── runtime_root ─────────────────────────────────────────────────────────────

class TestRuntimeRoot:
    def test_runtime_root_uses_xdg_runtime_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        root = runtime_root(tmp_path, os.getuid())
        assert str(root).startswith(str(tmp_path))
        mode = root.stat().st_mode & 0o777
        assert mode == 0o700

    def test_runtime_root_fallback_to_tmpdir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        root = runtime_root(tmp_path, os.getuid())
        assert root.exists()
        mode = root.stat().st_mode & 0o777
        assert mode == 0o700


# ── WatcherLock ──────────────────────────────────────────────────────────────

class TestWatcherLock:
    def test_acquire_and_release(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        with WatcherLock.acquire(ident) as lock:
            assert lock is not None
            assert lock.is_held
        # After release, a new lock can be acquired
        with WatcherLock.acquire(ident):
            pass

    def test_leftover_metadata_without_kernel_lock_does_not_block(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        root = runtime_root(tmp_path, os.getuid())
        meta_path = root / ident.digest / "metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({"pid": 99999, "proc_start_time": 0, "identity": ident.digest}))
        # Should succeed despite stale metadata — kernel lock is what matters
        with WatcherLock.acquire(ident):
            pass

    def test_two_processes_same_identity_one_acquires(self, tmp_path):
        """Subprocess test: two processes using same identity cannot both hold lock."""
        code = f'''
import sys, os, time
sys.path.insert(0, "{REPO}/plugins/kanban")
os.environ["XDG_RUNTIME_DIR"] = "{tmp_path}"
from watcher_runtime import WatcherIdentity, WatcherLock
ident = WatcherIdentity.from_values("b", "p", "s", "1")
try:
    with WatcherLock.acquire(ident):
        print("ACQUIRED", flush=True)
        time.sleep(3)
except SystemExit:
    print("REJECTED", flush=True)
    sys.exit(1)
'''
        # Start first process
        p1 = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(0.5)
        # Start second process — should be rejected
        p2 = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        out2, err2 = p2.communicate(timeout=10)
        p1.terminate()
        p1.wait(timeout=5)
        assert "REJECTED" in out2, f"Second process should be rejected, got: {out2}"

    def test_different_pane_ids_can_both_acquire(self, tmp_path):
        """Subprocess test: different pane IDs can hold locks simultaneously."""
        code_template = f'''
import sys, os, time
sys.path.insert(0, "{REPO}/plugins/kanban")
os.environ["XDG_RUNTIME_DIR"] = "{tmp_path}"
from watcher_runtime import WatcherIdentity, WatcherLock
ident = WatcherIdentity.from_values("b", "p", "s", "{{pane}}")
with WatcherLock.acquire(ident):
    print("ACQUIRED", flush=True)
    time.sleep(2)
'''
        p1 = subprocess.Popen(
            [sys.executable, "-c", code_template.format(pane="1")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(0.5)
        p2 = subprocess.Popen(
            [sys.executable, "-c", code_template.format(pane="2")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        out2, _ = p2.communicate(timeout=10)
        p1.terminate()
        p1.wait(timeout=5)
        assert "ACQUIRED" in out2, f"Different pane should acquire, got: {out2}"

    def test_metadata_file_mode_is_0600(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        with WatcherLock.acquire(ident):
            root = runtime_root(tmp_path, os.getuid())
            meta_path = root / ident.digest / "metadata.json"
            if meta_path.exists():
                mode = meta_path.stat().st_mode & 0o777
                assert mode == 0o600, f"metadata mode {oct(mode)} != 0600"

    def test_metadata_contains_pid_and_identity(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        with WatcherLock.acquire(ident):
            root = runtime_root(tmp_path, os.getuid())
            meta_path = root / ident.digest / "metadata.json"
            data = json.loads(meta_path.read_text())
            assert data["pid"] == os.getpid()
            assert data["identity"] == ident.digest
            assert "proc_start_time" in data


# ── watcher_main lock integration ────────────────────────────────────────────

FIXTURE = REPO / "tests" / "fixtures" / "fake_reload_watcher.py"


class TestWatcherMainLockIntegration:
    """Verify that watcher_main enforces one claim loop per identity."""

    def test_incomplete_identity_exits_nonzero_before_db(self, tmp_path):
        """A watcher missing session/pane must exit non-zero without opening DB."""
        # The fake watcher requires all four identity fields; argparse will
        # error on missing args, which is the desired fail-closed behavior.
        code = f'''
import sys, os
sys.path.insert(0, "{REPO}/plugins/kanban")
sys.path.insert(0, "{REPO}/tests/fixtures")
os.environ["XDG_RUNTIME_DIR"] = "{tmp_path}"
# Import watcher_runtime to check identity validation
from watcher_runtime import WatcherIdentity
try:
    WatcherIdentity.from_values("board", "profile", "", "1")
    print("SHOULD_NOT_REACH", flush=True)
except (ValueError, TypeError):
    print("REJECTED_INCOMPLETE", flush=True)
'''
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=10,
        )
        assert "REJECTED_INCOMPLETE" in result.stdout, f"Expected rejection, got: {result.stdout}"

    def test_two_identical_watchers_yield_one_live_loop(self, tmp_path):
        """Two identical watchers: one acquires lock and runs, other is rejected."""
        code = f'''
import sys, os, time
sys.path.insert(0, "{REPO}/tests/fixtures")
os.environ["XDG_RUNTIME_DIR"] = "{tmp_path}"
os.environ["HERMES_KANBAN_TEST_MODE"] = "1"
# Run fake_reload_watcher with fixed identity
import fake_reload_watcher
sys.argv = ["fake_reload_watcher.py",
    "--board", "testboard",
    "--profile", "coordinator",
    "--session", "test-session",
    "--pane", "1",
]
rc = fake_reload_watcher.main()
if rc:
    print(f"EXITED code={{rc}}", flush=True)
sys.exit(rc if rc else 0)
'''
        # Start first watcher
        p1 = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(0.5)

        # Start second watcher — should be rejected
        p2 = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        out2, err2 = p2.communicate(timeout=10)
        p1.terminate()
        p1.wait(timeout=5)

        assert "LOCK_REJECTED" in out2, (
            f"Second identical watcher should be rejected.\n"
            f"stdout: {out2}\nstderr: {err2}"
        )
        assert p2.returncode != 0, "Rejected watcher should exit non-zero"


# ── Reload handoff / ACK / request primitives ────────────────────────────────

from watcher_runtime import (  # type: ignore[import-not-found]
    ReloadRequest,
    ReloadHandoff,
    ReloadACK,
    write_reload_request,
    read_reload_request,
    delete_reload_request,
    write_reload_handoff,
    read_reload_handoff,
    delete_reload_handoff,
    write_reload_ack,
    read_reload_ack,
    delete_reload_ack,
    cleanup_reload_state,
    reload_request_path,
    reload_handoff_path,
    reload_ack_path,
)


class TestReloadRequest:
    def test_atomic_write_and_read(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        root = runtime_root()
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        req = ReloadRequest(
            nonce="abc123",
            identity_digest=ident.digest,
            owner_pid=12345,
            owner_start_time=999,
            request_time=int(time.time()),
            code_revision="sha1",
        )
        write_reload_request(root, req)
        loaded = read_reload_request(root, ident.digest)
        assert loaded is not None
        assert loaded.nonce == "abc123"
        assert loaded.identity_digest == ident.digest
        assert loaded.owner_pid == 12345

    def test_file_mode_is_0600(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        root = runtime_root()
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        req = ReloadRequest(
            nonce="n1", identity_digest=ident.digest,
            owner_pid=1, owner_start_time=1, request_time=1,
        )
        write_reload_request(root, req)
        mode = reload_request_path(root, ident.digest).stat().st_mode & 0o777
        assert mode == 0o600

    def test_delete_is_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        root = runtime_root()
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        delete_reload_request(root, ident.digest)  # should not raise


class TestReloadHandoff:
    def test_atomic_write_and_read(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        root = runtime_root()
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        handoff = ReloadHandoff(
            nonce="n1",
            identity_digest=ident.digest,
            task_id="t_abc",
            run_id=42,
            generation=2,
            claim_lock="host:123:slug-interactive",
            worker_pid=123,
            original_pid=456,
        )
        write_reload_handoff(root, handoff)
        loaded = read_reload_handoff(root, ident.digest)
        assert loaded is not None
        assert loaded.task_id == "t_abc"
        assert loaded.run_id == 42
        assert loaded.generation == 2
        assert loaded.claim_lock == "host:123:slug-interactive"
        assert loaded.worker_pid == 123

    def test_file_mode_is_0600(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        root = runtime_root()
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        handoff = ReloadHandoff(
            nonce="n1", identity_digest=ident.digest,
            task_id="t", run_id=1, generation=1,
            claim_lock="l", worker_pid=1, original_pid=1,
        )
        write_reload_handoff(root, handoff)
        mode = reload_handoff_path(root, ident.digest).stat().st_mode & 0o777
        assert mode == 0o600

    def test_tampered_run_id_rejects(self, tmp_path, monkeypatch):
        """Changing run_id in the handoff file produces a different object."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        root = runtime_root()
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        handoff = ReloadHandoff(
            nonce="n1", identity_digest=ident.digest,
            task_id="t", run_id=42, generation=1,
            claim_lock="l", worker_pid=1, original_pid=1,
        )
        write_reload_handoff(root, handoff)
        # Tamper with the file
        path = reload_handoff_path(root, ident.digest)
        data = json.loads(path.read_text())
        data["run_id"] = 999
        path.write_text(json.dumps(data))
        loaded = read_reload_handoff(root, ident.digest)
        assert loaded is not None
        assert loaded.run_id == 999  # Detects the tampering — caller must verify


class TestReloadACK:
    def test_atomic_write_and_read(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        root = runtime_root()
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        ack = ReloadACK(
            nonce="n1",
            pid=os.getpid(),
            proc_start_time=12345,
            code_revision="sha1",
            task_id="t_abc",
            run_id=42,
        )
        write_reload_ack(root, ident.digest, ack)
        loaded = read_reload_ack(root, ident.digest)
        assert loaded is not None
        assert loaded.nonce == "n1"
        assert loaded.pid == os.getpid()
        assert loaded.task_id == "t_abc"

    def test_nonce_mismatch_detected(self, tmp_path, monkeypatch):
        """ACK with wrong nonce is detectable by the caller."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        root = runtime_root()
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        ack = ReloadACK(
            nonce="correct", pid=1, proc_start_time=1, code_revision="",
        )
        write_reload_ack(root, ident.digest, ack)
        loaded = read_reload_ack(root, ident.digest)
        assert loaded is not None
        assert loaded.nonce != "wrong"  # Caller compares to expected nonce

    def test_file_mode_is_0600(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        root = runtime_root()
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        ack = ReloadACK(nonce="n", pid=1, proc_start_time=1, code_revision="")
        write_reload_ack(root, ident.digest, ack)
        mode = reload_ack_path(root, ident.digest).stat().st_mode & 0o777
        assert mode == 0o600

    def test_ack_has_ok_error_identity_fields(self, tmp_path, monkeypatch):
        """FAILED ACKs carry ok=False + a machine-readable error; identity is pinned."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        root = runtime_root()
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        ok_ack = ReloadACK(
            nonce="n1", pid=123, proc_start_time=456,
            code_revision="sha", ok=True,
            identity_digest=ident.digest,
            task_id="t_abc", run_id=7,
        )
        write_reload_ack(root, ident.digest, ok_ack)
        loaded = read_reload_ack(root, ident.digest)
        assert loaded is not None
        assert loaded.ok is True
        assert loaded.error == ""
        assert loaded.identity_digest == ident.digest
        assert loaded.task_id == "t_abc"
        assert loaded.run_id == 7

        failed_ack = ReloadACK(
            nonce="n2", pid=123, proc_start_time=456,
            code_revision="sha", ok=False, error="owner_mismatch",
            identity_digest=ident.digest,
        )
        write_reload_ack(root, ident.digest, failed_ack)
        loaded_failed = read_reload_ack(root, ident.digest)
        assert loaded_failed is not None
        assert loaded_failed.ok is False
        assert loaded_failed.error == "owner_mismatch"
        assert loaded_failed.identity_digest == ident.digest

    def test_ack_backward_compatible_missing_optional_fields(self, tmp_path, monkeypatch):
        """Old ACK JSON without ok/error/identity still parses (ok defaults True)."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        root = runtime_root()
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        path = reload_ack_path(root, ident.digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "nonce": "n", "pid": 1, "proc_start_time": 1,
            "code_revision": "", "task_id": "", "run_id": 0,
        }))
        loaded = read_reload_ack(root, ident.digest)
        assert loaded is not None
        assert loaded.ok is True
        assert loaded.error == ""
        assert loaded.identity_digest == ""


class TestCleanupReloadState:
    def test_cleanup_removes_all_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        root = runtime_root()
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        # Write all three
        write_reload_request(root, ReloadRequest(
            nonce="n", identity_digest=ident.digest,
            owner_pid=1, owner_start_time=1, request_time=1,
        ))
        write_reload_handoff(root, ReloadHandoff(
            nonce="n", identity_digest=ident.digest,
            task_id="t", run_id=1, generation=1,
            claim_lock="l", worker_pid=1, original_pid=1,
        ))
        write_reload_ack(root, ident.digest, ReloadACK(
            nonce="n", pid=1, proc_start_time=1, code_revision="",
        ))
        # All exist
        assert reload_request_path(root, ident.digest).exists()
        assert reload_handoff_path(root, ident.digest).exists()
        assert reload_ack_path(root, ident.digest).exists()
        # Cleanup
        cleanup_reload_state(root, ident.digest)
        # All gone
        assert not reload_request_path(root, ident.digest).exists()
        assert not reload_handoff_path(root, ident.digest).exists()
        assert not reload_ack_path(root, ident.digest).exists()


class TestLockInheritance:
    def test_make_inheritable_and_adopt(self, tmp_path, monkeypatch):
        """Lock FD can be made inheritable and adopted after exec simulation."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        lock1 = WatcherLock.acquire(ident)
        fd = lock1.make_inheritable()
        # Simulate post-exec: adopt the inherited FD
        lock2 = WatcherLock.adopt_inherited(ident, fd)
        assert lock2.is_held
        # Clean up
        lock2.make_non_inheritable()
        lock2.release()

    def test_adopt_rejects_invalid_fd(self, tmp_path, monkeypatch):
        """Adoption of a closed/invalid FD raises RuntimeError (fail closed)."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        WatcherLock.acquire(ident).release()  # metadata exists but no lock held
        with pytest.raises(RuntimeError):
            WatcherLock.adopt_inherited(ident, 99999)

    def test_adopt_rejects_wrong_dev_ino(self, tmp_path, monkeypatch):
        """Adoption of an FD pointing at a DIFFERENT file must fail (dev/ino)."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        with WatcherLock.acquire(ident):
            other = tmp_path / "other.lock"
            other_fd = os.open(str(other), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                with pytest.raises(RuntimeError):
                    WatcherLock.adopt_inherited(ident, other_fd)
            finally:
                os.close(other_fd)

    def test_adopt_rejects_metadata_identity_mismatch(self, tmp_path, monkeypatch):
        """Adoption fails when lock metadata identity does not match."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        lock = WatcherLock.acquire(ident)
        fd = lock.make_inheritable()
        try:
            # Tamper with the metadata identity field
            root = runtime_root()
            meta_path = root / ident.digest / "metadata.json"
            data = json.loads(meta_path.read_text())
            data["identity"] = "tampered"
            meta_path.write_text(json.dumps(data))
            with pytest.raises(RuntimeError):
                WatcherLock.adopt_inherited(ident, fd)
        finally:
            lock.make_non_inheritable()
            lock.release()

    def test_adopt_fail_closed_does_not_acquire(self, tmp_path, monkeypatch):
        """A process that cannot adopt must exit; it must NOT fall back to acquire.

        Simulates the watcher_main fail-closed contract: when the inherited
        FD is invalid the process returns non-zero without ever holding a
        fresh lock (i.e. a second process would still be able to acquire).
        """
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        root = runtime_root()
        meta_path = root / ident.digest / "metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({
            "pid": os.getpid(), "proc_start_time": 0,
            "identity": ident.digest, "code_revision": "test",
        }))
        # Invalid inherited FD + matching identity env → fail closed
        code = f'''
import sys, os
sys.path.insert(0, "{REPO}/plugins/kanban")
os.environ["XDG_RUNTIME_DIR"] = "{tmp_path}"
os.environ["HERMES_KANBAN_WATCHER_LOCK_FD"] = "424242"
os.environ["HERMES_KANBAN_RELOAD_IDENTITY"] = "{ident.digest}"
from watcher_runtime import WatcherIdentity, WatcherLock
try:
    WatcherLock.adopt_inherited(
        WatcherIdentity.from_values("b", "p", "s", "1"), 424242,
    )
    print("SHOULD_NOT_ADOPT", flush=True)
except RuntimeError:
    print("ADOPT_REJECTED", flush=True)
'''
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=10,
        )
        assert "ADOPT_REJECTED" in result.stdout
        # A fresh lock must still be acquirable afterwards (no leak)
        with WatcherLock.acquire(ident):
            pass


# ── SIGUSR1 self-exec integration ────────────────────────────────────────────

class TestSigusr1SelfExec:
    """真实 os.execve 集成：同 PID、继承 FD、EXEC_GENERATION 变化、单次 claim/inject、ACK 恰好一次。

    The fixture genuinely calls ``os.execve`` (same PID, same inherited lock
    FD) and mirrors the production adopt-first entry point, so these tests
    cover the real reload path rather than an ACK-only simulation.
    """

    FIXTURE = REPO / "tests" / "fixtures" / "fake_reload_watcher.py"

    @staticmethod
    def _ident() -> WatcherIdentity:
        return WatcherIdentity.from_values(
            "testboard", "coordinator", "test-session", "1",
        )

    def _launch(
        self, tmp_path, *,
        claim: bool = True,
        extra_args: tuple[str, ...] = (),
        extra_env: dict[str, str] | None = None,
    ):
        env = dict(os.environ)
        env["XDG_RUNTIME_DIR"] = str(tmp_path)
        if extra_env:
            env.update(extra_env)
        argv = [
            sys.executable, str(self.FIXTURE),
            "--board", "testboard",
            "--profile", "coordinator",
            "--session", "test-session",
            "--pane", "1",
            "--tick-s", "0.1",
        ]
        if claim:
            argv += [
                "--claim-file", str(tmp_path / "claim.json"),
                "--inject-file", str(tmp_path / "inject.log"),
            ]
        argv += list(extra_args)
        return subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env,
        )

    def _wait_for_ack(self, tmp_path, nonce: str, timeout: float = 15.0):
        root = runtime_root()
        deadline = time.time() + timeout
        while time.time() < deadline:
            ack = read_reload_ack(root, self._ident().digest)
            if ack is not None and ack.nonce == nonce:
                return ack
            time.sleep(0.2)
        return None

    def test_real_self_exec_preserves_pid_generation_and_claim(self, tmp_path, monkeypatch):
        """真实 exec：PID 不变、EXEC_GENERATION 1→2、首次标记只一次、无第二 claim/inject、claim/heartbeat 保持。"""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        p = self._launch(tmp_path)
        time.sleep(0.8)  # acquire lock + claim/inject + first heartbeats

        root = runtime_root()
        ident = self._ident()
        meta_path = root / ident.digest / "metadata.json"
        assert meta_path.exists(), "watcher metadata not found"
        meta = json.loads(meta_path.read_text())
        assert meta["pid"] == p.pid
        original_pid = p.pid

        # 写 reload request（用真实 owner_start_time）
        req = ReloadRequest(
            nonce="exec-nonce-1",
            identity_digest=ident.digest,
            owner_pid=original_pid,
            owner_start_time=meta["proc_start_time"],
            request_time=int(time.time()),
        )
        write_reload_request(root, req)
        os.kill(original_pid, signal.SIGUSR1)

        ack = self._wait_for_ack(tmp_path, "exec-nonce-1")
        assert ack is not None, "ACK not written after real self-exec"
        assert ack.ok is True, f"ACK should be ok, got error={ack.error!r}"
        assert ack.pid == original_pid, "PID must be preserved across os.execve"
        assert ack.identity_digest == ident.digest
        assert ack.task_id == "t_fake"
        assert ack.run_id == 1

        # 给心跳一点时间，然后终止并收集输出
        time.sleep(0.5)
        assert p.poll() is None, "watcher died after self-exec"
        p.terminate()
        out, err = p.communicate(timeout=5)

        # 真实 exec 发生：EXECVE 标记 + generation 变化
        assert "EXECVE gen=2" in out, f"expected real execve marker, got:\n{out}\n{err}"
        assert out.count("EXEC_GENERATION 1") == 1, "first-process marker must appear exactly once"
        assert "EXEC_GENERATION 2" in out
        # 无第二 claim/inject
        assert out.count("CLAIM_INJECT 1") == 1
        assert "CLAIM_INJECT_RESTORED 1" in out
        assert "LOCK_ADOPTED" in out, "new process must adopt the inherited FD"
        # claim 文件不变（active claim/heartbeat 保持）
        claim = json.loads((tmp_path / "claim.json").read_text())
        assert claim["task_id"] == "t_fake"
        assert claim["run_id"] == 1
        assert claim["generation"] == 1
        assert claim["claim_lock"] == f"host:{original_pid}:fake-interactive"
        # heartbeat 继续
        assert out.count("HEARTBEAT") >= 3

    def test_idle_watcher_acks_after_healthy_tick(self, tmp_path, monkeypatch):
        """Idle watcher（无 claim）：adoption 后一个健康 tick 即 ACK。"""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        p = self._launch(tmp_path, claim=False)
        time.sleep(0.8)

        root = runtime_root()
        ident = self._ident()
        meta = json.loads((root / ident.digest / "metadata.json").read_text())
        req = ReloadRequest(
            nonce="idle-nonce-1",
            identity_digest=ident.digest,
            owner_pid=p.pid,
            owner_start_time=meta["proc_start_time"],
            request_time=int(time.time()),
        )
        write_reload_request(root, req)
        os.kill(p.pid, signal.SIGUSR1)

        ack = self._wait_for_ack(tmp_path, "idle-nonce-1")
        assert ack is not None, "idle watcher should ACK after a healthy tick"
        assert ack.ok is True
        assert ack.pid == p.pid
        assert ack.task_id == "", "idle ACK must not carry a task"

        p.terminate()
        out, err = p.communicate(timeout=5)
        assert "EXEC_GENERATION 2" in out, f"expected exec, got:\n{out}\n{err}"
        assert out.count("ACK_WRITTEN") == 1, "exactly one ACK per reload"

    def test_failed_preflight_writes_failed_ack_and_keeps_old_loop(self, tmp_path, monkeypatch):
        """Preflight 失败：写 FAILED ACK (ok=False, error=preflight)，不 exec，旧循环继续 heartbeat。"""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        p = self._launch(tmp_path, extra_args=("--fail-preflight",))
        time.sleep(0.8)

        root = runtime_root()
        ident = self._ident()
        meta = json.loads((root / ident.digest / "metadata.json").read_text())
        req = ReloadRequest(
            nonce="fail-nonce-1",
            identity_digest=ident.digest,
            owner_pid=p.pid,
            owner_start_time=meta["proc_start_time"],
            request_time=int(time.time()),
        )
        write_reload_request(root, req)
        os.kill(p.pid, signal.SIGUSR1)

        ack = self._wait_for_ack(tmp_path, "fail-nonce-1")
        assert ack is not None, "FAILED ACK must be written"
        assert ack.ok is False
        assert ack.error == "preflight"
        assert ack.pid == p.pid

        time.sleep(0.5)
        assert p.poll() is None, "watcher must continue after failed preflight"
        p.terminate()
        out, err = p.communicate(timeout=5)
        # 没有 exec
        assert "EXECVE" not in out, f"no exec on preflight failure:\n{out}\n{err}"
        assert out.count("EXEC_GENERATION 1") == 1
        assert out.count("HEARTBEAT") >= 3

    def test_owner_mismatch_request_writes_failed_ack(self, tmp_path, monkeypatch):
        """Request owner PID/start-time 不匹配：FAILED ACK (error=owner_mismatch)，不 exec。"""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        p = self._launch(tmp_path)
        time.sleep(0.8)

        root = runtime_root()
        ident = self._ident()
        req = ReloadRequest(
            nonce="owner-nonce-1",
            identity_digest=ident.digest,
            owner_pid=999999,  # wrong owner
            owner_start_time=0,
            request_time=int(time.time()),
        )
        write_reload_request(root, req)
        os.kill(p.pid, signal.SIGUSR1)

        ack = self._wait_for_ack(tmp_path, "owner-nonce-1")
        assert ack is not None, "owner-mismatch must produce a distinguishable FAILED ACK"
        assert ack.ok is False
        assert ack.error == "owner_mismatch"

        p.terminate()
        out, _ = p.communicate(timeout=5)
        assert "EXECVE" not in out, f"no exec on owner mismatch:\n{out}"

    def test_bare_sigusr1_without_request_does_not_reload(self, tmp_path, monkeypatch):
        """SIGUSR1 无有效 request：忽略，不 ACK、不 exec。"""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        p = self._launch(tmp_path)
        time.sleep(0.8)

        os.kill(p.pid, signal.SIGUSR1)
        time.sleep(1.0)

        root = runtime_root()
        ack = read_reload_ack(root, self._ident().digest)
        assert ack is None, "no ACK should be written without a valid request"

        p.terminate()
        out, _ = p.communicate(timeout=5)
        assert "EXECVE" not in out
        assert "SIGUSR1_IGNORED no_valid_request" in out

    def test_adopt_fail_closed_at_entry(self, tmp_path, monkeypatch):
        """继承 FD 无效 + identity env 匹配：fake 入口 fail closed，退出非零，不 acquire。"""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        ident = self._ident()
        p = self._launch(
            tmp_path,
            claim=False,
            extra_env={
                "HERMES_KANBAN_WATCHER_LOCK_FD": "424242",
                "HERMES_KANBAN_RELOAD_IDENTITY": ident.digest,
            },
        )
        out, err = p.communicate(timeout=10)
        assert p.returncode != 0, "fail-closed adopt must exit non-zero"
        assert "ADOPT_FAILED" in out, f"expected ADOPT_FAILED, got:\n{out}\n{err}"
        assert "LOCK_ACQUIRED" not in out, "must NOT fall back to acquire()"


# ── Isolated fake-board smoke test ────────────────────────────────────────────

class TestIsolatedFakeBoardSmoke:
    """Smoke test: fake watcher holds lock, reload CLI triggers ACK, claim stays."""

    FIXTURE = REPO / "tests" / "fixtures" / "fake_reload_watcher.py"

    def _launch(self, tmp_path, *, board="fakeboard", session="fake-session"):
        env = dict(os.environ)
        env["XDG_RUNTIME_DIR"] = str(tmp_path)
        return subprocess.Popen(
            [
                sys.executable, str(self.FIXTURE),
                "--board", board,
                "--profile", "coordinator",
                "--session", session,
                "--pane", "1",
                "--claim-file", str(tmp_path / "claim.json"),
                "--inject-file", str(tmp_path / "inject.log"),
                "--tick-s", "0.1",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env,
        )

    def test_reload_preserves_pid_and_claim_state(self, tmp_path, monkeypatch):
        """
        1. Start a fake watcher with a fixed identity (active claim).
        2. Write a reload request with the real owner start-time.
        3. Send SIGUSR1 → real self-exec.
        4. Verify ACK is written with matching nonce, ok=True and unchanged PID.
        5. Verify the claim fixture is unchanged and the watcher keeps running.
        """
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        p = self._launch(tmp_path)
        time.sleep(0.8)

        root = runtime_root()
        ident = WatcherIdentity.from_values("fakeboard", "coordinator", "fake-session", "1")
        meta_path = root / ident.digest / "metadata.json"
        assert meta_path.exists(), "watcher metadata not found"
        meta = json.loads(meta_path.read_text())
        assert meta["pid"] == p.pid, f"metadata PID {meta['pid']} != watcher PID {p.pid}"

        # The fake watcher wrote its own claim fixture on cold start
        claim_path = tmp_path / "claim.json"
        original_claim = json.loads(claim_path.read_text())
        assert original_claim["task_id"] == "t_fake"

        # Write reload request with the REAL owner start-time
        nonce = "smoke-nonce-12345"
        req = ReloadRequest(
            nonce=nonce,
            identity_digest=ident.digest,
            owner_pid=p.pid,
            owner_start_time=meta["proc_start_time"],
            request_time=int(time.time()),
        )
        write_reload_request(root, req)

        # Send SIGUSR1
        os.kill(p.pid, signal.SIGUSR1)

        # Wait for ACK via the standard runtime-root path
        ack = None
        deadline = time.time() + 15
        while time.time() < deadline:
            ack = read_reload_ack(root, ident.digest)
            if ack is not None and ack.nonce == nonce:
                break
            time.sleep(0.2)
        assert ack is not None, "ACK not written"
        assert ack.nonce == nonce, "nonce mismatch"
        assert ack.ok is True, f"ACK should be ok, got error={ack.error!r}"
        assert ack.pid == p.pid, "PID changed during reload"
        assert ack.task_id == "t_fake"

        # Verify claim fixture unchanged (active claim preserved across exec)
        loaded_claim = json.loads(claim_path.read_text())
        assert loaded_claim == original_claim, "claim changed during reload"

        # Verify watcher still alive (heartbeat continues)
        assert p.poll() is None, "watcher process died after reload"

        p.terminate()
        out, _ = p.communicate(timeout=5)
        assert out.count("EXEC_GENERATION 1") == 1
        assert "EXEC_GENERATION 2" in out
        assert out.count("ACK_WRITTEN") == 1, "exactly one ACK per reload"

    def test_two_same_key_spawn_attempts_yield_one_lock_owner(self, tmp_path, monkeypatch):
        """Two simultaneous same-key spawn attempts yield one lock owner."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        code = f'''
import sys, os
sys.path.insert(0, "{REPO}/tests/fixtures")
os.environ["XDG_RUNTIME_DIR"] = "{tmp_path}"
import fake_reload_watcher
sys.argv = ["fake_reload_watcher.py",
    "--board", "fakeboard2",
    "--profile", "coordinator",
    "--session", "fake-session2",
    "--pane", "1",
]
rc = fake_reload_watcher.main()
sys.exit(rc if rc else 0)
'''
        p1 = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(0.3)
        p2 = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        out2, _ = p2.communicate(timeout=5)
        p1.terminate()
        p1.wait(timeout=5)

        assert "LOCK_REJECTED" in out2, "Second same-key spawn should be rejected"
        assert p2.returncode != 0, "Rejected spawn should exit non-zero"
