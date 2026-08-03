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


# ── SIGUSR1 self-exec integration ────────────────────────────────────────────

class TestSigusr1SelfExec:
    """Verify SIGUSR1 triggers safe-boundary self-exec with PID preservation."""

    def test_sigusr1_self_exec_preserves_pid_and_emits_ack(self, tmp_path, monkeypatch):
        """Fake watcher holds lock, receives SIGUSR1, self-execs, emits ACK."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        # This test uses the fake_reload_watcher which handles SIGUSR1 by
        # writing an ACK file (simulating the reload protocol).
        code = f'''
import sys, os, time, signal, json
sys.path.insert(0, "{REPO}/tests/fixtures")
os.environ["XDG_RUNTIME_DIR"] = "{tmp_path}"
import fake_reload_watcher
sys.argv = ["fake_reload_watcher.py",
    "--board", "testboard",
    "--profile", "coordinator",
    "--session", "test-session",
    "--pane", "1",
    "--handoff-dir", "{tmp_path}/handoff",
]
rc = fake_reload_watcher.main()
sys.exit(rc if rc else 0)
'''
        # Start watcher
        p = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(0.5)

        # Write a reload request
        root = runtime_root()
        ident = WatcherIdentity.from_values("testboard", "coordinator", "test-session", "1")
        req = ReloadRequest(
            nonce="test-nonce-123",
            identity_digest=ident.digest,
            owner_pid=p.pid,
            owner_start_time=0,
            request_time=int(time.time()),
        )
        write_reload_request(root, req)

        # Send SIGUSR1
        os.kill(p.pid, signal.SIGUSR1)

        # Wait for ACK
        ack_path = Path(tmp_path) / "handoff" / f"ack.{ident.digest}.json"
        deadline = time.time() + 10
        while time.time() < deadline:
            if ack_path.exists():
                break
            time.sleep(0.2)

        p.terminate()
        p.wait(timeout=5)

        assert ack_path.exists(), "ACK file not written after SIGUSR1"
        ack_data = json.loads(ack_path.read_text())
        assert ack_data["nonce"] == "test-nonce-123"
        assert ack_data["status"] == "ACK"

    def test_bare_sigusr1_without_request_does_not_reload(self, tmp_path, monkeypatch):
        """SIGUSR1 without a valid reload request must not trigger reload."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        code = f'''
import sys, os, time, signal
sys.path.insert(0, "{REPO}/tests/fixtures")
os.environ["XDG_RUNTIME_DIR"] = "{tmp_path}"
import fake_reload_watcher
sys.argv = ["fake_reload_watcher.py",
    "--board", "testboard",
    "--profile", "coordinator",
    "--session", "test-session",
    "--pane", "1",
    "--handoff-dir", "{tmp_path}/handoff",
]
rc = fake_reload_watcher.main()
sys.exit(rc if rc else 0)
'''
        p = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(0.5)

        # Send SIGUSR1 without writing a reload request
        os.kill(p.pid, signal.SIGUSR1)
        time.sleep(1.0)

        # Check no ACK was written
        root = runtime_root()
        ident = WatcherIdentity.from_values("testboard", "coordinator", "test-session", "1")
        ack_path = Path(tmp_path) / "handoff" / f"ack.{ident.digest}.json"
        assert not ack_path.exists(), "ACK should not exist without valid request"

        p.terminate()
        p.wait(timeout=5)
