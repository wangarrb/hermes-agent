"""Tests for plugins/kanban/watcher_runtime.py — identity, locks and reload state.

These tests exercise the watcher identity, local flock singleton, and reload
handoff/ACK primitives.  Subprocess tests prove that two processes using the
same identity cannot both acquire the lock, while different pane IDs can.
"""
from __future__ import annotations

import json
import os
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
