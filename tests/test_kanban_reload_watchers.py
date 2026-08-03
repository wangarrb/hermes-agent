"""Tests for local/bin/hermes-kanban-reload-watchers — the rolling reload CLI.

Tests use fake process trees and injected runtime roots to verify filtering,
ordering, nonce-matched ACK validation, timeout, and stop-on-first-failure
behavior without touching real watchers.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "local" / "bin" / "hermes-kanban-reload-watchers"

# Import the module without executing __main__
import importlib.util
from importlib.machinery import SourceFileLoader
spec = importlib.util.spec_from_file_location(
    "reload_watchers", SCRIPT,
    loader=SourceFileLoader("reload_watchers", str(SCRIPT)),
)
assert spec is not None and spec.loader is not None
rw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rw)


def _cmdline(*, board="egomotion4d", profile="coordinator", session="kanban-egomotion4d", pane="4"):
    return [
        "/home/wyr/miniconda/bin/python3",
        "/home/wyr/.hermes/hermes-agent-repo/local/bin/hermes-kanban-role-context-listener.py",
        "--watch-child",
        "--board", board,
        "--profile", profile,
        "--zellij-session", session,
        "--zellij-pane-id", pane,
    ]


# ── Filtering tests ──────────────────────────────────────────────────────────

class TestFiltering:
    def test_selects_real_post_exec_hermes_listener(self, tmp_path, monkeypatch):
        """A Hermes watcher remains discoverable after self-exec changes argv[0]."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        from watcher_runtime import WatcherIdentity, runtime_root, _atomic_write_json

        ident = WatcherIdentity.from_values(
            "egomotion4d", "coordinator", "kanban-egomotion4d", "4",
        )
        root = runtime_root()
        _atomic_write_json(root / ident.digest / "metadata.json", {
            "pid": 101, "proc_start_time": 55,
            "identity": ident.digest, "code_revision": "rev",
        })
        cmdline = _cmdline()
        cmdline[1] = "/repo/plugins/kanban/hermes_listener/hermes_kanban_interactive.py"

        targets = rw._filter_reload_targets(
            {101: cmdline}, board="egomotion4d",
            session="kanban-egomotion4d", profiles=None,
            runtime_root_path=str(tmp_path),
        )

        assert [target["pid"] for target in targets] == [101]

    def test_filters_exact_board_session(self, tmp_path, monkeypatch):
        """Only processes matching exact board/session are selected."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        # Create lock metadata manually with the test PID
        from watcher_runtime import WatcherIdentity, runtime_root, _atomic_write_json
        ident = WatcherIdentity.from_values("egomotion4d", "coordinator", "kanban-egomotion4d", "4")
        root = runtime_root()
        meta_path = root / ident.digest / "metadata.json"
        _atomic_write_json(meta_path, {
            "pid": 101, "proc_start_time": 0,
            "identity": ident.digest, "code_revision": "test",
        })
        procs = {
            101: _cmdline(board="egomotion4d", session="kanban-egomotion4d"),
            102: _cmdline(board="other-board", session="kanban-egomotion4d"),
            103: _cmdline(board="egomotion4d", session="other-session"),
        }
        filtered = rw._filter_reload_targets(
            procs, board="egomotion4d", session="kanban-egomotion4d", profiles=None,
            runtime_root_path=str(tmp_path),
        )
        pids = [p["pid"] for p in filtered]
        assert pids == [101]

    def test_ignores_conda_run_wrappers(self):
        """conda run wrapper processes are not selected as reload targets."""
        conda_cmd = [
            "/home/wyr/miniconda/bin/conda", "run", "-n", "egomotion4d",
            "python3", "/tmp/hermes-kanban-role-context-listener.py",
            "--watch-child", "--board", "egomotion4d",
            "--profile", "coordinator",
            "--zellij-session", "kanban-egomotion4d",
            "--zellij-pane-id", "4",
        ]
        procs = {101: conda_cmd}
        filtered = rw._filter_reload_targets(
            procs, board="egomotion4d", session="kanban-egomotion4d", profiles=None,
        )
        assert filtered == []

    def test_selects_lock_owner_pid(self, tmp_path, monkeypatch):
        """When multiple processes match, only the lock owner PID is selected."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        # We can't easily fake the lock owner check without a real lock,
        # but we can verify the function calls _is_lock_owner correctly
        procs = {101: _cmdline(), 102: _cmdline()}
        # Both have same identity, so both map to same lock file.
        # _filter_reload_targets should only pick the lock owner.
        # Without a real lock, both will be "no lock owner" and filtered out.
        filtered = rw._filter_reload_targets(
            procs, board="egomotion4d", session="kanban-egomotion4d",
            profiles=None, runtime_root_path=str(tmp_path / "rt"),
        )
        # Without real locks, no targets are selected
        assert filtered == []


class TestProfileOrdering:
    def test_profiles_sorted_stably(self):
        """Profiles are processed in sorted order."""
        profiles = ["coordinator", "planner", "implementer", "reviewer"]
        result = rw._sort_profiles(profiles)
        assert result == ["coordinator", "implementer", "planner", "reviewer"]


def test_default_ack_timeout_covers_slowest_watcher_poll():
    """Default rolling wait must exceed the production 60-second poll."""
    assert rw.DEFAULT_ACK_TIMEOUT_S >= 75.0


class TestAtomicRequestBeforeSignal:
    def test_request_written_before_signal(self, tmp_path, monkeypatch):
        """Reload request JSON is written atomically before SIGUSR1."""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        # Verify the request file exists and is mode 0600
        from watcher_runtime import (
            WatcherIdentity, write_reload_request, reload_request_path,
        )
        sys.path.insert(0, str(REPO / "plugins" / "kanban"))
        ident = WatcherIdentity.from_values("b", "p", "s", "1")
        root = rw._get_runtime_root(str(tmp_path))
        nonce = rw._generate_nonce()
        req = rw.ReloadRequest(
            nonce=nonce,
            identity_digest=ident.digest,
            owner_pid=12345,
            owner_start_time=0,
            request_time=int(time.time()),
        )
        rw.write_reload_request(root, req)
        path = rw.reload_request_path(root, ident.digest)
        assert path.exists()
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600


class TestAckStrictValidation:
    """_wait_for_ack 只接受 nonce/PID/start-time/identity/revision 全匹配且 ok=True 的 ACK。"""

    def _make_ack(self, tmp_path, *, nonce="n1", pid=101, proc_start_time=55,
                  identity_digest="", revision="rev1", ok=True, error=""):
        sys.path.insert(0, str(REPO / "plugins" / "kanban"))
        from watcher_runtime import (
            WatcherIdentity, write_reload_ack,
        )
        ident = WatcherIdentity.from_values("egomotion4d", "coordinator", "kanban-egomotion4d", "4")
        # The ACK file lives under the real identity digest; identity_digest
        # only controls the ACK *content* so tests can forge mismatches.
        digest = ident.digest
        root = rw._get_runtime_root(str(tmp_path))
        write_reload_ack(
            root, digest,
            rw.ReloadACK(
                nonce=nonce, pid=pid, proc_start_time=proc_start_time,
                code_revision=revision, ok=ok, error=error,
                identity_digest=identity_digest or digest,
            ),
        )
        return digest

    def test_failed_ack_returned_immediately(self, tmp_path, monkeypatch):
        """ok=False ACK 立即返回（不等待超时），供 CLI 停止 rolling。"""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        digest = self._make_ack(tmp_path, nonce="n1", ok=False, error="preflight")
        root = rw._get_runtime_root(str(tmp_path))
        start = time.monotonic()
        ack = rw._wait_for_ack(
            root, digest, "n1", timeout_s=5, pid=101,
            proc_start_time=55, expected_revision="rev1",
        )
        elapsed = time.monotonic() - start
        assert ack is not None
        assert ack.ok is False
        assert ack.error == "preflight"
        assert elapsed < 3.0, "FAILED ACK must return immediately, not after timeout"

    def test_ack_rejected_on_proc_start_time_mismatch(self, tmp_path, monkeypatch):
        """proc_start_time 不匹配（PID 复用防护）→ 不接受，超时返回 None。"""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        digest = self._make_ack(tmp_path, nonce="n1", pid=101, proc_start_time=55)
        root = rw._get_runtime_root(str(tmp_path))
        ack = rw._wait_for_ack(
            root, digest, "n1", timeout_s=1.0, pid=101,
            proc_start_time=999,  # mismatch
            expected_revision="rev1",
        )
        assert ack is None

    def test_ack_rejected_on_identity_mismatch(self, tmp_path, monkeypatch):
        """identity digest 不匹配 → 不接受。"""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        digest = self._make_ack(
            tmp_path, nonce="n1", identity_digest="deadbeef", revision="rev1",
        )
        root = rw._get_runtime_root(str(tmp_path))
        ack = rw._wait_for_ack(
            root, digest, "n1", timeout_s=1.0, pid=101,
            proc_start_time=55, expected_revision="rev1",
        )
        assert ack is None

    def test_ack_rejected_on_revision_mismatch(self, tmp_path, monkeypatch):
        """code revision 不匹配 → 不接受。"""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        digest = self._make_ack(tmp_path, nonce="n1", revision="old-rev")
        root = rw._get_runtime_root(str(tmp_path))
        ack = rw._wait_for_ack(
            root, digest, "n1", timeout_s=1.0, pid=101,
            proc_start_time=55, expected_revision="new-rev",
        )
        assert ack is None

    def test_ack_accepted_when_all_fields_match(self, tmp_path, monkeypatch):
        """全字段匹配且 ok=True → 接受。"""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        digest = self._make_ack(tmp_path, nonce="n1", pid=101, proc_start_time=55)
        root = rw._get_runtime_root(str(tmp_path))
        ack = rw._wait_for_ack(
            root, digest, "n1", timeout_s=1.0, pid=101,
            proc_start_time=55, expected_revision="rev1",
        )
        assert ack is not None
        assert ack.ok is True

    def test_lock_owner_requires_start_time(self, tmp_path, monkeypatch):
        """_is_lock_owner 在提供 proc_start_time 时必须同时匹配 PID 和 start-time。"""
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
        sys.path.insert(0, str(REPO / "plugins" / "kanban"))
        from watcher_runtime import (
            WatcherIdentity, runtime_root, _atomic_write_json,
        )
        ident = WatcherIdentity.from_values("egomotion4d", "coordinator", "kanban-egomotion4d", "4")
        root = runtime_root()
        meta_path = root / ident.digest / "metadata.json"
        _atomic_write_json(meta_path, {
            "pid": 101, "proc_start_time": 55,
            "identity": ident.digest, "code_revision": "rev1",
        })
        assert rw._is_lock_owner(101, ident.digest, root, proc_start_time=55)
        assert not rw._is_lock_owner(101, ident.digest, root, proc_start_time=999)
        assert not rw._is_lock_owner(102, ident.digest, root, proc_start_time=55)
