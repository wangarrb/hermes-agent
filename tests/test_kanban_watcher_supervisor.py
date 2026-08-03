from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "local" / "bin" / "kanban-watcher-supervisor.py"


def _load_supervisor():
    spec = importlib.util.spec_from_file_location("kanban_watcher_supervisor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cmd(*, board="egomotion4d", profile="coordinator", session="kanban-egomotion4d", pane="4"):
    return [
        "python3",
        "/tmp/deepseek_kanban_interactive.py",
        "--watch-child",
        "--board",
        board,
        "--profile",
        profile,
        "--zellij-session",
        session,
        "--zellij-pane-id",
        pane,
    ]


def test_same_logical_watcher_is_a_live_replacement():
    supervisor = _load_supervisor()
    current = {101: _cmd(), 202: _cmd()}

    assert supervisor._has_live_replacement(current, 101, _cmd())


def test_different_pane_is_not_a_replacement():
    supervisor = _load_supervisor()
    current = {101: _cmd(pane="4"), 202: _cmd(pane="5")}

    assert not supervisor._has_live_replacement(current, 101, _cmd(pane="4"))


def test_restart_target_requires_same_live_zellij_pane_identity():
    supervisor = _load_supervisor()

    def run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "id": 4,
                        "is_plugin": False,
                        "terminal_command": (
                            "bash -lc HERMES_KANBAN_BOARD=egomotion4d "
                            "hermes-kanban-continue -p coordinator"
                        ),
                    }
                ]
            ),
        )

    assert supervisor._watcher_target_pane_is_current(_cmd(), run=run)


def test_restart_target_rejects_removed_or_reused_pane():
    supervisor = _load_supervisor()

    def removed(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="[]")

    def reused(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "id": 4,
                        "is_plugin": False,
                        "terminal_command": (
                            "bash -lc HERMES_KANBAN_BOARD=egomotion4d "
                            "hermes-kanban-continue -p designer"
                        ),
                    }
                ]
            ),
        )

    assert not supervisor._watcher_target_pane_is_current(_cmd(), run=removed)
    assert not supervisor._watcher_target_pane_is_current(_cmd(), run=reused)


def test_shell_command_mentioning_watcher_is_not_a_watcher():
    supervisor = _load_supervisor()
    cmdline = [
        "/bin/bash",
        "-c",
        "python3 /tmp/deepseek_kanban_interactive.py --watch-child --profile planner",
    ]

    assert not supervisor._is_watcher(cmdline)


def test_conda_wrapper_with_listener_argument_is_a_watcher():
    supervisor = _load_supervisor()
    cmdline = [
        "/home/wyr/miniconda/bin/conda",
        "run",
        "-n",
        "egomotion4d",
        "python3",
        "/tmp/hermes-kanban-role-context-listener.py",
        "--watch-child",
        "--profile",
        "planner",
    ]

    assert supervisor._is_watcher(cmdline)


def test_cleanup_prefers_launcher_and_only_targets_supervisor_children():
    supervisor = _load_supervisor()
    current = {101: _cmd(), 102: _cmd(), 201: _cmd()}
    parent_by_pid = {101: 50, 102: 50, 201: 999}

    candidates = supervisor._duplicate_cleanup_candidates(
        current,
        parent_by_pid=parent_by_pid,
        supervisor_pid=50,
    )

    assert candidates == [101, 102]


def test_cleanup_never_targets_foreign_duplicates():
    supervisor = _load_supervisor()
    current = {201: _cmd(), 202: _cmd()}
    parent_by_pid = {201: 998, 202: 999}

    candidates = supervisor._duplicate_cleanup_candidates(
        current,
        parent_by_pid=parent_by_pid,
        supervisor_pid=50,
    )

    assert candidates == []


def test_cleanup_targets_marked_fallback_after_supervisor_reparenting():
    supervisor = _load_supervisor()
    current = {101: _cmd(), 201: _cmd()}
    parent_by_pid = {101: 1, 201: 999}

    candidates = supervisor._duplicate_cleanup_candidates(
        current,
        parent_by_pid=parent_by_pid,
        supervisor_pid=50,
        supervisor_spawned_pids={101},
    )

    assert candidates == [101]


# ── Board/session scoping and singleton ──────────────────────────────────────

def test_discovery_filters_by_board():
    """Discovery excludes watchers from other boards."""
    supervisor = _load_supervisor()
    # The _has_live_replacement function already checks identity key which
    # includes board.  Verify it rejects different board.
    assert not supervisor._has_live_replacement(
        {101: _cmd(board="other-board")}, 101, _cmd(),
    )


def test_discovery_filters_by_session():
    """Discovery excludes watchers from other sessions."""
    supervisor = _load_supervisor()
    assert not supervisor._has_live_replacement(
        {101: _cmd(session="other-session")}, 101, _cmd(),
    )


def test_supervisor_identity_key_includes_board_and_session():
    """The watcher key tuple includes board and session for scoping."""
    supervisor = _load_supervisor()
    key = supervisor._watcher_key(_cmd())
    assert key is not None
    board, profile, session, pane = key
    assert board == "egomotion4d"
    assert session == "kanban-egomotion4d"


def test_different_board_supervisors_coexist():
    """Supervisors for different boards should not interfere."""
    supervisor = _load_supervisor()
    # Same pane but different board — not a replacement
    current = {101: _cmd(board="board-a"), 102: _cmd(board="board-b")}
    assert not supervisor._has_live_replacement(
        current, 101, _cmd(board="board-a"),
    )


def test_launcher_replacement_during_restart_delay():
    """If a launcher replacement appears during restart_delay, no spawn occurs."""
    supervisor = _load_supervisor()
    # Simulate: a dead watcher, but a new live process with same identity
    # already exists → _has_live_replacement returns True
    current = {201: _cmd()}
    dead_cmdline = _cmd()
    assert supervisor._has_live_replacement(current, 101, dead_cmdline)
