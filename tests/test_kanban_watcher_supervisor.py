from __future__ import annotations

import importlib.util
from pathlib import Path


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
