"""Pane title sync reflects running foreign claims without touching execution."""

from __future__ import annotations

import json
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from plugins.kanban import base_listener as bl


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    monkeypatch.setattr(bl, "zellij_submit", lambda **_: True)
    return home


class _Listener(bl.BaseInteractiveListener):
    agent_name = "Hermes"
    agent_slug = "hermes"
    idle_markers = ("❯",)
    busy_markers = ("working",)

    def build_tui_cmd(self, workspace: Path, **kwargs):  # type: ignore[no-untyped-def]
        return []

    def has_saved_sessions(self, workspace: Path) -> bool:
        return True

    def inject_text(self, *args, **kwargs) -> str:  # type: ignore[no-untyped-def]
        return ""

    def pane_label(self, task_id: str | None = None) -> str:
        if task_id:
            return f"hermes-kanban [{task_id}]"
        return "hermes-kanban"


def _claim_running(assignee: str = "coordinator") -> str:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="work", assignee=assignee)
        assert kb.claim_task(conn, task_id, claimer="someone-else") is not None
    return task_id


def _sync(listener: _Listener, conn, current_title: str | None, renamed: list) -> None:
    listener._sync_pane_title_to_running_claim(
        conn=conn,
        assignees=["coordinator"],
        session="s",
        pane_id="4",
        log_path=Path("/tmp/title-sync-test.log"),
    )


def _patch_title(
    monkeypatch: pytest.MonkeyPatch, current_title: str | None, renamed: list
) -> None:
    monkeypatch.setattr(
        bl, "_zellij_get_pane_title", lambda **_: current_title
    )
    monkeypatch.setattr(
        bl,
        "zellij_rename_pane",
        lambda **kw: renamed.append(kw["name"]) or True,
    )


def test_sync_shows_foreign_running_claim_without_db_write(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_id = _claim_running()
    with kb.connect() as conn:
        before = kb.get_task(conn, task_id)
    renamed: list = []
    _patch_title(monkeypatch, "hermes-kanban", renamed)
    with kb.connect() as conn:
        _sync(_Listener(), conn, "hermes-kanban", renamed)
    assert renamed == [f"hermes-kanban [{task_id}]"]
    with kb.connect() as conn:
        after = kb.get_task(conn, task_id)
    assert after is not None and before is not None
    assert after.status == "running"
    assert after.current_run_id == before.current_run_id
    assert after.claim_lock == before.claim_lock


def test_sync_clears_stale_title_when_nothing_running(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renamed: list = []
    _patch_title(monkeypatch, "hermes-kanban [t_old]", renamed)
    with kb.connect() as conn:
        _sync(_Listener(), conn, "hermes-kanban [t_old]", renamed)
    assert renamed == ["hermes-kanban"]


def test_sync_noop_when_title_already_correct(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_id = _claim_running()
    renamed: list = []
    _patch_title(monkeypatch, f"hermes-kanban [{task_id}]", renamed)
    with kb.connect() as conn:
        _sync(_Listener(), conn, f"hermes-kanban [{task_id}]", renamed)
    assert renamed == []


def test_sync_ignores_foreign_pane_title(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _claim_running()
    renamed: list = []
    _patch_title(monkeypatch, "planner \u276f idle", renamed)
    with kb.connect() as conn:
        _sync(_Listener(), conn, "planner \u276f idle", renamed)
    assert renamed == []


def test_sync_skipped_while_control_pause_owns_title(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _claim_running()
    renamed: list = []
    _patch_title(monkeypatch, "hermes-kanban", renamed)
    listener = _Listener()
    listener._active_control_id = 7
    with kb.connect() as conn:
        _sync(listener, conn, "hermes-kanban", renamed)
    assert renamed == []


def test_sync_never_overrides_live_pause_title(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _claim_running()
    renamed: list = []
    _patch_title(monkeypatch, "hermes-kanban [PAUSE t_x]", renamed)
    with kb.connect() as conn:
        _sync(_Listener(), conn, "hermes-kanban [PAUSE t_x]", renamed)
    assert renamed == []


def test_get_pane_title_skips_plugin_and_exited_panes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = [
        {"id": "plugin_0", "title": "x", "is_plugin": True},
        {"id": "4", "title": "dead", "exited": True},
        {"id": "5", "title": "hermes-kanban [t_1]"},
    ]
    monkeypatch.setattr(
        bl.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=json.dumps(payload)),
    )
    assert (
        bl._zellij_get_pane_title(session="s", pane_id="5")
        == "hermes-kanban [t_1]"
    )
    assert bl._zellij_get_pane_title(session="s", pane_id="4") is None
    assert bl._zellij_get_pane_title(session="s", pane_id="99") is None


def test_watcher_syncs_title_while_own_claim_is_active() -> None:
    """A TUI may reset the title after injection; active loops must restore it."""
    tree = ast.parse(Path(bl.__file__).read_text(encoding="utf-8"))
    watcher = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "watcher_main"
    )
    active_branch = next(
        node for node in ast.walk(watcher)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "active_task"
    )
    calls = [
        node
        for node in ast.walk(active_branch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_sync_pane_title_to_running_claim"
    ]
    assert calls, "active claim loop must continuously restore task-id pane title"
