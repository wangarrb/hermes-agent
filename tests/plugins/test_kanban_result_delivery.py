"""Interactive watcher delivery of durable publisher result callbacks."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    agent_name = "Test"
    agent_slug = "test"
    idle_markers = ("❯",)
    busy_markers = ("working",)

    def build_tui_cmd(self, workspace: Path, **kwargs):
        return []

    def has_saved_sessions(self, workspace: Path) -> bool:
        return True

    def inject_text(self, *args, **kwargs) -> str:
        return ""

    def pane_label(self, task_id: str | None = None) -> str:
        return "test"


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        profile="planner",
        zellij_session="s",
        zellij_pane_id="0",
    )


def _enqueue_results(count: int = 2) -> list[str]:
    task_ids: list[str] = []
    with kb.connect() as conn:
        for index in range(count):
            task_id = kb.create_task(
                conn,
                title=f"review {index}",
                assignee="reviewer",
                result_subscriber="planner",
            )
            assert kb.complete_task(conn, task_id, summary=f"result {index}")
            task_ids.append(task_id)
    return task_ids


def test_busy_pane_keeps_result_queue_pending(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enqueue_results(1)
    listener = _Listener()
    injected: list[str] = []
    monkeypatch.setattr(listener, "wait_for_stable_composer_input", lambda **_: False)
    monkeypatch.setattr(bl, "zellij_inject", lambda **kw: injected.append(kw["text"]) or True)

    with kb.connect() as conn:
        assert listener.pump_result_notifications(_args(), conn, tmp_path / "watch.log")
        row = conn.execute("SELECT status FROM kanban_result_queue").fetchone()
        assert row["status"] == "pending"
    assert injected == []


def test_safe_pane_injects_bounded_fifo_batch_and_marks_delivered(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_ids = _enqueue_results(2)
    listener = _Listener()
    injected: list[str] = []
    monkeypatch.setattr(listener, "wait_for_stable_composer_input", lambda **_: True)
    monkeypatch.setattr(bl, "zellij_inject", lambda **kw: injected.append(kw["text"]) or True)

    with kb.connect() as conn:
        assert listener.pump_result_notifications(_args(), conn, tmp_path / "watch.log")
        rows = conn.execute(
            "SELECT id, event_id, status FROM kanban_result_queue ORDER BY id"
        ).fetchall()
    assert all(row["status"] == "delivered" for row in rows)
    assert len(injected) == 1
    assert injected[0].startswith("[TASK_RESULTS_READY]")
    assert injected[0].endswith("[by watcher]")
    assert "\n" not in injected[0]
    assert injected[0].index(task_ids[0]) < injected[0].index(task_ids[1])
    for row in rows:
        assert f"q{row['id']}/e{row['event_id']}" in injected[0]


def test_failed_injection_releases_result_lease(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enqueue_results(1)
    listener = _Listener()
    monkeypatch.setattr(listener, "wait_for_stable_composer_input", lambda **_: True)
    monkeypatch.setattr(bl, "zellij_inject", lambda **_: False)

    with kb.connect() as conn:
        assert listener.pump_result_notifications(_args(), conn, tmp_path / "watch.log")
        row = conn.execute(
            "SELECT status, lease_owner, lease_expires FROM kanban_result_queue"
        ).fetchone()
        assert tuple(row) == ("pending", None, None)


def test_disabled_result_delivery_leaves_queue_untouched(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _enqueue_results(1)
    listener = _Listener()
    monkeypatch.setenv("HERMES_KANBAN_RESULT_NOTIFICATIONS", "off")
    monkeypatch.setattr(
        listener,
        "wait_for_stable_composer_input",
        lambda **_: pytest.fail("disabled delivery must not inspect the pane"),
    )
    with kb.connect() as conn:
        assert not listener.pump_result_notifications(_args(), conn, tmp_path / "watch.log")
        row = conn.execute("SELECT status FROM kanban_result_queue").fetchone()
        assert row["status"] == "pending"
