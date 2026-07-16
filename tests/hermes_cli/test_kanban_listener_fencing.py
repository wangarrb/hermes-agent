"""Generation authority held by the in-process Kanban safety listener."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_listener as listener


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_delayed_old_listener_callback_cannot_complete_new_run(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="implementation", assignee="implementer")
        first = kb.claim_task(conn, task_id, claimer="listener-v1")
        returned = kb.return_task_for_rework(
            conn, task_id, actor="reviewer", reason="retry"
        )
        kb.ack_control_message(conn, returned.control_ids[0], receiver="listener-v1")
        second = kb.claim_task(conn, task_id, claimer="listener-v2")
        now = int(time.time())
        conn.execute(
            "UPDATE task_runs SET summary = ?, started_at = ?, ended_at = ? "
            "WHERE id = ?",
            ("old productive output", now - 120, now, first.current_run_id),
        )
        conn.execute(
            "UPDATE task_runs SET summary = ?, started_at = ?, ended_at = ? "
            "WHERE id = ?",
            ("new productive output", now - 120, now, second.current_run_id),
        )
        conn.commit()

    state = listener.ListenerState(board="default", current_task_id=task_id)
    state.current_run_id = first.current_run_id
    state.current_generation = first.generation
    listener._auto_complete_if_still_running(state)

    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)

    assert task.status == "running"
    assert task.current_run_id == second.current_run_id
