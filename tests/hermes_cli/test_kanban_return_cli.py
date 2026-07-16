"""CLI contracts for reviewer-directed Kanban rework."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "reviewer")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_return_for_rework_and_control_ack_round_trip(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(
            conn, title="implementation", assignee="implementer"
        )
        child = kb.create_task(
            conn, title="review", assignee="reviewer", parents=[parent]
        )
        assert kb.complete_task(conn, parent, summary="v1")
        assert kb.claim_task(conn, child, claimer="review-pane")

    out = kc.run_slash(
        f"return-for-rework {parent} --reason 'implementation violates contract'"
    )
    assert "Returned" in out
    control_id = int(re.search(r"control=(\d+)", out).group(1))

    with kb.connect() as conn:
        assert kb.get_task(conn, parent).status == "todo"
        assert kb.get_task(conn, parent).rework_hold is True
        comments = kb.list_comments(conn, parent)
        assert "implementation violates contract" in comments[-1].body

    ack = kc.run_slash(f"control-ack {control_id}")
    assert "Acknowledged" in ack
    with kb.connect() as conn:
        assert kb.get_task(conn, parent).status == "ready"
        assert kb.get_task(conn, parent).rework_hold is False


def test_cli_uses_env_generation_fence(kanban_home, monkeypatch):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="implementation", assignee="implementer"
        )
        first = kb.claim_task(conn, task_id, claimer="old-pane")
        old_run = first.current_run_id
        old_generation = first.generation
        returned = kb.return_task_for_rework(
            conn, task_id, actor="reviewer", reason="retry"
        )
        kb.ack_control_message(conn, returned.control_ids[0], receiver="old-pane")
        second = kb.claim_task(conn, task_id, claimer="new-pane")

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(old_run))
    monkeypatch.setenv("HERMES_KANBAN_GENERATION", str(old_generation))
    old_out = kc.run_slash(f"complete {task_id} --summary 'late result'")
    assert "cannot complete" in old_out

    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(second.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_GENERATION", str(second.generation))
    new_out = kc.run_slash(f"complete {task_id} --summary 'correct result'")
    assert "Completed" in new_out


def test_explicit_fence_flags_work_without_worker_env(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="implementation")
        first = kb.claim_task(conn, task_id, claimer="pane")
        returned = kb.return_task_for_rework(
            conn, task_id, actor="reviewer", reason="retry"
        )
        kb.ack_control_message(conn, returned.control_ids[0], receiver="pane")
        second = kb.claim_task(conn, task_id, claimer="pane-v2")

    rejected = kc.run_slash(
        f"complete {task_id} --run-id {first.current_run_id} "
        f"--generation {first.generation}"
    )
    assert "cannot complete" in rejected
    accepted = kc.run_slash(
        f"complete {task_id} --run-id {second.current_run_id} "
        f"--generation {second.generation} --summary 'v2'"
    )
    assert "Completed" in accepted


def test_update_reopen_returns_to_todo_and_invalidates_done_descendant(
    kanban_home,
):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="implementation")
        child = kb.create_task(conn, title="review", parents=[parent])
        assert kb.complete_task(conn, parent, summary="v1")
        assert kb.complete_task(conn, child, summary="accepted v1")

    out = kc.run_slash(f"update {parent} --reopen --body 'fix contract'"
    )
    assert "reopened" in out
    with kb.connect() as conn:
        parent_after = kb.get_task(conn, parent)
        child_after = kb.get_task(conn, child)

    assert parent_after.status == "todo"
    assert parent_after.generation == 2
    assert parent_after.body == "fix contract"
    assert child_after.status == "stale"
    assert child_after.generation == 2
