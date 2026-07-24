"""Tests for designer task creation validation."""
from __future__ import annotations

import pytest
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import pathlib
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_designer_requires_worktree(kanban_home):
    with kb.connect() as conn:
        with pytest.raises(ValueError, match="workspace_kind=worktree"):
            kb.create_task(conn, title="designer task", assignee="designer", workspace_kind="dir")


def test_designer_requires_base_commit(kanban_home):
    with kb.connect() as conn:
        with pytest.raises(ValueError, match="base_commit"):
            kb.create_task(
                conn, title="designer task", assignee="designer",
                workspace_kind="worktree", workspace_path="/home/wyr/code/Egomotion4D-designer",
            )


def test_designer_requires_workspace_path(kanban_home):
    with kb.connect() as conn:
        with pytest.raises(ValueError, match="workspace_path"):
            kb.create_task(
                conn, title="designer task", assignee="designer",
                workspace_kind="worktree", base_commit="a" * 40,
            )


def test_designer_valid_creation(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="designer task", assignee="designer",
            workspace_kind="worktree", workspace_path="/home/wyr/code/Egomotion4D-designer",
            base_commit="a" * 40, branch_name="designer/test",
        )
        assert task_id


def test_non_designer_does_not_require_worktree(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review task", assignee="reviewer", workspace_kind="dir")
        assert task_id