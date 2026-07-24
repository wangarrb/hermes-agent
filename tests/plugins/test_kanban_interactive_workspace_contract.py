"""Workspace identity contract for interactive Kanban listeners."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from plugins.kanban import base_listener as bl


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


class DummyListener(bl.BaseInteractiveListener):
    agent_name = "Dummy"
    agent_slug = "dummy"

    def build_tui_cmd(self, workspace, **kwargs):
        return []

    def has_saved_sessions(self, workspace):
        return False

    def inject_text(self, task_id, title, assignee, profile, prompt_path, board):
        return str(prompt_path)

    def pane_label(self, task_id=None):
        return f"dummy:{task_id or 'idle'}"


def _listener(pane_workspace: Path) -> DummyListener:
    listener = DummyListener()
    listener._profile = "implementer"
    listener._board = "default"
    listener._workspace = pane_workspace
    listener._log_path = pane_workspace / "listener.log"
    return listener


def _args(pane_workspace: Path) -> Namespace:
    return Namespace(
        profile="implementer",
        claim_assignees="implementer",
        assist_role=None,
        zellij_session="test-session",
        zellij_pane_id="7",
        workspace=str(pane_workspace),
        board="default",
        ttl=900,
    )


def _ready_worktree_task(
    conn, *, branch_name: str | None = "implementer/t_demo/g1",
) -> str:
    return kb.create_task(
        conn,
        title="honor task worktree",
        assignee="implementer",
        workspace_kind="worktree",
        workspace_path="/tmp/project-worktrees/t_demo",
        branch_name=branch_name,
    )


@pytest.mark.parametrize(
    ("stored_branch", "resolved_branch"),
    [
        (None, None),
        ("implementer/stale/g1", "implementer/t_demo/g1"),
    ],
)
def test_claim_persists_resolved_worktree_and_actual_branch_in_prompt(
    kanban_home, tmp_path, monkeypatch, stored_branch, resolved_branch,
):
    pane_workspace = tmp_path / "pane-workspace"
    pane_workspace.mkdir()
    resolved_workspace = tmp_path / "project-worktrees" / "t_demo"
    resolved_workspace.mkdir(parents=True)
    listener = _listener(pane_workspace)
    resolve_calls = []
    persisted_paths = []
    branch_calls = []
    persisted_branches = []
    injected = []
    actual_branch = []

    original_set_workspace_path = kb.set_workspace_path
    original_set_branch_name = kb.set_branch_name

    def resolve_workspace(task, *, board=None):
        resolve_calls.append((task.id, task.workspace_path, board))
        return resolved_workspace

    def set_workspace_path(conn, task_id, path):
        persisted_paths.append((task_id, str(path)))
        original_set_workspace_path(conn, task_id, path)

    def git_current_branch(path):
        branch_calls.append(path)
        return actual_branch[0]

    def set_branch_name(conn, task_id, branch_name):
        persisted_branches.append((task_id, branch_name))
        original_set_branch_name(conn, task_id, branch_name)

    monkeypatch.setattr(kb, "resolve_workspace", resolve_workspace)
    monkeypatch.setattr(kb, "set_workspace_path", set_workspace_path)
    monkeypatch.setattr(kb, "_git_current_branch", git_current_branch)
    monkeypatch.setattr(kb, "set_branch_name", set_branch_name)
    monkeypatch.setattr(
        bl, "zellij_inject",
        lambda **kwargs: injected.append(kwargs["text"]) or True,
    )
    monkeypatch.setattr(bl, "zellij_rename_pane", lambda **kwargs: True)

    with kb.connect() as conn:
        task_id = _ready_worktree_task(conn, branch_name=stored_branch)
        expected_branch = resolved_branch or f"wt/{task_id}"
        actual_branch.append(expected_branch)
        claimed_id, run_id = listener.claim_and_inject_one(
            _args(pane_workspace), log_path=listener._log_path, conn=conn,
        )
        claimed = kb.get_task(conn, task_id)

    assert claimed_id == task_id
    assert run_id is not None
    assert resolve_calls == [
        (task_id, "/tmp/project-worktrees/t_demo", "default"),
    ]
    assert persisted_paths == [(task_id, str(resolved_workspace))]
    assert branch_calls == [resolved_workspace]
    assert persisted_branches == [(task_id, expected_branch)]
    assert claimed.workspace_path == str(resolved_workspace)
    assert claimed.workspace_path != str(pane_workspace)
    assert claimed.branch_name == expected_branch

    assert len(injected) == 1
    prompt_path = Path(injected[0])
    assert prompt_path.is_relative_to(resolved_workspace)
    assert not prompt_path.is_relative_to(pane_workspace)
    prompt = prompt_path.read_text(encoding="utf-8")
    assert f"Workspace: worktree @ {resolved_workspace}" in prompt
    assert f"Branch:   {expected_branch}" in prompt
    assert "generation=1" in prompt


@pytest.mark.parametrize("failure_stage", ["resolve", "identity", "branch"])
def test_workspace_resolution_or_identity_failure_reclaims_without_injecting(
    kanban_home, tmp_path, monkeypatch, failure_stage,
):
    pane_workspace = tmp_path / "pane-workspace"
    pane_workspace.mkdir()
    resolved_workspace = tmp_path / "project-worktrees" / "t_demo"
    resolved_workspace.mkdir(parents=True)
    listener = _listener(pane_workspace)
    injected = []
    monkeypatch.setattr(
        kb, "_git_current_branch", lambda path: "implementer/t_demo/g1",
    )

    if failure_stage == "resolve":
        def resolve_workspace(task, *, board=None):
            raise RuntimeError("worktree unavailable")

        monkeypatch.setattr(kb, "resolve_workspace", resolve_workspace)
    else:
        monkeypatch.setattr(
            kb, "resolve_workspace", lambda task, *, board=None: resolved_workspace,
        )
    if failure_stage == "identity":
        monkeypatch.setattr(kb, "set_workspace_path", lambda conn, task_id, path: None)
    if failure_stage == "branch":
        monkeypatch.setattr(kb, "_git_current_branch", lambda path: None)

    monkeypatch.setattr(
        bl, "zellij_inject",
        lambda **kwargs: injected.append(kwargs["text"]) or True,
    )
    monkeypatch.setattr(bl, "zellij_rename_pane", lambda **kwargs: True)

    with kb.connect() as conn:
        task_id = _ready_worktree_task(conn)
        claimed_id, run_id = listener.claim_and_inject_one(
            _args(pane_workspace), log_path=listener._log_path, conn=conn,
        )
        task = kb.get_task(conn, task_id)

    assert claimed_id is None
    assert run_id is None
    assert task.status == "ready"
    assert injected == []
    log_text = listener._log_path.read_text(encoding="utf-8")
    assert task_id in log_text
    assert "workspace resolution/identity failed" in log_text
    if failure_stage == "resolve":
        assert "RuntimeError: worktree unavailable" in log_text
    elif failure_stage == "identity":
        assert "workspace identity mismatch" in log_text
    else:
        assert "could not resolve actual branch" in log_text
