"""Immutable repository identity contracts for Kanban worktree tasks."""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _contracts():
    try:
        from hermes_cli import kanban_workspace_contract as contracts
    except ImportError as exc:  # RED until the contract module exists.
        pytest.fail(f"workspace contract module missing: {exc}")
    return contracts


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_repo(path: Path) -> str:
    path.mkdir()
    _git(path, "init")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test User")
    _git(path, "branch", "-M", "main")
    (path / "README.md").write_text("base\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "base")
    return _git(path, "rev-parse", "HEAD")


def _materialize_contract_task(tmp_path: Path):
    repo = tmp_path / "repo"
    base_commit = _init_repo(repo)
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="immutable worktree",
            assignee="Implementer",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name="impl/{assignee}/{task_id}/g{generation}",
            base_commit=base_commit,
            target_branch="main",
        )
        task = kb.get_task(conn, task_id)
        workspace = kb.resolve_workspace(task)
        assert kb.set_workspace_path(conn, task_id, workspace)
        task = kb.get_task(conn, task_id)
    return repo, workspace, task


def test_create_cli_persists_repo_identity_and_resolves_safe_branch_template(
    kanban_home, tmp_path,
):
    target = tmp_path / "project-worktrees" / "task"
    base_commit = "a" * 40
    output = kc.run_slash(
        "create 'templated worktree' --assignee Implementer "
        f"--workspace worktree:{target} "
        "--branch 'impl/{assignee}/{task_id}/g{generation}' "
        f"--base-commit {base_commit} --target-branch main --json"
    )
    payload = json.loads(output)

    assert payload["base_commit"] == base_commit
    assert payload["target_branch"] == "main"
    assert payload["branch_name"] == f"impl/implementer/{payload['id']}/g1"
    assert payload["generation"] == 1
    assert payload["common_dir"] is None
    assert payload["workspace_contract"] is None

    with kb.connect() as conn:
        task = kb.get_task(conn, payload["id"])
    assert task.base_commit == base_commit
    assert task.target_branch == "main"


@pytest.mark.parametrize(
    ("branch", "error"),
    [
        ("impl/{unknown}/{task_id}", "unknown branch placeholder"),
        ("bad..{task_id}", "invalid git branch"),
    ],
)
def test_create_rejects_unknown_placeholder_and_invalid_rendered_branch(
    kanban_home, tmp_path, branch, error,
):
    output = kc.run_slash(
        "create bad --workspace worktree:"
        f"{tmp_path / 'wt'} --branch '{branch}' --base-commit {'a' * 40}"
    )
    assert error in output
    with kb.connect() as conn:
        assert kb.list_tasks(conn) == []


def test_resolved_contract_is_shared_by_task_show_json_and_worker_context(
    kanban_home, tmp_path,
):
    repo, workspace, task = _materialize_contract_task(tmp_path)
    contracts = _contracts()
    contract = contracts.contract_for_task(task)

    assert contract["version"] == "workspace_contract.v1"
    assert contract["valid"] is True
    assert contract["repository"] == str(repo.resolve())
    assert contract["worktree"] == str(workspace.resolve())
    assert contract["common_dir"] == str((repo / ".git").resolve())
    assert contract["base_commit"] == _git(repo, "rev-parse", "HEAD")
    assert contract["target_branch"] == "main"
    assert contract["branch"] == f"impl/implementer/{task.id}/g1"
    assert contract["task_id"] == task.id
    assert contract["generation"] == 1
    assert contract["write_set"] is None
    assert contract["artifact_namespace"] is None
    assert task.common_dir == contract["common_dir"]

    task_json = kc._task_to_dict(task)
    show_json = json.loads(kc.run_slash(f"show {task.id} --json"))["task"]
    assert task_json["workspace_contract"] == contract
    assert show_json["workspace_contract"] == contract
    assert show_json["common_dir"] == contract["common_dir"]
    assert show_json["base_commit"] == contract["base_commit"]

    with kb.connect() as conn:
        context = kb.build_worker_context(conn, task.id)
    assert "workspace_contract.v1" in context
    assert json.dumps(contract, ensure_ascii=False, sort_keys=True) in context


def test_contract_resolver_fails_closed_for_dirty_branch_and_base_mismatch(
    kanban_home, tmp_path,
):
    _, workspace, task = _materialize_contract_task(tmp_path)
    contracts = _contracts()

    # Worktree dirty (untracked or tracked modifications) must NOT block claim.
    (workspace / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    tracked = workspace / "tracked.txt"
    tracked.write_text("v1\n", encoding="utf-8")
    _git(workspace, "add", "tracked.txt")
    _git(workspace, "commit", "-m", "add tracked.txt")
    tracked.write_text("dirty\n", encoding="utf-8")
    contracts.resolve_workspace_contract(task, workspace)  # should succeed
    _git(workspace, "checkout", "--", "tracked.txt")
    (workspace / "untracked.txt").unlink()

    _git(workspace, "checkout", "-b", "wrong-branch")
    with pytest.raises(contracts.WorkspaceContractError, match="branch mismatch"):
        contracts.resolve_workspace_contract(task, workspace)

    _git(workspace, "checkout", task.branch_name)
    bad_base = replace(task, base_commit="0" * 40)
    with pytest.raises(contracts.WorkspaceContractError, match="base commit"):
        contracts.resolve_workspace_contract(bad_base, workspace)


def test_contract_resolver_rejects_reachable_base_that_is_not_head_ancestor(
    kanban_home, tmp_path,
):
    repo, workspace, task = _materialize_contract_task(tmp_path)
    contracts = _contracts()

    (repo / "new-main.txt").write_text("new main\n", encoding="utf-8")
    _git(repo, "add", "new-main.txt")
    _git(repo, "commit", "-m", "advance main")
    newer_main = _git(repo, "rev-parse", "HEAD")

    assert _git(workspace, "rev-parse", "HEAD") != newer_main
    non_ancestor_base = replace(task, base_commit=newer_main)
    with pytest.raises(contracts.WorkspaceContractError, match="not an ancestor"):
        contracts.resolve_workspace_contract(non_ancestor_base, workspace)


def test_rework_generation_does_not_invalidate_workspace_contract(
    kanban_home, tmp_path,
):
    """Rework bumps task generation, but generation is not a workspace
    identity field.  The stored contract should remain valid and
    set_workspace_path should succeed without raising.
    """
    _, workspace, task = _materialize_contract_task(tmp_path)
    contracts = _contracts()

    with kb.connect() as conn:
        kb.return_task_for_rework(
            conn,
            task.id,
            actor="reviewer",
            reason="new generation",
        )
        reworked = kb.get_task(conn, task.id)
        # Contract should still be valid (generation excluded from comparison).
        stale_contract = contracts.contract_for_task(reworked)
        assert stale_contract["valid"] is True
        assert "generation" not in stale_contract["mismatches"]
        # set_workspace_path should succeed without raising.
        assert kb.set_workspace_path(conn, task.id, workspace)
        refreshed = kb.get_task(conn, task.id)

    final_contract = contracts.contract_for_task(refreshed)
    assert final_contract["valid"] is True
    assert final_contract["mismatches"] == []


def test_legacy_db_migrates_contract_columns_and_old_tasks_remain_readable(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE tasks ("
        "id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT, assignee TEXT, "
        "status TEXT NOT NULL, priority INTEGER DEFAULT 0, created_by TEXT, "
        "created_at INTEGER NOT NULL, started_at INTEGER, completed_at INTEGER, "
        "workspace_kind TEXT NOT NULL DEFAULT 'scratch', workspace_path TEXT, "
        "claim_lock TEXT, claim_expires INTEGER)"
    )
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('legacy', 'old task', 'ready', 1)"
    )
    conn.commit()
    conn.close()

    with kb.connect(db_path) as migrated:
        columns = {
            row["name"] for row in migrated.execute("PRAGMA table_info(tasks)")
        }
        task = kb.get_task(migrated, "legacy")

    assert {"base_commit", "target_branch", "workspace_contract_json"} <= columns
    assert task.base_commit is None
    assert task.target_branch is None
    assert task.workspace_contract_json is None
    assert task.common_dir is None


def test_render_branch_template_exposes_only_safe_identity_placeholders():
    contracts = _contracts()
    rendered = contracts.render_branch_template(
        "impl/{assignee}/{task_id}/g{generation}",
        task_id="t_demo123",
        generation=4,
        assignee="Implementer",
    )
    assert rendered == "impl/implementer/t_demo123/g4"
    assert re.fullmatch(r"[A-Za-z0-9._/-]+", rendered)
