"""Generation lifecycle for long-lived Kanban Git worktrees."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_workspace_contract as contracts


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(path: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


def _init_upstream(tmp_path: Path):
    publisher = tmp_path / "publisher"
    publisher.mkdir()
    _git(publisher, "init")
    _git(publisher, "config", "user.email", "test@example.com")
    _git(publisher, "config", "user.name", "Test User")
    _git(publisher, "branch", "-M", "main")
    (publisher / "README.md").write_text("base\n", encoding="utf-8")
    _git(publisher, "add", "README.md")
    _git(publisher, "commit", "-m", "base")
    first = _git(publisher, "rev-parse", "HEAD")

    upstream = tmp_path / "upstream.git"
    subprocess.run(
        ["git", "init", "--bare", str(upstream)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(publisher, "remote", "add", "origin", str(upstream))
    _git(publisher, "push", "-u", "origin", "main")
    _git(upstream, "symbolic-ref", "HEAD", "refs/heads/main")

    repo = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", str(upstream), str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    return publisher, upstream, repo, first


def _advance_upstream(publisher: Path, label: str) -> str:
    path = publisher / f"{label}.txt"
    path.write_text(f"{label}\n", encoding="utf-8")
    _git(publisher, "add", path.name)
    _git(publisher, "commit", "-m", label)
    _git(publisher, "push", "origin", "main")
    return _git(publisher, "rev-parse", "HEAD")


def _create_task_and_long_worktree(
    conn,
    repo: Path,
    tmp_path: Path,
    *,
    base_commit: str,
    seed_branch: str | None = None,
    precreate_delivery_branch: bool = False,
):
    worktree = tmp_path / "long-lived-worktree"
    task_id = kb.create_task(
        conn,
        title="generation lifecycle",
        assignee="implementer",
        workspace_kind="worktree",
        workspace_path=str(worktree),
        branch_name="implementer/{task_id}/g{generation}",
        base_commit=base_commit,
        target_branch="main",
    )
    task = kb.get_task(conn, task_id)
    seed = seed_branch or f"implementer/{task.id}/g0"
    _git(repo, "branch", seed, base_commit)
    if precreate_delivery_branch:
        _git(repo, "branch", task.branch_name, base_commit)
    _git(repo, "worktree", "add", str(worktree), seed)
    return task, worktree


def test_fetched_base_resolution_and_explicit_override(kanban_home, tmp_path):
    publisher, _, repo, first = _init_upstream(tmp_path)
    latest = _advance_upstream(publisher, "upstream-2")

    fetched = contracts.resolve_fetched_base(
        repo, source_branch="main", upstream="origin",
    )
    explicit = contracts.resolve_fetched_base(
        repo,
        source_branch="main",
        upstream="origin",
        base_commit=first,
    )

    assert fetched["base_commit"] == latest
    assert fetched["upstream_sha"] == latest
    assert fetched["source_branch"] == "main"
    assert explicit["base_commit"] == first
    assert explicit["upstream_sha"] == latest


def test_prepare_switches_clean_long_lived_worktree_and_resumes_exact_manifest(
    kanban_home, tmp_path,
):
    _, _, repo, base = _init_upstream(tmp_path)
    with kb.connect() as conn:
        task, worktree = _create_task_and_long_worktree(
            conn,
            repo,
            tmp_path,
            base_commit=base,
            precreate_delivery_branch=True,
        )
        prepared = kb.prepare_task_worktree(
            conn,
            task.id,
            source_branch="main",
            upstream="origin",
        )
        persisted = kb.get_task(conn, task.id)

    assert _git(worktree, "branch", "--show-current") == task.branch_name
    assert _git(worktree, "rev-parse", "HEAD") == base
    assert prepared["version"] == "workspace_contract.v1"
    assert prepared["branch"] == task.branch_name
    assert prepared["generation"] == 1
    assert prepared["source_branch"] == "main"
    assert prepared["upstream"] == "origin"
    assert persisted.workspace_contract == prepared

    manifest_path = Path(prepared["manifest_path"])
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == prepared
    assert not list(manifest_path.parent.glob(f".{manifest_path.name}.*.tmp"))
    before_mtime = manifest_path.stat().st_mtime_ns

    resumed = contracts.prepare_generation_worktree(
        persisted,
        worktree,
        source_branch="main",
        upstream="origin",
    )
    assert resumed == prepared
    assert manifest_path.stat().st_mtime_ns == before_mtime


@pytest.mark.parametrize("failure", ["dirty", "ownership", "base"])
def test_prepare_fails_closed_without_reset_stash_or_delete(
    kanban_home, tmp_path, failure,
):
    _, _, repo, base = _init_upstream(tmp_path)
    with kb.connect() as conn:
        seed = "foreign/owned" if failure == "ownership" else None
        task, worktree = _create_task_and_long_worktree(
            conn,
            repo,
            tmp_path,
            base_commit=base,
            seed_branch=seed,
        )

    original_branch = _git(worktree, "branch", "--show-current")
    dirty_file = worktree / "keep-me.txt"
    if failure == "dirty":
        dirty_file.write_text("must survive\n", encoding="utf-8")
    override = "0" * 40 if failure == "base" else None
    expected = {
        "dirty": "clean",
        "ownership": "ownership",
        "base": "base commit",
    }[failure]

    with pytest.raises(contracts.WorkspaceContractError, match=expected):
        contracts.prepare_generation_worktree(
            task,
            worktree,
            source_branch="main",
            upstream="origin",
            base_commit=override,
        )

    assert _git(worktree, "branch", "--show-current") == original_branch
    if failure == "dirty":
        assert dirty_file.read_text(encoding="utf-8") == "must survive\n"
        assert "keep-me.txt" in _git(worktree, "status", "--porcelain")


def test_generation_two_rejects_generation_one_manifest(kanban_home, tmp_path):
    _, _, repo, base = _init_upstream(tmp_path)
    with kb.connect() as conn:
        task, worktree = _create_task_and_long_worktree(
            conn, repo, tmp_path, base_commit=base,
        )
        prepared = kb.prepare_task_worktree(
            conn, task.id, source_branch="main", upstream="origin",
        )
        conn.execute(
            "UPDATE tasks SET generation = 2, branch_name = ? WHERE id = ?",
            (f"implementer/{task.id}/g2", task.id),
        )
        conn.commit()
        generation_two = kb.get_task(conn, task.id)

    with pytest.raises(contracts.WorkspaceContractError, match="generation"):
        contracts.prepare_generation_worktree(
            generation_two,
            worktree,
            source_branch="main",
            upstream="origin",
        )
    assert json.loads(
        Path(prepared["manifest_path"]).read_text(encoding="utf-8")
    )["generation"] == 1


def test_freeze_records_delivery_and_release_preserves_delivery_branch(
    kanban_home, tmp_path,
):
    _, _, repo, base = _init_upstream(tmp_path)
    with kb.connect() as conn:
        task, worktree = _create_task_and_long_worktree(
            conn, repo, tmp_path, base_commit=base,
        )
        kb.prepare_task_worktree(
            conn, task.id, source_branch="main", upstream="origin",
        )
        task = kb.get_task(conn, task.id)

        delivery_file = worktree / "delivery.txt"
        delivery_file.write_text("delivery\n", encoding="utf-8")
        with pytest.raises(contracts.WorkspaceContractError, match="clean"):
            contracts.freeze_delivery(task, worktree)
        _git(worktree, "add", "delivery.txt")
        _git(worktree, "commit", "-m", "delivery")

        frozen = kb.freeze_task_delivery(conn, task.id)
        delivery_commit = _git(worktree, "rev-parse", "HEAD")
        delivery_tree = _git(worktree, "rev-parse", "HEAD^{tree}")
        assert frozen["delivery_commit"] == delivery_commit
        assert frozen["delivery_tree"] == delivery_tree

        released = kb.release_task_worktree_lease(conn, task.id)
        persisted = kb.get_task(conn, task.id)

    assert _git(worktree, "branch", "--show-current") == ""
    assert _git(repo, "show-ref", "--verify", f"refs/heads/{task.branch_name}")
    assert _git(repo, "rev-parse", task.branch_name) == delivery_commit
    assert released["lease_released"] is True
    assert released["delivery_commit"] == delivery_commit
    assert persisted.workspace_contract == released


def test_upstream_change_during_resolution_is_rejected(
    kanban_home, tmp_path, monkeypatch,
):
    publisher, _, repo, _ = _init_upstream(tmp_path)
    original_snapshot = contracts._snapshot_source_refs
    calls = 0

    def racing_snapshot(repository, *, source_branch, upstream):
        nonlocal calls
        calls += 1
        if calls == 2:
            _advance_upstream(publisher, "race")
            _git(repo, "fetch", "origin", "main")
        return original_snapshot(
            repository,
            source_branch=source_branch,
            upstream=upstream,
        )

    monkeypatch.setattr(contracts, "_snapshot_source_refs", racing_snapshot)
    with pytest.raises(contracts.WorkspaceContractError, match="changed during resolution"):
        contracts.resolve_fetched_base(
            repo, source_branch="main", upstream="origin",
        )
