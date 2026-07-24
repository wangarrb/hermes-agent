"""All interactive backends share one effective-role context renderer."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from plugins.kanban.claude_listener.claude_kanban_interactive import (
    ClaudeInteractiveListener,
)
from plugins.kanban.codex_listener.codex_kanban_interactive import (
    CodexInteractiveListener,
)
from plugins.kanban.deepseek_listener.deepseek_kanban_interactive import (
    CodeWhaleInteractiveListener,
)
from plugins.kanban.hermes_listener.hermes_kanban_interactive import (
    HermesInteractiveListener,
)
from plugins.kanban.reasonix_listener.reasonix_kanban_interactive import (
    ReasonixInteractiveListener,
)


BACKENDS = [
    ("hermes", HermesInteractiveListener),
    ("codex", CodexInteractiveListener),
    ("deepseek", CodeWhaleInteractiveListener),
    ("codewhale", CodeWhaleInteractiveListener),
    ("claude", ClaudeInteractiveListener),
    ("reasonix", ReasonixInteractiveListener),
]


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    prompt = (
        repo
        / ".hermes-kanban"
        / "board"
        / "implementer"
        / "kanban-system-prompt.md"
    )
    prompt.parent.mkdir(parents=True)
    prompt.write_text("TRUSTED PROJECT PROMPT\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "control prompt")
    control_commit = _git(repo, "rev-parse", "HEAD")
    prompt.write_text("WRITABLE TASK BRANCH PROMPT\n", encoding="utf-8")

    profiles = tmp_path / "profiles"
    role_root = profiles / "implementer"
    skill = role_root / "skills" / "delivery" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# Delivery\nRun focused tests.\n", encoding="utf-8")
    skill_sha = hashlib.sha256(skill.read_bytes()).hexdigest()
    (role_root / "role-context.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "role": "implementer",
                "description": "Implement the frozen contract.",
                "skills": [
                    {
                        "name": "delivery",
                        "path": "skills/delivery/SKILL.md",
                        "sha256": skill_sha,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    shared = tmp_path / "shared-skills"
    task_skill = shared / "tdd" / "SKILL.md"
    task_skill.parent.mkdir(parents=True)
    task_skill.write_text("# TDD\nRed then green.\n", encoding="utf-8")

    contract = {
        "version": "workspace_contract.v1",
        "valid": True,
        "repository": str(repo),
        "worktree": str(repo),
        "common_dir": str(repo / ".git"),
        "base_commit": control_commit,
        "target_branch": "main",
        "branch": "implementer/t_demo/g1",
        "task_id": "t_demo",
        "generation": 1,
        "write_set": ["plugins/kanban"],
        "artifact_namespace": "/tmp/designer-role/H8/g1",
    }
    task = SimpleNamespace(
        id="t_demo",
        assignee="implementer",
        skills=["tdd"],
        workspace_contract=contract,
    )
    return repo, profiles, shared, task, control_commit, skill


@pytest.mark.parametrize("backend,listener_class", BACKENDS)
def test_primary_and_assist_backends_share_structured_role_context(
    tmp_path, backend, listener_class,
):
    repo, profiles, shared, task, control_commit, _ = _fixture(tmp_path)
    listener = listener_class()
    output = tmp_path / backend / "role-context.json"

    rendered = listener.render_effective_role_context(
        board="board",
        workspace=repo,
        pane_profile="coordinator",
        task=task,
        output_path=output,
        profiles_root=profiles,
        shared_skills_root=shared,
        backend=backend,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["pane_profile"] == "coordinator"
    assert payload["effective_role"] == "implementer"
    assert payload["assist"] is True
    assert payload["fixed_control_commit"] == control_commit
    assert payload["workspace_contract"] == task.workspace_contract
    assert payload["project_prompt"]["content"] == "TRUSTED PROJECT PROMPT\n"
    assert payload["control_prompt_sha256"] == payload["project_prompt"]["sha256"]
    assert (
        f"HERMES_KANBAN_CONTROL_PROMPT_SHA256={payload['control_prompt_sha256']}"
        in rendered
    )
    assert "WRITABLE TASK BRANCH PROMPT" not in rendered
    assert payload["declared_skills"][0]["name"] == "delivery"
    assert payload["declared_skills"][0]["sha256"]
    assert payload["task_skills"][0]["name"] == "tdd"
    assert payload["sources"]["role_manifest"]["path"]
    assert payload["sources"]["role_manifest"]["sha256"]


def test_task_assignee_is_effective_role_for_primary_claim(tmp_path):
    repo, profiles, shared, task, _, _ = _fixture(tmp_path)
    listener = CodexInteractiveListener()
    output = tmp_path / "primary" / "role-context.json"

    listener.render_effective_role_context(
        board="board",
        workspace=repo,
        pane_profile="implementer",
        task=task,
        output_path=output,
        profiles_root=profiles,
        shared_skills_root=shared,
        backend="codex",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["effective_role"] == "implementer"
    assert payload["assist"] is False


@pytest.mark.parametrize("pane_profile", ["implementer", "coordinator"])
def test_claim_path_emits_manifest_and_exposes_control_sha_in_prompt(
    tmp_path, monkeypatch, pane_profile,
):
    db_path = tmp_path / "kanban.db"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    listener = CodexInteractiveListener()
    listener._board = "board"
    listener._profile = pane_profile
    listener._workspace = workspace
    listener._log_path = tmp_path / "listener.log"
    injected: list[str] = []
    rendered_calls = []

    monkeypatch.setattr(listener, "on_claim_pre_check", lambda args, log_path: True)
    monkeypatch.setattr(listener, "on_claim_post_confirm", lambda args, log_path: True)

    def render_context(**kwargs):
        rendered_calls.append(kwargs)
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_text(
            json.dumps({"control_prompt_sha256": "a" * 64}),
            encoding="utf-8",
        )
        return f"HERMES_KANBAN_CONTROL_PROMPT_SHA256={'a' * 64}"

    monkeypatch.setattr(listener, "render_effective_role_context", render_context)
    monkeypatch.setattr(listener, "inject_text", lambda **kwargs: str(kwargs["prompt_path"]))
    listener_base = sys.modules[type(listener).__mro__[1].__module__]
    monkeypatch.setattr(
        listener_base,
        "zellij_inject",
        lambda **kwargs: injected.append(kwargs["text"]) or True,
    )
    monkeypatch.setattr(listener_base, "zellij_rename_pane", lambda **kwargs: True)

    kb.init_db(db_path)
    with kb.connect(db_path) as conn:
        task_id = kb.create_task(
            conn,
            title="role context claim",
            assignee="implementer",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        claimed_id, run_id = listener.claim_and_inject_one(
            Namespace(
                profile=pane_profile,
                claim_assignees="implementer",
                assist_role=None,
                zellij_session="test-session",
                zellij_pane_id="7",
                workspace=str(workspace),
                board="board",
                ttl=900,
            ),
            log_path=listener._log_path,
            conn=conn,
        )

    assert claimed_id == task_id
    assert run_id is not None
    assert rendered_calls[0]["task"].assignee == "implementer"
    manifest = rendered_calls[0]["output_path"]
    assert manifest.name == "role-context.json"
    assert manifest.is_file()
    prompt = Path(injected[0]).read_text(encoding="utf-8")
    assert f"HERMES_KANBAN_CONTROL_PROMPT_SHA256={'a' * 64}" in prompt


def test_declared_skill_sha_mismatch_fails_visibly_and_emits_json(tmp_path):
    repo, profiles, shared, task, _, skill = _fixture(tmp_path)
    skill.write_text("tampered\n", encoding="utf-8")
    listener = HermesInteractiveListener()
    output = tmp_path / "blocked" / "role-context.json"

    with pytest.raises(RuntimeError, match="sha_mismatch:declared_skill:delivery"):
        listener.render_effective_role_context(
            board="board",
            workspace=repo,
            pane_profile="coordinator",
            task=task,
            output_path=output,
            profiles_root=profiles,
            shared_skills_root=shared,
            backend="hermes",
        )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert "sha_mismatch:declared_skill:delivery" in payload["errors"]


def test_missing_fixed_commit_prompt_fails_instead_of_reading_workspace(tmp_path):
    repo, profiles, shared, task, _, _ = _fixture(tmp_path)
    task.workspace_contract = {
        **task.workspace_contract,
        "base_commit": "f" * 40,
    }
    listener = ClaudeInteractiveListener()
    output = tmp_path / "missing" / "role-context.json"

    with pytest.raises(RuntimeError, match="fixed_control_prompt"):
        listener.render_effective_role_context(
            board="board",
            workspace=repo,
            pane_profile="coordinator",
            task=task,
            output_path=output,
            profiles_root=profiles,
            shared_skills_root=shared,
            backend="claude",
        )

    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "blocked"
